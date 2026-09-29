from __future__ import annotations

import copy
import math
import os
import re
import time

import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

from .obs_strategy import sample_and_pad_images, describe as describe_obs
from .cot_prompt import build_question, parse_cot_output, describe as describe_cot
from .perturb import load_injector
from .trace import TraceRecorder, softmax_entropy_first


_FALLBACK_KEYWORDS = {
    0: re.compile(r"\bstop\b", re.IGNORECASE),
    1: re.compile(r"\bis move forward\b", re.IGNORECASE),
    2: re.compile(r"\bis turn left\b", re.IGNORECASE),
    3: re.compile(r"\bis turn right\b", re.IGNORECASE),
}


def _fallback_action(outputs: str) -> int:
    for k, pat in _FALLBACK_KEYWORDS.items():
        if pat.search(outputs):
            return k
    return 1



def _m4_heading_from_quat(rotation) -> float:
    x, y, z, w = [float(v) for v in rotation]
    siny = 2.0 * (w * y + x * z)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def _m4_new_manager(budget_tokens: int):
    from scripts.m4_memory.manager import MemoryManager

    return MemoryManager(budget_tokens=budget_tokens)


def _navcot_question(
    instruction: str,
    num_images: int,
    scene_id: str,
    timestep: int,
    heading_rad=None,
    geodesic=None,
) -> str:
    tokens = "\n".join(DEFAULT_IMAGE_TOKEN for _ in range(num_images))
    chunks = [f"RGB observations:\n{tokens}\n"]
    chunks.append(f"Scene {scene_id} | timestep {timestep}.")
    if heading_rad is not None:
        chunks.append(f"Heading (rad): {heading_rad:.3f}.")
    if geodesic is not None and geodesic == geodesic:
        chunks.append(f"Geodesic distance: {geodesic:.2f}m.")
    chunks.append(
        "You are a quadruped robot navigating a Matterport3D house. "
        "Fuse the provided RGB, Depth-Anything v2, and point cloud cues to reason about the next action."
    )
    chunks.append(f'Instruction: "{instruction}"')
    return "\n".join(chunks)


class PatchedNaVILATrainer:

    def __init__(self, config=None, num_chunks: int = 1, chunk_idx: int = 0):
        self.config = config
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self.device = None

    def _eval_checkpoint(self, checkpoint_path: str, writer) -> None:
        from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
        from habitat_baselines.common.environments import get_env_class
        from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
        from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
        from vlnce_baselines.common.utils import extract_instruction_tokens
        from llava.model.builder import load_pretrained_model

        import json

        logger.info(f"[patched] checkpoint_path: {checkpoint_path}")

        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            checkpoint_path, model_name, device_map="cpu"
        )
        model = model.cuda()

        config = self.config.clone()
        split = config.EVAL.SPLIT
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = self.num_chunks
        config.TASK_CONFIG.DATASET.CHUNK_IDX = self.chunk_idx
        config.RESULTS_DIR = os.path.join(
            config.RESULTS_DIR, model_name, config.TASK_CONFIG.DATASET.TYPE, config.TASK_CONFIG.DATASET.SPLIT
        )
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        memory_mode = os.environ.get("NAVI_MEMORY", "off")
        memory_budget = int(os.environ.get("NAVI_MEMORY_BUDGET", "400"))
        navcot_prompt = os.environ.get("NAVI_PROMPT", "navila") == "navcot"
        trace_on = os.environ.get("NAVI_TRACE", "0") == "1"
        if trace_on:
            logger.info("[patched] NAVI_TRACE=1: 逐步轨迹 + 动作熵记录开启")
        exp_lib = None
        exp_inject_mod = None
        exp_k = int(os.environ.get("NAVI_EXP_K", "5"))
        _exp_path = os.environ.get("NAVI_EXP", "")
        if _exp_path:
            from . import exp_inject as exp_inject_mod
            exp_lib = exp_inject_mod.load_library(_exp_path)
            logger.info(
                f"[patched] NAVI_EXP on: {_exp_path} "
                f"({len(exp_lib['rules'])} rules, top-{exp_k})"
            )
        if memory_mode != "off" or navcot_prompt or trace_on:
            from . import m4_measure

            config.TASK_CONFIG.TASK.MEASUREMENTS.append("M4_POSE")
            m4_pose_cfg = type(config.TASK_CONFIG.TASK.SUCCESS)()
            m4_pose_cfg.TYPE = "M4Pose"
            config.TASK_CONFIG.TASK.M4_POSE = m4_pose_cfg
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR, f"{split}_{self.num_chunks}-{self.chunk_idx}.json"
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes = {}
        past_rgbs = [[] for _ in range(envs.num_envs)]
        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)
        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_path}]" " [Episodes evaluated: {evaluated}/{total}]" " [Time elapsed (s): {time}]"
        )
        start_time = time.time()
        assert envs.num_envs == 1

        queue_actions = []
        cot_max_tokens = int(os.environ.get("NAVI_COT_MAX_TOKENS", "256"))
        cot_enabled = os.environ.get("NAVI_COT", "0") == "1"
        recorder = None
        t_env_step = 0
        iter_kind = "?"
        if trace_on:
            trace_path = os.path.join(
                config.RESULTS_DIR, f"trace_{split}_{self.num_chunks}-{self.chunk_idx}.jsonl"
            )
            recorder = TraceRecorder(trace_path)
            logger.info(f"[patched] trace file: {trace_path}")

        perturb = load_injector()
        if perturb is not None:
            logger.info(
                f"[patched] T2 perturb on: p={perturb.p} deg={perturb.deg} "
                f"log={perturb.log_path}"
            )
        step_cnt = 0

        memory_mgr = _m4_new_manager(memory_budget) if memory_mode != "off" else None
        memory_step = 0
        if memory_mgr is not None:
            logger.info(f"[patched] M4 memory on: mode={memory_mode}, budget={memory_budget}")
        navcot_step = 0
        navcot_pose = None
        if navcot_prompt:
            logger.info("[patched] NAVI_PROMPT=navcot: 评测问题复刻训练格式")

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()
            forced = None
            dec_info = None
            iter_kind = "decision"

            if len(queue_actions) > 0:
                iter_kind = "queue"
                print(f"using queue...{queue_actions[0]}")
                outputs = envs.step([queue_actions[0]])
                dec_info = {"action": None, "q": queue_actions[0]}
                queue_actions.pop(0)
            else:
                with torch.no_grad():
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")
                    past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
                    num_video_frames = model.config.num_video_frames
                    past_and_current_rgbs = sample_and_pad_images(
                        past_and_current_rgbs,
                        num_frames=num_video_frames,
                    )

                    instruction = current_episodes[0].instruction.instruction_text
                    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)

                    if navcot_prompt:
                        pose = navcot_pose
                        if pose is None:
                            ep = current_episodes[0]
                            pose = {
                                "rotation": list(ep.start_rotation),
                                "geodesic_to_goal": float("nan"),
                            }
                        scene_id = os.path.basename(
                            str(current_episodes[0].scene_id)
                        ).replace(".glb", "")
                        question = _navcot_question(
                            instruction=instruction,
                            num_images=len(past_and_current_rgbs),
                            scene_id=scene_id,
                            timestep=navcot_step + 1,
                            heading_rad=_m4_heading_from_quat(pose["rotation"]),
                            geodesic=pose.get("geodesic_to_goal"),
                        )
                        navcot_step += 1
                    else:
                        memory_text = ""
                        if memory_mgr is not None:
                            memory_text = memory_mgr.context_text(
                                include_notebook=(memory_mode == "level2")
                            )
                        question = build_question(instruction=instruction, interleaved_images=interleaved_images)
                        if memory_text:
                            question = memory_text + "\n" + question
                        if exp_lib is not None:
                            _exp_rules = exp_inject_mod.retrieve(exp_lib, instruction, top_k=exp_k)
                            question = exp_inject_mod.format_context(_exp_rules) + "\n" + question

                    conv_mode = "llama_3"
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(
                        model.device, dtype=torch.float16
                    )
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        gen_kwargs = dict(
                            images=images_tensor.half().cuda(),
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=cot_max_tokens if (cot_enabled or navcot_prompt) else 32,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        ent_first = top1_p = None
                        if recorder is not None:
                            gen = model.generate(
                                input_ids,
                                return_dict_in_generate=True,
                                output_scores=True,
                                **gen_kwargs,
                            )
                            output_ids = gen.sequences
                            if gen.scores:
                                fe = softmax_entropy_first(gen.scores[0][0])
                                if fe is not None:
                                    ent_first, top1_p = fe["ent_first"], fe["top1_p"]
                        else:
                            output_ids = model.generate(input_ids, **gen_kwargs)
                    outputs_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    if outputs_text.endswith(stop_str):
                        outputs_text = outputs_text[: -len(stop_str)]
                    outputs_text = outputs_text.strip()
                    new_tokens = output_ids.shape[1] - input_ids.shape[1]
                    print(
                        f"[VLM] newtok={new_tokens} has_action={'<action>' in outputs_text} "
                        f"head={outputs_text[:150]!r} tail={outputs_text[-120:]!r}",
                        flush=True,
                    )

                    action, extras = parse_cot_output(outputs_text)
                    if action is None:
                        action = _fallback_action(outputs_text)
                    distance = extras.get("distance")
                    degree = extras.get("degree")
                    actions = [action]
                    step_cnt += 1
                    dec_info = {
                        "action": action,
                        "distance": distance,
                        "degree": degree,
                        "q": None,
                        "raw": outputs_text[:200],
                        "think": (extras.get("think") or "")[:200],
                        "ent_first": ent_first,
                        "top1_p": top1_p,
                    }

                if perturb is not None:
                    forced = perturb.maybe_force()
                if forced is not None:
                    side, n = forced
                    outputs = envs.step([side])
                    for _ in range(n - 1):
                        outputs = envs.step([side])
                elif actions[0] == 1:
                    if distance is None:
                        distance = 25
                    if (distance % 25) != 0:
                        distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                    outputs = envs.step([1])
                    for _ in range(int(distance // 25) - 1):
                        queue_actions.append(1)
                elif actions[0] == 2:
                    if degree is None:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([2])
                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(2)
                elif actions[0] == 3:
                    if degree is None:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([3])
                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(3)
                else:
                    outputs = envs.step(actions)

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            if recorder is not None:
                t_env_step += 1
                _epc = current_episodes[0]
                _pose = infos[0].get("m4_pose")
                _pos = list(_pose["position"]) if _pose else None
                _rot = list(_pose["rotation"]) if _pose else None
                _hdg = _m4_heading_from_quat(_pose["rotation"]) if _pose else None
                recorder.write(
                    kind=iter_kind,
                    ep=str(_epc.episode_id),
                    scene=os.path.basename(str(_epc.scene_id)).replace(".glb", ""),
                    t_env=t_env_step,
                    t_dec=step_cnt,
                    action=dec_info.get("action") if dec_info else None,
                    q=dec_info.get("q") if dec_info else None,
                    distance=dec_info.get("distance") if dec_info else None,
                    degree=dec_info.get("degree") if dec_info else None,
                    ent_first=dec_info.get("ent_first") if dec_info else None,
                    top1_p=dec_info.get("top1_p") if dec_info else None,
                    raw=dec_info.get("raw") if dec_info else None,
                    think=dec_info.get("think") if dec_info else None,
                    pos=_pos,
                    heading=_hdg,
                    geo=float(infos[0].get("distance_to_goal") or float("nan")),
                )

            if perturb is not None:
                _ep = current_episodes[0].episode_id
                _geo = float(infos[0].get("distance_to_goal", 0.0) or 0.0)
                if forced is not None:
                    perturb.log_perturb(_ep, step_cnt, forced[0], _geo)
                perturb.log_step(_ep, step_cnt, _geo)

            if navcot_prompt:
                _p = infos[0].get("m4_pose")
                if _p is not None:
                    navcot_pose = _p
                if dones[0]:
                    navcot_pose = None
                    navcot_step = 0

            if memory_mgr is not None:
                try:
                    pose = infos[0].get("m4_pose")
                    if pose is None:
                        raise KeyError("infos[0] 缺 m4_pose（M4_POSE measure 未生效？）")
                    pos = list(pose["position"])
                    heading = _m4_heading_from_quat(pose["rotation"])
                    memory_step += 1
                    if memory_step == 1:
                        print(
                            f"[M4] first update ok: pos={[round(v,2) for v in pos]} "
                            f"heading={heading:.2f}",
                            flush=True,
                        )
                    memory_mgr.update(
                        pos,
                        heading,
                        memory_step,
                        done_info=infos[0] if dones[0] else None,
                    )
                    if dones[0]:
                        memory_mgr = _m4_new_manager(memory_budget)
                        memory_step = 0
                except Exception as e:
                    logger.warn(f"[patched] M4 memory update failed: {e}")

            for i in range(envs.num_envs):
                past_rgbs[i].append(Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB"))
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(frame, current_episodes[i].instruction.instruction_text)
                    rgb_frames[i].append(frame)
                if not dones[i]:
                    continue
                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                past_rgbs[i] = []
                queue_actions = []
                t_env_step = 0
                if perturb is not None:
                    perturb.end_episode(ep_id)
                step_cnt = 0
                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(log_str.format(
                        evaluated=len(stats_episodes), total=num_eps, time=round(time.time() - start_time)
                    ))
                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx="0",
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause, next_episodes = [], envs.current_episodes()
            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)
            (envs, batch, rgb_frames,) = self._pause_envs(envs_to_pause, envs, batch, rgb_frames)

        envs.close()
        if recorder is not None:
            recorder.close()
            logger.info(f"[patched] NAVI_TRACE closed: {recorder.n} rows → {recorder.path}")
        if perturb is not None:
            perturb.close()
            logger.info(f"[patched] T2 perturb stats: {perturb.stats()}")
        if config.use_pbar:
            pbar.close()
        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)

    @staticmethod
    def _pause_envs(envs_to_pause, envs, batch, rgb_frames=None):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            for k, v in batch.items():
                batch[k] = v[state_index]
            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]
        return envs, batch, rgb_frames



_PATCHED = False


def monkey_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        from vlnce_baselines import navila_trainer as nt
    except ImportError as e:
        raise ImportError(
            "monkey_patch() 必须在 NaVILA 环境下调用："
            "conda activate <navila-env> 且 cd 到 NaVILA 仓库根目录后 import。"
        ) from e

    nt.sample_and_pad_images = sample_and_pad_images
    nt.NaVILATrainer._eval_checkpoint = PatchedNaVILATrainer._eval_checkpoint
    nt.NaVILATrainer._pause_envs = staticmethod(PatchedNaVILATrainer._pause_envs)

    _PATCHED = True
    logger.info("[navila_patches] monkey_patch applied: " + describe_obs() + " | " + describe_cot())
