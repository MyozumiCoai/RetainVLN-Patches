from __future__ import annotations

import os
import re

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

_LOOSE_PATTERNS = {
    0: re.compile(r"\bstop\b", re.IGNORECASE),
    1: re.compile(r"\bmove forward\b", re.IGNORECASE),
    2: re.compile(r"\bturn left\b", re.IGNORECASE),
    3: re.compile(r"\bturn right\b", re.IGNORECASE),
}
_NAVI_FALLBACK_PATTERNS = {
    0: re.compile(r"\bstop\b", re.IGNORECASE),
    1: re.compile(r"\bis move forward\b", re.IGNORECASE),
    2: re.compile(r"\bis turn left\b", re.IGNORECASE),
    3: re.compile(r"\bis turn right\b", re.IGNORECASE),
}

_FORWARD_RE = re.compile(r"move forward (\d+) cm", re.IGNORECASE)
_TURN_LEFT_RE = re.compile(r"turn left (\d+) degree", re.IGNORECASE)
_TURN_RIGHT_RE = re.compile(r"turn right (\d+) degree", re.IGNORECASE)


def _cfg():
    return {
        "enabled":   os.environ.get("NAVI_COT", "0") == "1",
        "max_tokens": int(os.environ.get("NAVI_COT_MAX_TOKENS", "256")),
        "template":  os.environ.get("NAVI_COT_TEMPLATE", "minimal"),
    }



_BASE_PROMPT = (
    'Imagine you are a robot programmed for navigation tasks. You have been given a video '
    'of historical observations {interleaved_images}, and current observation <image>\n. '
    'Your assigned task is: "{instruction}" '
    'Analyze this series of images to decide your next action, which could be turning left or right by a specific '
    'degree, moving forward a certain distance, or stop if the task is completed.'
)


_COT_SUFFIX_MINIMAL = (
    "\n\nFirst, briefly reason about the current scene and your progress toward the task "
    "(1-2 sentences). Then output exactly one action in the required format.\n"
    "Output format:\n"
    "<think>your brief reasoning here</think>\n"
    "<action>the next action, e.g. \"move forward 25 cm\", \"turn left 15 degree\", or \"stop\"</action>"
)

_COT_SUFFIX_DETAILED = (
    "\n\nReason step by step before acting:\n"
    "1. Scene: what do you see in the current frame (objects, layout)?\n"
    "2. Progress: how far have you come; are you approaching the goal?\n"
    "3. Decision: what is the next best action?\n\n"
    "Output format:\n"
    "<think>step-by-step reasoning covering the three points above</think>\n"
    "<action>the next action, e.g. \"move forward 25 cm\", \"turn left 15 degree\", or \"stop\"</action>"
)


def build_question(instruction: str, interleaved_images: str) -> str:
    base = _BASE_PROMPT.format(instruction=instruction, interleaved_images=interleaved_images)
    cfg = _cfg()
    if not cfg["enabled"]:
        return base
    suffix = _COT_SUFFIX_DETAILED if cfg["template"] == "detailed" else _COT_SUFFIX_MINIMAL
    return base + suffix



def parse_cot_output(outputs: str) -> tuple[int | None, dict]:
    extras = {"distance": None, "degree": None, "think": None, "action_text": None}

    think_m = THINK_RE.search(outputs)
    if think_m:
        extras["think"] = think_m.group(1).strip()

    action_m = ACTION_RE.search(outputs)
    if action_m:
        action_text = action_m.group(1).strip()
        action = _match_action(action_text, _LOOSE_PATTERNS)
    else:
        action_text = outputs
        action = None

    extras["action_text"] = action_text

    if action is None:
        action = _match_action(outputs, _NAVI_FALLBACK_PATTERNS)
    if action is None:
        action = _match_action(outputs, _LOOSE_PATTERNS)

    if action == 1:
        m = _FORWARD_RE.search(action_text) or _FORWARD_RE.search(outputs)
        if m:
            extras["distance"] = int(m.group(1))
    elif action == 2:
        m = _TURN_LEFT_RE.search(action_text) or _TURN_LEFT_RE.search(outputs)
        if m:
            extras["degree"] = int(m.group(1))
    elif action == 3:
        m = _TURN_RIGHT_RE.search(action_text) or _TURN_RIGHT_RE.search(outputs)
        if m:
            extras["degree"] = int(m.group(1))

    return action, extras


def _match_action(text: str, patterns: dict) -> int | None:
    for k, pat in patterns.items():
        if pat.search(text):
            return k
    return None


def describe() -> str:
    cfg = _cfg()
    return f"cot(enabled={cfg['enabled']}, template={cfg['template']}, max_tokens={cfg['max_tokens']})"
