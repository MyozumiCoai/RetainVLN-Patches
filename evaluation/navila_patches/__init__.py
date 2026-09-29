from .obs_strategy import sample_and_pad_images, describe as describe_obs
from .cot_prompt import build_question, parse_cot_output, describe as describe_cot

__all__ = [
    "sample_and_pad_images",
    "build_question",
    "parse_cot_output",
    "describe_obs",
    "describe_cot",
]


def __getattr__(name):
    if name in ("monkey_patch", "PatchedNaVILATrainer"):
        from . import apply_patch as _ap
        return getattr(_ap, name)
    raise AttributeError(f"module 'navila_patches' has no attribute {name!r}")
