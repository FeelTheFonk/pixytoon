"""Scheduler factory — maps string identifiers to diffusers scheduler instances."""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("sddj.scheduler_factory")

# Scheduler registry: name -> (class_path, extra_kwargs)
SCHEDULER_REGISTRY = {
    "dpm++_sde_karras": {
        "cls": "DPMSolverMultistepScheduler",
        "kwargs": {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    },
    "dpm++_2m_karras": {
        "cls": "DPMSolverMultistepScheduler",
        "kwargs": {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
    },
    "ddim": {
        "cls": "DDIMScheduler",
        "kwargs": {},
    },
    "euler_a": {
        "cls": "EulerAncestralDiscreteScheduler",
        "kwargs": {},
    },
    "euler": {
        "cls": "EulerDiscreteScheduler",
        "kwargs": {},
    },
    "unipc": {
        "cls": "UniPCMultistepScheduler",
        "kwargs": {},
    },
    "lms": {
        "cls": "LMSDiscreteScheduler",
        "kwargs": {},
    },
}

SCHEDULER_NAMES = list(SCHEDULER_REGISTRY.keys())


def create_scheduler(name: str, base_config: dict) -> object:
    """Create a scheduler instance from a registry name and base config.

    Args:
        name: Scheduler name (e.g. "dpm++_sde_karras")
        base_config: Base scheduler config dict (from pipe.scheduler.config)

    Returns:
        Scheduler instance ready for pipeline assignment.
    """
    entry = SCHEDULER_REGISTRY.get(name)
    if entry is None:
        log.warning("Unknown scheduler '%s', falling back to dpm++_sde_karras", name)
        entry = SCHEDULER_REGISTRY["dpm++_sde_karras"]

    cls_name = entry["cls"]
    extra_kwargs = entry["kwargs"]

    # Dynamic import from diffusers
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    cls_map = {
        "DDIMScheduler": DDIMScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "LMSDiscreteScheduler": LMSDiscreteScheduler,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
    }

    scheduler_cls = cls_map[cls_name]
    return scheduler_cls.from_config(
        base_config,
        timestep_spacing="trailing",
        **extra_kwargs,
    )
