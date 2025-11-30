"""
Brain-inspired action generator module.

Contains:
- Motion Primitives (CPGs)
- Primitive Selector (Premotor Cortex)
- Primitive Modulator (M1)
- Forward Model (Cerebellum)
- Plan Encoder
"""

from .brain_model import (
    BrainInspiredActionGenerator,
    MotionPrimitiveLibrary,
    PrimitiveSelector,
    PrimitiveModulator,
)
from .forward_model import CerebellumForwardModel
from .plan_encoder import RelativePlanEncoder

__all__ = [
    "BrainInspiredActionGenerator",
    "MotionPrimitiveLibrary",
    "PrimitiveSelector",
    "PrimitiveModulator",
    "CerebellumForwardModel",
    "RelativePlanEncoder",
]
