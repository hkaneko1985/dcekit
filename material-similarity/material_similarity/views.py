"""Default purpose-specific views; none is designated as universally best."""

from __future__ import annotations

from .distance import ViewSpec


DEFAULT_VIEWS = {
    "composition": ViewSpec("composition", {"composition": 1.0}),
    "material_identity": ViewSpec(
        "material_identity",
        {"composition": 0.40, "material_identity": 0.60},
        soft_identity=False,
    ),
    "material_identity_soft": ViewSpec(
        "material_identity_soft",
        {"composition": 0.40, "material_identity": 0.60},
        soft_identity=True,
    ),
    "synthesis_pathway": ViewSpec(
        "synthesis_pathway",
        {"process_method": 0.25, "step_type": 0.25, "sequence": 0.50},
        soft_identity=False,
    ),
    "experimental_protocol": ViewSpec(
        "experimental_protocol",
        {
            "process_method": 0.15,
            "step_type": 0.15,
            "sequence": 0.25,
            "settings": 0.45,
        },
        soft_identity=True,
    ),
    "balanced_instance": ViewSpec(
        "balanced_instance",
        {
            "composition": 0.20,
            "material_identity": 0.25,
            "process_method": 0.10,
            "step_type": 0.10,
            "sequence": 0.15,
            "settings": 0.20,
        },
        soft_identity=True,
    ),
}
