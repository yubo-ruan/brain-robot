"""Enriched object representation for grounding.

Converts LIBERO object IDs to human-readable descriptions with spatial context.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from ..world_model.state import WorldState, ObjectState


@dataclass
class EnrichedObject:
    """Object with human-readable descriptions for Qwen grounding."""

    id: str                                    # "akita_black_bowl_1_main"
    description: str                           # "black bowl"
    type: Optional[str] = None                 # "bowl"
    color: Optional[str] = None                # "black"
    material: Optional[str] = None             # "wooden"
    position: Optional[Tuple[float, float, float]] = None
    spatial_id: str = "on table"               # "on cookies_1_main" (for logging)
    spatial_text: str = "on table"             # "on cookie box" (for Qwen)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "description": self.description,
            "spatial": self.spatial_text,
        }
        if self.position:
            result["position"] = f"({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})"
        return result


# LIBERO naming quirks - map internal names to task language
# Single-word mappings
LIBERO_NAME_MAP = {
    "cookies": "cookie box",
}

# Multi-word compound mappings (checked first, order matters)
LIBERO_COMPOUND_MAP = {
    "flat_stove": "stove",
    "burner_plate": "burner",
}

# Known object types
OBJECT_TYPES = [
    "bowl", "plate", "mug", "cup", "drawer", "cabinet", "box",
    "can", "bottle", "ramekin", "stove", "burner", "cookies"
]
COLORS = ["black", "white", "red", "blue", "green", "yellow", "brown"]
MATERIALS = ["wooden", "metal", "plastic", "glass", "ceramic", "porcelain", "glazed"]
IGNORE_PARTS = ["akita", "rim", "flat", "main", "base", "top", "middle", "bottom"]


def parse_object_description(obj_id: str) -> str:
    """Parse LIBERO object ID into human-readable description.

    Examples:
        "akita_black_bowl_1_main" → "black bowl"
        "plate_1_main" → "plate"
        "cookies_1_main" → "cookie box"
        "wooden_cabinet_1_base" → "wooden cabinet"
        "flat_stove_1_burner" → "stove burner"
    """
    name = obj_id.lower()

    # Remove common suffixes
    for suffix in ["_main", "_base", "_top", "_middle", "_bottom"]:
        name = name.replace(suffix, "")

    # Check for compound mappings first (e.g., "flat_stove" → "stove")
    obj_types = []
    for compound, mapped in LIBERO_COMPOUND_MAP.items():
        if compound in name:
            obj_types.append(mapped)
            # Remove the compound from name to avoid double-matching
            name = name.replace(compound, "")

    # Split remaining into parts
    parts = [p for p in name.split("_") if p]  # Filter empty strings

    # Remove trailing numbers
    if parts and parts[-1].isdigit():
        parts = parts[:-1]

    color = None
    material = None

    for part in parts:
        if part in OBJECT_TYPES:
            # Apply single-word mapping
            mapped_type = LIBERO_NAME_MAP.get(part, part)
            obj_types.append(mapped_type)
        elif part in COLORS:
            color = part
        elif part in MATERIALS:
            material = part

    # Build description
    desc_parts = []
    if color:
        desc_parts.append(color)
    if material:
        desc_parts.append(material)
    # Add all object types (handles compound objects like "stove burner")
    desc_parts.extend(obj_types)

    if desc_parts:
        return " ".join(desc_parts)

    # Fallback: use cleaned name
    clean_parts = [p for p in parts if p not in IGNORE_PARTS and not p.isdigit()]
    return " ".join(clean_parts) if clean_parts else obj_id


def get_spatial_descriptions(
    obj_id: str,
    on: Dict[str, str],
    inside: Dict[str, str],
) -> Tuple[str, str]:
    """Get spatial descriptions for an object.

    Returns:
        (spatial_id, spatial_text) tuple where:
        - spatial_id: uses object IDs for logging ("on cookies_1_main")
        - spatial_text: uses human language for Qwen ("on cookie box")
    """
    if obj_id in on:
        support_id = on[obj_id]
        if support_id == "table":
            return "on table", "on table"
        support_desc = parse_object_description(support_id).lower()
        return f"on {support_id}", f"on {support_desc}"

    if obj_id in inside:
        container_id = inside[obj_id]
        container_desc = parse_object_description(container_id).lower()
        return f"inside {container_id}", f"inside {container_desc}"

    return "on table", "on table"


def enrich_objects(
    world_state: WorldState,
) -> List[EnrichedObject]:
    """Convert WorldState objects to enriched representations.

    Args:
        world_state: Current world state with objects

    Returns:
        List of EnrichedObject with human-readable descriptions
    """
    enriched = []

    for obj_id, obj_state in world_state.objects.items():
        # Skip robot/gripper/fixture bodies
        if any(skip in obj_id.lower() for skip in ["robot", "gripper", "mount", "floor"]):
            continue

        # Parse description
        description = parse_object_description(obj_id)

        # Extract type, color, material
        obj_type = None
        color = None
        material = None

        name_lower = obj_id.lower()
        for t in OBJECT_TYPES:
            if t in name_lower:
                obj_type = t
                break
        for c in COLORS:
            if c in name_lower:
                color = c
                break
        for m in MATERIALS:
            if m in name_lower:
                material = m
                break

        # Get position
        position = None
        if obj_state.pose is not None:
            position = tuple(obj_state.pose[:3])

        # Get spatial context
        spatial_id, spatial_text = get_spatial_descriptions(
            obj_id,
            world_state.on or {},
            world_state.inside or {}
        )

        enriched.append(EnrichedObject(
            id=obj_id,
            description=description,
            type=obj_type,
            color=color,
            material=material,
            position=position,
            spatial_id=spatial_id,
            spatial_text=spatial_text,
        ))

    return enriched
