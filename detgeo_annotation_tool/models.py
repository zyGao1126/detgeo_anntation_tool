from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PAIR_STATUSES = ["raw", "in_progress", "done"]

ABSENCE_REASONS = [
    "",
    "temporal_missing",
    "too_small_in_sat",
    "occluded_in_sat",
    "ambiguous_unreliable",
]

QUERY_TYPES = [
    "both_exist_single",
    "both_exist_multi",
    "uav_only_single",
    "uav_only_multi",
    "neither_exist",
    "partial_match",
]

QA_STATUSES = ["raw", "checked", "passed"]
CASE_STATUSES = ["raw", "in_progress", "done"]
CASE_TYPES = [
    "both_exist_single",
    "both_exist_multi",
    "uav_only_single",
    "uav_only_multi",
    "neither_exist",
]


def default_attributes() -> dict[str, str]:
    return {
        "color": "",
        "roof_color": "",
        "size": "",
        "shape": "",
        "position_hint": "",
    }


@dataclass
class Pair:
    pair_id: str
    split: str
    uav_image_path: str
    sat_image_path: str
    original_click_xy: list[float] = field(default_factory=list)
    original_gt_bbox: list[float] = field(default_factory=list)
    original_class: str = ""
    status: str = "raw"
    query_center_xy: list[float] = field(default_factory=list)
    original_polygon_xy: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UAVObject:
    obj_id: str
    pair_id: str
    bbox: list[float]
    center_point: list[float]
    category: str = ""
    subtype: str = ""
    attributes: dict[str, str] = field(default_factory=default_attributes)
    is_anchor: bool = False
    referable: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SatObject:
    obj_id: str
    pair_id: str
    bbox: list[float]
    rbox: list[float] = field(default_factory=list)
    mask_path: str = ""
    category: str = ""
    subtype: str = ""
    attributes: dict[str, str] = field(default_factory=default_attributes)
    is_anchor: bool = False
    is_distractor: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Link:
    link_id: str
    pair_id: str
    uav_obj_id: str
    sat_exists: bool = True
    sat_obj_id: str | None = None
    absence_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetQuery:
    query_id: str
    pair_id: str
    query_type: str
    uav_target_ids: list[str] = field(default_factory=list)
    sat_target_ids: list[str] = field(default_factory=list)
    text: str = ""
    anchors: list[str] = field(default_factory=list)
    union_mask_path: str = ""
    exportable: bool = True
    qa_status: str = "raw"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationCase:
    case_id: str
    pair_id: str
    case_name: str
    case_type: str = "both_exist_single"
    category: str = ""
    status: str = "raw"
    description: str = ""
    notes: str = ""
    color_hex: str = "#4E79A7"
    uav_annotations: list[dict[str, Any]] = field(default_factory=list)
    sat_annotations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
