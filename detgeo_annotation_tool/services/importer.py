from __future__ import annotations

import json
from pathlib import Path

import torch

from ..models import Pair
from ..repository import AnnotationRepository


def import_pairs_from_pth(
    repo: AnnotationRepository,
    data_root: Path,
    data_name: str,
    splits: list[str],
) -> int:
    data_root = Path(data_root)
    dataset_dir = data_root / data_name
    query_root = dataset_dir / "query"
    sat_root = dataset_dir / "satellite"

    imported = 0
    for split in splits:
        split_path = dataset_dir / f"{data_name}_{split}.pth"
        if not split_path.exists():
            continue
        records = torch.load(split_path)
        for index, row in enumerate(records):
            (
                sample_id,
                query_name,
                satellite_name,
                query_center_xy,
                click_xy,
                bbox_xyxy,
                polygon_xy,
                class_name,
            ) = row
            pair = Pair(
                pair_id=f"{split}_{index:06d}_{sample_id}",
                split=split,
                uav_image_path=str(query_root / query_name),
                sat_image_path=str(sat_root / satellite_name),
                original_click_xy=[float(click_xy[0]), float(click_xy[1])],
                original_gt_bbox=[float(v) for v in bbox_xyxy],
                original_class=str(class_name),
                status="raw",
                query_center_xy=[float(query_center_xy[0]), float(query_center_xy[1])],
                original_polygon_xy=[float(v) for v in polygon_xy],
            )
            repo.upsert_pair(pair)
            imported += 1
    return imported


def import_pairs_from_manifest(repo: AnnotationRepository, manifest_path: Path) -> int:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split = payload.get("split", "train")
    imported = 0
    for index, item in enumerate(payload.get("items", [])):
        pair = Pair(
            pair_id=f"{split}_{index:06d}_{item['sample_id']}",
            split=split,
            uav_image_path=item["query_image"],
            sat_image_path=item["satellite_image"],
            original_click_xy=item.get("click_xy", []),
            original_gt_bbox=item.get("bbox_xyxy", []),
            original_class=item.get("class_name", ""),
            status="raw",
            query_center_xy=item.get("query_center_xy", []),
            original_polygon_xy=item.get("polygon_xy", []),
        )
        repo.upsert_pair(pair)
        imported += 1
    return imported
