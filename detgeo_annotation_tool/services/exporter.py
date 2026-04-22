from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..repository import AnnotationRepository


class Exporter:
    def __init__(self, repo: AnnotationRepository, workspace_dir: Path):
        self.repo = repo
        self.workspace_dir = Path(workspace_dir)

    def export_all(self, output_dir: Path, include_partial_match: bool = True) -> dict[str, int]:
        del include_partial_match
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_pairs = self.repo.list_pairs()
        all_cases = self.repo.list_all_annotation_cases()
        exported_case_records = []

        for case in all_cases:
            pair = self.repo.get_pair(case.pair_id)
            if not pair:
                continue
            case_dir = output_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            uav_dst = case_dir / Path(pair.uav_image_path).name
            sat_dst = case_dir / Path(pair.sat_image_path).name
            if Path(pair.uav_image_path).exists():
                shutil.copy2(pair.uav_image_path, uav_dst)
            if Path(pair.sat_image_path).exists():
                shutil.copy2(pair.sat_image_path, sat_dst)

            sat_annotations = []
            for ann in case.sat_annotations:
                ann_copy = dict(ann)
                mask_path = ann_copy.get("mask_path", "")
                if mask_path and Path(mask_path).exists():
                    dst = case_dir / Path(mask_path).name
                    shutil.copy2(mask_path, dst)
                    ann_copy["mask_path"] = str(dst)
                sat_annotations.append(ann_copy)

            exported_case_records.append(
                {
                    "case_id": case.case_id,
                    "pair_id": case.pair_id,
                    "split": pair.split,
                    "original_class": pair.original_class,
                    "case_name": case.case_name,
                    "case_type": case.case_type,
                    "category": case.category,
                    "status": case.status,
                    "description": case.description,
                    "notes": case.notes,
                    "color_hex": case.color_hex,
                    "uav_image_path": str(uav_dst),
                    "sat_image_path": str(sat_dst),
                    "uav_annotations": case.uav_annotations,
                    "sat_annotations": sat_annotations,
                }
            )

        (output_dir / "pairs.json").write_text(
            json.dumps([pair.to_dict() for pair in all_pairs], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "annotation_cases.json").write_text(
            json.dumps([case.to_dict() for case in all_cases], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "benchmark_cases.jsonl").open("w", encoding="utf-8") as fp:
            for record in exported_case_records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "pairs": len(all_pairs),
            "cases": len(all_cases),
            "exported_case_records": len(exported_case_records),
        }
