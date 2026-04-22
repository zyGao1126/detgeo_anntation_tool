from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

from ..models import SetQuery
from ..repository import AnnotationRepository


def validate_query(repo: AnnotationRepository, query: SetQuery, workspace_dir: Path | None = None) -> list[str]:
    issues: list[str] = []
    uav_count = len(query.uav_target_ids)
    sat_count = len(query.sat_target_ids)

    if not query.text.strip():
        issues.append("text must not be empty")

    if query.query_type == "both_exist_single" and not (uav_count == 1 and sat_count == 1):
        issues.append("both_exist_single requires 1 UAV target and 1 satellite target")
    if query.query_type == "both_exist_multi" and not (uav_count > 1 and sat_count > 1):
        issues.append("both_exist_multi requires more than 1 UAV target and more than 1 satellite target")
    if query.query_type == "uav_only_single" and not (uav_count == 1 and sat_count == 0):
        issues.append("uav_only_single requires 1 UAV target and 0 satellite targets")
    if query.query_type == "uav_only_multi" and not (uav_count > 1 and sat_count == 0):
        issues.append("uav_only_multi requires more than 1 UAV target and 0 satellite targets")
    if query.query_type == "partial_match" and not (0 < sat_count < uav_count):
        issues.append("partial_match requires only part of the UAV targets to exist in satellite")
    if query.query_type == "neither_exist" and (uav_count != 0 or sat_count != 0):
        issues.append("neither_exist must not bind real UAV or satellite targets")

    if query.sat_target_ids:
        if not query.union_mask_path or not Path(query.union_mask_path).exists():
            issues.append("union mask is missing")
        elif workspace_dir is not None:
            rebuilt_path = repo.update_union_mask(query.query_id, workspace_dir)
            rebuilt = Image.open(rebuilt_path).convert("L")
            current = Image.open(query.union_mask_path).convert("L")
            diff = ImageChops.difference(rebuilt, current).getbbox()
            if diff is not None:
                issues.append("union mask is inconsistent with satellite target masks")

    if not query.exportable and query.query_type != "partial_match":
        issues.append("non-partial queries should normally remain exportable unless manually disabled")

    return issues


def run_pair_qa(
    repo: AnnotationRepository,
    pair_id: str,
    workspace_dir: Path | None = None,
) -> dict[str, list[str]]:
    issues_by_query: dict[str, list[str]] = {}
    for query in repo.list_queries(pair_id):
        issues = validate_query(repo, query, workspace_dir=workspace_dir)
        issues_by_query[query.query_id] = issues
        query.qa_status = "passed" if not issues else "checked"
        repo.save_set_query(query)
    repo.refresh_pair_status(pair_id)
    return issues_by_query
