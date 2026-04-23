from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from .models import (
    AnnotationCase,
    Link,
    Pair,
    SetQuery,
    SatObject,
    UAVObject,
    default_attributes,
    normalize_case_type,
)
from .storage import Database


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _row_to_pair(row: Any) -> Pair:
    return Pair(
        pair_id=row["pair_id"],
        split=row["split"],
        uav_image_path=row["uav_image_path"],
        sat_image_path=row["sat_image_path"],
        original_click_xy=_json_loads(row["original_click_xy"], []),
        original_gt_bbox=_json_loads(row["original_gt_bbox"], []),
        original_class=row["original_class"],
        status=row["status"],
        query_center_xy=_json_loads(row["query_center_xy"], []),
        original_polygon_xy=_json_loads(row["original_polygon_xy"], []),
    )


def _row_to_uav(row: Any) -> UAVObject:
    return UAVObject(
        obj_id=row["obj_id"],
        pair_id=row["pair_id"],
        bbox=_json_loads(row["bbox"], []),
        center_point=_json_loads(row["center_point"], []),
        category=row["category"],
        subtype=row["subtype"],
        attributes=_json_loads(row["attributes"], default_attributes()),
        is_anchor=bool(row["is_anchor"]),
        referable=bool(row["referable"]),
        notes=row["notes"],
    )


def _row_to_sat(row: Any) -> SatObject:
    return SatObject(
        obj_id=row["obj_id"],
        pair_id=row["pair_id"],
        bbox=_json_loads(row["bbox"], []),
        rbox=_json_loads(row["rbox"], []),
        mask_path=row["mask_path"],
        category=row["category"],
        subtype=row["subtype"],
        attributes=_json_loads(row["attributes"], default_attributes()),
        is_anchor=bool(row["is_anchor"]),
        is_distractor=bool(row["is_distractor"]),
        notes=row["notes"],
    )


def _row_to_link(row: Any) -> Link:
    return Link(
        link_id=row["link_id"],
        pair_id=row["pair_id"],
        uav_obj_id=row["uav_obj_id"],
        sat_exists=bool(row["sat_exists"]),
        sat_obj_id=row["sat_obj_id"],
        absence_reason=row["absence_reason"],
    )


def _row_to_query(row: Any) -> SetQuery:
    return SetQuery(
        query_id=row["query_id"],
        pair_id=row["pair_id"],
        query_type=row["query_type"],
        uav_target_ids=_json_loads(row["uav_target_ids"], []),
        sat_target_ids=_json_loads(row["sat_target_ids"], []),
        text=row["text"],
        anchors=_json_loads(row["anchors"], []),
        union_mask_path=row["union_mask_path"],
        exportable=bool(row["exportable"]),
        qa_status=row["qa_status"],
    )


def _row_to_case(row: Any) -> AnnotationCase:
    return AnnotationCase(
        case_id=row["case_id"],
        pair_id=row["pair_id"],
        case_name=row["case_name"],
        case_type=normalize_case_type(row["case_type"]),
        category=row["category"] if "category" in row.keys() else "",
        status=row["status"],
        description=row["description"],
        notes=row["notes"],
        color_hex=row["color_hex"],
        uav_annotations=_json_loads(row["uav_annotations"], []),
        sat_annotations=_json_loads(row["sat_annotations"], []),
        hard_negative_image_path=row["hard_negative_image_path"] if "hard_negative_image_path" in row.keys() else "",
        hard_negative_bbox=_json_loads(row["hard_negative_bbox"], []),
    )


def compute_center_from_bbox(bbox: list[float]) -> list[float]:
    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]


def bbox_to_rbox(bbox: list[float]) -> list[float]:
    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1, x2, y2, x1, y2]


class AnnotationRepository:
    def __init__(self, db_path: Path):
        self.db = Database(db_path)
        self.db_path = Path(db_path)

    def generate_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def upsert_pair(self, pair: Pair) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO pairs (
                    pair_id, split, uav_image_path, sat_image_path, original_click_xy,
                    original_gt_bbox, original_class, status, query_center_xy, original_polygon_xy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair_id) DO UPDATE SET
                    split=excluded.split,
                    uav_image_path=excluded.uav_image_path,
                    sat_image_path=excluded.sat_image_path,
                    original_click_xy=excluded.original_click_xy,
                    original_gt_bbox=excluded.original_gt_bbox,
                    original_class=excluded.original_class,
                    status=excluded.status,
                    query_center_xy=excluded.query_center_xy,
                    original_polygon_xy=excluded.original_polygon_xy
                """,
                (
                    pair.pair_id,
                    pair.split,
                    pair.uav_image_path,
                    pair.sat_image_path,
                    _json_dumps(pair.original_click_xy),
                    _json_dumps(pair.original_gt_bbox),
                    pair.original_class,
                    pair.status,
                    _json_dumps(pair.query_center_xy),
                    _json_dumps(pair.original_polygon_xy),
                ),
            )

    def list_pairs(
        self,
        split: str = "",
        class_name: str = "",
        status: str = "",
        search_text: str = "",
    ) -> list[Pair]:
        clauses = []
        params: list[Any] = []
        if split:
            clauses.append("split = ?")
            params.append(split)
        if class_name:
            clauses.append("original_class = ?")
            params.append(class_name)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if search_text:
            clauses.append("(pair_id LIKE ? OR uav_image_path LIKE ? OR sat_image_path LIKE ?)")
            needle = f"%{search_text}%"
            params.extend([needle, needle, needle])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.db.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pairs {where} ORDER BY split, original_class, pair_id",
                params,
            ).fetchall()
        return [_row_to_pair(row) for row in rows]

    def get_pair(self, pair_id: str) -> Pair | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM pairs WHERE pair_id = ?", (pair_id,)).fetchone()
        return _row_to_pair(row) if row else None

    def list_distinct_classes(self) -> list[str]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT original_class FROM pairs WHERE original_class != '' ORDER BY original_class"
            ).fetchall()
        return [row["original_class"] for row in rows]

    def count_pairs(self) -> int:
        with self.db.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM pairs").fetchone()
        return int(row["count"])

    def save_annotation_case(self, case: AnnotationCase) -> None:
        case.case_type = normalize_case_type(case.case_type)
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO annotation_cases (
                    case_id, pair_id, case_name, case_type, category, status, description, notes,
                    color_hex, uav_annotations, sat_annotations, hard_negative_image_path,
                    hard_negative_bbox, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(case_id) DO UPDATE SET
                    pair_id=excluded.pair_id,
                    case_name=excluded.case_name,
                    case_type=excluded.case_type,
                    category=excluded.category,
                    status=excluded.status,
                    description=excluded.description,
                    notes=excluded.notes,
                    color_hex=excluded.color_hex,
                    uav_annotations=excluded.uav_annotations,
                    sat_annotations=excluded.sat_annotations,
                    hard_negative_image_path=excluded.hard_negative_image_path,
                    hard_negative_bbox=excluded.hard_negative_bbox,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    case.case_id,
                    case.pair_id,
                    case.case_name,
                    case.case_type,
                    case.category,
                    case.status,
                    case.description,
                    case.notes,
                    case.color_hex,
                    _json_dumps(case.uav_annotations),
                    _json_dumps(case.sat_annotations),
                    case.hard_negative_image_path,
                    _json_dumps(case.hard_negative_bbox),
                ),
            )
        self.refresh_pair_status(case.pair_id)

    def get_annotation_case(self, case_id: str) -> AnnotationCase | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM annotation_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        return _row_to_case(row) if row else None

    def list_annotation_cases(self, pair_id: str) -> list[AnnotationCase]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotation_cases WHERE pair_id = ? ORDER BY created_at, case_id",
                (pair_id,),
            ).fetchall()
        return [_row_to_case(row) for row in rows]

    def list_all_annotation_cases(self) -> list[AnnotationCase]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotation_cases ORDER BY pair_id, created_at, case_id"
            ).fetchall()
        return [_row_to_case(row) for row in rows]

    def delete_annotation_case(self, case_id: str) -> None:
        case = self.get_annotation_case(case_id)
        if not case:
            return
        with self.db.connect() as conn:
            conn.execute("DELETE FROM annotation_cases WHERE case_id = ?", (case_id,))
        self.refresh_pair_status(case.pair_id)

    def create_annotation_case(
        self,
        pair_id: str,
        case_name: str,
        case_type: str,
        color_hex: str,
    ) -> AnnotationCase:
        case = AnnotationCase(
            case_id=self.generate_id("case"),
            pair_id=pair_id,
            case_name=case_name,
            case_type=normalize_case_type(case_type),
            category="",
            color_hex=color_hex,
        )
        self.save_annotation_case(case)
        return case

    def save_uav_object(self, obj: UAVObject) -> None:
        obj.center_point = compute_center_from_bbox(obj.bbox)
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO uav_objects (
                    obj_id, pair_id, bbox, center_point, category, subtype, attributes,
                    is_anchor, referable, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(obj_id) DO UPDATE SET
                    pair_id=excluded.pair_id,
                    bbox=excluded.bbox,
                    center_point=excluded.center_point,
                    category=excluded.category,
                    subtype=excluded.subtype,
                    attributes=excluded.attributes,
                    is_anchor=excluded.is_anchor,
                    referable=excluded.referable,
                    notes=excluded.notes,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    obj.obj_id,
                    obj.pair_id,
                    _json_dumps(obj.bbox),
                    _json_dumps(obj.center_point),
                    obj.category,
                    obj.subtype,
                    _json_dumps(obj.attributes),
                    _bool_to_int(obj.is_anchor),
                    _bool_to_int(obj.referable),
                    obj.notes,
                ),
            )
        self.refresh_pair_status(obj.pair_id)

    def save_sat_object(self, obj: SatObject) -> None:
        if not obj.rbox and obj.bbox:
            obj.rbox = bbox_to_rbox(obj.bbox)
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO sat_objects (
                    obj_id, pair_id, bbox, rbox, mask_path, category, subtype, attributes,
                    is_anchor, is_distractor, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(obj_id) DO UPDATE SET
                    pair_id=excluded.pair_id,
                    bbox=excluded.bbox,
                    rbox=excluded.rbox,
                    mask_path=excluded.mask_path,
                    category=excluded.category,
                    subtype=excluded.subtype,
                    attributes=excluded.attributes,
                    is_anchor=excluded.is_anchor,
                    is_distractor=excluded.is_distractor,
                    notes=excluded.notes,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    obj.obj_id,
                    obj.pair_id,
                    _json_dumps(obj.bbox),
                    _json_dumps(obj.rbox),
                    obj.mask_path,
                    obj.category,
                    obj.subtype,
                    _json_dumps(obj.attributes),
                    _bool_to_int(obj.is_anchor),
                    _bool_to_int(obj.is_distractor),
                    obj.notes,
                ),
            )
        self.refresh_pair_status(obj.pair_id)

    def save_link(self, link: Link) -> None:
        if not link.sat_exists:
            link.sat_obj_id = None
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO links (
                    link_id, pair_id, uav_obj_id, sat_exists, sat_obj_id, absence_reason, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(link_id) DO UPDATE SET
                    pair_id=excluded.pair_id,
                    uav_obj_id=excluded.uav_obj_id,
                    sat_exists=excluded.sat_exists,
                    sat_obj_id=excluded.sat_obj_id,
                    absence_reason=excluded.absence_reason,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    link.link_id,
                    link.pair_id,
                    link.uav_obj_id,
                    _bool_to_int(link.sat_exists),
                    link.sat_obj_id,
                    link.absence_reason,
                ),
            )
        self.refresh_pair_status(link.pair_id)

    def save_set_query(self, query: SetQuery) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO set_queries (
                    query_id, pair_id, query_type, uav_target_ids, sat_target_ids, text, anchors,
                    union_mask_path, exportable, qa_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(query_id) DO UPDATE SET
                    pair_id=excluded.pair_id,
                    query_type=excluded.query_type,
                    uav_target_ids=excluded.uav_target_ids,
                    sat_target_ids=excluded.sat_target_ids,
                    text=excluded.text,
                    anchors=excluded.anchors,
                    union_mask_path=excluded.union_mask_path,
                    exportable=excluded.exportable,
                    qa_status=excluded.qa_status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    query.query_id,
                    query.pair_id,
                    query.query_type,
                    _json_dumps(query.uav_target_ids),
                    _json_dumps(query.sat_target_ids),
                    query.text,
                    _json_dumps(query.anchors),
                    query.union_mask_path,
                    _bool_to_int(query.exportable),
                    query.qa_status,
                ),
            )
        self.refresh_pair_status(query.pair_id)

    def list_uav_objects(self, pair_id: str) -> list[UAVObject]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM uav_objects WHERE pair_id = ? ORDER BY created_at, obj_id",
                (pair_id,),
            ).fetchall()
        return [_row_to_uav(row) for row in rows]

    def list_sat_objects(self, pair_id: str) -> list[SatObject]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sat_objects WHERE pair_id = ? ORDER BY created_at, obj_id",
                (pair_id,),
            ).fetchall()
        return [_row_to_sat(row) for row in rows]

    def list_links(self, pair_id: str) -> list[Link]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM links WHERE pair_id = ? ORDER BY created_at, link_id",
                (pair_id,),
            ).fetchall()
        return [_row_to_link(row) for row in rows]

    def list_queries(self, pair_id: str) -> list[SetQuery]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM set_queries WHERE pair_id = ? ORDER BY created_at, query_id",
                (pair_id,),
            ).fetchall()
        return [_row_to_query(row) for row in rows]

    def get_uav_object(self, obj_id: str) -> UAVObject | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM uav_objects WHERE obj_id = ?", (obj_id,)).fetchone()
        return _row_to_uav(row) if row else None

    def get_sat_object(self, obj_id: str) -> SatObject | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM sat_objects WHERE obj_id = ?", (obj_id,)).fetchone()
        return _row_to_sat(row) if row else None

    def get_link(self, link_id: str) -> Link | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM links WHERE link_id = ?", (link_id,)).fetchone()
        return _row_to_link(row) if row else None

    def get_query(self, query_id: str) -> SetQuery | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM set_queries WHERE query_id = ?", (query_id,)).fetchone()
        return _row_to_query(row) if row else None

    def delete_uav_object(self, obj_id: str) -> None:
        obj = self.get_uav_object(obj_id)
        if not obj:
            return
        with self.db.connect() as conn:
            conn.execute("DELETE FROM uav_objects WHERE obj_id = ?", (obj_id,))
            conn.execute("DELETE FROM links WHERE uav_obj_id = ?", (obj_id,))
        self.refresh_pair_status(obj.pair_id)

    def delete_sat_object(self, obj_id: str) -> None:
        obj = self.get_sat_object(obj_id)
        if not obj:
            return
        with self.db.connect() as conn:
            conn.execute("DELETE FROM sat_objects WHERE obj_id = ?", (obj_id,))
            conn.execute("UPDATE links SET sat_obj_id = NULL, sat_exists = 0 WHERE sat_obj_id = ?", (obj_id,))
        self.refresh_pair_status(obj.pair_id)

    def delete_link(self, link_id: str) -> None:
        link = self.get_link(link_id)
        if not link:
            return
        with self.db.connect() as conn:
            conn.execute("DELETE FROM links WHERE link_id = ?", (link_id,))
        self.refresh_pair_status(link.pair_id)

    def delete_query(self, query_id: str) -> None:
        query = self.get_query(query_id)
        if not query:
            return
        with self.db.connect() as conn:
            conn.execute("DELETE FROM set_queries WHERE query_id = ?", (query_id,))
        self.refresh_pair_status(query.pair_id)

    def refresh_pair_status(self, pair_id: str) -> None:
        cases = self.list_annotation_cases(pair_id)
        if cases:
            if all(case.status == "done" for case in cases):
                status = "done"
            elif any(
                case.uav_annotations or case.sat_annotations or case.description.strip()
                for case in cases
            ):
                status = "in_progress"
            else:
                status = "raw"
            with self.db.connect() as conn:
                conn.execute("UPDATE pairs SET status = ? WHERE pair_id = ?", (status, pair_id))
            return

        queries = self.list_queries(pair_id)
        links = self.list_links(pair_id)
        uav_objects = self.list_uav_objects(pair_id)
        sat_objects = self.list_sat_objects(pair_id)

        status = "raw"
        if uav_objects or sat_objects:
            status = "in_progress"
        if uav_objects and sat_objects:
            status = "in_progress"
        if links:
            status = "in_progress"
        if queries:
            status = "in_progress"
        if queries and all(query.qa_status == "passed" for query in queries if query.exportable):
            status = "done"

        with self.db.connect() as conn:
            conn.execute("UPDATE pairs SET status = ? WHERE pair_id = ?", (status, pair_id))

    def ensure_seed_objects(self, pair_id: str, workspace_dir: Path) -> tuple[UAVObject | None, SatObject | None]:
        pair = self.get_pair(pair_id)
        if not pair:
            return None, None

        uav_existing = self.list_uav_objects(pair_id)
        sat_existing = self.list_sat_objects(pair_id)
        created_uav = None
        created_sat = None

        if not uav_existing and len(pair.original_click_xy) == 2:
            cx, cy = pair.original_click_xy
            seed_half = 12.0
            bbox = [cx - seed_half, cy - seed_half, cx + seed_half, cy + seed_half]
            created_uav = UAVObject(
                obj_id=self.generate_id("uav"),
                pair_id=pair_id,
                bbox=bbox,
                center_point=[cx, cy],
                category=pair.original_class,
                is_anchor=False,
                referable=True,
            )
            self.save_uav_object(created_uav)

        if not sat_existing and len(pair.original_gt_bbox) == 4:
            sat_mask_dir = Path(workspace_dir) / "masks" / pair_id
            sat_mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = sat_mask_dir / f"{pair_id}_seed.png"
            self._write_rect_mask(pair.sat_image_path, pair.original_gt_bbox, mask_path)
            created_sat = SatObject(
                obj_id=self.generate_id("sat"),
                pair_id=pair_id,
                bbox=pair.original_gt_bbox,
                rbox=bbox_to_rbox(pair.original_gt_bbox),
                mask_path=str(mask_path),
                category=pair.original_class,
                is_anchor=False,
                is_distractor=False,
            )
            self.save_sat_object(created_sat)

        self.refresh_pair_status(pair_id)
        return created_uav, created_sat

    def _write_rect_mask(self, sat_image_path: str, bbox: list[float], output_path: Path) -> None:
        image = Image.open(sat_image_path).convert("RGB")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(tuple(bbox), fill=255)
        mask.save(output_path)

    def auto_query_type(self, uav_target_ids: list[str], sat_target_ids: list[str]) -> tuple[str, bool]:
        uav_count = len(uav_target_ids)
        sat_count = len(sat_target_ids)
        exportable = True
        if uav_count == 0 and sat_count == 0:
            return "neither_exist", True
        if uav_count == 1 and sat_count == 1:
            return "both_exist_single", True
        if uav_count > 1 and sat_count == uav_count:
            return "both_exist_multi", True
        if uav_count == 1 and sat_count == 0:
            return "uav_only_single", True
        if uav_count > 1 and sat_count == 0:
            return "uav_only_multi", True
        if 0 < sat_count < uav_count:
            return "partial_match", False
        if uav_count > 1 and sat_count > 1:
            return "both_exist_multi", True
        return "partial_match", False

    def find_link_for_uav(self, pair_id: str, uav_obj_id: str) -> Link | None:
        for link in self.list_links(pair_id):
            if link.uav_obj_id == uav_obj_id:
                return link
        return None

    def create_query_from_uav_ids(self, pair_id: str, uav_target_ids: list[str]) -> SetQuery:
        sat_target_ids: list[str] = []
        for uav_id in uav_target_ids:
            link = self.find_link_for_uav(pair_id, uav_id)
            if link and link.sat_exists and link.sat_obj_id:
                sat_target_ids.append(link.sat_obj_id)
        query_type, exportable = self.auto_query_type(uav_target_ids, sat_target_ids)
        query = SetQuery(
            query_id=self.generate_id("query"),
            pair_id=pair_id,
            query_type=query_type,
            uav_target_ids=uav_target_ids,
            sat_target_ids=sat_target_ids,
            exportable=exportable,
        )
        self.save_set_query(query)
        return query

    def create_neither_exist_query(self, pair_id: str, text: str) -> SetQuery:
        query = SetQuery(
            query_id=self.generate_id("query"),
            pair_id=pair_id,
            query_type="neither_exist",
            text=text,
            exportable=True,
        )
        self.save_set_query(query)
        return query

    def update_union_mask(self, query_id: str, workspace_dir: Path) -> str:
        query = self.get_query(query_id)
        if not query:
            raise ValueError(f"Query not found: {query_id}")
        pair = self.get_pair(query.pair_id)
        if not pair:
            raise ValueError(f"Pair not found for query: {query_id}")

        sat_image = Image.open(pair.sat_image_path).convert("RGB")
        union = Image.new("L", sat_image.size, 0)
        for sat_obj_id in query.sat_target_ids:
            sat_obj = self.get_sat_object(sat_obj_id)
            if not sat_obj:
                continue
            if sat_obj.mask_path and Path(sat_obj.mask_path).exists():
                mask = Image.open(sat_obj.mask_path).convert("L").resize(sat_image.size)
            else:
                mask = Image.new("L", sat_image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle(tuple(sat_obj.bbox), fill=255)
            union = ImageChops.lighter(union, mask)

        union_dir = Path(workspace_dir) / "union_masks" / query.pair_id
        union_dir.mkdir(parents=True, exist_ok=True)
        output_path = union_dir / f"{query.query_id}.png"
        union.save(output_path)
        query.union_mask_path = str(output_path)
        self.save_set_query(query)
        return str(output_path)

    def get_pair_bundle(self, pair_id: str) -> dict[str, Any]:
        pair = self.get_pair(pair_id)
        if not pair:
            raise ValueError(f"Unknown pair: {pair_id}")
        return {
            "pair": pair,
            "cases": self.list_annotation_cases(pair_id),
            "uav_objects": self.list_uav_objects(pair_id),
            "sat_objects": self.list_sat_objects(pair_id),
            "links": self.list_links(pair_id),
            "queries": self.list_queries(pair_id),
        }
