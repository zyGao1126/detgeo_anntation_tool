from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pairs (
    pair_id TEXT PRIMARY KEY,
    split TEXT NOT NULL,
    uav_image_path TEXT NOT NULL,
    sat_image_path TEXT NOT NULL,
    original_click_xy TEXT NOT NULL DEFAULT '[]',
    original_gt_bbox TEXT NOT NULL DEFAULT '[]',
    original_class TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'raw',
    query_center_xy TEXT NOT NULL DEFAULT '[]',
    original_polygon_xy TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS uav_objects (
    obj_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    bbox TEXT NOT NULL,
    center_point TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    subtype TEXT NOT NULL DEFAULT '',
    attributes TEXT NOT NULL DEFAULT '{}',
    is_anchor INTEGER NOT NULL DEFAULT 0,
    referable INTEGER NOT NULL DEFAULT 1,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pair_id) REFERENCES pairs(pair_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sat_objects (
    obj_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    bbox TEXT NOT NULL,
    rbox TEXT NOT NULL DEFAULT '[]',
    mask_path TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    subtype TEXT NOT NULL DEFAULT '',
    attributes TEXT NOT NULL DEFAULT '{}',
    is_anchor INTEGER NOT NULL DEFAULT 0,
    is_distractor INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pair_id) REFERENCES pairs(pair_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS links (
    link_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    uav_obj_id TEXT NOT NULL,
    sat_exists INTEGER NOT NULL DEFAULT 1,
    sat_obj_id TEXT,
    absence_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pair_id) REFERENCES pairs(pair_id) ON DELETE CASCADE,
    FOREIGN KEY(uav_obj_id) REFERENCES uav_objects(obj_id) ON DELETE CASCADE,
    FOREIGN KEY(sat_obj_id) REFERENCES sat_objects(obj_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS set_queries (
    query_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    query_type TEXT NOT NULL,
    uav_target_ids TEXT NOT NULL DEFAULT '[]',
    sat_target_ids TEXT NOT NULL DEFAULT '[]',
    text TEXT NOT NULL DEFAULT '',
    anchors TEXT NOT NULL DEFAULT '[]',
    union_mask_path TEXT NOT NULL DEFAULT '',
    exportable INTEGER NOT NULL DEFAULT 1,
    qa_status TEXT NOT NULL DEFAULT 'raw',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pair_id) REFERENCES pairs(pair_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS annotation_cases (
    case_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    case_name TEXT NOT NULL DEFAULT '',
    case_type TEXT NOT NULL DEFAULT 'both_exist_single',
    category TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'raw',
    description TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    color_hex TEXT NOT NULL DEFAULT '#4E79A7',
    uav_annotations TEXT NOT NULL DEFAULT '[]',
    sat_annotations TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pair_id) REFERENCES pairs(pair_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pairs_split ON pairs(split);
CREATE INDEX IF NOT EXISTS idx_pairs_status ON pairs(status);
CREATE INDEX IF NOT EXISTS idx_pairs_class ON pairs(original_class);
CREATE INDEX IF NOT EXISTS idx_uav_pair ON uav_objects(pair_id);
CREATE INDEX IF NOT EXISTS idx_sat_pair ON sat_objects(pair_id);
CREATE INDEX IF NOT EXISTS idx_links_pair ON links(pair_id);
CREATE INDEX IF NOT EXISTS idx_queries_pair ON set_queries(pair_id);
CREATE INDEX IF NOT EXISTS idx_cases_pair ON annotation_cases(pair_id);
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(annotation_cases)").fetchall()
        }
        if columns and "category" not in columns:
            conn.execute(
                "ALTER TABLE annotation_cases ADD COLUMN category TEXT NOT NULL DEFAULT ''"
            )

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
