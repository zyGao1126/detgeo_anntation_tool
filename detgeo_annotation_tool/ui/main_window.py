from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..models import AnnotationCase, CASE_STATUSES, CASE_TYPES, PAIR_STATUSES, normalize_case_type
from ..repository import AnnotationRepository
from ..services.exporter import Exporter
from ..services.sam3_process import SAM3ProcessClient
from .viewer import AnnotatedImageView


CASE_COLORS = [
    "#4E79A7",
    "#E15759",
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
]

CATEGORY_OPTIONS = [
    "building",
    "junction_roundabout",
    "garage",
    "sport_baseball",
    "man_made_bridge",
    "sport_tennis",
    "man_made_storage_tank",
    "sport_soccer",
    "leisure_track",
    "man_made_water_tower",
    "sport_swimming",
    "sport_basketball",
    "vehicle",
    "ship",
]

DETGEO_CATEGORY_MAPPING = {
    "building_apartments": "building",
    "building_building": "building",
    "building_church": "building",
    "building_house": "building",
    "building_office": "building",
    "building_retail": "building",
    "building_garages": "garage",
    "building_industrial": "garage",
    "building_roof": "garage",
    "junction_roundabout": "junction_roundabout",
    "leisure_track": "leisure_track",
    "man_made_bridge": "man_made_bridge",
    "man_made_storage_tank": "man_made_storage_tank",
    "man_made_water_tower": "man_made_water_tower",
    "sport_baseball": "sport_baseball",
    "sport_basketball": "sport_basketball",
    "sport_soccer": "sport_soccer",
    "sport_swimming": "sport_swimming",
    "sport_tennis": "sport_tennis",
}


class MainWindow(QMainWindow):
    def __init__(self, repo: AnnotationRepository, workspace_dir: Path):
        super().__init__()
        self.repo = repo
        self.project_root = Path(__file__).resolve().parents[2]
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.saved_anno_root = self.workspace_dir.parent / "saved_anno"
        self.saved_anno_root.mkdir(parents=True, exist_ok=True)
        self.ui_state_path = self.workspace_dir / "ui_state.json"
        self.ui_state = self._load_ui_state()
        sam3_repo_root, sam3_checkpoint_path = self._resolve_sam3_paths()
        self.sam3_client = SAM3ProcessClient(
            project_root=self.project_root,
            repo_root=sam3_repo_root,
            checkpoint_path=sam3_checkpoint_path,
            parent=self,
        )
        self.sam3_preload_scheduled = False
        self.sam3_state = "idle"
        self.sam3_message = ""
        self.sam3_device = ""
        self.context_menu_open = False
        self.sam3_loading_dialog: QProgressDialog | None = None
        self.sam3_pending_preview: dict | None = None

        self.current_pair_id = ""
        self.current_case_id = ""
        self.active_viewer_name = "uav"
        self.sat_annotation_mode = "sam3"
        self.satellite_view_mode = "original"
        self.pending_hard_negative_by_case: dict[str, dict] = {}
        self.hard_negative_capture_active = False

        self.setWindowTitle("DetGeo 二次加工标注工具")
        self._build_ui()
        self._bind_signals()
        self.refresh_pair_filters()
        self.refresh_pair_list()
        self._install_shortcuts()
        self.log(f"SAM3 repo root: {sam3_repo_root}")
        self.log(f"SAM3 checkpoint: {sam3_checkpoint_path}")

    def _load_ui_state(self) -> dict:
        if not self.ui_state_path.exists():
            return {}
        try:
            return json.loads(self.ui_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_ui_state(self) -> None:
        self.ui_state_path.write_text(
            json.dumps(self.ui_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _default_category_for_class(self, original_class: str) -> str:
        return DETGEO_CATEGORY_MAPPING.get(original_class, CATEGORY_OPTIONS[0])

    def _default_download_dir(self) -> Path:
        saved_dir = self.ui_state.get("uav_export_dir", "")
        if saved_dir:
            path = Path(saved_dir)
            if path.exists():
                return path
        return self.saved_anno_root

    def _pair_export_stem(self) -> str:
        pair = self.repo.get_pair(self.current_pair_id) if self.current_pair_id else None
        if pair and pair.uav_image_path:
            return Path(pair.uav_image_path).stem
        return self.current_pair_id or "uav_view"

    def _resolve_sam3_paths(self) -> tuple[Path, Path]:
        project_root = Path(__file__).resolve().parents[2]
        bundled_repo_root = project_root.parent / "sam3"
        fallback_repo_root = Path("/home/gaoziyang/research/RRSIS/sam3")
        repo_root_env = os.environ.get("DETGEO_SAM3_REPO")
        if repo_root_env:
            repo_root = Path(repo_root_env).expanduser()
        elif bundled_repo_root.exists():
            repo_root = bundled_repo_root
        else:
            repo_root = fallback_repo_root
        default_checkpoint = repo_root / "checkpoints" / "sam3.pt"
        checkpoint_path = Path(
            os.environ.get("DETGEO_SAM3_CHECKPOINT", str(default_checkpoint))
        ).expanduser()
        return repo_root, checkpoint_path

    def _get_pending_hard_negative(self, case_id: str) -> dict | None:
        if not case_id:
            return None
        return self.pending_hard_negative_by_case.get(case_id)

    def _get_case_hard_negative_path(self, case: AnnotationCase | None) -> str:
        if not case:
            return ""
        pending = self._get_pending_hard_negative(case.case_id)
        if pending:
            return pending.get("image_path", "")
        return case.hard_negative_image_path

    def _case_has_hard_negative(self, case: AnnotationCase | None) -> bool:
        path = self._get_case_hard_negative_path(case)
        return bool(path and Path(path).exists())

    def _focus_bbox_for_hard_negative(self, case: AnnotationCase | None, pair=None) -> list[float]:
        if case:
            boxes: list[list[float]] = []
            for ann in case.sat_annotations:
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    boxes.append([float(v) for v in bbox])
            if boxes:
                return [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ]
        pair = pair or (self.repo.get_pair(self.current_pair_id) if self.current_pair_id else None)
        if pair and len(pair.original_gt_bbox) == 4:
            return [float(v) for v in pair.original_gt_bbox]
        return []

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_sam3_preload()

    def closeEvent(self, event) -> None:
        self.sam3_client.shutdown()
        super().closeEvent(event)

    def _schedule_sam3_preload(self) -> None:
        if self.sam3_preload_scheduled or self.sam3_state in {"starting", "ready", "busy"}:
            return
        self.sam3_preload_scheduled = True
        self.log("SAM3 preload scheduled after GUI is fully shown")
        QTimer.singleShot(2500, self._start_sam3_preload)

    def _show_sam3_loading_notice(self) -> None:
        if self.sam3_loading_dialog is not None:
            return
        self.sam3_loading_dialog = QProgressDialog("正在加载 SAM3，请稍候...", None, 0, 0, self)
        self.sam3_loading_dialog.setWindowTitle("SAM3")
        self.sam3_loading_dialog.setCancelButton(None)
        self.sam3_loading_dialog.setMinimumDuration(0)
        self.sam3_loading_dialog.setWindowModality(Qt.NonModal)
        self.sam3_loading_dialog.setAutoClose(False)
        self.sam3_loading_dialog.setAutoReset(False)
        self.sam3_loading_dialog.show()

    def _hide_sam3_loading_notice(self) -> None:
        if self.sam3_loading_dialog is not None:
            self.sam3_loading_dialog.close()
            self.sam3_loading_dialog.deleteLater()
            self.sam3_loading_dialog = None

    def _start_sam3_preload(self) -> None:
        if self.sam3_state in {"starting", "ready", "busy"}:
            return
        if self.context_menu_open or QApplication.activePopupWidget() is not None or QApplication.activeModalWidget() is not None:
            self.log("Postponing SAM3 preload because a menu or dialog is active")
            QTimer.singleShot(800, self._start_sam3_preload)
            return
        self.sam3_preload_scheduled = False
        self._apply_annotation_mode()
        self.log("Starting delayed SAM3 preload...")
        self._show_sam3_loading_notice()
        self.sam3_client.start()

    def _on_sam3_state_changed(self, state: str, message: str) -> None:
        self.sam3_state = state
        self.sam3_message = message
        self.sam3_device = self.sam3_client.device
        if state == "starting":
            self.log(message)
            self._show_sam3_loading_notice()
        else:
            self._hide_sam3_loading_notice()
        if state == "ready":
            device = self.sam3_device or "unknown"
            self.log(f"SAM3 model loaded successfully on {device}")
        elif state == "failed":
            self.log(f"SAM3 unavailable: {message}")
        self._apply_annotation_mode()

    def _on_sam3_segment_finished(self, ok: bool, payload: dict) -> None:
        pending = self.sam3_pending_preview
        self.sam3_pending_preview = None
        if not pending:
            return
        if not ok:
            message = str(payload.get("message", "SAM3 segmentation failed."))
            QMessageBox.warning(self, "SAM3 Failed", message)
            self.log(f"SAM3 failed: {message}")
            return
        case = self.repo.get_annotation_case(pending["case_id"])
        if not case:
            return
        bbox_xyxy = [float(v) for v in payload.get("bbox_xyxy", pending["bbox"])]
        mask_path = str(payload.get("mask_path", pending["mask_path"]))
        self.sat_view.set_draft_mask(mask_path, bbox_xyxy, case.color_hex)
        score = float(payload.get("score", 0.0))
        self.log(f"SAM3 preview ready | score={score:.4f} | {mask_path}")

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        self.setCentralWidget(root)

        main_splitter = QSplitter(Qt.Vertical)
        root_layout.addWidget(main_splitter)

        top_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(top_splitter)
        top_splitter.addWidget(self._build_navigator())
        top_splitter.addWidget(self._build_viewers())
        top_splitter.addWidget(self._build_case_panel())

        self.bottom_tabs = self._build_bottom_tabs()
        main_splitter.addWidget(self.bottom_tabs)
        top_splitter.setSizes([280, 1100, 380])
        main_splitter.setSizes([790, 250])

    def _build_navigator(self) -> QWidget:
        box = QGroupBox("Pair Navigator")
        layout = QVBoxLayout(box)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search pair_id / image path")
        self.split_filter = QComboBox()
        self.split_filter.addItems(["", "train", "val", "test"])
        self.class_filter = QComboBox()
        self.status_filter = QComboBox()
        self.status_filter.addItems([""] + PAIR_STATUSES)
        self.refresh_pairs_button = QPushButton("Refresh Pairs")

        layout.addWidget(self.search_edit)
        layout.addWidget(QLabel("Split"))
        layout.addWidget(self.split_filter)
        layout.addWidget(QLabel("Class"))
        layout.addWidget(self.class_filter)
        layout.addWidget(QLabel("Status"))
        layout.addWidget(self.status_filter)

        self.pair_list = QListWidget()
        layout.addWidget(self.pair_list, 1)
        layout.addWidget(self.refresh_pairs_button)
        return box

    def _build_viewers(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        controls = QHBoxLayout()
        self.annotation_toggle = QToolButton()
        self.annotation_toggle.setText("Annotation Mode")
        self.annotation_toggle.setCheckable(True)
        self.reference_toggle = QToolButton()
        self.reference_toggle.setText("DetGeo Reference")
        self.reference_toggle.setCheckable(True)
        self.reference_toggle.setChecked(True)
        self.annotation_hint = QLabel("Normal browse mode")
        self.sat_mode_label = QLabel("Satellite Mode: SAM3")
        controls.addWidget(self.annotation_toggle)
        controls.addWidget(self.reference_toggle)
        controls.addWidget(self.annotation_hint, 1)
        controls.addWidget(self.sat_mode_label)
        layout.addLayout(controls)

        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split, 1)

        uav_box = QGroupBox("UAV Viewer")
        uav_layout = QVBoxLayout(uav_box)
        self.uav_view = AnnotatedImageView("uav")
        uav_layout.addWidget(self.uav_view)
        split.addWidget(uav_box)

        sat_box = QGroupBox("Satellite Viewer")
        sat_layout = QVBoxLayout(sat_box)
        sat_toolbar = QHBoxLayout()
        self.generate_hard_negative_button = QPushButton("Generate Hard Negative")
        self.toggle_satellite_view_button = QToolButton()
        self.toggle_satellite_view_button.setText("<->")
        self.sat_view_mode_label = QLabel("View: original")
        sat_toolbar.addWidget(self.generate_hard_negative_button)
        sat_toolbar.addStretch(1)
        sat_toolbar.addWidget(self.sat_view_mode_label)
        sat_toolbar.addWidget(self.toggle_satellite_view_button)
        sat_layout.addLayout(sat_toolbar)
        self.sat_view = AnnotatedImageView("sat")
        sat_layout.addWidget(self.sat_view)
        split.addWidget(sat_box)
        split.setSizes([430, 700])
        return container

    def _build_case_panel(self) -> QWidget:
        box = QGroupBox("Case Panel")
        layout = QVBoxLayout(box)

        button_row = QHBoxLayout()
        self.new_case_button = QPushButton("New Case")
        self.delete_case_button = QPushButton("Delete Case")
        self.save_anno_button = QPushButton("Save Anno")
        button_row.addWidget(self.new_case_button)
        button_row.addWidget(self.delete_case_button)
        button_row.addWidget(self.save_anno_button)
        layout.addLayout(button_row)

        form = QFormLayout()
        self.case_name_edit = QLineEdit()
        self.case_type_combo = QComboBox()
        self.case_type_combo.addItems(CASE_TYPES)
        self.case_category_combo = QComboBox()
        self.case_category_combo.addItems(CATEGORY_OPTIONS)
        self.case_status_combo = QComboBox()
        self.case_status_combo.addItems(CASE_STATUSES)
        self.case_color_label = QLabel("-")
        self.hard_negative_label = QLabel("-")
        self.case_description_edit = QPlainTextEdit()
        self.case_description_edit.setFixedHeight(120)
        self.case_notes_edit = QPlainTextEdit()
        self.case_notes_edit.setFixedHeight(120)
        form.addRow("Case Name", self.case_name_edit)
        form.addRow("Case Type", self.case_type_combo)
        form.addRow("Category", self.case_category_combo)
        form.addRow("Case Status", self.case_status_combo)
        form.addRow("Case Color", self.case_color_label)
        form.addRow("Hard Negative", self.hard_negative_label)
        form.addRow("Description", self.case_description_edit)
        form.addRow("Notes", self.case_notes_edit)
        layout.addLayout(form)

        self.save_case_button = QPushButton("Save Case Metadata")
        layout.addWidget(self.save_case_button)
        layout.addStretch(1)
        return box

    def _build_bottom_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        cases_tab = QWidget()
        cases_layout = QVBoxLayout(cases_tab)
        self.cases_table = QTableWidget(0, 8)
        self.cases_table.setHorizontalHeaderLabels(
            ["case_id", "name", "type", "category", "status", "uav_n", "sat_n", "color"]
        )
        self.cases_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cases_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cases_table.setEditTriggers(QTableWidget.NoEditTriggers)
        cases_layout.addWidget(self.cases_table)
        tabs.addTab(cases_tab, "Cases")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        export_row = QHBoxLayout()
        self.export_button = QPushButton("Export Workspace")
        export_row.addWidget(self.export_button)
        export_row.addStretch(1)
        log_layout.addLayout(export_row)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        tabs.addTab(log_tab, "Logs")
        return tabs

    def _bind_signals(self) -> None:
        self.refresh_pairs_button.clicked.connect(self.refresh_pair_list)
        self.search_edit.textChanged.connect(self.refresh_pair_list)
        self.split_filter.currentTextChanged.connect(self.refresh_pair_list)
        self.class_filter.currentTextChanged.connect(self.refresh_pair_list)
        self.status_filter.currentTextChanged.connect(self.refresh_pair_list)
        self.reference_toggle.toggled.connect(self.refresh_viewers)
        self.pair_list.currentItemChanged.connect(self._on_pair_changed)

        self.annotation_toggle.toggled.connect(self._apply_annotation_mode)
        self.uav_view.viewerActivated.connect(self._on_viewer_activated)
        self.sat_view.viewerActivated.connect(self._on_viewer_activated)
        self.uav_view.contextMenuRequested.connect(self._show_viewer_context_menu)
        self.sat_view.contextMenuRequested.connect(self._show_viewer_context_menu)
        self.uav_view.draftCompleted.connect(self._on_draft_completed)
        self.sat_view.draftCompleted.connect(self._on_draft_completed)

        self.new_case_button.clicked.connect(self.create_case)
        self.delete_case_button.clicked.connect(self.delete_current_case)
        self.save_anno_button.clicked.connect(self.save_case_annotation_bundle)
        self.save_case_button.clicked.connect(self.save_case_metadata)
        self.generate_hard_negative_button.clicked.connect(self.start_hard_negative_capture)
        self.toggle_satellite_view_button.clicked.connect(self.toggle_satellite_view)
        self.cases_table.itemSelectionChanged.connect(self._on_case_table_selected)
        self.export_button.clicked.connect(self.export_workspace)
        self.sam3_client.stateChanged.connect(self._on_sam3_state_changed)
        self.sam3_client.segmentFinished.connect(self._on_sam3_segment_finished)
        self.sam3_client.logMessage.connect(self.log)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence.Save, self, activated=self.handle_ctrl_save)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.clear_current_draft)
        QShortcut(QKeySequence.Delete, self, activated=self.delete_current_case)

    def log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def refresh_pair_filters(self) -> None:
        current = self.class_filter.currentText()
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItems([""] + self.repo.list_distinct_classes())
        index = self.class_filter.findText(current)
        self.class_filter.setCurrentIndex(max(index, 0))
        self.class_filter.blockSignals(False)

    def refresh_pair_list(self) -> None:
        pairs = self.repo.list_pairs(
            split=self.split_filter.currentText(),
            class_name=self.class_filter.currentText(),
            status=self.status_filter.currentText(),
            search_text=self.search_edit.text().strip(),
        )
        selected_pair_id = self.current_pair_id
        self.pair_list.blockSignals(True)
        self.pair_list.clear()
        for pair in pairs:
            label = f"[{pair.split}] {pair.original_class} | {pair.status} | {pair.pair_id}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, pair.pair_id)
            self.pair_list.addItem(item)
            if pair.pair_id == selected_pair_id:
                self.pair_list.setCurrentItem(item)
        self.pair_list.blockSignals(False)
        if not selected_pair_id and self.pair_list.count() > 0:
            self.pair_list.setCurrentRow(0)

    def _on_pair_changed(self, current: QListWidgetItem | None) -> None:
        if current:
            self.load_pair(current.data(Qt.UserRole))

    def load_pair(self, pair_id: str, keep_case_id: str = "") -> None:
        self.current_pair_id = pair_id
        self.hard_negative_capture_active = False
        self.satellite_view_mode = "original"
        pair = self.repo.get_pair(pair_id)
        if not pair:
            return

        self.uav_view.set_image(pair.uav_image_path)
        if keep_case_id:
            self.current_case_id = keep_case_id
        cases = self.repo.list_annotation_cases(pair_id)
        if self.current_case_id and not any(case.case_id == self.current_case_id for case in cases):
            self.current_case_id = ""
        if not self.current_case_id and cases:
            self.current_case_id = cases[0].case_id
        self.refresh_case_table(cases)
        self.refresh_case_editor()
        self.refresh_viewers()
        self._apply_annotation_mode()
        self.log(
            f"Loaded pair {pair_id} | class={pair.original_class} | status={pair.status}\n"
            f"uav={pair.uav_image_path}\nsat={pair.sat_image_path}"
        )

    def refresh_case_table(self, cases: list[AnnotationCase] | None = None) -> None:
        cases = cases if cases is not None else self.repo.list_annotation_cases(self.current_pair_id)
        pair = self.repo.get_pair(self.current_pair_id) if self.current_pair_id else None
        self.cases_table.blockSignals(True)
        self.cases_table.setRowCount(len(cases))
        selected_row = -1
        for row, case in enumerate(cases):
            category = case.category or self._default_category_for_class(pair.original_class if pair else "")
            values = [
                case.case_id,
                case.case_name,
                case.case_type,
                category,
                case.status,
                str(len(case.uav_annotations)),
                str(len(case.sat_annotations)),
                case.color_hex,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 7:
                    item.setBackground(QColor(case.color_hex))
                self.cases_table.setItem(row, col, item)
            if case.case_id == self.current_case_id:
                selected_row = row
        self.cases_table.blockSignals(False)
        if selected_row >= 0:
            self.cases_table.selectRow(selected_row)

    def refresh_case_editor(self) -> None:
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        widgets = [
            self.case_name_edit,
            self.case_type_combo,
            self.case_category_combo,
            self.case_status_combo,
            self.case_description_edit,
            self.case_notes_edit,
            self.save_case_button,
            self.delete_case_button,
            self.save_anno_button,
            self.generate_hard_negative_button,
            self.toggle_satellite_view_button,
        ]
        enabled = case is not None
        for widget in widgets:
            widget.setEnabled(enabled)
        if not case:
            self.case_name_edit.clear()
            self.case_type_combo.setCurrentIndex(0)
            self.case_category_combo.setCurrentIndex(0)
            self.case_status_combo.setCurrentIndex(0)
            self.case_description_edit.clear()
            self.case_notes_edit.clear()
            self.case_color_label.setText("-")
            self.hard_negative_label.setText("-")
            self.hard_negative_label.setToolTip("")
            self.sat_view_mode_label.setText("View: original")
            return
        self.case_name_edit.setText(case.case_name)
        self.case_type_combo.setCurrentText(normalize_case_type(case.case_type))
        pair = self.repo.get_pair(case.pair_id)
        category = case.category or self._default_category_for_class(pair.original_class if pair else "")
        self.case_category_combo.setCurrentText(category)
        self.case_status_combo.setCurrentText(case.status)
        self.case_description_edit.setPlainText(case.description)
        self.case_notes_edit.setPlainText(case.notes)
        self.case_color_label.setText(case.color_hex)
        hard_negative_path = self._get_case_hard_negative_path(case)
        if hard_negative_path and Path(hard_negative_path).exists():
            self.hard_negative_label.setText(Path(hard_negative_path).name)
            self.hard_negative_label.setToolTip(hard_negative_path)
        else:
            self.hard_negative_label.setText("not generated")
            self.hard_negative_label.setToolTip("")

    def refresh_viewers(self) -> None:
        if not self.current_pair_id:
            return
        pair = self.repo.get_pair(self.current_pair_id)
        cases = self.repo.list_annotation_cases(self.current_pair_id)
        self.uav_view.set_saved_cases(cases, side="uav", selected_case_id=self.current_case_id)
        self.uav_view.set_reference_data(pair, self.reference_toggle.isChecked())
        self.refresh_satellite_view(pair=pair, cases=cases)

    def refresh_satellite_view(self, pair=None, cases: list[AnnotationCase] | None = None) -> None:
        if not self.current_pair_id:
            return
        pair = pair or self.repo.get_pair(self.current_pair_id)
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        cases = cases if cases is not None else self.repo.list_annotation_cases(self.current_pair_id)
        if not pair:
            return

        if self.satellite_view_mode == "hard_negative" and not self._case_has_hard_negative(case):
            self.satellite_view_mode = "original"

        if self.satellite_view_mode == "hard_negative":
            sat_path = self._get_case_hard_negative_path(case)
            self.sat_view.set_image(sat_path)
            self.sat_view.set_saved_cases([], side="sat", selected_case_id=self.current_case_id)
            self.sat_view.set_reference_data(None, False)
            self.sat_view_mode_label.setText("View: hard negative")
        else:
            self.sat_view.set_image(pair.sat_image_path)
            self.sat_view.set_saved_cases(cases, side="sat", selected_case_id=self.current_case_id)
            self.sat_view.set_reference_data(pair, self.reference_toggle.isChecked())
            self.sat_view_mode_label.setText("View: original")

        self.toggle_satellite_view_button.setEnabled(case is not None and self._case_has_hard_negative(case))

    def _on_case_table_selected(self) -> None:
        rows = self.cases_table.selectionModel().selectedRows()
        if not rows:
            return
        self.current_case_id = self.cases_table.item(rows[0].row(), 0).text()
        self.hard_negative_capture_active = False
        self.refresh_case_editor()
        self.refresh_viewers()

    def _on_viewer_activated(self, viewer_name: str) -> None:
        self.active_viewer_name = viewer_name
        self._apply_annotation_mode()

    def _apply_annotation_mode(self) -> None:
        enabled = self.annotation_toggle.isChecked()
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        pair = self.repo.get_pair(self.current_pair_id) if self.current_pair_id else None
        if self.hard_negative_capture_active:
            focus_bbox = self._focus_bbox_for_hard_negative(case, pair)
            self.sat_view.set_crop_assist(True, focus_bbox=focus_bbox)
            self.uav_view.set_annotation_tool("pan", enabled=False)
            self.sat_view.set_annotation_tool("bbox", enabled=True)
            if focus_bbox:
                self.annotation_hint.setText(
                    "Hard negative capture: yellow box marks the target area, drag a crop box that only partially contains it"
                )
            else:
                self.annotation_hint.setText(
                    "Hard negative capture: drag a crop box on the original satellite image"
                )
            self.sat_mode_label.setText("Satellite Mode: Hard Negative Capture")
            return
        self.sat_view.set_crop_assist(False)
        if not enabled:
            self.uav_view.set_annotation_tool("pan", enabled=False)
            self.sat_view.set_annotation_tool("pan", enabled=False)
            self.annotation_hint.setText("Normal browse mode")
            if self.satellite_view_mode == "hard_negative":
                self.sat_mode_label.setText("Satellite Mode: Hard Negative Preview")
            else:
                mode_name = "SAM3" if self.sat_annotation_mode == "sam3" else "Manual Polygon"
                self.sat_mode_label.setText(f"Satellite Mode: {mode_name}")
            return

        if self.active_viewer_name == "uav":
            self.uav_view.set_annotation_tool("bbox", enabled=True)
            self.sat_view.set_annotation_tool("pan", enabled=False)
            self.annotation_hint.setText("UAV annotation mode: drag a box, Ctrl+S save, Esc clear")
        else:
            if self.satellite_view_mode == "hard_negative":
                self.uav_view.set_annotation_tool("pan", enabled=False)
                self.sat_view.set_annotation_tool("pan", enabled=False)
                self.annotation_hint.setText("Hard negative preview is read-only. Switch back to original view to annotate.")
                self.sat_mode_label.setText("Satellite Mode: Hard Negative Preview")
                return
            sat_tool = "bbox" if self.sat_annotation_mode == "sam3" else "polygon"
            self.uav_view.set_annotation_tool("pan", enabled=False)
            self.sat_view.set_annotation_tool(sat_tool, enabled=True)
            if self.sat_annotation_mode == "sam3":
                if self.sam3_state == "ready":
                    self.annotation_hint.setText("Satellite SAM3 mode: drag a box, model ready, Ctrl+S save, Esc clear")
                elif self.sam3_state in {"starting", "busy"}:
                    self.annotation_hint.setText("Satellite SAM3 mode: model is loading, please wait")
                else:
                    self.annotation_hint.setText("Satellite SAM3 mode: model not ready yet")
            else:
                self.annotation_hint.setText("Satellite manual mode: left click points, double click/ Ctrl+S save, Esc clear")
        mode_name = "SAM3" if self.sat_annotation_mode == "sam3" else "Manual Polygon"
        if self.sat_annotation_mode == "sam3":
            if self.sam3_state == "ready":
                suffix = "ready"
            elif self.sam3_state in {"starting", "busy"}:
                suffix = "loading"
            elif self.sam3_state == "failed":
                suffix = "unavailable"
            else:
                suffix = "pending"
            self.sat_mode_label.setText(f"Satellite Mode: {mode_name} ({suffix})")
        else:
            self.sat_mode_label.setText(f"Satellite Mode: {mode_name}")

    def _show_viewer_context_menu(self, viewer_name: str, global_pos) -> None:
        menu = QMenu(self)
        self.context_menu_open = True
        try:
            if viewer_name == "uav":
                export_action = menu.addAction("下载图像")
                selected_action = menu.exec(global_pos)
                if selected_action == export_action:
                    self.export_uav_view_image()
                return

            sam_action = menu.addAction("SAM3标注")
            manual_action = menu.addAction("手动标注")
            delete_hard_negative_action = None
            current_case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
            if self._case_has_hard_negative(current_case):
                menu.addSeparator()
                delete_hard_negative_action = menu.addAction("删除难负样本")
            selected_action = menu.exec(global_pos)
            if selected_action == sam_action:
                self.sat_annotation_mode = "sam3"
                self.log("Satellite annotation mode switched to SAM3")
            elif selected_action == manual_action:
                self.sat_annotation_mode = "manual"
                self.log("Satellite annotation mode switched to manual polygon")
            elif delete_hard_negative_action is not None and selected_action == delete_hard_negative_action:
                self.delete_hard_negative()
                return
            self.active_viewer_name = "sat"
            self._apply_annotation_mode()
        finally:
            self.context_menu_open = False

    def next_case_color(self) -> str:
        cases = self.repo.list_annotation_cases(self.current_pair_id) if self.current_pair_id else []
        return CASE_COLORS[len(cases) % len(CASE_COLORS)]

    def toggle_satellite_view(self) -> None:
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        if not self._case_has_hard_negative(case):
            QMessageBox.information(self, "Hard Negative", "Current case does not have a hard negative image yet.")
            return
        self.hard_negative_capture_active = False
        self.satellite_view_mode = "hard_negative" if self.satellite_view_mode == "original" else "original"
        self.active_viewer_name = "sat"
        self.refresh_viewers()
        self._apply_annotation_mode()

    def start_hard_negative_capture(self) -> None:
        pair = self.repo.get_pair(self.current_pair_id) if self.current_pair_id else None
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        if not pair or not case:
            QMessageBox.information(self, "Hard Negative", "Select a pair and case first.")
            return
        self.satellite_view_mode = "original"
        self.hard_negative_capture_active = True
        self.active_viewer_name = "sat"
        self.sat_view.clear_draft()
        self.refresh_satellite_view(pair=pair)
        self.sat_view.fit_image()
        self.sat_view.setFocus()
        self._apply_annotation_mode()
        self.log(f"Hard negative capture armed for case {case.case_name}")

    def delete_hard_negative(self) -> None:
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        if not case:
            QMessageBox.information(self, "Hard Negative", "Select a case first.")
            return
        pending = self._get_pending_hard_negative(case.case_id)
        if not pending and not self._case_has_hard_negative(case):
            QMessageBox.information(self, "Hard Negative", "Current case does not have a hard negative image.")
            return
        reply = QMessageBox.question(self, "Delete Hard Negative", "Delete the current case hard negative image?")
        if reply != QMessageBox.Yes:
            return

        paths_to_remove = set()
        if pending:
            pending_path = pending.get("image_path", "")
            if pending_path:
                paths_to_remove.add(pending_path)
        if case.hard_negative_image_path:
            paths_to_remove.add(case.hard_negative_image_path)

        self.pending_hard_negative_by_case.pop(case.case_id, None)
        for path_str in paths_to_remove:
            path = Path(path_str)
            if path.exists():
                path.unlink()

        case.hard_negative_image_path = ""
        case.hard_negative_bbox = []
        self.hard_negative_capture_active = False
        self.satellite_view_mode = "original"
        self.repo.save_annotation_case(case)
        self.load_pair(self.current_pair_id, keep_case_id=case.case_id)
        self.log(f"Deleted hard negative for case {case.case_name}")

    def _render_hard_negative_preview(self, bbox: list[float]) -> None:
        pair = self.repo.get_pair(self.current_pair_id)
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        if not pair or not case:
            return

        output_dir = self.workspace_dir / "cases" / case.case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hard_negative.png"

        with Image.open(pair.sat_image_path) as raw_image:
            image = raw_image.convert("RGB")
            image_width, image_height = image.size
            x1 = max(0, min(int(round(bbox[0])), image_width - 1))
            y1 = max(0, min(int(round(bbox[1])), image_height - 1))
            x2 = max(x1 + 1, min(int(round(bbox[2])), image_width))
            y2 = max(y1 + 1, min(int(round(bbox[3])), image_height))
            cropped = image.crop((x1, y1, x2, y2))
            resized = cropped.resize((image_width, image_height), Image.Resampling.LANCZOS)
            resized.save(output_path)

        self.pending_hard_negative_by_case[case.case_id] = {
            "image_path": str(output_path),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        }
        self.hard_negative_capture_active = False
        self.satellite_view_mode = "hard_negative"
        self.refresh_case_editor()
        self.refresh_viewers()
        self._apply_annotation_mode()
        self.log(f"Generated hard negative preview for case {case.case_name}: {output_path}")

    def create_case(self) -> None:
        if not self.current_pair_id:
            QMessageBox.information(self, "New Case", "Select a pair first.")
            return
        case_type, ok = QInputDialog.getItem(
            self,
            "Case Type",
            "Choose case type:",
            CASE_TYPES,
            editable=False,
        )
        if not ok:
            return
        suggested_name = f"{case_type}_{len(self.repo.list_annotation_cases(self.current_pair_id)) + 1}"
        case_name, ok = QInputDialog.getText(self, "Case Name", "Input case name:", text=suggested_name)
        if not ok or not case_name.strip():
            return
        case = self.repo.create_annotation_case(
            pair_id=self.current_pair_id,
            case_name=case_name.strip(),
            case_type=case_type,
            color_hex=self.next_case_color(),
        )
        pair = self.repo.get_pair(self.current_pair_id)
        if pair:
            case.category = self._default_category_for_class(pair.original_class)
            self.repo.save_annotation_case(case)
        self.current_case_id = case.case_id
        self.load_pair(self.current_pair_id, keep_case_id=case.case_id)

    def delete_current_case(self) -> None:
        if not self.current_case_id:
            return
        reply = QMessageBox.question(self, "Delete Case", "Delete the current case?")
        if reply != QMessageBox.Yes:
            return
        case_id = self.current_case_id
        self.current_case_id = ""
        self.pending_hard_negative_by_case.pop(case_id, None)
        self.repo.delete_annotation_case(case_id)
        self.load_pair(self.current_pair_id)

    def save_case_metadata(self) -> None:
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        if not case:
            return
        case.case_name = self.case_name_edit.text().strip() or case.case_name
        case.case_type = normalize_case_type(self.case_type_combo.currentText())
        case.category = self.case_category_combo.currentText().strip()
        case.status = self.case_status_combo.currentText()
        case.description = self.case_description_edit.toPlainText().strip()
        case.notes = self.case_notes_edit.toPlainText().strip()
        pending_hard_negative = self._get_pending_hard_negative(case.case_id)
        if pending_hard_negative:
            case.hard_negative_image_path = pending_hard_negative.get("image_path", "")
            case.hard_negative_bbox = list(pending_hard_negative.get("bbox", []))
        self.repo.save_annotation_case(case)
        if case.hard_negative_image_path:
            self.log(
                "Saved case metadata | "
                f"case={case.case_name} | hard_negative={case.hard_negative_image_path} | "
                "note=Save Case Metadata only stores metadata; use Save Anno to export into saved_anno/"
            )
        else:
            self.log(f"Saved case metadata | case={case.case_name} | hard_negative=none")
        self.load_pair(self.current_pair_id, keep_case_id=case.case_id)

    def handle_ctrl_save(self) -> None:
        if self.annotation_toggle.isChecked():
            target_view = self.uav_view if self.active_viewer_name == "uav" else self.sat_view
            if target_view.has_draft():
                self.save_current_draft()
                return
        self.save_case_metadata()

    def clear_current_draft(self) -> None:
        target_view = self.uav_view if self.active_viewer_name == "uav" else self.sat_view
        if target_view.has_draft():
            target_view.clear_draft()
            self.log(f"Cleared draft on {self.active_viewer_name} viewer")
            return
        if self.hard_negative_capture_active and self.active_viewer_name == "sat":
            self.hard_negative_capture_active = False
            self._apply_annotation_mode()
            self.log("Cancelled hard negative capture mode")

    def _on_draft_completed(self, viewer_name: str, draft_type: str) -> None:
        if viewer_name == "sat" and draft_type == "bbox" and self.hard_negative_capture_active:
            draft = self.sat_view.get_draft()
            self.sat_view.clear_draft()
            if draft.get("kind") == "bbox":
                self._render_hard_negative_preview(draft["bbox"])
            return
        if viewer_name == "sat" and draft_type == "bbox" and self.sat_annotation_mode == "sam3":
            self.run_sam3_preview()

    def run_sam3_preview(self) -> None:
        if not self.current_case_id:
            QMessageBox.information(self, "SAM3", "Create or select a case before running SAM3.")
            self.sat_view.clear_draft()
            return
        if self.sam3_state != "ready":
            self._schedule_sam3_preload()
            self._apply_annotation_mode()
            if self.sam3_state in {"starting", "busy"}:
                QMessageBox.information(self, "SAM3", "SAM3 is still loading in background. Please wait a moment.")
            else:
                message = self.sam3_message or "SAM3 is not ready yet."
                QMessageBox.warning(self, "SAM3 Unavailable", message)
            return
        draft = self.sat_view.get_draft()
        if draft.get("kind") != "bbox":
            return

        pair = self.repo.get_pair(self.current_pair_id)
        case = self.repo.get_annotation_case(self.current_case_id)
        if not pair or not case:
            return

        preview_dir = self.workspace_dir / "sam3_preview" / case.case_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{self.repo.generate_id('preview')}.png"
        request_id = self.repo.generate_id("sam3req")
        self.sam3_pending_preview = {
            "request_id": request_id,
            "case_id": case.case_id,
            "bbox": [float(v) for v in draft["bbox"]],
            "mask_path": str(preview_path),
        }
        self.log("Running SAM3 on current satellite bbox...")
        ok, message = self.sam3_client.request_segment(
            request_id=request_id,
            image_path=pair.sat_image_path,
            bbox_xyxy=draft["bbox"],
            output_mask_path=str(preview_path),
        )
        if not ok:
            self.sam3_pending_preview = None
            QMessageBox.warning(self, "SAM3 Unavailable", message)
            self.log(f"SAM3 request rejected: {message}")

    def save_current_draft(self) -> None:
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        if not case:
            QMessageBox.information(self, "Save Draft", "Create or select a case first.")
            return

        target_view = self.uav_view if self.active_viewer_name == "uav" else self.sat_view
        draft = target_view.get_draft()
        if not draft:
            return

        annotation_id = self.repo.generate_id("ann")
        if self.active_viewer_name == "uav":
            case.uav_annotations.append(
                {
                    "annotation_id": annotation_id,
                    "kind": "bbox",
                    "bbox": [float(v) for v in draft["bbox"]],
                }
            )
            self.log(f"Saved UAV bbox to case {case.case_name}")
        else:
            if draft["kind"] == "polygon":
                if len(draft["points"]) < 3:
                    QMessageBox.information(self, "Save Draft", "Manual polygon needs at least 3 points.")
                    return
                mask_path = self._save_polygon_mask(case.case_id, annotation_id, draft["points"])
                case.sat_annotations.append(
                    {
                        "annotation_id": annotation_id,
                        "kind": "polygon",
                        "points": draft["points"],
                        "bbox": draft["bbox"],
                        "mask_path": str(mask_path),
                        "source": "manual",
                    }
                )
                self.log(f"Saved manual satellite mask to case {case.case_name}")
            elif draft["kind"] == "sam_mask":
                final_mask_path = self._copy_mask_into_case(case.case_id, annotation_id, draft["mask_path"])
                case.sat_annotations.append(
                    {
                        "annotation_id": annotation_id,
                        "kind": "mask",
                        "bbox": draft["bbox"],
                        "mask_path": str(final_mask_path),
                        "source": "sam3",
                    }
                )
                self.log(f"Saved SAM3 satellite mask to case {case.case_name}")
            else:
                QMessageBox.information(self, "Save Draft", "Satellite draft is not ready yet.")
                return

        if case.status == "raw":
            case.status = "in_progress"
        self.repo.save_annotation_case(case)
        target_view.clear_draft()
        self.load_pair(self.current_pair_id, keep_case_id=case.case_id)

    def _save_polygon_mask(self, case_id: str, annotation_id: str, points: list[list[float]]) -> Path:
        pair = self.repo.get_pair(self.current_pair_id)
        image = Image.open(pair.sat_image_path).convert("RGB")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
        output_dir = self.workspace_dir / "cases" / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{annotation_id}.png"
        mask.save(output_path)
        return output_path

    def _copy_mask_into_case(self, case_id: str, annotation_id: str, source_mask_path: str) -> Path:
        output_dir = self.workspace_dir / "cases" / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{annotation_id}.png"
        shutil.copy2(source_mask_path, output_path)
        return output_path

    def save_case_annotation_bundle(self) -> None:
        case = self.repo.get_annotation_case(self.current_case_id) if self.current_case_id else None
        pair = self.repo.get_pair(self.current_pair_id) if self.current_pair_id else None
        if not case or not pair:
            QMessageBox.information(self, "Save Anno", "Select a case first.")
            return

        self.save_case_metadata()
        case = self.repo.get_annotation_case(self.current_case_id)
        safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", case.case_name).strip("_") or case.case_id
        case_dir = self.saved_anno_root / f"{pair.pair_id}__{case.case_id}__{safe_name}"

        if case_dir.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite Existing Annotation",
                f"{case_dir} already exists.\nDo you want to overwrite it?",
            )
            if reply != QMessageBox.Yes:
                return
            shutil.rmtree(case_dir)

        case_dir.mkdir(parents=True, exist_ok=True)
        uav_dst = case_dir / Path(pair.uav_image_path).name
        sat_dst = case_dir / Path(pair.sat_image_path).name
        shutil.copy2(pair.uav_image_path, uav_dst)
        shutil.copy2(pair.sat_image_path, sat_dst)
        hard_negative_dst = ""
        if case.hard_negative_image_path and Path(case.hard_negative_image_path).exists():
            hard_negative_path = Path(case.hard_negative_image_path)
            hard_negative_dst_path = case_dir / hard_negative_path.name
            shutil.copy2(hard_negative_path, hard_negative_dst_path)
            hard_negative_dst = str(hard_negative_dst_path)

        summary = {
            "pair_id": pair.pair_id,
            "split": pair.split,
            "original_class": pair.original_class,
            "case_id": case.case_id,
            "case_name": case.case_name,
            "case_type": case.case_type,
            "category": case.category,
            "status": "done",
            "description": case.description,
            "notes": case.notes,
            "color_hex": case.color_hex,
            "uav_image_path": str(uav_dst),
            "sat_image_path": str(sat_dst),
            "hard_negative_image_path": hard_negative_dst,
            "hard_negative_bbox": case.hard_negative_bbox,
            "uav_count": len(case.uav_annotations),
            "sat_count": len(case.sat_annotations),
        }
        (case_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for idx, ann in enumerate(case.uav_annotations, start=1):
            payload = {
                "pair_id": pair.pair_id,
                "case_id": case.case_id,
                "case_type": case.case_type,
                "category": case.category,
                "uav_image_path": str(uav_dst),
                **ann,
            }
            (case_dir / f"uav_obj_{idx:03d}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        for idx, ann in enumerate(case.sat_annotations, start=1):
            payload = {
                "pair_id": pair.pair_id,
                "case_id": case.case_id,
                "case_type": case.case_type,
                "category": case.category,
                "sat_image_path": str(sat_dst),
                **ann,
            }
            mask_path = ann.get("mask_path", "")
            if mask_path and Path(mask_path).exists():
                dst = case_dir / f"sat_mask_{idx:03d}.png"
                shutil.copy2(mask_path, dst)
                payload["mask_path"] = str(dst)
            (case_dir / f"sat_obj_{idx:03d}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        case.status = "done"
        self.repo.save_annotation_case(case)
        self.load_pair(self.current_pair_id, keep_case_id=case.case_id)
        QMessageBox.information(
            self,
            "Save Anno",
            f"Annotation saved successfully.\nPath:\n{case_dir}",
        )
        self.log(f"Saved annotation bundle to {case_dir}")

    def export_uav_view_image(self) -> None:
        pair = self.repo.get_pair(self.current_pair_id) if self.current_pair_id else None
        if not pair:
            QMessageBox.information(self, "下载图像", "请先选择一个 pair。")
            return
        default_path = self._default_download_dir() / f"{self._pair_export_stem()}.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 UAV 图像",
            str(default_path),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not save_path:
            return
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = self.uav_view.render_to_image()
        if not image.save(str(path)):
            QMessageBox.warning(self, "下载图像", f"保存失败：{path}")
            return
        self.ui_state["uav_export_dir"] = str(path.parent)
        self._save_ui_state()
        self.log(f"Saved UAV viewer image to {path}")
        QMessageBox.information(self, "下载图像", f"图像已保存到：\n{path}")

    def export_workspace(self) -> None:
        output_dir = QFileDialog.getExistingDirectory(self, "Choose Export Directory", str(self.workspace_dir))
        if not output_dir:
            return
        exporter = Exporter(self.repo, self.workspace_dir)
        result = exporter.export_all(Path(output_dir))
        self.log("Export finished:")
        for key, value in result.items():
            self.log(f"{key}: {value}")
