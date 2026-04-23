from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QContextMenuEvent,
    QFocusEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)


def _bbox_from_points(points: list[QPointF]) -> list[float]:
    xs = [point.x() for point in points]
    ys = [point.y() for point in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


class AnnotatedImageView(QGraphicsView):
    viewerActivated = Signal(str)
    draftCompleted = Signal(str, str)
    contextMenuRequested = Signal(str, object)

    def __init__(self, viewer_name: str, parent=None):
        super().__init__(parent)
        self.viewer_name = viewer_name
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor(28, 30, 36))

        self.image_item: QGraphicsPixmapItem | None = None
        self.image_size = (0, 0)
        self.active_tool = "pan"
        self.annotation_mode = False

        self.saved_items: list = []
        self.reference_items: list = []
        self.draft_items: list = []
        self.assist_items: list = []

        self.draw_start: QPointF | None = None
        self.draw_rect_item: QGraphicsRectItem | None = None
        self.crop_overlay_item: QGraphicsPathItem | None = None
        self.draft_kind = ""
        self.draft_bbox: list[float] = []
        self.draft_points: list[QPointF] = []
        self.preview_pos: QPointF | None = None
        self.draft_mask_path = ""
        self.crop_assist_enabled = False
        self.crop_focus_bbox: list[float] = []

    def focusInEvent(self, event: QFocusEvent) -> None:
        self.viewerActivated.emit(self.viewer_name)
        super().focusInEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self.viewerActivated.emit(self.viewer_name)
        self.contextMenuRequested.emit(self.viewer_name, event.globalPos())
        event.accept()

    def set_image(self, image_path: str) -> None:
        self.scene().clear()
        self.saved_items.clear()
        self.reference_items.clear()
        self.draft_items.clear()
        self.assist_items.clear()
        self.image_item = None
        self.image_size = (0, 0)
        self.crop_overlay_item = None
        self.clear_draft()

        if not image_path or not Path(image_path).exists():
            return

        pixmap = QPixmap(image_path)
        self.image_item = self.scene().addPixmap(pixmap)
        self.image_item.setZValue(0)
        self.image_size = (pixmap.width(), pixmap.height())
        self.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def fit_image(self) -> None:
        if self.image_item:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def set_annotation_tool(self, tool_name: str, enabled: bool) -> None:
        self.active_tool = tool_name
        self.annotation_mode = enabled
        self.setDragMode(QGraphicsView.NoDrag if enabled and tool_name != "pan" else QGraphicsView.ScrollHandDrag)

    def set_crop_assist(self, enabled: bool, focus_bbox: list[float] | None = None) -> None:
        self.crop_assist_enabled = enabled
        self.crop_focus_bbox = [float(v) for v in (focus_bbox or [])]
        self._clear_assist_items()
        self._clear_crop_overlay()
        if not enabled or len(self.crop_focus_bbox) != 4 or not self.image_item:
            return

        x1, y1, x2, y2 = self.crop_focus_bbox
        rect = QRectF(x1, y1, x2 - x1, y2 - y1).normalized()
        focus_rect = QGraphicsRectItem(rect)
        focus_rect.setPen(QPen(QColor(255, 215, 0), 3.0, Qt.DashLine))
        focus_rect.setZValue(40)
        self.scene().addItem(focus_rect)
        self.assist_items.append(focus_rect)

        cx = rect.center().x()
        cy = rect.center().y()
        cross_color = QColor(255, 215, 0, 180)
        h_line = QGraphicsLineItem(rect.left(), cy, rect.right(), cy)
        v_line = QGraphicsLineItem(cx, rect.top(), cx, rect.bottom())
        for line in (h_line, v_line):
            line.setPen(QPen(cross_color, 1.5, Qt.DotLine))
            line.setZValue(40)
            self.scene().addItem(line)
            self.assist_items.append(line)

    def clear_draft(self) -> None:
        if self.draw_rect_item:
            self.scene().removeItem(self.draw_rect_item)
            self.draw_rect_item = None
        self._clear_crop_overlay()
        for item in self.draft_items:
            self.scene().removeItem(item)
        self.draft_items.clear()
        self.draw_start = None
        self.draft_kind = ""
        self.draft_bbox = []
        self.draft_points = []
        self.preview_pos = None
        self.draft_mask_path = ""

    def has_draft(self) -> bool:
        return bool(self.draft_kind)

    def get_draft(self) -> dict:
        if self.draft_kind == "bbox":
            return {"kind": "bbox", "bbox": list(self.draft_bbox)}
        if self.draft_kind == "polygon":
            return {
                "kind": "polygon",
                "points": [[float(point.x()), float(point.y())] for point in self.draft_points],
                "bbox": _bbox_from_points(self.draft_points),
            }
        if self.draft_kind == "sam_mask":
            return {
                "kind": "sam_mask",
                "bbox": list(self.draft_bbox),
                "mask_path": self.draft_mask_path,
            }
        return {}

    def set_draft_mask(self, mask_path: str, bbox: list[float], color_hex: str) -> None:
        self.clear_draft()
        self.draft_kind = "sam_mask"
        self.draft_bbox = [float(v) for v in bbox]
        self.draft_mask_path = mask_path
        color = QColor(color_hex)
        self._add_mask_overlay(mask_path, color, alpha=120, z_value=60, store_draft=True)
        rect = QRectF(
            self.draft_bbox[0],
            self.draft_bbox[1],
            self.draft_bbox[2] - self.draft_bbox[0],
            self.draft_bbox[3] - self.draft_bbox[1],
        )
        item = QGraphicsRectItem(rect.normalized())
        item.setPen(QPen(color, 2.5, Qt.DashLine))
        item.setZValue(61)
        self.scene().addItem(item)
        self.draft_items.append(item)

    def set_saved_cases(self, cases: list, side: str, selected_case_id: str) -> None:
        for item in self.saved_items:
            self.scene().removeItem(item)
        self.saved_items.clear()

        for case in cases:
            color = QColor(case.color_hex)
            selected = case.case_id == selected_case_id
            pen = QPen(color, 3.2 if selected else 2.0)
            brush_color = QColor(color)
            brush_color.setAlpha(50 if selected else 28)
            annotations = case.uav_annotations if side == "uav" else case.sat_annotations
            for ann in annotations:
                bbox = ann.get("bbox", [])
                points = ann.get("points", [])
                if points:
                    polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
                    poly_item = QGraphicsPolygonItem(polygon)
                    poly_item.setPen(pen)
                    poly_item.setBrush(brush_color)
                    poly_item.setZValue(10)
                    self.scene().addItem(poly_item)
                    self.saved_items.append(poly_item)
                elif len(bbox) == 4:
                    rect = QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]).normalized()
                    rect_item = QGraphicsRectItem(rect)
                    rect_item.setPen(pen)
                    rect_item.setZValue(10)
                    self.scene().addItem(rect_item)
                    self.saved_items.append(rect_item)
                if side == "sat" and ann.get("mask_path"):
                    self._add_mask_overlay(
                        ann["mask_path"],
                        color,
                        alpha=80 if selected else 45,
                        z_value=8,
                        store_draft=False,
                    )

    def set_reference_data(self, pair, enabled: bool) -> None:
        for item in self.reference_items:
            self.scene().removeItem(item)
        self.reference_items.clear()
        if not enabled or not pair or not self.image_item:
            return

        ref_color = QColor(230, 60, 60)
        ref_pen = QPen(ref_color, 2.0, Qt.DashLine)

        if self.viewer_name == "uav" and len(pair.original_click_xy) == 2:
            cx, cy = pair.original_click_xy
            lines = [
                QGraphicsLineItem(cx - 10, cy, cx + 10, cy),
                QGraphicsLineItem(cx, cy - 10, cx, cy + 10),
            ]
            for line in lines:
                line.setPen(ref_pen)
                line.setZValue(12)
                self.scene().addItem(line)
                self.reference_items.append(line)
            if len(pair.query_center_xy) == 2:
                qx, qy = pair.query_center_xy
                circle = QGraphicsEllipseItem(qx - 5, qy - 5, 10, 10)
                circle.setPen(ref_pen)
                circle.setZValue(12)
                self.scene().addItem(circle)
                self.reference_items.append(circle)

        if self.viewer_name == "sat" and len(pair.original_gt_bbox) == 4:
            x1, y1, x2, y2 = pair.original_gt_bbox
            rect = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1).normalized())
            rect.setPen(ref_pen)
            rect.setZValue(12)
            self.scene().addItem(rect)
            self.reference_items.append(rect)
            if pair.original_polygon_xy:
                polygon = QPolygonF(
                    [
                        QPointF(pair.original_polygon_xy[idx], pair.original_polygon_xy[idx + 1])
                        for idx in range(0, len(pair.original_polygon_xy), 2)
                    ]
                )
                poly_item = QGraphicsPolygonItem(polygon)
                poly_item.setPen(ref_pen)
                poly_item.setBrush(QColor(220, 220, 220, 30))
                poly_item.setZValue(11)
                self.scene().addItem(poly_item)
                self.reference_items.append(poly_item)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.viewerActivated.emit(self.viewer_name)
        scene_pos = self._clamp(self.mapToScene(event.position().toPoint()))

        if self.annotation_mode and self.active_tool == "bbox" and event.button() == Qt.LeftButton:
            self.draw_start = scene_pos
            if self.draw_rect_item:
                self.scene().removeItem(self.draw_rect_item)
            self.draw_rect_item = QGraphicsRectItem()
            pen_color = QColor(255, 215, 0) if self.crop_assist_enabled else QColor(80, 210, 255)
            self.draw_rect_item.setPen(QPen(pen_color, 2.0, Qt.DashLine))
            self.draw_rect_item.setZValue(50)
            self.scene().addItem(self.draw_rect_item)
            return

        if self.annotation_mode and self.active_tool == "polygon" and event.button() == Qt.LeftButton:
            self.draft_kind = "polygon"
            self.draft_points.append(scene_pos)
            self.preview_pos = scene_pos
            self._render_polygon_draft()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        scene_pos = self._clamp(self.mapToScene(event.position().toPoint()))
        if self.draw_start is not None and self.draw_rect_item is not None:
            rect = QRectF(self.draw_start, scene_pos).normalized()
            self.draw_rect_item.setRect(rect)
            if self.crop_assist_enabled:
                self._update_crop_overlay(rect)
            return
        if self.annotation_mode and self.active_tool == "polygon" and self.draft_points:
            self.preview_pos = scene_pos
            self._render_polygon_draft()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.annotation_mode and self.active_tool == "polygon" and len(self.draft_points) >= 3:
            self.preview_pos = None
            self._render_polygon_draft(closed=True)
            self.draftCompleted.emit(self.viewer_name, "polygon")
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.draw_start is not None and self.draw_rect_item is not None:
            rect = self.draw_rect_item.rect().normalized()
            self.scene().removeItem(self.draw_rect_item)
            self.draw_rect_item = None
            self.draw_start = None
            self._clear_crop_overlay()
            if rect.width() > 3 and rect.height() > 3:
                self.clear_draft()
                self.draft_kind = "bbox"
                self.draft_bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                draft_rect = QGraphicsRectItem(rect)
                pen_color = QColor(255, 215, 0) if self.crop_assist_enabled else QColor(80, 210, 255)
                draft_rect.setPen(QPen(pen_color, 2.0, Qt.DashLine))
                draft_rect.setZValue(51)
                self.scene().addItem(draft_rect)
                self.draft_items.append(draft_rect)
                self.draftCompleted.emit(self.viewer_name, "bbox")
            return
        super().mouseReleaseEvent(event)

    def _clamp(self, point: QPointF) -> QPointF:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            return point
        return QPointF(
            min(max(point.x(), 0.0), width - 1.0),
            min(max(point.y(), 0.0), height - 1.0),
        )

    def _render_polygon_draft(self, closed: bool = False) -> None:
        for item in self.draft_items:
            self.scene().removeItem(item)
        self.draft_items.clear()

        if not self.draft_points:
            return

        polygon_points = list(self.draft_points)
        if not closed and self.preview_pos is not None:
            polygon_points.append(self.preview_pos)

        if len(polygon_points) >= 2:
            path = QPainterPath(polygon_points[0])
            for point in polygon_points[1:]:
                path.lineTo(point)
            if len(self.draft_points) >= 3:
                path.closeSubpath()
            path_item = QGraphicsPathItem(path)
            path_item.setPen(QPen(QColor(255, 190, 80), 2.0, Qt.DashLine))
            path_item.setZValue(52)
            self.scene().addItem(path_item)
            self.draft_items.append(path_item)

        for point in self.draft_points:
            dot = QGraphicsEllipseItem(point.x() - 3, point.y() - 3, 6, 6)
            dot.setPen(QPen(QColor(255, 190, 80), 1.5))
            dot.setBrush(QColor(255, 190, 80))
            dot.setZValue(53)
            self.scene().addItem(dot)
            self.draft_items.append(dot)

    def _add_mask_overlay(
        self,
        mask_path: str,
        color: QColor,
        alpha: int,
        z_value: float,
        store_draft: bool,
    ) -> None:
        if not mask_path or not Path(mask_path).exists():
            return
        mask_image = QImage(mask_path).convertToFormat(QImage.Format_Grayscale8)
        overlay = QImage(mask_image.size(), QImage.Format_ARGB32)
        overlay.fill(Qt.transparent)
        rgba = QColor(color)
        rgba.setAlpha(alpha)
        for y in range(mask_image.height()):
            for x in range(mask_image.width()):
                value = QColor(mask_image.pixel(x, y)).red()
                if value > 0:
                    overlay.setPixelColor(x, y, rgba)
        item = self.scene().addPixmap(QPixmap.fromImage(overlay))
        item.setZValue(z_value)
        if store_draft:
            self.draft_items.append(item)
        else:
            self.saved_items.append(item)

    def _update_crop_overlay(self, rect: QRectF) -> None:
        if not self.crop_assist_enabled or not self.image_item:
            return
        self._clear_crop_overlay()
        image_rect = QRectF(0, 0, self.image_size[0], self.image_size[1])
        path = QPainterPath()
        path.addRect(image_rect)
        hole = QPainterPath()
        hole.addRect(rect.normalized())
        path = path.subtracted(hole)
        overlay_item = QGraphicsPathItem(path)
        overlay_item.setPen(QPen(Qt.NoPen))
        overlay_item.setBrush(QColor(0, 0, 0, 110))
        overlay_item.setZValue(49)
        self.scene().addItem(overlay_item)
        self.crop_overlay_item = overlay_item

    def _clear_crop_overlay(self) -> None:
        if self.crop_overlay_item is not None:
            self.scene().removeItem(self.crop_overlay_item)
            self.crop_overlay_item = None

    def _clear_assist_items(self) -> None:
        for item in self.assist_items:
            self.scene().removeItem(item)
        self.assist_items.clear()

    def render_to_image(self) -> QImage:
        rect = self.sceneRect().toRect()
        image = QImage(rect.size(), QImage.Format_ARGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        self.scene().render(painter, target=QRectF(image.rect()), source=QRectF(rect))
        painter.end()
        return image
