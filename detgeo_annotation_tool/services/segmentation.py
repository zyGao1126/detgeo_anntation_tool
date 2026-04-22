from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass
class SegmentationCandidate:
    mask: Image.Image
    source: str


class SegmentationBackend:
    def run(
        self,
        satellite_image_path: str,
        bbox: list[float] | None = None,
        rbox: list[float] | None = None,
    ) -> SegmentationCandidate:
        raise NotImplementedError


class DummyRectangleSegmentationBackend(SegmentationBackend):
    def run(
        self,
        satellite_image_path: str,
        bbox: list[float] | None = None,
        rbox: list[float] | None = None,
    ) -> SegmentationCandidate:
        image = Image.open(satellite_image_path).convert("RGB")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        if rbox and len(rbox) >= 8:
            polygon = [(rbox[idx], rbox[idx + 1]) for idx in range(0, 8, 2)]
            draw.polygon(polygon, fill=255)
            source = "dummy_rbox_fill"
        elif bbox and len(bbox) == 4:
            draw.rectangle(tuple(bbox), fill=255)
            source = "dummy_bbox_fill"
        else:
            raise ValueError("Segmentation backend needs bbox or rbox input")
        return SegmentationCandidate(mask=mask, source=source)

    @staticmethod
    def save_candidate(candidate: SegmentationCandidate, output_path: Path) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.mask.save(output_path)
        return str(output_path)
