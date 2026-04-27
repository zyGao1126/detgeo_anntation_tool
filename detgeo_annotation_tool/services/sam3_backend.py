from __future__ import annotations

from contextlib import nullcontext
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class SAM3Result:
    bbox_xyxy: list[float]
    mask: Image.Image
    score: float


class SAM3Backend:
    def __init__(self, repo_root: Path, checkpoint_path: Path):
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)
        self._processor = None
        self._device = None
        self._load_error = ""
        self._torch = None
        self._autocast_dtype = None

    def is_available(self) -> tuple[bool, str]:
        if not self.repo_root.exists():
            return False, f"SAM3 repo not found: {self.repo_root}"
        if not self.checkpoint_path.exists():
            return False, f"SAM3 checkpoint not found: {self.checkpoint_path}"
        return True, ""

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cuda":
                self._autocast_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                self._autocast_dtype = None
            model = build_sam3_image_model(
                checkpoint_path=str(self.checkpoint_path),
                load_from_HF=False,
                device=self._device,
                eval_mode=True,
            )
            self._processor = Sam3Processor(model, device=self._device, confidence_threshold=0.0)
            self._load_error = ""
        except Exception as exc:
            self._load_error = str(exc)
            raise

    def _inference_context(self):
        if self._torch is None or self._device != "cuda" or self._autocast_dtype is None:
            return nullcontext()
        return self._torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

    def preload(self) -> tuple[bool, str]:
        available, reason = self.is_available()
        if not available:
            self._load_error = reason
            return False, reason
        try:
            self._ensure_loaded()
            return True, ""
        except Exception as exc:
            self._load_error = str(exc)
            return False, self._load_error

    def is_loaded(self) -> bool:
        return self._processor is not None

    @property
    def device(self) -> str:
        return self._device or ""

    @property
    def load_error(self) -> str:
        return self._load_error

    def segment_from_bbox(self, image_path: str, bbox_xyxy: list[float]) -> SAM3Result:
        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        x1, y1, x2, y2 = bbox_xyxy
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        bw = abs(x2 - x1) / img_w
        bh = abs(y2 - y1) / img_h
        with self._inference_context():
            state = self._processor.set_image(image)
            output = self._processor.add_geometric_prompt(
                box=[cx, cy, bw, bh],
                label=True,
                state=state,
            )

        scores = output["scores"].detach().float().cpu().numpy()
        if scores.size == 0:
            raise RuntimeError("SAM3 returned no mask candidates for this box.")
        best_index = int(np.argmax(scores))
        mask_np = (
            output["masks"][best_index]
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
            * 255
        )
        box = output["boxes"][best_index].detach().float().cpu().numpy().tolist()
        return SAM3Result(
            bbox_xyxy=[float(v) for v in box],
            mask=Image.fromarray(mask_np, mode="L"),
            score=float(scores[best_index]),
        )
