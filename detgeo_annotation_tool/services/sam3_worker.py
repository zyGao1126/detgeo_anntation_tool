from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .sam3_backend import SAM3Backend


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DetGeo SAM3 worker")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = SAM3Backend(repo_root=args.repo_root, checkpoint_path=args.checkpoint)
    ok, message = backend.preload()
    if not ok:
        _emit({"event": "ready", "ok": False, "message": message})
        return 1

    _emit({"event": "ready", "ok": True, "device": backend.device})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"event": "error", "message": f"Invalid JSON command: {exc}"})
            continue

        cmd = command.get("cmd", "")
        if cmd == "shutdown":
            _emit({"event": "shutdown", "ok": True})
            return 0

        if cmd != "segment":
            _emit({"event": "error", "message": f"Unknown command: {cmd}"})
            continue

        request_id = command.get("request_id", "")
        image_path = command.get("image_path", "")
        bbox_xyxy = command.get("bbox_xyxy", [])
        output_mask_path = command.get("output_mask_path", "")

        try:
            result = backend.segment_from_bbox(image_path=image_path, bbox_xyxy=bbox_xyxy)
            output_path = Path(output_mask_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.mask.save(output_path)
            _emit(
                {
                    "event": "segment_result",
                    "ok": True,
                    "request_id": request_id,
                    "bbox_xyxy": result.bbox_xyxy,
                    "score": result.score,
                    "mask_path": str(output_path),
                }
            )
        except Exception as exc:
            _emit(
                {
                    "event": "segment_result",
                    "ok": False,
                    "request_id": request_id,
                    "message": str(exc),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
