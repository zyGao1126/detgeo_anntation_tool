from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal


class SAM3ProcessClient(QObject):
    stateChanged = Signal(str, str)
    segmentFinished = Signal(bool, dict)
    logMessage = Signal(str)

    def __init__(self, project_root: Path, repo_root: Path, checkpoint_path: Path, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.project_root))
        self.process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self.process.readyReadStandardError.connect(self._on_stderr_ready)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._state = "idle"
        self._message = ""
        self._device = ""
        self._request_in_flight = ""

    @property
    def state(self) -> str:
        return self._state

    @property
    def message(self) -> str:
        return self._message

    @property
    def device(self) -> str:
        return self._device

    def is_ready(self) -> bool:
        return self._state == "ready"

    def start(self) -> None:
        if self._state in {"starting", "ready", "busy"}:
            return
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._request_in_flight = ""
        self._set_state("starting", "Starting SAM3 worker...")
        self.process.start(
            sys.executable,
            [
                "-u",
                "-m",
                "detgeo_annotation_tool.services.sam3_worker",
                "--repo-root",
                str(self.repo_root),
                "--checkpoint",
                str(self.checkpoint_path),
            ],
        )

    def request_segment(
        self,
        request_id: str,
        image_path: str,
        bbox_xyxy: list[float],
        output_mask_path: str,
    ) -> tuple[bool, str]:
        if not self.is_ready():
            return False, self._message or "SAM3 worker is not ready."
        payload = {
            "cmd": "segment",
            "request_id": request_id,
            "image_path": image_path,
            "bbox_xyxy": [float(v) for v in bbox_xyxy],
            "output_mask_path": output_mask_path,
        }
        self._request_in_flight = request_id
        self._set_state("busy", "Running SAM3 segmentation...")
        self.process.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        return True, ""

    def shutdown(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        self.process.write(b'{"cmd":"shutdown"}\n')
        self.process.waitForFinished(1000)
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            self.process.waitForFinished(1000)
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(1000)

    def _set_state(self, state: str, message: str) -> None:
        self._state = state
        self._message = message
        self.stateChanged.emit(state, message)

    def _on_stdout_ready(self) -> None:
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not chunk:
            return
        self._stdout_buffer += chunk
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_stdout_message(line)

    def _handle_stdout_message(self, line: str) -> None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.logMessage.emit(f"SAM3 worker stdout: {line}")
            return

        event = payload.get("event", "")
        if event == "ready":
            if payload.get("ok"):
                self._device = str(payload.get("device", ""))
                self._set_state("ready", f"SAM3 ready on {self._device or 'unknown'}")
            else:
                self._set_state("failed", str(payload.get("message", "SAM3 failed to start.")))
            return

        if event == "segment_result":
            ok = bool(payload.get("ok"))
            if ok:
                self._set_state("ready", f"SAM3 ready on {self._device or 'unknown'}")
            else:
                self._set_state("ready", self._message or "SAM3 worker is ready.")
            self.segmentFinished.emit(ok, payload)
            self._request_in_flight = ""
            return

        if event == "shutdown":
            self._set_state("stopped", "SAM3 worker stopped.")
            return

        if event == "error":
            message = str(payload.get("message", "SAM3 worker error."))
            self.logMessage.emit(message)
            if self._state == "starting":
                self._set_state("failed", message)
            return

        self.logMessage.emit(f"SAM3 worker event: {payload}")

    def _on_stderr_ready(self) -> None:
        chunk = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if not chunk:
            return
        self._stderr_buffer += chunk
        while "\n" in self._stderr_buffer:
            line, self._stderr_buffer = self._stderr_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logMessage.emit(f"SAM3 stderr: {line}")

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        if self._state not in {"failed", "stopped"}:
            message = f"SAM3 worker exited unexpectedly (code={exit_code})."
            if self._state == "busy":
                self.segmentFinished.emit(False, {"message": message, "request_id": self._request_in_flight})
                self._request_in_flight = ""
            self._set_state("failed", message)

    def _on_process_error(self, error) -> None:
        self._set_state("failed", f"SAM3 worker process error: {error}")
