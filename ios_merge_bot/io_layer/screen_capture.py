"""Continuous frame ingestion for the vision pipeline.

Two interchangeable backends behind a common :class:`FrameSource` interface:

* :class:`WDAMJPEGSource` - reads the WebDriverAgent MJPEG stream over HTTP.
* :class:`MSSWindowSource` - grabs a region of the macOS desktop (a QuickTime
  iPhone mirror window) using ``mss`` for low-latency native capture.

Both run a background grabber thread and expose the most recent BGR frame via
:meth:`FrameSource.latest`, so the perception loop never blocks on I/O.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..config import BotConfig, CaptureBackend

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract threaded source of BGR frames (``np.ndarray``, HxWx3, uint8)."""

    def __init__(self, target_fps: int = 10) -> None:
        self._target_fps = max(1, target_fps)
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    # -- subclasses implement frame acquisition ------------------------- #
    @abstractmethod
    def _open(self) -> None:
        """Open the underlying stream / capture handle."""

    @abstractmethod
    def _grab(self) -> Optional[np.ndarray]:
        """Return a single BGR frame, or None if unavailable this tick."""

    @abstractmethod
    def _release(self) -> None:
        """Release the underlying stream / capture handle."""

    # -- lifecycle ------------------------------------------------------ #
    def start(self) -> "FrameSource":
        if self._thread is not None:
            return self
        self._open()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name=type(self).__name__, daemon=True)
        self._thread.start()
        logger.info("%s started (target %d fps)", type(self).__name__, self._target_fps)
        return self

    def _loop(self) -> None:
        period = 1.0 / self._target_fps
        while self._running.is_set():
            tic = time.time()
            try:
                frame = self._grab()
                if frame is not None:
                    with self._lock:
                        self._latest = frame
            except Exception as err:  # noqa: BLE001 - keep the grabber alive
                logger.warning("%s grab error: %s", type(self).__name__, err)
            elapsed = time.time() - tic
            if elapsed < period:
                time.sleep(period - elapsed)

    def latest(self, timeout: float = 5.0) -> np.ndarray:
        """Return the most recent frame, waiting up to ``timeout`` for the first."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.01)
        raise TimeoutError("No frame received from capture source within timeout")

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()
        logger.info("%s stopped", type(self).__name__)

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class WDAMJPEGSource(FrameSource):
    """Reads frames from the WebDriverAgent MJPEG stream."""

    def __init__(self, mjpeg_url: str, target_fps: int = 10) -> None:
        super().__init__(target_fps)
        self._url = mjpeg_url
        self._cap = None  # cv2.VideoCapture

    def _open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self._url)
        if not self._cap.isOpened():
            raise ConnectionError(f"Unable to open MJPEG stream at {self._url}")

    def _grab(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()  # type: ignore[union-attr]
        if not ok:
            return None
        return frame  # already BGR

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class MSSWindowSource(FrameSource):
    """Captures a fixed region of the macOS desktop using ``mss``.

    Intended for a QuickTime iPhone mirror window. ``bbox`` is
    ``(left, top, width, height)`` in display pixels.
    """

    def __init__(self, bbox: Tuple[int, int, int, int], target_fps: int = 10) -> None:
        super().__init__(target_fps)
        if bbox is None:
            raise ValueError("MSSWindowSource requires a window_bbox (left, top, w, h)")
        self._monitor = {"left": bbox[0], "top": bbox[1], "width": bbox[2], "height": bbox[3]}
        self._sct = None

    def _open(self) -> None:
        import mss

        self._sct = mss.mss()

    def _grab(self) -> Optional[np.ndarray]:
        import cv2

        shot = self._sct.grab(self._monitor)  # type: ignore[union-attr]
        # mss returns BGRA; drop alpha to BGR for OpenCV consistency.
        frame = np.asarray(shot)[:, :, :3]
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame

    def _release(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


def build_frame_source(config: BotConfig, mjpeg_url: str) -> FrameSource:
    """Factory selecting a capture backend from configuration."""
    capture = config.capture
    if capture.backend == CaptureBackend.WDA_MJPEG:
        return WDAMJPEGSource(mjpeg_url, target_fps=capture.target_fps)
    if capture.backend == CaptureBackend.MSS_WINDOW:
        return MSSWindowSource(capture.window_bbox, target_fps=capture.target_fps)
    raise ValueError(f"Unknown capture backend: {capture.backend}")
