"""WebDriverAgent connection and session management.

Wraps ``facebook-wda`` (imported as ``wda``) to provide a resilient connection,
auto-detection of the device's logical screen size and Retina scale factor, and
a single ``CoordinateMapper`` instance shared across the codebase.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from ..config import BotConfig, DeviceConfig
from .coordinates import CoordinateMapper

logger = logging.getLogger(__name__)

try:  # facebook-wda is an optional runtime dependency.
    import wda  # type: ignore
except ImportError:  # pragma: no cover - import guard for environments without WDA
    wda = None  # type: ignore


class WDAClient:
    """Manages a WebDriverAgent client + session for an iOS device."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.device_cfg: DeviceConfig = config.device
        self._client: Optional["wda.Client"] = None
        self._session: Optional["wda.Session"] = None
        self._mapper: Optional[CoordinateMapper] = None

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(self, retries: int = 5, backoff: float = 1.5) -> "WDAClient":
        """Establish the WDA client + session, retrying with backoff."""
        if wda is None:
            raise RuntimeError(
                "facebook-wda is not installed. Run `pip install facebook-wda`."
            )

        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                logger.info("Connecting to WDA at %s (attempt %d/%d)",
                            self.device_cfg.wda_url, attempt, retries)
                self._client = wda.Client(self.device_cfg.wda_url)
                # status() raises if WDA is unreachable.
                status = self._client.status()
                logger.debug("WDA status: %s", status)
                self._session = self._client.session()
                self._detect_geometry()
                logger.info("WDA session established.")
                return self
            except Exception as err:  # noqa: BLE001 - surface after retries
                last_err = err
                wait = backoff ** attempt
                logger.warning("WDA connection failed: %s (retrying in %.1fs)", err, wait)
                time.sleep(wait)

        raise ConnectionError(f"Could not connect to WDA after {retries} attempts") from last_err

    def _detect_geometry(self) -> None:
        """Resolve logical screen size + Retina scale factor and build mapper."""
        assert self._client is not None and self._session is not None

        logical_size = self.device_cfg.logical_size
        if logical_size is None:
            size = self._session.window_size()  # WDA Size(width, height) in points
            logical_size = (int(size.width), int(size.height))
            logger.info("Detected logical screen size: %s", logical_size)

        scale = self.device_cfg.scale_factor
        if scale is None:
            scale = self._estimate_scale_factor(logical_size)
            logger.info("Estimated Retina scale factor: %.2f", scale)

        self._mapper = CoordinateMapper(
            geometry=self.config.board,
            logical_size=logical_size,
            scale_factor=scale,
        )

    def _estimate_scale_factor(self, logical_size: Tuple[int, int]) -> float:
        """Estimate points->pixels ratio by comparing a screenshot to logical size."""
        try:
            import io

            from PIL import Image

            png_bytes = self._client.screenshot(format="raw")  # type: ignore[union-attr]
            img = Image.open(io.BytesIO(png_bytes))
            return img.width / logical_size[0]
        except Exception as err:  # noqa: BLE001
            logger.warning("Scale-factor auto-detect failed (%s); defaulting to 3.0", err)
            return 3.0

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> "wda.Client":
        if self._client is None:
            raise RuntimeError("WDAClient.connect() must be called first")
        return self._client

    @property
    def session(self) -> "wda.Session":
        if self._session is None:
            raise RuntimeError("WDAClient.connect() must be called first")
        return self._session

    @property
    def mapper(self) -> CoordinateMapper:
        if self._mapper is None:
            raise RuntimeError("WDAClient.connect() must be called first")
        return self._mapper

    @property
    def mjpeg_url(self) -> str:
        """MJPEG stream URL derived from the WDA base URL + configured port."""
        base = self.device_cfg.wda_url.rsplit(":", 1)[0]
        if not base.startswith("http"):
            base = self.device_cfg.wda_url
        return f"{base}:{self.config.capture.mjpeg_port}"

    def close(self) -> None:
        """Tear down the WDA session."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception as err:  # noqa: BLE001
                logger.debug("Error closing WDA session: %s", err)
        self._session = None
        self._client = None

    def __enter__(self) -> "WDAClient":
        return self.connect()

    def __exit__(self, *exc_info: object) -> None:
        self.close()
