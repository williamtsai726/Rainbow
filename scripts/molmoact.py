"""Remote MolmoAct HTTP client: build payloads, POST with ``json_numpy``, parse ``actions``.

Same wire format as YAM ``molmoact.MolmoAct`` (``left_cam`` / ``top_cam`` / ``right_cam``,
``instruction``, ``state``). Images are converted to NumPy **once** in ``send_request``, matching
``gello_software/experiments/molmoact.py``. Use with ``eval_molmoact.py`` for RB-Y1; state is 16-D
(left 8 + right 8) matching ``eval.py``."""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter

try:
    import json_numpy

    json_numpy.patch()
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "molmoact requires `json_numpy` for numpy arrays over HTTP. "
        "Install with: pip install json-numpy"
    ) from exc

LOGGER = logging.getLogger(__name__)


class MolmoAct:
    """HTTP client for a MolmoAct-style inference server."""

    def __init__(
        self,
        url: str,
        *,
        multi_views: bool = True,
        request_timeout_sec: float = 120.0,
        extra_headers: Optional[dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.url = url
        self.multi_views = multi_views
        self.request_timeout_sec = request_timeout_sec
        self.extra_headers = dict(extra_headers) if extra_headers else {}
        self._session = session if session is not None else _make_molmoact_session()
        LOGGER.info("MolmoAct client url=%s multi_views=%s timeout=%ss", url, multi_views, request_timeout_sec)

    def prepare_input(
        self,
        left_rgb: np.ndarray,
        front_rgb: np.ndarray,
        right_rgb: np.ndarray,
        state_16: np.ndarray,
        instruction: str,
    ) -> dict[str, Any]:
        """Pack observation for :meth:`inference` (no image conversion; that happens in ``send_request``)."""
        st = np.asarray(state_16, dtype=np.float32).reshape(-1)
        if st.size != 16:
            raise ValueError(f"state_16 must be 16-D (got {st.size})")
        return {
            "left_camera_rgb": left_rgb,
            "front_camera_rgb": front_rgb,
            "right_camera_rgb": right_rgb,
            "instruction": instruction,
            "state": st,
        }

    def inference(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Run remote policy; returns server JSON (must include ``actions``)."""
        images = [
            input_dict["left_camera_rgb"],
            input_dict["front_camera_rgb"],
            input_dict["right_camera_rgb"],
        ]
        instruction = input_dict["instruction"]
        state = input_dict["state"]
        LOGGER.info(
            "MolmoAct inference: instruction=%r state_dim=%s",
            instruction,
            np.asarray(state).size,
        )
        return self.send_request(images, instruction, state, self.url)

    def infer_from_observation(
        self,
        left_rgb: np.ndarray,
        front_rgb: np.ndarray,
        right_rgb: np.ndarray,
        state_16: np.ndarray,
        instruction: str,
    ) -> dict[str, Any]:
        """``prepare_input`` + ``inference`` (single call site for the eval loop)."""
        return self.inference(self.prepare_input(left_rgb, front_rgb, right_rgb, state_16, instruction))

    def send_request(
        self,
        images: List[np.ndarray],
        instruction: str,
        state: Any,
        server_url: str,
    ) -> dict[str, Any]:
        """Serialize payload with ``json_numpy``, POST, return parsed JSON (YAM: ``np.array`` here only)."""
        LOGGER.info("Sending request to server: %s", server_url)

        if not self.multi_views:
            LOGGER.info("Using single view mode")
            image_np = np.array(images[0])
            LOGGER.info("Single image shape: %s", image_np.shape)
            payload = {
                "image": image_np,
                "instruction": instruction,
                "state": state,
            }
        else:
            LOGGER.info("Using multi-view mode")
            left_img_np = np.array(images[0])
            front_img_np = np.array(images[1])
            right_img_np = np.array(images[2])
            LOGGER.info("Left image shape: %s", left_img_np.shape)
            LOGGER.info("Front image shape: %s", front_img_np.shape)
            LOGGER.info("Right image shape: %s", right_img_np.shape)
            payload = {
                "left_cam": left_img_np,
                "top_cam": front_img_np,
                "right_cam": right_img_np,
                "timestamp": time.time(),
                "instruction": instruction,
                "state": state,
            }

        headers = {"Content-Type": "application/json", **self.extra_headers}
        LOGGER.info("Preparing HTTP request")
        t_ser0 = time.time()
        serialized_payload = json_numpy.dumps(payload)
        ser_ms = (time.time() - t_ser0) * 1000.0
        LOGGER.info("Payload serialized in %.3fs", ser_ms / 1000.0)

        t_http0 = time.time()
        try:
            response = self._session.post(
                server_url,
                headers=headers,
                data=serialized_payload,
                timeout=self.request_timeout_sec,
            )
        except requests.exceptions.ConnectionError as e:
            LOGGER.error("Connection error to %s: %s", server_url, e)
            raise
        except requests.exceptions.Timeout as e:
            LOGGER.error("Timeout to %s: %s", server_url, e)
            raise
        except requests.exceptions.RequestException as e:
            LOGGER.error("Request error to %s: %s", server_url, e)
            raise

        http_ms = (time.time() - t_http0) * 1000.0
        LOGGER.info("HTTP request completed in %.3fs", http_ms / 1000.0)
        LOGGER.info("Response status code: %s", response.status_code)

        if response.status_code != 200:
            msg = f"Server error {response.status_code}: {response.text[:500]}"
            LOGGER.error(msg)
            raise RuntimeError(msg)

        t_parse0 = time.time()
        response_data = response.json()
        parse_ms = (time.time() - t_parse0) * 1000.0
        LOGGER.info("Response parsed in %.3fs", parse_ms)
        LOGGER.info("Server request completed successfully")
        return response_data


def _make_molmoact_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s
