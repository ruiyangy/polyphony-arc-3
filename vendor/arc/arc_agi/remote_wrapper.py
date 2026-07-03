"""Remote environment wrapper for ARC-AGI-3 environments."""

import json
import logging
import threading
from typing import Any, Callable, Optional

import numpy as np
import requests
from arcengine import FrameData, FrameDataRaw, GameAction
from requests.cookies import RequestsCookieJar

from .models import EnvironmentInfo
from .wrapper import EnvironmentWrapper


class RemoteEnvironmentWrapper(EnvironmentWrapper):
    """Wrapper for running ARC-AGI-3 environments remotely via API.

    This wrapper makes HTTP requests to the ARC-AGI-3 API to interact with
    environments that are hosted remotely.
    """

    def __init__(
        self,
        base_url: str,
        environment_info: EnvironmentInfo,
        arc_api_key: str,
        logger: logging.Logger,
        scorecard_id: str,
        save_recording: bool = False,
        include_frame_data: bool = True,
        recordings_dir: str = "recordings",
        scorecard_manager: Optional[Any] = None,
        renderer: Optional[Callable[[int, FrameDataRaw], None]] = None,
        master_cookie_jar: Optional[RequestsCookieJar] = None,
        cookie_lock: Optional[threading.Lock] = None,
    ) -> None:
        """Initialize the remote environment wrapper.

        Args:
            base_url: Base URL for the ARC-AGI-3 API (e.g., "https://three.arcprize.org").
            environment_info: EnvironmentInfo object with game metadata.
            arc_api_key: API key for authentication.
            logger: Logger instance for logging.
            scorecard_id: Scorecard ID for tracking runs.
            save_recording: Whether to save recordings to JSONL file.
            include_frame_data: Whether to include frame data in the recording file.
            recordings_dir: Directory to save recordings.
            scorecard_manager: Optional scorecard manager for tracking.
            renderer: Optional callable that accepts FrameDataRaw and performs custom rendering.
        """
        super().__init__(
            environment_info,
            logger,
            scorecard_id,
            save_recording,
            include_frame_data,
            recordings_dir,
            scorecard_manager,
            renderer,
        )
        self.base_url = base_url
        self.arc_api_key = arc_api_key
        self.headers = {
            "X-API-Key": arc_api_key,
            "Accept": "application/json",
        }
        self._session = requests.Session()
        self._session.headers.update(self.headers)

        self._master_cookie_jar = (
            master_cookie_jar if master_cookie_jar is not None else RequestsCookieJar()
        )
        self._cookie_lock = cookie_lock if cookie_lock is not None else threading.Lock()

        # Hotfix N D2: expose HTTP metadata for observability
        #   - _last_response_meta: per-call meta (status / trace_id / latency) — success AND failure
        #   - _last_server_error:  structured server error from 4xx/5xx bodies
        # Both are None when never populated; OFFLINE-side wrappers don't create
        # these attributes, so the executor's adapter will degrade to None cleanly.
        self._last_response_meta: Optional[dict[str, Any]] = None
        self._last_server_error: Optional[dict[str, Any]] = None
        # Hotfix T-2: structured reset failure info for InitialResetFailed
        self._last_reset_reason_code: Optional[str] = None
        self._last_reset_http_status: Optional[int] = None
        self._last_reset_body: Optional[str] = None

        self.reset()

    def _record_response_meta(self, response, elapsed_ms: Optional[float]) -> None:
        """Hotfix N D2: record trace-id / http_status / latency from a live response.

        Called on both success and failure paths (200/400/etc). Trace-id falls
        back through X-Amzn-Trace-Id → X-Request-Id. Any exception is swallowed —
        meta capture must never break the main request flow.
        """
        try:
            headers = getattr(response, "headers", {}) or {}
            trace_id = headers.get("X-Amzn-Trace-Id") or headers.get("X-Request-Id")
            status = getattr(response, "status_code", None)
            self._last_response_meta = {
                "http_status": int(status) if status is not None else None,
                "trace_id": trace_id,
                "response_time_ms": float(elapsed_ms) if elapsed_ms is not None else None,
            }
        except Exception:
            self._last_response_meta = None

    def _record_server_error(self, response, fallback_status: Optional[int] = None) -> None:
        """Hotfix N N-2.1: capture structured error info from a 4xx/5xx response.

        Populates self._last_server_error so ActionExecutor._get_last_server_error()
        can distinguish SERVER_GNS from plain network timeouts. Body is widened
        to 4000 chars (was 500 in Hotfix M3) to capture the full GNS / SERVER_ERROR
        payload, which runs ~200 chars in practice.
        """
        try:
            body = ""
            body_json: dict[str, Any] = {}
            status_code = fallback_status
            trace_id = None
            if response is not None:
                try:
                    body = response.text[:4000]
                except Exception:
                    body = "(unable to read response body)"
                try:
                    body_json = response.json() if body else {}
                except Exception:
                    body_json = {}
                status_code = getattr(response, "status_code", status_code)
                headers = getattr(response, "headers", {}) or {}
                trace_id = headers.get("X-Amzn-Trace-Id") or headers.get("X-Request-Id")
            self._last_server_error = {
                "http_status": int(status_code) if status_code is not None else None,
                "error_code": body_json.get("error") if isinstance(body_json, dict) else None,
                "message": (body_json.get("message", body)
                            if isinstance(body_json, dict) else body),
                "body_raw": body,
                "trace_id": trace_id,
            }
        except Exception:
            # Defensive: never raise from the error-capture path itself
            self._last_server_error = {
                "http_status": fallback_status,
                "error_code": None,
                "message": "(error-capture failed)",
                "body_raw": "",
                "trace_id": None,
            }

    def reset(self) -> Optional[FrameDataRaw]:
        """Reset the environment and return the initial frame data.

        Returns:
            FrameDataRaw object with initial game state, or None if reset failed.
        """
        # Hotfix T-2: clear previous reset error state
        self._last_reset_reason_code = None
        self._last_reset_http_status = None
        self._last_reset_body = None
        try:
            url = f"{self.base_url}/api/cmd/RESET"
            headers = {
                "X-Api-Key": self.arc_api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "card_id": self.scorecard_id,
                "game_id": self.environment_info.game_id,
            }
            if self._guid:
                payload["guid"] = self._guid

            with self._cookie_lock:
                self._session.cookies.update(self._master_cookie_jar)  # type: ignore[no-untyped-call]

            import time as _time
            _t0 = _time.time()
            response = self._session.post(
                url, json=payload, headers=headers, timeout=30
            )
            _elapsed_ms = (_time.time() - _t0) * 1000.0
            # Hotfix N D2: capture HTTP meta on the success path too
            self._record_response_meta(response, _elapsed_ms)
            # Reset last_server_error on the success path so stale entries
            # from a prior failed call don't mislead the executor
            self._last_server_error = None

            with self._cookie_lock:
                self._master_cookie_jar.update(self._session.cookies)  # type: ignore[no-untyped-call]

            response.raise_for_status()
            response_data = response.json()

            # Convert API response to FrameDataRaw
            frame_data_raw = self._convert_to_frame_data_raw(response_data)
            if frame_data_raw:
                # Store guid from response
                self._guid = response_data.get("guid")
                # Setup recording file now that guid is set
                if self.save_recording and self._guid:
                    self._setup_recording_file()
                self._set_last_response(frame_data_raw)
                self.logger.info(
                    f"Successfully reset game {self.environment_info.game_id}, guid={self._guid}, scorecard_id={self.scorecard_id}"
                )
                return frame_data_raw

            return None

        except requests.exceptions.RequestException as e:
            # M3 + Hotfix N N-0.d: response body 从 500 放宽到 4000，覆盖完整 GNS/SERVER_ERROR payload
            response_body = ""
            status_code = None
            if hasattr(e, 'response') and e.response is not None:
                status_code = getattr(e.response, 'status_code', None)
                try:
                    response_body = e.response.text[:4000]
                except Exception:
                    response_body = "(unable to read response body)"
                # Hotfix N D2 + N-2.1: capture structured meta + error for executor
                try:
                    self._record_response_meta(e.response, None)
                    self._record_server_error(e.response)
                except Exception:
                    pass
            else:
                # Network/connection error before any response — still leave a meta trace
                try:
                    self._record_response_meta(None, None)
                except Exception:
                    pass
                self._last_server_error = None
            # Hotfix T-2: persist structured reset failure info
            self._last_reset_http_status = status_code
            self._last_reset_body = response_body
            self._last_reset_reason_code = self._classify_reset_error(
                status_code, response_body)
            self.logger.error(
                f"Failed to reset game {self.environment_info.game_id}: {e}"
                + (f"\n  Response body: {response_body}" if response_body else ""),
                exc_info=True,
            )
            return None
        except Exception as e:
            self.logger.exception(
                f"Unexpected error while resetting game {self.environment_info.game_id}: {e}"
            )
            # Hotfix T-2: unknown error path
            self._last_reset_reason_code = "UNKNOWN_RESET_NONE"
            return None

    def _classify_reset_error(self, status: Optional[int], body: str) -> str:
        """Hotfix T-2: classify reset failure into reason_code enum."""
        body_lower = (body or "").lower()
        if status in (400, 404) and "not found" in body_lower:
            if "scorecard" in body_lower:
                return "SCORECARD_NOT_FOUND"
            return "GAME_NOT_FOUND"
        if status == 401:
            return "UNAUTHORIZED"
        if status == 403:
            return "FORBIDDEN"
        if status is None:
            return "CONNECTION_ERROR"
        return f"HTTP_{status}"

    def step(
        self,
        action: GameAction,
        data: Optional[dict[str, Any]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Optional[FrameDataRaw]:
        """Perform a step in the environment.

        Args:
            action: The game action to perform.
            data: Optional action data dictionary (for complex actions, should contain "x" and "y").
            reasoning: Optional reasoning dictionary.

        Returns:
            FrameDataRaw object with updated game state, or None if step failed.
        """
        if self._guid is None:
            self.logger.error("Cannot step: game not reset. Call reset() first.")
            return None

        try:
            # Determine action endpoint
            if action == GameAction.RESET:
                action_name = "RESET"
            else:
                action_name = f"ACTION{action.value}"

            url = f"{self.base_url}/api/cmd/{action_name}"
            headers = {
                "X-Api-Key": self.arc_api_key,
                "Content-Type": "application/json",
            }

            # Build payload
            payload = {
                "game_id": self.environment_info.game_id,
                "guid": self._guid,
            }

            # Add x, y coordinates for complex actions
            if data:
                if "x" in data:
                    payload["x"] = data["x"]
                if "y" in data:
                    payload["y"] = data["y"]

            # Add reasoning if provided
            if reasoning:
                payload["reasoning"] = json.dumps(reasoning)

            with self._cookie_lock:
                self._session.cookies.update(self._master_cookie_jar)  # type: ignore[no-untyped-call]

            import time as _time
            _t0 = _time.time()
            response = self._session.post(
                url, json=payload, headers=headers, timeout=30
            )
            _elapsed_ms = (_time.time() - _t0) * 1000.0
            # Hotfix N D2: meta captured for success AND failure
            self._record_response_meta(response, _elapsed_ms)
            self._last_server_error = None  # clear stale error on success

            with self._cookie_lock:
                self._master_cookie_jar.update(self._session.cookies)  # type: ignore[no-untyped-call]

            response.raise_for_status()
            response_data = response.json()

            # Convert API response to FrameDataRaw
            frame_data_raw = self._convert_to_frame_data_raw(response_data)
            if frame_data_raw:
                self._set_last_response(frame_data_raw, reasoning=reasoning)
                return frame_data_raw

            return None

        except requests.exceptions.RequestException as e:
            # M3 + Hotfix N N-0.d: response body 从 500 放宽到 4000
            response_body = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    response_body = e.response.text[:4000]
                except Exception:
                    response_body = "(unable to read response body)"
                # Hotfix N D2 + N-2.1: capture structured meta + server error
                try:
                    self._record_response_meta(e.response, None)
                    self._record_server_error(e.response)
                except Exception:
                    pass
            else:
                try:
                    self._record_response_meta(None, None)
                except Exception:
                    pass
                self._last_server_error = None
            self.logger.error(
                f"Failed to perform action {action.name} for game {self.environment_info.game_id}: {e}"
                + (f"\n  Response body: {response_body}" if response_body else ""),
                exc_info=True,
            )
            return None
        except Exception as e:
            self.logger.exception(
                f"Unexpected error while performing action {action.name}: {e}"
            )
            return None

    def _convert_to_frame_data_raw(
        self, response_data: dict[str, Any]
    ) -> Optional[FrameDataRaw]:
        """Convert API response dictionary to FrameDataRaw.

        Args:
            response_data: Dictionary from API response.

        Returns:
            FrameDataRaw object if successful, None otherwise.
        """
        try:
            # First, try to parse as FrameData (Pydantic model)
            frame_data = FrameData.model_validate(response_data)

            # Convert FrameData to FrameDataRaw
            frame_data_raw = FrameDataRaw()
            frame_data_raw.game_id = frame_data.game_id
            # Convert frame from list of lists of lists to list of ndarrays
            frame_data_raw.frame = [
                np.array(frame_layer, dtype=np.int8) for frame_layer in frame_data.frame
            ]
            frame_data_raw.state = frame_data.state
            frame_data_raw.levels_completed = frame_data.levels_completed
            frame_data_raw.win_levels = frame_data.win_levels
            frame_data_raw.action_input = frame_data.action_input
            frame_data_raw.guid = frame_data.guid
            frame_data_raw.full_reset = getattr(frame_data, "full_reset", False)
            frame_data_raw.available_actions = frame_data.available_actions

            return frame_data_raw

        except Exception as e:
            self.logger.error(
                f"Failed to convert API response to FrameDataRaw: {e}",
                exc_info=True,
            )
            return None
