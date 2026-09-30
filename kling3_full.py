"""Kling 3.0 Full nodes for ComfyUI-Kling-Direct.

Current API-key-first implementation for Kling's 2026 API:
- New-standard video endpoints (path-per-model + unified /tasks polling)
- Legacy image endpoints (current image API still uses /v1/...)
- Single KLING_API_KEY Bearer authentication

Designed to coexist with the upstream ComfyUI-Kling-Direct nodes without
changing their behavior.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import requests
import torch

try:
    from .kling_nodes import (
        tensor_to_base64_string,
        download_to_output,
        download_to_tensor,
        load_video_to_tensor,
        load_audio_to_tensor,
    )
except ImportError:
    from kling_nodes import (
        tensor_to_base64_string,
        download_to_output,
        download_to_tensor,
        load_video_to_tensor,
        load_audio_to_tensor,
    )


DEFAULT_BASE_URL = "https://api-singapore.klingai.com"
VIDEO_MODELS = ["kling-3.0", "kling-3.0-turbo"]
VIDEO_RESOLUTIONS = ["720p", "1080p", "4k"]
OMNI_RESOLUTIONS = ["720p", "1080p", "4k"]
VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
IMAGE_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"]
OMNI_IMAGE_ASPECT_RATIOS = IMAGE_ASPECT_RATIOS + ["auto"]
VIDEO_AUDIO = ["native", "off"]
OMNI_AUDIO = ["native", "original", "off"]


class Kling3APIError(RuntimeError):
    def __init__(self, message: str, code: Optional[int] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _interrupt_check() -> None:
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _sleep_interruptible(seconds: float) -> None:
    end = time.monotonic() + max(0.0, seconds)
    while True:
        _interrupt_check()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def _sanitize_debug_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_debug_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_debug_payload(v) for v in value]
    if isinstance(value, str) and len(value) > 160:
        return f"{value[:24]}… [{len(value)} chars]"
    return value


def _parse_element_ids(raw: str, max_count: int = 10) -> List[Any]:
    raw = (raw or "").strip()
    if not raw:
        return []
    values: Iterable[Any]
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"element_ids must be JSON array or comma-separated IDs: {exc}") from None
        if not isinstance(parsed, list):
            raise ValueError("element_ids JSON must be an array.")
        values = parsed
    else:
        values = [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]

    out: List[Any] = []
    for value in values:
        s = str(value).strip()
        if not s:
            continue
        out.append(int(s) if s.isdigit() else s)
    if len(out) > max_count:
        raise ValueError(f"Too many element IDs: {len(out)} supplied, max {max_count}.")
    return out


def _collect_images(*images: Any) -> List[str]:
    out: List[str] = []
    for image in images:
        if image is None:
            continue
        b64 = tensor_to_base64_string(image)
        if b64:
            out.append(b64)
    return out


def _auth_from_input(api_key: str, base_url: str, debug: bool) -> Dict[str, Any]:
    key = (api_key or "").strip() or os.environ.get("KLING_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "No Kling API key provided. Enter the current single API key or set KLING_API_KEY before starting ComfyUI."
        )
    base = (base_url or "").strip() or DEFAULT_BASE_URL
    return {"api_key": key, "base_url": base.rstrip("/"), "debug": bool(debug)}


def _client_from_auth(auth: Dict[str, Any]) -> "Kling3Client":
    key = (auth.get("api_key") or "").strip() or os.environ.get("KLING_API_KEY", "").strip()
    if not key:
        if auth.get("access_key") or auth.get("secret_key"):
            raise ValueError(
                "This Kling 3.0 Full node requires the current single API Key, not legacy Access Key + Secret Key. "
                "Use 'Kling 3.0 Full • API Key' or set KLING_API_KEY."
            )
        raise ValueError("No Kling API key found in auth input or KLING_API_KEY.")
    return Kling3Client(
        api_key=key,
        base_url=(auth.get("base_url") or DEFAULT_BASE_URL),
        debug=bool(auth.get("debug", False)),
    )


class Kling3Client:
    """Small direct Kling client covering the current new video standard and current image /v1 standard."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, debug: bool = False):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.debug = debug
        self.session = requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 180,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if self.debug:
            print(f"[KLING3 FULL] {method.upper()} {url}")
            if body is not None:
                print(json.dumps(_sanitize_debug_payload(body), indent=2, ensure_ascii=False))
        try:
            response = self.session.request(
                method.upper(),
                url,
                headers=self.headers,
                json=body,
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise Kling3APIError(f"Network error contacting Kling API: {type(exc).__name__}") from None

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if self.debug:
            print(f"[KLING3 FULL] HTTP {response.status_code}")
            if payload:
                print(json.dumps(_sanitize_debug_payload(payload), indent=2, ensure_ascii=False)[:12000])

        code = payload.get("code") if isinstance(payload, dict) else None
        if response.status_code >= 400 or (code is not None and code != 0):
            message = "Kling API request failed"
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("msg") or message
            raise Kling3APIError(
                f"{message} (HTTP {response.status_code}, code={code})",
                code=code if isinstance(code, int) else None,
                status_code=response.status_code,
            )
        if not isinstance(payload, dict):
            raise Kling3APIError("Kling API returned a non-object JSON response.")
        return payload

    # ---------- new video standard ----------

    def create_new_video(self, product: str, model: str, body: Dict[str, Any]) -> str:
        result = self._request("POST", f"/{product}/{model}", body=body, timeout=180)
        data = result.get("data") or {}
        task_id = data.get("id")
        if not task_id:
            raise Kling3APIError(f"Kling create response returned no data.id: {result}")
        print(f"[KLING3 FULL] Task submitted: {task_id}")
        return str(task_id)

    def get_new_task(self, task_id: str) -> Dict[str, Any]:
        result = self._request("GET", "/tasks", params={"task_ids": task_id}, timeout=60)
        data = result.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and str(item.get("id")) == str(task_id):
                    return item
            if data and isinstance(data[0], dict):
                return data[0]
        elif isinstance(data, dict):
            return data
        raise Kling3APIError(f"Task {task_id} not found in /tasks response.")

    def poll_new_task(self, task_id: str, timeout: int = 1800) -> Dict[str, Any]:
        start = time.monotonic()
        transient_failures = 0
        while True:
            _interrupt_check()
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Kling task {task_id} is still running after {timeout}s. Do not resubmit automatically; query the task later."
                )
            try:
                task = self.get_new_task(task_id)
                transient_failures = 0
            except Kling3APIError as exc:
                transient_failures += 1
                if transient_failures > 3:
                    raise
                print(f"[KLING3 FULL] Poll transient error ({transient_failures}/3): {exc}")
                _sleep_interruptible(3.0 * transient_failures)
                continue

            status = str(task.get("status") or "").lower()
            if status == "succeeded":
                return task
            if status == "failed":
                raise Kling3APIError(f"Kling task failed: {task.get('message') or 'unknown reason'}")
            elapsed = time.monotonic() - start
            wait = 3.0 if elapsed < 90 else 5.0 if elapsed < 300 else 10.0
            _sleep_interruptible(wait)

    # ---------- image / legacy standard ----------

    def create_legacy_task(self, endpoint: str, body: Dict[str, Any]) -> str:
        result = self._request("POST", endpoint, body=body, timeout=180)
        data = result.get("data") or {}
        task_id = data.get("task_id")
        if not task_id:
            raise Kling3APIError(f"Legacy create response returned no data.task_id: {result}")
        print(f"[KLING3 FULL] Task submitted: {task_id}")
        return str(task_id)

    def get_legacy_task(self, endpoint: str, task_id: str) -> Dict[str, Any]:
        result = self._request("GET", f"{endpoint.rstrip('/')}/{task_id}", timeout=60)
        data = result.get("data")
        if not isinstance(data, dict):
            raise Kling3APIError(f"Legacy task response contains no data object: {result}")
        return data

    def poll_legacy_task(self, endpoint: str, task_id: str, timeout: int = 1800) -> Dict[str, Any]:
        start = time.monotonic()
        while True:
            _interrupt_check()
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Kling task {task_id} is still running after {timeout}s.")
            data = self.get_legacy_task(endpoint, task_id)
            status = str(data.get("task_status") or "").lower()
            if status == "succeed":
                return data
            if status == "failed":
                raise Kling3APIError(f"Kling task failed: {data.get('task_status_msg') or 'unknown reason'}")
            _sleep_interruptible(4.0)


def _new_options(watermark: bool) -> Dict[str, Any]:
    return {
        "external_task_id": str(uuid.uuid4()),
        "watermark_info": {"enabled": bool(watermark)},
    }


def _video_output(task: Dict[str, Any]) -> Dict[str, Any]:
    outputs = task.get("outputs") or []
    for out in outputs:
        if isinstance(out, dict) and out.get("type") == "video" and out.get("url"):
            return out
    raise Kling3APIError(f"Succeeded Kling task returned no video output: {json.dumps(task)[:3000]}")


def _finish_video(task: Dict[str, Any], task_id: str):
    output = _video_output(task)
    url = output["url"]
    path, name = download_to_output(url)
    video = load_video_to_tensor(path)
    audio = load_audio_to_tensor(path)
    metadata = {
        "task_id": task_id,
        "status": task.get("status"),
        "output": output,
        "billing": task.get("billing"),
        "create_time": task.get("create_time"),
        "update_time": task.get("update_time"),
    }
    return video, name, audio, url, task_id, json.dumps(metadata, indent=2, ensure_ascii=False)


def _legacy_image_urls(data: Dict[str, Any]) -> List[str]:
    result = data.get("task_result") or {}
    rows = result.get("images") or result.get("series_images") or []
    urls = [r.get("url") for r in rows if isinstance(r, dict) and r.get("url")]
    if not urls:
        raise Kling3APIError(f"Succeeded image task returned no images: {json.dumps(data)[:3000]}")
    return urls


def _download_image_batch(urls: List[str]) -> torch.Tensor:
    tensors = [download_to_tensor(url) for url in urls]
    return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]


class Kling3Full_Auth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "api_key": ("STRING", {
                "default": "",
                "password": True,
                "tooltip": "Current single Kling API Key. Leave blank to use KLING_API_KEY.",
            }),
            "base_url": ("STRING", {
                "default": DEFAULT_BASE_URL,
                "tooltip": "Global Kling API endpoint. Singapore is the current endpoint for servers outside China.",
            }),
            "debug": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("KLING_AUTH",)
    RETURN_NAMES = ("auth",)
    FUNCTION = "build"
    CATEGORY = "Kling 3.0 Full/Config"

    def build(self, api_key: str, base_url: str, debug: bool):
        return (_auth_from_input(api_key, base_url, debug),)


class Kling3Full_HealthCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"auth": ("KLING_AUTH",)}}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("ok", "status")
    FUNCTION = "check"
    CATEGORY = "Kling 3.0 Full/Config"

    def check(self, auth):
        try:
            c = _client_from_auth(auth)
            c._request("GET", "/tasks", params={"task_ids": "0"}, timeout=30)
            return True, "Kling 3.0 API key accepted; /tasks is reachable."
        except Exception as exc:
            return False, f"Kling API health check failed: {exc}"


class Kling3Full_TaskStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "task_id": ("STRING", {"default": "", "forceInput": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "task_json", "task_id")
    FUNCTION = "check"
    CATEGORY = "Kling 3.0 Full/Config"

    def check(self, auth, task_id):
        c = _client_from_auth(auth)
        task = c.get_new_task(task_id.strip())
        return str(task.get("status", "unknown")), json.dumps(task, indent=2, ensure_ascii=False), task_id


class Kling3Full_Storyboard:
    @classmethod
    def INPUT_TYPES(cls):
        req = {}
        for i in range(1, 7):
            req[f"shot_{i}_duration"] = ("INT", {"default": 0 if i > 1 else 3, "min": 0, "max": 15})
            req[f"shot_{i}_prompt"] = ("STRING", {"default": "", "multiline": True})
        return {"required": req}

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("storyboard_prompt", "total_duration", "multi_shot")
    FUNCTION = "build"
    CATEGORY = "Kling 3.0 Full/Prompt"

    def build(self, **kwargs):
        shots = []
        total = 0
        for i in range(1, 7):
            dur = int(kwargs.get(f"shot_{i}_duration", 0))
            prompt = (kwargs.get(f"shot_{i}_prompt", "") or "").strip()
            if not prompt and dur == 0:
                continue
            if not prompt or dur <= 0:
                raise ValueError(f"Shot {i} needs both a prompt and a duration > 0.")
            shots.append(f"shot {len(shots)+1}, {dur}, {prompt}")
            total += dur
        if not shots:
            raise ValueError("Add at least one storyboard shot.")
        if total < 3 or total > 15:
            raise ValueError(f"Storyboard total duration must be 3–15 seconds; got {total}s.")
        return "; ".join(shots) + ";", total, True


class Kling3Full_TextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "model": (VIDEO_MODELS, {"default": "kling-3.0"}),
            "resolution": (VIDEO_RESOLUTIONS, {"default": "1080p"}),
            "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
            "duration": ("INT", {"default": 5, "min": 3, "max": 15}),
            "audio": (VIDEO_AUDIO, {"default": "native"}),
            "multi_shot": ("BOOLEAN", {"default": False}),
            "watermark": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling 3.0 Full/Video"

    def generate(self, auth, prompt, model, resolution, aspect_ratio, duration, audio, multi_shot, watermark):
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        settings: Dict[str, Any] = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        if model == "kling-3.0-turbo":
            if resolution == "4k":
                raise ValueError("kling-3.0-turbo supports 720p/1080p, not 4K.")
            if audio != "native":
                raise ValueError("kling-3.0-turbo native audio is always on; choose audio=native.")
            # Turbo multi-shot is prompt-syntax driven; no settings.multi_shot field.
        else:
            settings["audio"] = audio
            settings["multi_shot"] = bool(multi_shot)

        body = {"prompt": prompt.strip(), "settings": settings, "options": _new_options(watermark)}
        c = _client_from_auth(auth)
        task_id = c.create_new_video("text-to-video", model, body)
        task = c.poll_new_task(task_id)
        return _finish_video(task, task_id)


class Kling3Full_ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "first_frame": ("IMAGE",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "model": (VIDEO_MODELS, {"default": "kling-3.0"}),
            "resolution": (VIDEO_RESOLUTIONS, {"default": "1080p"}),
            "duration": ("INT", {"default": 5, "min": 3, "max": 15}),
            "audio": (VIDEO_AUDIO, {"default": "native"}),
            "multi_shot": ("BOOLEAN", {"default": False}),
            "element_ids": ("STRING", {
                "default": "",
                "tooltip": "Optional Kling element IDs, comma separated or JSON array. Prompt refs become @element_1, @element_2, ...",
            }),
            "watermark": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "last_frame": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling 3.0 Full/Video"

    def generate(self, auth, first_frame, prompt, model, resolution, duration, audio, multi_shot, element_ids, watermark, last_frame=None):
        first = tensor_to_base64_string(first_frame)
        last = tensor_to_base64_string(last_frame) if last_frame is not None else None
        elements = _parse_element_ids(element_ids, max_count=3)

        if model == "kling-3.0-turbo":
            if resolution == "4k":
                raise ValueError("kling-3.0-turbo supports 720p/1080p, not 4K.")
            if last:
                raise ValueError("kling-3.0-turbo I2V supports first frame only; use kling-3.0 for first+last frame.")
            if elements:
                raise ValueError("Element references on I2V are supported by kling-3.0, not kling-3.0-turbo.")
            if audio != "native":
                raise ValueError("kling-3.0-turbo native audio is always on; choose audio=native.")

        contents: List[Dict[str, Any]] = [
            {"type": "prompt", "text": prompt or ""},
            {"type": "first_frame", "url": first},
        ]
        if last:
            contents.append({"type": "last_frame", "url": last})
        for i, element_id in enumerate(elements, 1):
            contents.append({"type": "element", "element_id": element_id, "id": f"element_{i}"})

        settings: Dict[str, Any] = {"resolution": resolution, "duration": int(duration)}
        if model == "kling-3.0":
            settings["audio"] = audio
            settings["multi_shot"] = bool(multi_shot)

        body = {"contents": contents, "settings": settings, "options": _new_options(watermark)}
        c = _client_from_auth(auth)
        task_id = c.create_new_video("image-to-video", model, body)
        task = c.poll_new_task(task_id)
        return _finish_video(task, task_id)


class Kling3Full_OmniVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "first_frame": ("IMAGE",),
            "last_frame": ("IMAGE",),
            "refer_image_1": ("IMAGE",),
            "refer_image_2": ("IMAGE",),
            "refer_image_3": ("IMAGE",),
            "refer_image_4": ("IMAGE",),
            "refer_image_5": ("IMAGE",),
            "refer_image_6": ("IMAGE",),
            "refer_image_7": ("IMAGE",),
        }
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": "Reference inputs as @image_1, @image_2, @video_1, @element_1, ...",
            }),
            "resolution": (OMNI_RESOLUTIONS, {"default": "1080p"}),
            "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
            "duration": ("INT", {"default": 5, "min": 3, "max": 15}),
            "audio": (OMNI_AUDIO, {"default": "native"}),
            "multi_shot": ("BOOLEAN", {"default": False}),
            "feature_video_url": ("STRING", {
                "default": "",
                "tooltip": "Optional public reference-video URL. feature_video requires audio=off and multi_shot=true.",
            }),
            "base_video_url": ("STRING", {
                "default": "",
                "tooltip": "Optional public base-video URL for editing. Cannot be combined with first/last frame or feature_video.",
            }),
            "element_ids": ("STRING", {
                "default": "",
                "tooltip": "Optional element IDs, comma-separated or JSON array; referenced as @element_1 etc.",
            }),
            "watermark": ("BOOLEAN", {"default": False}),
        }, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling 3.0 Full/Video"

    def generate(
        self, auth, prompt, resolution, aspect_ratio, duration, audio, multi_shot,
        feature_video_url, base_video_url, element_ids, watermark,
        first_frame=None, last_frame=None,
        refer_image_1=None, refer_image_2=None, refer_image_3=None, refer_image_4=None,
        refer_image_5=None, refer_image_6=None, refer_image_7=None,
    ):
        feature = (feature_video_url or "").strip()
        base = (base_video_url or "").strip()
        if feature and base:
            raise ValueError("Use either feature_video_url or base_video_url, not both.")
        if feature:
            if audio != "off":
                raise ValueError("Kling 3.0 Omni feature_video requires audio=off.")
            if not multi_shot:
                raise ValueError("Kling 3.0 Omni feature_video requires multi_shot=true.")
        if base:
            if first_frame is not None or last_frame is not None:
                raise ValueError("base_video editing cannot be combined with first_frame or last_frame.")
            if multi_shot:
                raise ValueError("base_video editing cannot use multi_shot=true.")
            if audio == "native":
                raise ValueError("base_video editing does not support native audio; use original or off.")

        refs = _collect_images(
            refer_image_1, refer_image_2, refer_image_3, refer_image_4,
            refer_image_5, refer_image_6, refer_image_7,
        )
        elements = _parse_element_ids(element_ids, max_count=7)

        contents: List[Dict[str, Any]] = [{"type": "prompt", "text": prompt or ""}]
        image_idx = 0
        if first_frame is not None:
            image_idx += 1
            contents.append({"type": "first_frame", "url": tensor_to_base64_string(first_frame), "id": f"image_{image_idx}"})
        if last_frame is not None:
            image_idx += 1
            contents.append({"type": "last_frame", "url": tensor_to_base64_string(last_frame), "id": f"image_{image_idx}"})
        for b64 in refs:
            image_idx += 1
            contents.append({"type": "refer_image", "url": b64, "id": f"image_{image_idx}"})
        if feature:
            contents.append({"type": "feature_video", "url": feature, "id": "video_1"})
        if base:
            contents.append({"type": "base_video", "url": base, "id": "video_1"})
        for i, element_id in enumerate(elements, 1):
            contents.append({"type": "element", "element_id": element_id, "id": f"element_{i}"})

        settings: Dict[str, Any] = {"resolution": resolution, "audio": audio, "multi_shot": bool(multi_shot)}
        if not base:
            settings["duration"] = int(duration)
        if first_frame is None and not feature and not base:
            settings["aspect_ratio"] = aspect_ratio

        body = {"contents": contents, "settings": settings, "options": _new_options(watermark)}
        c = _client_from_auth(auth)
        task_id = c.create_new_video("omni-video", "kling-3.0-omni", body)
        task = c.poll_new_task(task_id)
        return _finish_video(task, task_id)


class Kling3Full_MotionControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "image": ("IMAGE",),
            "video_url": ("STRING", {"default": "", "tooltip": "Public URL of the motion-reference video."}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "character_orientation": (["image", "video"], {"default": "image"}),
            "resolution": (["720p", "1080p"], {"default": "1080p"}),
            "audio": (["original", "off"], {"default": "off"}),
            "element_id": ("STRING", {"default": ""}),
            "watermark": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling 3.0 Full/Video"

    def generate(self, auth, image, video_url, prompt, character_orientation, resolution, audio, element_id, watermark):
        video_url = (video_url or "").strip()
        if not video_url.startswith(("http://", "https://")):
            raise ValueError("Motion Control requires a public http(s) reference-video URL.")
        contents: List[Dict[str, Any]] = []
        if prompt.strip():
            contents.append({"type": "prompt", "text": prompt.strip()})
        contents.append({"type": "image", "url": tensor_to_base64_string(image)})
        contents.append({"type": "video", "url": video_url})
        if element_id.strip():
            parsed = _parse_element_ids(element_id, max_count=1)[0]
            contents.append({"type": "element", "element_id": parsed, "id": "element_1"})

        body = {
            "contents": contents,
            "settings": {
                "character_orientation": character_orientation,
                "audio": audio,
                "resolution": resolution,
            },
            "options": _new_options(watermark),
        }
        c = _client_from_auth(auth)
        task_id = c.create_new_video("motion-control", "kling-3.0", body)
        task = c.poll_new_task(task_id)
        return _finish_video(task, task_id)


class Kling3Full_ImageGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "resolution": (["1k", "2k"], {"default": "2k"}),
            "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
            "n": ("INT", {"default": 1, "min": 1, "max": 9}),
            "watermark": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling 3.0 Full/Image"

    def generate(self, auth, prompt, negative_prompt, resolution, aspect_ratio, n, watermark):
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        body: Dict[str, Any] = {
            "model_name": "kling-v3",
            "prompt": prompt.strip(),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "n": int(n),
            "watermark_info": {"enabled": bool(watermark)},
            "external_task_id": str(uuid.uuid4()),
        }
        if negative_prompt.strip():
            body["negative_prompt"] = negative_prompt.strip()
        c = _client_from_auth(auth)
        endpoint = "/v1/images/generations"
        task_id = c.create_legacy_task(endpoint, body)
        data = c.poll_legacy_task(endpoint, task_id)
        urls = _legacy_image_urls(data)
        return _download_image_batch(urls), json.dumps(urls, indent=2), task_id


class Kling3Full_OmniImage:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"refer_image_{i}": ("IMAGE",) for i in range(1, 8)}
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": "Reference images with <<<image_1>>>, <<<image_2>>>, ...",
            }),
            "resolution": (["1k", "2k", "4k"], {"default": "2k"}),
            "aspect_ratio": (OMNI_IMAGE_ASPECT_RATIOS, {"default": "auto"}),
            "result_type": (["single", "series"], {"default": "single"}),
            "n": ("INT", {"default": 1, "min": 1, "max": 9}),
            "series_amount": (["auto", "2", "3", "4", "5", "6", "7", "8", "9"], {"default": "auto"}),
            "element_ids": ("STRING", {"default": ""}),
            "watermark": ("BOOLEAN", {"default": False}),
        }, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling 3.0 Full/Image"

    def generate(
        self, auth, prompt, resolution, aspect_ratio, result_type, n, series_amount,
        element_ids, watermark,
        refer_image_1=None, refer_image_2=None, refer_image_3=None, refer_image_4=None,
        refer_image_5=None, refer_image_6=None, refer_image_7=None,
    ):
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        refs = _collect_images(
            refer_image_1, refer_image_2, refer_image_3, refer_image_4,
            refer_image_5, refer_image_6, refer_image_7,
        )
        elements = _parse_element_ids(element_ids, max_count=10)
        if len(refs) + len(elements) > 10:
            raise ValueError("Omni Image supports at most 10 reference images + elements in total.")

        body: Dict[str, Any] = {
            "model_name": "kling-v3-omni",
            "prompt": prompt.strip(),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "result_type": result_type,
            "watermark_info": {"enabled": bool(watermark)},
            "external_task_id": str(uuid.uuid4()),
        }
        if refs:
            body["image_list"] = [{"image": b64} for b64 in refs]
        if elements:
            body["element_list"] = [{"element_id": x} for x in elements]
        if result_type == "series":
            body["series_amount"] = "auto" if series_amount == "auto" else int(series_amount)
        else:
            body["n"] = int(n)

        c = _client_from_auth(auth)
        endpoint = "/v1/images/omni-image"
        task_id = c.create_legacy_task(endpoint, body)
        data = c.poll_legacy_task(endpoint, task_id)
        urls = _legacy_image_urls(data)
        return _download_image_batch(urls), json.dumps(urls, indent=2), task_id


NODE_CLASS_MAPPINGS = {
    "Kling3Full_Auth": Kling3Full_Auth,
    "Kling3Full_HealthCheck": Kling3Full_HealthCheck,
    "Kling3Full_TaskStatus": Kling3Full_TaskStatus,
    "Kling3Full_Storyboard": Kling3Full_Storyboard,
    "Kling3Full_TextToVideo": Kling3Full_TextToVideo,
    "Kling3Full_ImageToVideo": Kling3Full_ImageToVideo,
    "Kling3Full_OmniVideo": Kling3Full_OmniVideo,
    "Kling3Full_MotionControl": Kling3Full_MotionControl,
    "Kling3Full_ImageGeneration": Kling3Full_ImageGeneration,
    "Kling3Full_OmniImage": Kling3Full_OmniImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling3Full_Auth": "Kling 3.0 Full • API Key",
    "Kling3Full_HealthCheck": "Kling 3.0 Full • Health Check",
    "Kling3Full_TaskStatus": "Kling 3.0 Full • Task Status",
    "Kling3Full_Storyboard": "Kling 3.0 Full • Storyboard Builder",
    "Kling3Full_TextToVideo": "Kling 3.0 Full • Text to Video",
    "Kling3Full_ImageToVideo": "Kling 3.0 Full • Image to Video",
    "Kling3Full_OmniVideo": "Kling 3.0 Full • Omni Video",
    "Kling3Full_MotionControl": "Kling 3.0 Full • Motion Control",
    "Kling3Full_ImageGeneration": "Kling Image 3.0 Full • Generation",
    "Kling3Full_OmniImage": "Kling Image 3.0 Full • Omni",
}