"""Direct Kling Video 4.0 Early Access REST nodes.

Implements the temporary Kling 4.0 API specification supplied for Early Access:
- POST /text-to-video/kling-4.0
- POST /image-to-video/kling-4.0
- POST /omni-video/kling-4.0
- GET  /tasks?task_ids=...

Authentication uses the same single Bearer API key and Singapore base URL as the
existing Kling 3.0 Full REST layer. Existing CLI/MCP and 3.0 nodes are untouched.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import torch

try:
    from .kling_nodes import tensor_to_base64_string
    from .kling4_cli import Kling4CLIError, _run_cli
    from .kling3_full import (
        DEFAULT_BASE_URL,
        Kling3APIError,
        Kling3Full_Auth,
        Kling3Full_TaskStatus,
        _client_from_auth,
        _finish_video,
        _parse_element_ids,
    )
except ImportError:
    from kling_nodes import tensor_to_base64_string
    from kling4_cli import Kling4CLIError, _run_cli
    from kling3_full import (
        DEFAULT_BASE_URL,
        Kling3APIError,
        Kling3Full_Auth,
        Kling3Full_TaskStatus,
        _client_from_auth,
        _finish_video,
        _parse_element_ids,
    )


KLING4_MODEL = "kling-4.0"
KLING4_RESOLUTIONS = ["720p", "1080p"]
KLING4_T2V_ASPECT_RATIOS = ["21:9", "16:9", "9:16", "1:1"]
KLING4_OMNI_ASPECT_RATIOS = ["adaptive", "21:9", "16:9", "9:16", "1:1"]


def _validate_duration(duration: int) -> int:
    value = int(duration)
    if value < 3 or value > 30:
        raise ValueError("Kling 4.0 duration must be between 3 and 30 seconds.")
    return value


def _validate_prompt(prompt: str, *, required: bool) -> str:
    text = (prompt or "").strip()
    if required and not text:
        raise ValueError("Prompt cannot be empty.")
    if not text:
        return ""

    # Temporary EA spec: English-only <= 26k chars, non-English-only <= 11k,
    # mixed input must satisfy E/26000 + N/11000 <= 1.
    # ASCII is treated as the English bucket; non-ASCII as the non-English bucket.
    english = sum(1 for ch in text if ord(ch) < 128)
    non_english = len(text) - english
    weighted = english / 26000.0 + non_english / 11000.0
    if weighted > 1.0:
        raise ValueError(
            "Kling 4.0 prompt exceeds the Early Access mixed-language limit "
            "(E/26000 + N/11000 <= 1)."
        )
    return text


def _validate_image(image: torch.Tensor, label: str) -> None:
    if image is None:
        return
    sample = image[0] if image.dim() == 4 else image
    if sample.dim() != 3:
        raise ValueError(f"{label} must be a ComfyUI IMAGE tensor.")
    h, w = int(sample.shape[0]), int(sample.shape[1])
    if w < 300 or h < 300:
        raise ValueError(
            f"{label} is {w}x{h}. Kling 4.0 requires width and height >= 300px."
        )
    ratio = w / max(h, 1)
    if ratio < (2 / 5) or ratio > (5 / 2):
        raise ValueError(
            f"{label} aspect ratio {ratio:.3f} is outside Kling 4.0's 2:5 to 5:2 range."
        )


def _encode_image(image: torch.Tensor, label: str) -> str:
    _validate_image(image, label)
    # Existing helper keeps payloads comfortably below Kling's 30 MB image limit.
    return tensor_to_base64_string(image)


def _parse_string_ids(raw: str, max_count: int, label: str) -> List[str]:
    value = (raw or "").strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be a JSON array or comma/newline-separated IDs: {exc}") from None
        if not isinstance(parsed, list):
            raise ValueError(f"{label} JSON must be an array.")
        rows: Iterable[Any] = parsed
    else:
        rows = value.replace("\n", ",").split(",")

    out = [str(x).strip() for x in rows if str(x).strip()]
    if len(out) > max_count:
        raise ValueError(f"{label} supports at most {max_count} entries; got {len(out)}.")
    return out


def _parse_url_lines(raw: str, max_count: int, label: str) -> List[str]:
    rows = [x.strip() for x in (raw or "").splitlines() if x.strip()]
    if len(rows) > max_count:
        raise ValueError(f"{label} supports at most {max_count} URLs; got {len(rows)}.")
    for url in rows:
        _validate_public_url(url, label)
    return rows


def _validate_public_url(url: str, label: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{label} must be a public http(s) URL.")
    return value


_MATERIAL_URL_KEYS = (
    "url",
    "file_url",
    "fileUrl",
    "material_url",
    "materialUrl",
    "download_url",
    "downloadUrl",
    "resource_url",
    "resourceUrl",
    "cdn_url",
    "cdnUrl",
)
_MATERIAL_ID_KEYS = ("id", "materials_id", "material_id", "file_id", "fileId")


def _validate_local_video_path(file_path: str, label: str) -> str:
    path = os.path.abspath(os.path.expanduser((file_path or "").strip()))
    if not path or not os.path.isfile(path):
        raise ValueError(f"{label} must point to an existing local video file.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".mov"):
        raise ValueError(f"{label} must be an .mp4 or .mov file.")
    size = os.path.getsize(path)
    if size > 300 * 1024 * 1024:
        raise ValueError(f"{label} exceeds Kling 4.0's 300 MB video limit.")
    return path


def _find_nested_string(value: Any, keys: Iterable[str]) -> str:
    wanted = set(keys)
    if isinstance(value, dict):
        for key in wanted:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for child in value.values():
            found = _find_nested_string(child, wanted)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested_string(child, wanted)
            if found:
                return found
    return ""


def _extract_material_url(data: Any) -> str:
    value = _find_nested_string(data, _MATERIAL_URL_KEYS)
    if value:
        return _validate_public_url(value, "Kling material upload response URL")
    return ""


def _extract_material_id(data: Any) -> str:
    if not isinstance(data, (dict, list)):
        return ""
    return _find_nested_string(data, _MATERIAL_ID_KEYS)


def _official_upload_video(auth: Dict[str, Any], file_path: str) -> tuple[str, str, Dict[str, Any]]:
    """Upload a local video through Kling's official CLI/MCP file_upload flow.

    Kling's current public REST API accepts video references as URLs but does
    not document a single-API-key REST upload endpoint. The official CLI wraps
    Kling's two-step file_upload ticket flow and returns the final Kling-hosted
    public URL. The subsequent Kling 4.0 generation still uses the direct REST
    API and KLING_API_KEY.
    """
    path = _validate_local_video_path(file_path, "video_path")

    try:
        result = _run_cli(["file_upload", path, "--quiet"], timeout=600)
    except (Kling4CLIError, TimeoutError) as exc:
        raise Kling3APIError(
            "Official Kling file upload failed. Local video upload requires the "
            "official Kling CLI to be installed and logged in as the same OS user "
            "that runs ComfyUI. Run: npm i -g @klingai/cli-global && kling login. "
            f"Details: {exc}"
        ) from None

    material_url = _extract_material_url(result)
    material_id = _extract_material_id(result)
    if not material_url:
        keys = sorted(result.keys()) if isinstance(result, dict) else []
        raise Kling3APIError(
            "Official Kling CLI file_upload completed but returned no usable public URL. "
            f"Top-level response fields: {keys}"
        )

    return material_url, material_id, result


def _options(
    watermark: bool,
    callback_url: str,
    external_task_id: str,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "external_task_id": (external_task_id or "").strip() or str(uuid.uuid4()),
        "watermark_info": {"enabled": bool(watermark)},
    }
    callback = (callback_url or "").strip()
    if callback:
        options["callback_url"] = _validate_public_url(callback, "callback_url")
    return options


def _finish(task: Dict[str, Any], task_id: str):
    # Kling 4.0 uses the same /tasks output shape as the existing new-standard
    # REST layer, so reuse its secure download + ComfyUI video/audio conversion.
    return _finish_video(task, task_id)


class Kling4REST_Auth(Kling3Full_Auth):
    CATEGORY = "Kling 4.0 REST/Config"


class Kling4REST_AccessCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"auth": ("KLING_AUTH",)}}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("ok", "status")
    FUNCTION = "check"
    CATEGORY = "Kling 4.0 REST/Config"

    def check(self, auth):
        try:
            client = _client_from_auth(auth)
            # Non-billable connectivity/auth check only. The EA spec exposes no
            # dedicated entitlement endpoint, so this intentionally does not
            # submit a Kling 4.0 generation just to prove access.
            client._request("GET", "/tasks", params={"task_ids": "0"}, timeout=30)
            return (
                True,
                "Kling REST API key accepted and /tasks is reachable. "
                "Kling 4.0 entitlement is verified only when a real 4.0 task is submitted.",
            )
        except Exception as exc:
            return False, f"Kling REST access check failed: {exc}"


class Kling4REST_TaskStatus(Kling3Full_TaskStatus):
    CATEGORY = "Kling 4.0 REST/Config"


class Kling4REST_OfficialVideoUpload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auth": ("KLING_AUTH",),
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Local .mp4/.mov path. Uses official Kling CLI file_upload; run kling login once on the ComfyUI server.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("url", "material_id", "metadata_json")
    FUNCTION = "upload"
    CATEGORY = "Kling 4.0 REST/Config"

    def upload(self, auth, video_path):
        url, material_id, data = _official_upload_video(auth, video_path)
        return url, material_id, json.dumps(data, indent=2, ensure_ascii=False)


class Kling4REST_TextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auth": ("KLING_AUTH",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": (KLING4_RESOLUTIONS, {"default": "1080p"}),
                "aspect_ratio": (KLING4_T2V_ASPECT_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 10, "min": 3, "max": 30}),
                "watermark": ("BOOLEAN", {"default": False}),
                "callback_url": ("STRING", {"default": ""}),
                "external_task_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional unique task ID. Blank = auto-generated UUID.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling 4.0 REST/Video"

    def generate(
        self,
        auth,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        watermark,
        callback_url,
        external_task_id,
    ):
        text = _validate_prompt(prompt, required=True)
        body = {
            "prompt": text,
            "settings": {
                "duration": _validate_duration(duration),
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            },
            "options": _options(watermark, callback_url, external_task_id),
        }

        client = _client_from_auth(auth)
        task_id = client.create_new_video("text-to-video", KLING4_MODEL, body)
        task = client.poll_new_task(task_id)
        return _finish(task, task_id)


class Kling4REST_ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional: Dict[str, Any] = {
            "first_frame": ("IMAGE",),
            "last_frame": ("IMAGE",),
        }
        for i in range(1, 9):
            optional[f"middle_frame_{i}"] = ("IMAGE",)

        return {
            "required": {
                "auth": ("KLING_AUTH",),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional. Images are auto-ID'd @image_1, @image_2, ... in payload order.",
                    },
                ),
                "resolution": (KLING4_RESOLUTIONS, {"default": "1080p"}),
                "duration": ("INT", {"default": 10, "min": 3, "max": 30}),
                "watermark": ("BOOLEAN", {"default": False}),
                "callback_url": ("STRING", {"default": ""}),
                "external_task_id": ("STRING", {"default": ""}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling 4.0 REST/Video"

    def generate(
        self,
        auth,
        prompt,
        resolution,
        duration,
        watermark,
        callback_url,
        external_task_id,
        first_frame=None,
        last_frame=None,
        middle_frame_1=None,
        middle_frame_2=None,
        middle_frame_3=None,
        middle_frame_4=None,
        middle_frame_5=None,
        middle_frame_6=None,
        middle_frame_7=None,
        middle_frame_8=None,
    ):
        text = _validate_prompt(prompt, required=False)
        middle_frames = [
            middle_frame_1,
            middle_frame_2,
            middle_frame_3,
            middle_frame_4,
            middle_frame_5,
            middle_frame_6,
            middle_frame_7,
            middle_frame_8,
        ]

        image_rows: List[tuple[str, torch.Tensor, str]] = []
        if first_frame is not None:
            image_rows.append(("first_frame", first_frame, "first_frame"))
        for i, image in enumerate(middle_frames, 1):
            if image is not None:
                image_rows.append(("middle_frame", image, f"middle_frame_{i}"))
        if last_frame is not None:
            image_rows.append(("last_frame", last_frame, "last_frame"))

        if not image_rows:
            raise ValueError("Kling 4.0 Image to Video requires at least one keyframe image.")
        if len(image_rows) > 10:
            raise ValueError("Kling 4.0 Image to Video supports at most 10 images.")

        contents: List[Dict[str, Any]] = []
        if text:
            contents.append({"type": "prompt", "text": text})
        for index, (kind, image, label) in enumerate(image_rows, 1):
            contents.append(
                {
                    "type": kind,
                    "url": _encode_image(image, label),
                    "id": f"image_{index}",
                }
            )

        body = {
            "contents": contents,
            "settings": {
                "duration": _validate_duration(duration),
                "resolution": resolution,
            },
            "options": _options(watermark, callback_url, external_task_id),
        }

        client = _client_from_auth(auth)
        task_id = client.create_new_video("image-to-video", KLING4_MODEL, body)
        task = client.poll_new_task(task_id)
        return _finish(task, task_id)


class Kling4REST_OmniVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional: Dict[str, Any] = {
            "first_frame": ("IMAGE",),
            "last_frame": ("IMAGE",),
        }
        for i in range(1, 9):
            optional[f"middle_frame_{i}"] = ("IMAGE",)
        for i in range(1, 11):
            optional[f"refer_image_{i}"] = ("IMAGE",)
        optional["base_video_path"] = (
            "STRING",
            {
                "default": "",
                "forceInput": True,
                "tooltip": "Optional local .mp4/.mov. Connect video_path from Kling Video Loader; uploaded via official Kling CLI file_upload.",
            },
        )
        for i in range(1, 6):
            optional[f"feature_video_path_{i}"] = (
                "STRING",
                {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional local .mp4/.mov feature reference. Uploaded via official Kling CLI file_upload.",
                },
            )

        return {
            "required": {
                "auth": ("KLING_AUTH",),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Required. Auto IDs can be referenced as @image_1, @video_1, @element_1 and @voice_1.",
                    },
                ),
                "resolution": (KLING4_RESOLUTIONS, {"default": "1080p"}),
                "aspect_ratio": (KLING4_OMNI_ASPECT_RATIOS, {"default": "adaptive"}),
                "duration": ("INT", {"default": 10, "min": 3, "max": 30}),
                "feature_video_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Public .mp4/.mov URLs, one per line. Up to 5 videos total including base_video.",
                    },
                ),
                "base_video_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional public .mp4/.mov URL. At most one base_video per task.",
                    },
                ),
                "element_ids": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma/newline list or JSON array. Up to 7 Elements; auto IDs @element_1...",
                    },
                ),
                "voice_ids": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma/newline list or JSON array. Up to 3 Voices; auto IDs @voice_1...",
                    },
                ),
                "watermark": ("BOOLEAN", {"default": False}),
                "callback_url": ("STRING", {"default": ""}),
                "external_task_id": ("STRING", {"default": ""}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling 4.0 REST/Video"

    def generate(
        self,
        auth,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        feature_video_urls,
        base_video_url,
        element_ids,
        voice_ids,
        watermark,
        callback_url,
        external_task_id,
        first_frame=None,
        last_frame=None,
        middle_frame_1=None,
        middle_frame_2=None,
        middle_frame_3=None,
        middle_frame_4=None,
        middle_frame_5=None,
        middle_frame_6=None,
        middle_frame_7=None,
        middle_frame_8=None,
        refer_image_1=None,
        refer_image_2=None,
        refer_image_3=None,
        refer_image_4=None,
        refer_image_5=None,
        refer_image_6=None,
        refer_image_7=None,
        refer_image_8=None,
        refer_image_9=None,
        refer_image_10=None,
        base_video_path=None,
        feature_video_path_1=None,
        feature_video_path_2=None,
        feature_video_path_3=None,
        feature_video_path_4=None,
        feature_video_path_5=None,
    ):
        text = _validate_prompt(prompt, required=True)

        middle_frames = [
            middle_frame_1,
            middle_frame_2,
            middle_frame_3,
            middle_frame_4,
            middle_frame_5,
            middle_frame_6,
            middle_frame_7,
            middle_frame_8,
        ]
        refer_images = [
            refer_image_1,
            refer_image_2,
            refer_image_3,
            refer_image_4,
            refer_image_5,
            refer_image_6,
            refer_image_7,
            refer_image_8,
            refer_image_9,
            refer_image_10,
        ]

        image_rows: List[tuple[str, torch.Tensor, str]] = []
        if first_frame is not None:
            image_rows.append(("first_frame", first_frame, "first_frame"))
        for i, image in enumerate(middle_frames, 1):
            if image is not None:
                image_rows.append(("middle_frame", image, f"middle_frame_{i}"))
        if last_frame is not None:
            image_rows.append(("last_frame", last_frame, "last_frame"))
        for i, image in enumerate(refer_images, 1):
            if image is not None:
                image_rows.append(("refer_image", image, f"refer_image_{i}"))

        if len(image_rows) > 10:
            raise ValueError(
                f"Kling 4.0 Omni supports at most 10 images total; got {len(image_rows)}."
            )

        feature_videos = _parse_url_lines(
            feature_video_urls, 5, "feature_video_urls"
        )
        local_feature_paths = [
            p.strip()
            for p in (
                feature_video_path_1,
                feature_video_path_2,
                feature_video_path_3,
                feature_video_path_4,
                feature_video_path_5,
            )
            if isinstance(p, str) and p.strip()
        ]

        base_video = (base_video_url or "").strip()
        local_base_video = (
            base_video_path.strip()
            if isinstance(base_video_path, str) and base_video_path.strip()
            else ""
        )
        if base_video and local_base_video:
            raise ValueError(
                "Use either base_video_url or base_video_path, not both."
            )
        if base_video:
            _validate_public_url(base_video, "base_video_url")

        video_count = (
            len(feature_videos)
            + len(local_feature_paths)
            + (1 if (base_video or local_base_video) else 0)
        )
        if video_count > 5:
            raise ValueError(
                "Kling 4.0 Omni supports at most 5 reference videos total."
            )

        for path in local_feature_paths:
            _validate_local_video_path(path, "feature_video_path")
        if local_base_video:
            _validate_local_video_path(local_base_video, "base_video_path")

        if local_base_video:
            base_video, _, _ = _official_upload_video(auth, local_base_video)

        for path in local_feature_paths:
            uploaded_url, _, _ = _official_upload_video(auth, path)
            feature_videos.append(uploaded_url)

        elements = _parse_element_ids(element_ids, max_count=7)
        voices = _parse_string_ids(voice_ids, 3, "voice_ids")

        combined = len(image_rows) + video_count + len(elements) + len(voices)
        if combined > 15:
            raise ValueError(
                "Kling 4.0 Omni supports at most 15 combined images, videos, Elements and Voices; "
                f"got {combined}."
            )

        contents: List[Dict[str, Any]] = [{"type": "prompt", "text": text}]

        for index, (kind, image, label) in enumerate(image_rows, 1):
            contents.append(
                {
                    "type": kind,
                    "url": _encode_image(image, label),
                    "id": f"image_{index}",
                }
            )

        video_index = 1
        if base_video:
            contents.append(
                {
                    "type": "base_video",
                    "url": base_video,
                    "id": f"video_{video_index}",
                }
            )
            video_index += 1
        for url in feature_videos:
            contents.append(
                {
                    "type": "feature_video",
                    "url": url,
                    "id": f"video_{video_index}",
                }
            )
            video_index += 1

        for index, element_id in enumerate(elements, 1):
            contents.append(
                {
                    "type": "element",
                    "element_id": str(element_id),
                    "id": f"element_{index}",
                }
            )

        for index, voice_id in enumerate(voices, 1):
            contents.append(
                {
                    "type": "voice",
                    "voice_id": voice_id,
                    "id": f"voice_{index}",
                }
            )

        body = {
            "contents": contents,
            "settings": {
                "duration": _validate_duration(duration),
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            },
            "options": _options(watermark, callback_url, external_task_id),
        }

        client = _client_from_auth(auth)
        task_id = client.create_new_video("omni-video", KLING4_MODEL, body)
        task = client.poll_new_task(task_id)
        return _finish(task, task_id)


NODE_CLASS_MAPPINGS = {
    "Kling4REST_Auth": Kling4REST_Auth,
    "Kling4REST_AccessCheck": Kling4REST_AccessCheck,
    "Kling4REST_TaskStatus": Kling4REST_TaskStatus,
    "Kling4REST_OfficialVideoUpload": Kling4REST_OfficialVideoUpload,
    "Kling4REST_TextToVideo": Kling4REST_TextToVideo,
    "Kling4REST_ImageToVideo": Kling4REST_ImageToVideo,
    "Kling4REST_OmniVideo": Kling4REST_OmniVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling4REST_Auth": "Kling 4.0 REST • API Key",
    "Kling4REST_AccessCheck": "Kling 4.0 REST • Access Check",
    "Kling4REST_TaskStatus": "Kling 4.0 REST • Task Status",
    "Kling4REST_OfficialVideoUpload": "Kling 4.0 REST • Official Video Upload",
    "Kling4REST_TextToVideo": "Kling 4.0 REST • Text to Video",
    "Kling4REST_ImageToVideo": "Kling 4.0 REST • Image to Video / Multi-Keyframes",
    "Kling4REST_OmniVideo": "Kling 4.0 REST • Omni Reference / Video Edit",
}
