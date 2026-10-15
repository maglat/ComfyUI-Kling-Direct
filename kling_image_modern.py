"""Modern Kling image nodes.

Adds:
- Direct REST Omni Image with selectable kling-v3-omni / kling-image-o1.
- Official Kling CLI/MCP image capability discovery.
- Dynamic text-to-image and image-to-image nodes that validate models and
  parameters against live ``kling who_am_i`` data.

The CLI nodes intentionally use a free-form model name instead of hard-coding
the current image generation. If Kling later advertises e.g.
``kling-image-v4_0`` for the signed-in account, the same nodes can use it
without a code update.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

try:
    from .kling_nodes import download_to_tensor
    from .kling3_full import (
        OMNI_IMAGE_ASPECT_RATIOS,
        _client_from_auth,
        _collect_images,
        _download_image_batch,
        _legacy_image_urls,
        _parse_element_ids,
    )
    from .kling4_cli import (
        Kling4CLIError,
        _FAILED,
        _SUCCESS,
        _append_advanced_args,
        _available_models,
        _declared_arguments,
        _find_declared_name,
        _flag_for_argument,
        _generation_id,
        _model_spec,
        _run_cli,
        _save_image,
        _status,
        _unwrap_payload,
        _who_am_i,
        _works,
    )
except ImportError:
    from kling_nodes import download_to_tensor
    from kling3_full import (
        OMNI_IMAGE_ASPECT_RATIOS,
        _client_from_auth,
        _collect_images,
        _download_image_batch,
        _legacy_image_urls,
        _parse_element_ids,
    )
    from kling4_cli import (
        Kling4CLIError,
        _FAILED,
        _SUCCESS,
        _append_advanced_args,
        _available_models,
        _declared_arguments,
        _find_declared_name,
        _flag_for_argument,
        _generation_id,
        _model_spec,
        _run_cli,
        _save_image,
        _status,
        _unwrap_payload,
        _who_am_i,
        _works,
    )


CURRENT_CLI_IMAGE_MODEL = "kling-image-v3_0"
REST_OMNI_IMAGE_MODELS = ["kling-v3-omni", "kling-image-o1"]
IMAGE_RESOLUTIONS = ["auto", "1k", "2k", "4k", "8k"]
IMAGE_ASPECT_RATIOS = [
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "21:9",
]


def _allowed_values(argument: Dict[str, Any]) -> List[str]:
    values = argument.get("allowedValues") or argument.get("allowed_values") or []
    return [str(v) for v in values] if isinstance(values, list) else []


def _append_checked_argument(
    cmd: List[str],
    spec: Dict[str, Any],
    candidates: Sequence[str],
    value: Any,
    *,
    require: bool = False,
    label: str = "",
) -> bool:
    arguments = _declared_arguments(spec)
    name = _find_declared_name(arguments.keys(), candidates)
    if not name:
        if require:
            raise Kling4CLIError(
                "The live Kling schema for %s does not expose %r. "
                "No paid request was submitted."
                % (spec.get("model"), label or candidates[0])
            )
        return False

    allowed = _allowed_values(arguments[name])
    string_value = str(value)
    if allowed and string_value not in allowed:
        raise Kling4CLIError(
            "%s=%r is not allowed by the live %s schema. Allowed values: %s. "
            "No paid request was submitted."
            % (label or name, string_value, spec.get("model"), allowed)
        )

    cmd.extend([_flag_for_argument(name), string_value])
    return True


def _image_urls(payload: Dict[str, Any]) -> List[str]:
    payload = _unwrap_payload(payload)
    urls: List[str] = []
    for work in _works(payload):
        content_type = str(
            work.get("contentType") or work.get("content_type") or ""
        ).lower()
        if content_type and "image" not in content_type:
            continue
        url = (
            work.get("urlWithoutWatermark")
            or work.get("url_without_watermark")
            or work.get("url")
        )
        if url:
            urls.append(str(url))
    if not urls:
        for work in _works(payload):
            url = (
                work.get("urlWithoutWatermark")
                or work.get("url_without_watermark")
                or work.get("url")
            )
            if url:
                urls.append(str(url))
    return urls


def _finish_images(payload: Dict[str, Any]):
    payload = _unwrap_payload(payload)
    gid = _generation_id(payload)
    state = _status(payload).lower()
    if state in _FAILED:
        raise Kling4CLIError(
            "Kling image generation failed: " + json.dumps(payload)[:2500]
        )
    if state not in _SUCCESS and not _works(payload):
        raise Kling4CLIError(
            "Kling image generation is not complete (status=%s, generationId=%s). "
            "Use the Kling CLI query_tasks command to inspect it."
            % (state, gid or "unknown")
        )

    urls = _image_urls(payload)
    if not urls:
        raise Kling4CLIError(
            "Kling image generation completed without image URLs: "
            + json.dumps(payload)[:2500]
        )
    tensors = [download_to_tensor(url) for url in urls]
    images = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
    return (
        images,
        json.dumps(urls, indent=2, ensure_ascii=False),
        gid,
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


def _add_image_generation_arguments(
    cmd: List[str],
    spec: Dict[str, Any],
    image_count: int,
    resolution: str,
    aspect_ratio: str,
) -> None:
    if int(image_count) != 1:
        _append_checked_argument(
            cmd,
            spec,
            ["imageCount", "image_count"],
            int(image_count),
            require=True,
            label="image count",
        )
    else:
        _append_checked_argument(
            cmd,
            spec,
            ["imageCount", "image_count"],
            1,
            require=False,
            label="image count",
        )

    if resolution != "auto":
        _append_checked_argument(
            cmd,
            spec,
            ["imgResolution", "image_resolution", "resolution"],
            resolution,
            require=True,
            label="image resolution",
        )
    if aspect_ratio != "auto":
        _append_checked_argument(
            cmd,
            spec,
            ["aspectRatio", "aspect_ratio"],
            aspect_ratio,
            require=True,
            label="aspect ratio",
        )


class KlingImageCLI_Capabilities:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("ready", "models", "capabilities_json")
    FUNCTION = "check"
    CATEGORY = "Kling Image/Config"

    def check(self):
        try:
            who = _who_am_i()
            available = _available_models(who)
            filtered = {
                "text_to_image": available.get("text_to_image") or [],
                "image_to_image": available.get("image_to_image") or [],
            }
            names = sorted(
                {
                    str(row.get("model"))
                    for rows in filtered.values()
                    for row in rows
                    if isinstance(row, dict) and row.get("model")
                }
            )
            ready = bool(names)
            return (
                ready,
                ", ".join(names) if names else "No image models advertised.",
                json.dumps(filtered, indent=2, ensure_ascii=False),
            )
        except Exception as exc:
            return False, "Kling image capability check failed: %s" % exc, "{}"


class KlingImageCLI_TextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    "STRING",
                    {
                        "default": CURRENT_CLI_IMAGE_MODEL,
                        "tooltip": (
                            "Live Kling CLI model name. Use Models & Capabilities to inspect "
                            "your account. Future models such as a possible kling-image-v4_0 "
                            "can be entered here once Kling advertises them."
                        ),
                    },
                ),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 9}),
                "resolution": (IMAGE_RESOLUTIONS, {"default": "auto"}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "auto"}),
                "poll_seconds": ("INT", {"default": 300, "min": 30, "max": 1800}),
                "advanced_cli_args": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional argv from kling text_to_image --help. No shell is used.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "generation_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Image/CLI"

    def generate(
        self,
        prompt,
        model,
        image_count,
        resolution,
        aspect_ratio,
        poll_seconds,
        advanced_cli_args,
    ):
        prompt = (prompt or "").strip()
        model = (model or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if not model:
            raise ValueError("Model cannot be empty.")

        spec = _model_spec("text_to_image", model, _who_am_i())
        cmd = ["text_to_image", "--model", model]
        _add_image_generation_arguments(
            cmd, spec, image_count, resolution, aspect_ratio
        )
        _append_advanced_args(cmd, advanced_cli_args)
        cmd += ["--poll", str(int(poll_seconds)), "--quiet", prompt]
        result = _run_cli(cmd, timeout=int(poll_seconds) + 90)
        return _finish_images(result)


class KlingImageCLI_ImageToImage:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "refer_image_%d" % i: ("IMAGE",)
            for i in range(1, 11)
        }
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    "STRING",
                    {
                        "default": CURRENT_CLI_IMAGE_MODEL,
                        "tooltip": "Live image_to_image model from Kling Image Models & Capabilities.",
                    },
                ),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 9}),
                "resolution": (IMAGE_RESOLUTIONS, {"default": "auto"}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "auto"}),
                "elements_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional JSON array. Only submitted when the live model declares elements.",
                    },
                ),
                "poll_seconds": ("INT", {"default": 300, "min": 30, "max": 1800}),
                "advanced_cli_args": ("STRING", {"default": ""}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "generation_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Image/CLI"

    def generate(
        self,
        prompt,
        model,
        image_count,
        resolution,
        aspect_ratio,
        elements_json,
        poll_seconds,
        advanced_cli_args,
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
    ):
        prompt = (prompt or "").strip()
        model = (model or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if not model:
            raise ValueError("Model cannot be empty.")

        images = [
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
        images = [image for image in images if image is not None]
        if not images:
            raise ValueError("Image-to-Image requires at least one reference image.")

        spec = _model_spec("image_to_image", model, _who_am_i())
        with tempfile.TemporaryDirectory(prefix="kling-image-") as tmp:
            root = Path(tmp)
            cmd = ["image_to_image", "--model", model]
            for i, image in enumerate(images, 1):
                path = root / ("reference_%02d.png" % i)
                _save_image(image, path)
                cmd += ["--image", str(path)]

            _add_image_generation_arguments(
                cmd, spec, image_count, resolution, aspect_ratio
            )
            if (elements_json or "").strip():
                try:
                    elements = json.loads(elements_json)
                except json.JSONDecodeError as exc:
                    raise ValueError("elements_json must be valid JSON: %s" % exc) from None
                if not isinstance(elements, list):
                    raise ValueError("elements_json must be a JSON array.")
                _append_checked_argument(
                    cmd,
                    spec,
                    ["elements"],
                    json.dumps(elements, separators=(",", ":")),
                    require=True,
                    label="elements",
                )

            _append_advanced_args(cmd, advanced_cli_args)
            cmd += ["--poll", str(int(poll_seconds)), "--quiet", prompt]
            result = _run_cli(cmd, timeout=int(poll_seconds) + 120)
        return _finish_images(result)


class KlingImageREST_Omni:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"refer_image_%d" % i: ("IMAGE",) for i in range(1, 8)}
        return {
            "required": {
                "auth": ("KLING_AUTH",),
                "model": (
                    REST_OMNI_IMAGE_MODELS,
                    {"default": "kling-v3-omni"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": (["1k", "2k", "4k"], {"default": "2k"}),
                "aspect_ratio": (
                    OMNI_IMAGE_ASPECT_RATIOS,
                    {"default": "auto"},
                ),
                "result_type": (
                    ["single", "series"],
                    {"default": "single"},
                ),
                "n": ("INT", {"default": 1, "min": 1, "max": 9}),
                "series_amount": (
                    ["auto", "2", "3", "4", "5", "6", "7", "8", "9"],
                    {"default": "auto"},
                ),
                "element_ids": ("STRING", {"default": ""}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling Image/REST"

    def generate(
        self,
        auth,
        model,
        prompt,
        resolution,
        aspect_ratio,
        result_type,
        n,
        series_amount,
        element_ids,
        watermark,
        refer_image_1=None,
        refer_image_2=None,
        refer_image_3=None,
        refer_image_4=None,
        refer_image_5=None,
        refer_image_6=None,
        refer_image_7=None,
    ):
        if not (prompt or "").strip():
            raise ValueError("Prompt cannot be empty.")

        refs = _collect_images(
            refer_image_1,
            refer_image_2,
            refer_image_3,
            refer_image_4,
            refer_image_5,
            refer_image_6,
            refer_image_7,
        )
        elements = _parse_element_ids(element_ids, max_count=10)
        if len(refs) + len(elements) > 10:
            raise ValueError(
                "Omni Image supports at most 10 reference images + elements in total."
            )

        body: Dict[str, Any] = {
            "model_name": model,
            "prompt": prompt.strip(),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "result_type": result_type,
            "watermark_info": {"enabled": bool(watermark)},
        }
        if refs:
            body["image_list"] = [{"image": value} for value in refs]
        if elements:
            body["element_list"] = [{"element_id": value} for value in elements]
        if result_type == "series":
            body["series_amount"] = (
                "auto" if series_amount == "auto" else int(series_amount)
            )
        else:
            body["n"] = int(n)

        client = _client_from_auth(auth)
        endpoint = "/v1/images/omni-image"
        task_id = client.create_legacy_task(endpoint, body)
        data = client.poll_legacy_task(endpoint, task_id)
        urls = _legacy_image_urls(data)
        return (
            _download_image_batch(urls),
            json.dumps(urls, indent=2, ensure_ascii=False),
            task_id,
        )


NODE_CLASS_MAPPINGS = {
    "KlingImageCLI_Capabilities": KlingImageCLI_Capabilities,
    "KlingImageCLI_TextToImage": KlingImageCLI_TextToImage,
    "KlingImageCLI_ImageToImage": KlingImageCLI_ImageToImage,
    "KlingImageREST_Omni": KlingImageREST_Omni,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingImageCLI_Capabilities": "Kling Image - Models & Capabilities (Official CLI)",
    "KlingImageCLI_TextToImage": "Kling Image - Text to Image (Official CLI)",
    "KlingImageCLI_ImageToImage": "Kling Image - Image to Image (Official CLI)",
    "KlingImageREST_Omni": "Kling Image - Omni REST (3.0 / O1)",
}
