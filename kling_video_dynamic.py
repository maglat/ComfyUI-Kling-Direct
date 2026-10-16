"""Dynamic Kling video nodes using the official Kling CLI/MCP transport.

Unlike the compatibility Kling4CLI_* nodes, this module does not hard-code
Kling 4.0 model IDs or model-specific limits. It treats kling who_am_i as
the source of truth for model names, arguments, allowed values, input slots and
combined-item caps.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .kling4_cli import (
        Kling4CLIError,
        _append_advanced_args,
        _available_models,
        _declared_arguments,
        _declared_inputs,
        _element_count,
        _find_declared_name,
        _finish_video,
        _flag_for_argument,
        _model_spec,
        _run_cli,
        _save_image,
        _who_am_i,
    )
except ImportError:
    from kling4_cli import (
        Kling4CLIError,
        _append_advanced_args,
        _available_models,
        _declared_arguments,
        _declared_inputs,
        _element_count,
        _find_declared_name,
        _finish_video,
        _flag_for_argument,
        _model_spec,
        _run_cli,
        _save_image,
        _who_am_i,
    )


VIDEO_COMMANDS = (
    "text_to_video",
    "image_to_video",
    "omni_ref_video",
    "motion_control",
)

ASPECT_RATIOS = ["auto", "21:9", "16:9", "1:1", "9:16", "4:3", "3:4", "3:2", "2:3"]
RESOLUTIONS = ["auto", "480p", "540p", "720p", "1080p", "2k", "4k", "8k"]
AUDIO_MODES = ["auto", "on", "off"]
DYNAMIC_RANGES = ["auto", "sdr", "hdr"]


def _require_model(model: str) -> str:
    model = (model or "").strip()
    if not model:
        raise ValueError(
            "Model is required. Run 'Kling Video - Models & Capabilities' "
            "or 'kling who_am_i' and paste a model advertised for this command."
        )
    return model


def _argument_allowed_values(argument: Dict[str, Any]) -> List[str]:
    values = argument.get("allowedValues")
    if values is None:
        values = argument.get("allowed_values")
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def _match_allowed_value(value: Any, allowed: Sequence[str]) -> str:
    text = str(value)
    if not allowed:
        return text
    if text in allowed:
        return text

    lower = {str(v).lower(): str(v) for v in allowed}
    key = text.lower()
    if key in lower:
        return lower[key]

    aliases = {
        "on": ("true", "1", "yes", "enabled", "native"),
        "off": ("false", "0", "no", "disabled"),
        "true": ("on", "1", "yes", "enabled", "native"),
        "false": ("off", "0", "no", "disabled"),
        "sdr": ("standard",),
        "hdr": ("high",),
    }
    for alias in aliases.get(key, ()):
        if alias in lower:
            return lower[alias]

    raise Kling4CLIError(
        "Requested value %r is not allowed by the live Kling schema. "
        "Allowed values: %s. No paid request was submitted."
        % (text, list(allowed))
    )


def _append_checked_argument(
    cmd: List[str],
    spec: Dict[str, Any],
    candidates: Sequence[str],
    value: Any,
    *,
    require: bool = False,
    label: Optional[str] = None,
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

    checked = _match_allowed_value(
        value, _argument_allowed_values(arguments[name])
    )
    cmd.extend([_flag_for_argument(name), checked])
    return True


def _append_dynamic_range(
    cmd: List[str], spec: Dict[str, Any], dynamic_range: str
) -> bool:
    value = (dynamic_range or "auto").strip().lower()
    if value == "auto":
        return False

    arguments = _declared_arguments(spec)
    name = _find_declared_name(
        arguments.keys(),
        ["dynamicRange", "dynamic_range", "color_range", "colorRange"],
    )
    if name:
        checked = _match_allowed_value(
            value, _argument_allowed_values(arguments[name])
        )
        cmd.extend([_flag_for_argument(name), checked])
        return True

    hdr_name = _find_declared_name(
        arguments.keys(), ["hdr", "enable_hdr", "enableHdr"]
    )
    if hdr_name:
        boolean_value = "true" if value == "hdr" else "false"
        checked = _match_allowed_value(
            boolean_value, _argument_allowed_values(arguments[hdr_name])
        )
        cmd.extend([_flag_for_argument(hdr_name), checked])
        return True

    raise Kling4CLIError(
        "The live Kling schema for %s does not expose a dynamic-range/HDR "
        "argument. No paid request was submitted." % spec.get("model")
    )


def _add_common_video_arguments(
    cmd: List[str],
    spec: Dict[str, Any],
    duration: int,
    aspect_ratio: str,
    resolution: str,
    audio: str,
    dynamic_range: str,
) -> None:
    if int(duration) > 0:
        _append_checked_argument(
            cmd, spec, ["duration"], int(duration),
            require=True, label="duration"
        )

    if aspect_ratio != "auto":
        _append_checked_argument(
            cmd, spec, ["aspectRatio", "aspect_ratio"], aspect_ratio,
            require=True, label="aspect ratio"
        )

    if resolution != "auto":
        _append_checked_argument(
            cmd, spec, ["resolution", "videoResolution", "video_resolution"],
            resolution, require=True, label="resolution"
        )

    if audio != "auto":
        _append_checked_argument(
            cmd, spec, ["enableAudio", "enable_audio", "audio"],
            "true" if audio == "on" else "false",
            require=True, label="audio"
        )

    _append_dynamic_range(cmd, spec, dynamic_range)


def _input_specs(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in spec.get("inputs") or []:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = row
    return out


def _parse_named_inputs(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("named_inputs_json must be valid JSON: %s" % exc) from None
    if not isinstance(data, dict):
        raise ValueError(
            'named_inputs_json must be a JSON object like '
            '{"slot_name":"public URL or local path"}.'
        )
    out: Dict[str, str] = {}
    for key, value in data.items():
        name = str(key).strip()
        path = str(value).strip()
        if not name or not path:
            raise ValueError("named_inputs_json cannot contain empty names/values.")
        out[name] = path
    return out


def _append_named_inputs(
    cmd: List[str],
    spec: Dict[str, Any],
    values: Dict[str, str],
    used: set[str],
) -> None:
    declared = _input_specs(spec)
    for name, value in values.items():
        if name not in declared:
            raise Kling4CLIError(
                "Input slot %r is not declared by live model %s. "
                "Declared inputs: %s. No paid request was submitted."
                % (name, spec.get("model"), list(declared))
            )
        if name in used:
            raise ValueError("Input slot %r was supplied more than once." % name)
        cmd += ["--input", "%s=%s" % (name, value)]
        used.add(name)


def _validate_required_inputs(spec: Dict[str, Any], used: set[str]) -> None:
    missing = [
        name for name, row in _input_specs(spec).items()
        if bool(row.get("required")) and name not in used
    ]
    if missing:
        raise Kling4CLIError(
            "The live %s schema requires input slot(s) %s. Supply them with "
            "the image inputs or named_inputs_json. No paid request was submitted."
            % (spec.get("model"), missing)
        )


def _mark_convenience_inputs(
    spec: Dict[str, Any], used: set[str], media_kind: str, count: int
) -> set[str]:
    """Account for CLI --image/--video convenience flags during required-slot validation."""
    marked = set(used)
    if count <= 0:
        return marked
    token = media_kind.lower()
    candidates = [
        name for name in _declared_inputs(spec)
        if name not in marked and token in name.lower()
    ]
    for name in candidates[:count]:
        marked.add(name)
    return marked


def _resolve_slot(
    spec: Dict[str, Any],
    candidates: Sequence[str],
    used: set[str],
    *,
    fallback_single: bool = False,
) -> Optional[str]:
    names = [name for name in _declared_inputs(spec) if name not in used]
    found = _find_declared_name(names, candidates)
    if found:
        return found
    if fallback_single and len(names) == 1:
        return names[0]
    return None


def _combined_item_cap(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cap = spec.get("combinedItemCap") or spec.get("combined_item_cap")
    return cap if isinstance(cap, dict) else None


def _validate_combined_cap(
    spec: Dict[str, Any],
    *,
    used_input_names: Iterable[str] = (),
    convenience_images: int = 0,
    convenience_videos: int = 0,
    argument_arrays: Optional[Dict[str, int]] = None,
) -> None:
    cap = _combined_item_cap(spec)
    if not cap:
        return
    try:
        maximum = int(cap.get("maxItems"))
    except (TypeError, ValueError):
        return

    prefixes = [str(x) for x in (cap.get("inputPrefixes") or [])]
    array_names = [str(x) for x in (cap.get("arguments") or [])]
    total = 0

    for name in used_input_names:
        if any(str(name).startswith(prefix) for prefix in prefixes):
            total += 1

    lower_prefixes = [p.lower() for p in prefixes]
    if any("image" in p for p in lower_prefixes):
        total += int(convenience_images)
    if any("video" in p for p in lower_prefixes):
        total += int(convenience_videos)

    for name, count in (argument_arrays or {}).items():
        if name in array_names:
            total += int(count)

    if total > maximum:
        raise Kling4CLIError(
            "The live %s schema allows at most %d combined reference items; "
            "%d were supplied. No paid request was submitted."
            % (spec.get("model"), maximum, total)
        )


def _model_summary(who: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    available = _available_models(who)
    filtered: Dict[str, Any] = {}
    lines: List[str] = []
    for command in VIDEO_COMMANDS:
        rows = available.get(command) or []
        if not isinstance(rows, list):
            rows = []
        filtered[command] = rows
        names = [
            str(row.get("model"))
            for row in rows
            if isinstance(row, dict) and row.get("model")
        ]
        if names:
            lines.append("%s: %s" % (command, ", ".join(names)))
    ready = any(bool(rows) for rows in filtered.values())
    return ready, (
        "\n".join(lines) if lines else "No video generation models advertised."
    ), filtered


class KlingVideoCLI_Capabilities:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("ready", "models_by_command", "capabilities_json")
    FUNCTION = "check"
    CATEGORY = "Kling Video/Config"

    def check(self):
        try:
            who = _who_am_i()
            ready, summary, filtered = _model_summary(who)
            return ready, summary, json.dumps(filtered, indent=2, ensure_ascii=False)
        except Exception as exc:
            return False, "Kling video capability check failed: %s" % exc, "{}"


class KlingVideoCLI_TextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Paste a text_to_video model shown by Models & Capabilities.",
                    },
                ),
                "duration": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 120,
                        "tooltip": "0 = use the live model default.",
                    },
                ),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto"}),
                "resolution": (RESOLUTIONS, {"default": "auto"}),
                "audio": (AUDIO_MODES, {"default": "auto"}),
                "dynamic_range": (DYNAMIC_RANGES, {"default": "auto"}),
                "poll_seconds": ("INT", {"default": 600, "min": 30, "max": 1800}),
                "advanced_cli_args": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "generation_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Video/Dynamic CLI"

    def generate(
        self, prompt, model, duration, aspect_ratio, resolution, audio,
        dynamic_range, poll_seconds, advanced_cli_args
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        model = _require_model(model)
        spec = _model_spec("text_to_video", model, _who_am_i())

        cmd = ["text_to_video", "--model", model]
        _add_common_video_arguments(
            cmd, spec, duration, aspect_ratio, resolution, audio, dynamic_range
        )
        _append_advanced_args(cmd, advanced_cli_args)
        cmd += ["--poll", str(int(poll_seconds)), "--quiet", prompt]
        result = _run_cli(cmd, timeout=int(poll_seconds) + 90)
        return _finish_video(result)


class KlingVideoCLI_ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional: Dict[str, Any] = {
            "first_frame": ("IMAGE",),
            "tail_frame": ("IMAGE",),
        }
        for i in range(1, 9):
            optional["keyframe_%d" % i] = ("IMAGE",)
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Paste an image_to_video model shown by Models & Capabilities.",
                    },
                ),
                "duration": ("INT", {"default": 0, "min": 0, "max": 120}),
                "resolution": (RESOLUTIONS, {"default": "auto"}),
                "audio": (AUDIO_MODES, {"default": "auto"}),
                "dynamic_range": (DYNAMIC_RANGES, {"default": "auto"}),
                "named_inputs_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Exact live input slots mapped to public URLs or local paths.",
                    },
                ),
                "poll_seconds": ("INT", {"default": 600, "min": 30, "max": 1800}),
                "advanced_cli_args": ("STRING", {"default": ""}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "generation_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Video/Dynamic CLI"

    def generate(
        self, prompt, model, duration, resolution, audio, dynamic_range,
        named_inputs_json, poll_seconds, advanced_cli_args,
        first_frame=None, tail_frame=None,
        keyframe_1=None, keyframe_2=None, keyframe_3=None, keyframe_4=None,
        keyframe_5=None, keyframe_6=None, keyframe_7=None, keyframe_8=None,
    ):
        model = _require_model(model)
        spec = _model_spec("image_to_video", model, _who_am_i())
        named = _parse_named_inputs(named_inputs_json)
        used: set[str] = set()
        keyframes = [
            keyframe_1, keyframe_2, keyframe_3, keyframe_4,
            keyframe_5, keyframe_6, keyframe_7, keyframe_8,
        ]

        with tempfile.TemporaryDirectory(prefix="kling-video-dynamic-") as tmp:
            root = Path(tmp)
            cmd = ["image_to_video", "--model", model]
            _append_named_inputs(cmd, spec, named, used)

            if first_frame is not None:
                slot = _resolve_slot(
                    spec,
                    ["first_image", "firstImage", "start_image", "startImage", "image_1", "image"],
                    used,
                    fallback_single=True,
                )
                if not slot:
                    raise Kling4CLIError(
                        "The live %s schema exposes no unused first/start-image slot. "
                        "Declared inputs: %s" % (model, _declared_inputs(spec))
                    )
                path = root / "first.png"
                _save_image(first_frame, path)
                cmd += ["--input", "%s=%s" % (slot, path)]
                used.add(slot)

            for index, image in enumerate(keyframes, 1):
                if image is None:
                    continue
                slot = _resolve_slot(
                    spec,
                    [
                        "keyframe_%d" % index, "keyframe%d" % index,
                        "keyFrame_%d" % index, "middle_image_%d" % index,
                        "middleImage%d" % index, "image_%d" % (index + 1),
                    ],
                    used,
                )
                if not slot:
                    raise Kling4CLIError(
                        "The live %s schema exposes no slot for keyframe %d. "
                        "Use named_inputs_json with an exact live slot. Declared inputs: %s"
                        % (model, index, _declared_inputs(spec))
                    )
                path = root / ("keyframe_%02d.png" % index)
                _save_image(image, path)
                cmd += ["--input", "%s=%s" % (slot, path)]
                used.add(slot)

            if tail_frame is not None:
                slot = _resolve_slot(
                    spec,
                    ["tail_image", "tailImage", "last_image", "lastImage", "end_image", "endImage"],
                    used,
                )
                if not slot:
                    raise Kling4CLIError(
                        "The live %s schema exposes no tail/end-frame slot. "
                        "Declared inputs: %s" % (model, _declared_inputs(spec))
                    )
                path = root / "tail.png"
                _save_image(tail_frame, path)
                cmd += ["--input", "%s=%s" % (slot, path)]
                used.add(slot)

            _validate_required_inputs(spec, used)
            _validate_combined_cap(spec, used_input_names=used)
            _add_common_video_arguments(
                cmd, spec, duration, "auto", resolution, audio, dynamic_range
            )
            _append_advanced_args(cmd, advanced_cli_args)
            cmd += ["--poll", str(int(poll_seconds)), "--quiet"]
            if (prompt or "").strip():
                cmd.append(prompt.strip())
            result = _run_cli(cmd, timeout=int(poll_seconds) + 120)

        return _finish_video(result)


class KlingVideoCLI_OmniReference:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"refer_image_%d" % i: ("IMAGE",) for i in range(1, 11)}
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Paste an omni_ref_video model shown by Models & Capabilities.",
                    },
                ),
                "duration": ("INT", {"default": 0, "min": 0, "max": 120}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto"}),
                "resolution": (RESOLUTIONS, {"default": "auto"}),
                "audio": (AUDIO_MODES, {"default": "auto"}),
                "dynamic_range": (DYNAMIC_RANGES, {"default": "auto"}),
                "keep_original_sound": ("BOOLEAN", {"default": False}),
                "primary_video": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional primary/base video URL or local path for video-edit/reference models.",
                    },
                ),
                "video_refs": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Additional public video URLs or local paths, one per line.",
                    },
                ),
                "elements_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional Elements JSON array supported by the selected live model.",
                    },
                ),
                "named_inputs_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Exact live input slots mapped to public URLs or local paths.",
                    },
                ),
                "poll_seconds": ("INT", {"default": 600, "min": 30, "max": 1800}),
                "advanced_cli_args": ("STRING", {"default": ""}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "generation_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Video/Dynamic CLI"

    def generate(
        self, prompt, model, duration, aspect_ratio, resolution, audio,
        dynamic_range, keep_original_sound, primary_video, video_refs,
        elements_json, named_inputs_json, poll_seconds, advanced_cli_args,
        refer_image_1=None, refer_image_2=None, refer_image_3=None,
        refer_image_4=None, refer_image_5=None, refer_image_6=None,
        refer_image_7=None, refer_image_8=None, refer_image_9=None,
        refer_image_10=None,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        model = _require_model(model)
        spec = _model_spec("omni_ref_video", model, _who_am_i())
        named = _parse_named_inputs(named_inputs_json)
        used: set[str] = set()

        images = [
            refer_image_1, refer_image_2, refer_image_3, refer_image_4, refer_image_5,
            refer_image_6, refer_image_7, refer_image_8, refer_image_9, refer_image_10,
        ]
        images = [image for image in images if image is not None]
        videos = [x.strip() for x in (video_refs or "").splitlines() if x.strip()]
        elements_count = _element_count(elements_json)

        with tempfile.TemporaryDirectory(prefix="kling-video-omni-") as tmp:
            root = Path(tmp)
            cmd = ["omni_ref_video", "--model", model]
            _append_named_inputs(cmd, spec, named, used)

            primary = (primary_video or "").strip()
            primary_via_named_slot = False
            if primary:
                primary_slot = _resolve_slot(
                    spec,
                    ["primary_video", "primaryVideo", "base_video", "baseVideo", "source_video", "sourceVideo"],
                    used,
                )
                if primary_slot:
                    cmd += ["--input", "%s=%s" % (primary_slot, primary)]
                    used.add(primary_slot)
                    primary_via_named_slot = True
                else:
                    cmd += ["--video", primary]

            for index, image in enumerate(images, 1):
                path = root / ("reference_%02d.png" % index)
                _save_image(image, path)
                cmd += ["--image", str(path)]

            for ref in videos:
                cmd += ["--video", ref]

            if (elements_json or "").strip():
                try:
                    parsed_elements = json.loads(elements_json)
                except json.JSONDecodeError as exc:
                    raise ValueError("elements_json must be valid JSON: %s" % exc) from None
                if not isinstance(parsed_elements, list):
                    raise ValueError("elements_json must be a JSON array.")
                _append_checked_argument(
                    cmd, spec, ["elements"],
                    json.dumps(parsed_elements, separators=(",", ":")),
                    require=True, label="elements"
                )

            validation_used = _mark_convenience_inputs(
                spec, used, "image", len(images)
            )
            validation_used = _mark_convenience_inputs(
                spec,
                validation_used,
                "video",
                len(videos) + (1 if primary and not primary_via_named_slot else 0),
            )
            _validate_required_inputs(spec, validation_used)
            _validate_combined_cap(
                spec,
                used_input_names=used,
                convenience_images=len(images),
                convenience_videos=len(videos) + (1 if primary and not primary_via_named_slot else 0),
                argument_arrays={"elements": elements_count},
            )

            _add_common_video_arguments(
                cmd, spec, duration, aspect_ratio, resolution, audio, dynamic_range
            )

            if keep_original_sound:
                _append_checked_argument(
                    cmd, spec, ["keepOriginalSound", "keep_original_sound"],
                    "true", require=True, label="keep original sound"
                )

            _append_advanced_args(cmd, advanced_cli_args)
            cmd += ["--poll", str(int(poll_seconds)), "--quiet", prompt]
            result = _run_cli(cmd, timeout=int(poll_seconds) + 120)

        return _finish_video(result)


class KlingVideoCLI_MotionControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_image": ("IMAGE",),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Paste a motion_control model shown by Models & Capabilities.",
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "motion_video": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Public URL/local path. Use this OR motion_id.",
                    },
                ),
                "motion_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Saved Kling motion ID. Use this OR motion_video.",
                    },
                ),
                "poll_seconds": ("INT", {"default": 600, "min": 30, "max": 1800}),
                "advanced_cli_args": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "generation_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Video/Dynamic CLI"

    def generate(
        self, subject_image, model, prompt, motion_video, motion_id,
        poll_seconds, advanced_cli_args
    ):
        model = _require_model(model)
        _model_spec("motion_control", model, _who_am_i())

        video = (motion_video or "").strip()
        motion = (motion_id or "").strip()
        if bool(video) == bool(motion):
            raise ValueError("Provide exactly one of motion_video or motion_id.")

        with tempfile.TemporaryDirectory(prefix="kling-motion-") as tmp:
            image_path = Path(tmp) / "subject.png"
            _save_image(subject_image, image_path)
            cmd = ["motion_control", "--model", model, "--image", str(image_path)]
            if video:
                cmd += ["--video", video]
            else:
                cmd += ["--motionId", motion]

            _append_advanced_args(cmd, advanced_cli_args)
            cmd += ["--poll", str(int(poll_seconds)), "--quiet"]
            if (prompt or "").strip():
                cmd.append(prompt.strip())
            result = _run_cli(cmd, timeout=int(poll_seconds) + 120)

        return _finish_video(result)


NODE_CLASS_MAPPINGS = {
    "KlingVideoCLI_Capabilities": KlingVideoCLI_Capabilities,
    "KlingVideoCLI_TextToVideo": KlingVideoCLI_TextToVideo,
    "KlingVideoCLI_ImageToVideo": KlingVideoCLI_ImageToVideo,
    "KlingVideoCLI_OmniReference": KlingVideoCLI_OmniReference,
    "KlingVideoCLI_MotionControl": KlingVideoCLI_MotionControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingVideoCLI_Capabilities": "Kling Video - Models & Capabilities (Official CLI)",
    "KlingVideoCLI_TextToVideo": "Kling Video - Text to Video (Dynamic CLI)",
    "KlingVideoCLI_ImageToVideo": "Kling Video - Image to Video / Keyframes (Dynamic CLI)",
    "KlingVideoCLI_OmniReference": "Kling Video - Omni / Video Edit (Dynamic CLI)",
    "KlingVideoCLI_MotionControl": "Kling Video - Motion Control (Dynamic CLI)",
}
