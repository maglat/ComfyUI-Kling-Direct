"""Kling 4.0 nodes using the official Kling CLI/MCP transport.

As of 2026-10-15 Kling 4.0 is available through Kling's Creator/CLI/MCP
surface while the public paid REST API still marks 4.0 as Preview/Coming Soon.
These nodes therefore use the official OAuth CLI and never guess private REST
endpoints. Before every paid request they validate the selected model and the
relevant parameter/input names against kling who_am_i.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image

try:
    from .kling_nodes import download_to_output, load_audio_to_tensor, load_video_to_tensor
except ImportError:
    from kling_nodes import download_to_output, load_audio_to_tensor, load_video_to_tensor


KLING4_STANDARD = "kling-video-v4_0"
KLING4_FLASH = "kling-video-v4_0_flash"
KLING4_MODELS = [KLING4_STANDARD, KLING4_FLASH]
ASPECT_RATIOS = ["auto", "21:9", "16:9", "1:1", "9:16"]
RESOLUTIONS = ["auto", "720p", "1080p", "4k"]
AUDIO_MODES = ["auto", "on", "off"]
MIN_KLING_CLI_VERSION = "0.2.3"

_SUCCESS = {"completed", "partial_completed", "succeed", "succeeded", "success"}
_FAILED = {"failed", "error", "cancelled", "canceled"}


class Kling4CLIError(RuntimeError):
    pass


def _interrupt_check() -> None:
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _kling_executable() -> str:
    exe = shutil.which("kling")
    if not exe:
        raise Kling4CLIError(
            "Official Kling CLI not found. On the ComfyUI server install/upgrade "
            "@klingai/cli-global (>= %s), then run kling login."
            % MIN_KLING_CLI_VERSION
        )
    return exe


def _parse_json_output(stdout: str) -> Dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise Kling4CLIError("Kling CLI returned empty stdout.")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise Kling4CLIError("Could not parse Kling CLI JSON output: " + text[-1200:])


def _run_cli(args: Sequence[str], timeout: int = 60) -> Dict[str, Any]:
    cmd = [_kling_executable(), *[str(x) for x in args]]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise Kling4CLIError(
            "Failed to start Kling CLI: %s: %s" % (type(exc).__name__, exc)
        ) from None

    # communicate() must drain the pipes while the CLI runs. Output larger
    # than the 64 KiB pipe buffer (e.g. `who_am_i`, ~65 KiB) deadlocks a
    # poll-only reader: the child blocks on write, never exits, and the call
    # dies at the timeout with an empty stderr.
    import threading

    result: Dict[str, Any] = {}

    def _drain() -> None:
        result["out"], result["err"] = proc.communicate()

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    start_t = time.monotonic()
    while reader.is_alive():
        _interrupt_check()
        if time.monotonic() - start_t > timeout:
            proc.kill()
            reader.join(timeout=5)
            raise TimeoutError(
                "Kling CLI timed out after %ss. The generation may still exist. "
                "Use Kling 4.0 Task Status if you have a generationId. stderr: %s"
                % (timeout, (result.get("err") or "")[-800:])
            )
        time.sleep(0.25)
    reader.join(timeout=5)

    out = result.get("out") or ""
    err = result.get("err") or ""
    if proc.returncode != 0:
        safe_err = (err or "").strip()[-2500:]
        raise Kling4CLIError(
            "Kling CLI failed with exit code %s. %s"
            % (proc.returncode, safe_err or "No error text returned.")
        )
    return _parse_json_output(out)


def _unwrap_payload(value: Any) -> Dict[str, Any]:
    """Unwrap the official CLI envelope and polled generation wrapper.

    The CLI normally returns {ok, status, body}.  Polling may additionally
    return body.generations[].result.  Prefer the nested tool payload before
    treating a wrapper-level status as the generation status.
    """
    if not isinstance(value, dict):
        return {}

    if any(k in value for k in ("availableModels", "available_models", "generationId", "generation_id", "works")):
        return value

    generations = value.get("generations")
    if isinstance(generations, list) and generations:
        row = generations[0]
        if isinstance(row, dict):
            result = row.get("result")
            if isinstance(result, dict):
                payload = dict(_unwrap_payload(result))
                payload.setdefault("generationId", row.get("generationId") or row.get("generation_id"))
                payload.setdefault("status", row.get("status"))
                return payload
            if any(k in row for k in ("generationId", "generation_id", "status", "works")):
                return row

    for key in ("body", "result", "data", "content"):
        child = value.get(key)
        if isinstance(child, dict):
            found = _unwrap_payload(child)
            if found:
                return found

    if "status" in value:
        return value
    return value


def _who_am_i() -> Dict[str, Any]:
    return _unwrap_payload(_run_cli(["who_am_i", "--quiet"], timeout=45))


def _available_models(who: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    models = who.get("availableModels") or who.get("available_models") or {}
    if not isinstance(models, dict):
        return {}

    # Normalize per-command entries: Kling CLI >= 0.2.3 returns one object per
    # command ({"defaultModel": ..., "models": [...]}), while the rest of this
    # module expects a flat list of model dicts per command.
    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for command, entry in models.items():
        if isinstance(entry, dict):
            rows = entry.get("models")
        elif isinstance(entry, list):
            rows = entry
        else:
            rows = None
        normalized[command] = [row for row in (rows or []) if isinstance(row, dict)]
    return normalized


def _model_spec(
    command: str, model: str, who: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    who = who or _who_am_i()
    rows = _available_models(who).get(command) or []
    for row in rows:
        if isinstance(row, dict) and row.get("model") == model:
            return row
    offered = [
        row.get("model")
        for row in rows
        if isinstance(row, dict) and row.get("model")
    ]
    raise Kling4CLIError(
        "Model %r is not available for %r on this Kling account. "
        "Live models: %s. Run kling who_am_i on the ComfyUI server to verify rollout."
        % (model, command, offered or "none")
    )


def _declared_arguments(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in spec.get("arguments") or []:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = row
    return out


def _declared_inputs(spec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for row in spec.get("inputs") or []:
        if isinstance(row, dict) and row.get("name"):
            out.append(str(row["name"]))
    return out


def _find_declared_name(
    names: Iterable[str], candidates: Iterable[str]
) -> Optional[str]:
    names = list(names)
    lower = {n.lower(): n for n in names}
    for candidate in candidates:
        if candidate in names:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _flag_for_argument(name: str) -> str:
    key = name.replace("-", "_").lower()
    known = {
        "duration": "--duration",
        "aspect_ratio": "--aspectRatio",
        "aspectratio": "--aspectRatio",
        "resolution": "--resolution",
        "video_resolution": "--resolution",
        "videoresolution": "--resolution",
        "enable_audio": "--enableAudio",
        "enableaudio": "--enableAudio",
        "keep_original_sound": "--keepOriginalSound",
        "keeporiginalsound": "--keepOriginalSound",
        "hdr": "--hdr",
        "dynamic_range": "--dynamicRange",
        "dynamicrange": "--dynamicRange",
    }
    return known.get(key, "--" + name)


def _append_if_declared(
    cmd: List[str],
    spec: Dict[str, Any],
    candidates: Sequence[str],
    value: Any,
    *,
    require: bool = False,
    label: Optional[str] = None,
) -> bool:
    args = _declared_arguments(spec)
    name = _find_declared_name(args.keys(), candidates)
    if not name:
        if require:
            raise Kling4CLIError(
                "The live Kling schema for %s does not expose %r. "
                "No paid request was submitted."
                % (spec.get("model"), label or candidates[0])
            )
        return False
    cmd.extend([_flag_for_argument(name), str(value)])
    return True


def _append_schema_prompt(
    cmd: List[str],
    spec: Dict[str, Any],
    prompt: Optional[str],
    *,
    positional_fallback: bool = True,
) -> bool:
    """Respect CLI 0.2.3's distinction between omitted and explicit-empty prompt.

    If the live model declares a prompt argument, a required prompt is always
    passed explicitly with --prompt, including an empty string. Optional empty
    prompts are omitted so the server can apply its default behavior.
    """
    arguments = _declared_arguments(spec)
    name = _find_declared_name(arguments.keys(), ["prompt"])
    value = "" if prompt is None else str(prompt)

    if name:
        row = arguments[name]
        required = bool(row.get("required"))
        if value != "" or required:
            cmd.extend([_flag_for_argument(name), value])
            return True
        return False

    if positional_fallback and value.strip():
        cmd.append(value.strip())
        return True
    return False


def _parse_elements_json(elements_json: str) -> List[Dict[str, Any]]:
    """Parse Elements and validate CLI 0.2.3 binding objects.

    The official guidance now binds persistent Elements with id, bindName and
    type. Keep non-id future payloads pass-through compatible, but when an
    object is an Element binding (contains id), require the complete triplet.
    """
    raw = (elements_json or "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("elements_json must be valid JSON: %s" % exc) from None
    if not isinstance(value, list):
        raise ValueError("elements_json must be a JSON array.")

    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise ValueError("elements_json item %d must be an object." % index)
        if "id" in item:
            missing = [
                key for key in ("id", "bindName", "type")
                if not str(item.get(key, "")).strip()
            ]
            if missing:
                raise ValueError(
                    "Element binding %d is missing %s. Kling CLI 0.2.3 "
                    "documents Element bindings as {id, bindName, type}."
                    % (index, ", ".join(missing))
                )
    return value


def _append_advanced_args(cmd: List[str], raw: str) -> None:
    raw = (raw or "").strip()
    if not raw:
        return
    parts = shlex.split(raw)
    forbidden = {"--model", "--poll", "--quiet"}
    if any(part in forbidden for part in parts):
        raise ValueError(
            "advanced_cli_args may not override --model, --poll or --quiet."
        )
    cmd.extend(parts)


def _save_image(image: torch.Tensor, path: Path) -> None:
    if image is None:
        raise ValueError("Image input is required.")
    if image.dim() == 4:
        image = image[0]
    arr = (image.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(
            "Expected ComfyUI IMAGE tensor [H,W,3/4], got %r" % (tuple(arr.shape),)
        )
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")


def _generation_id(payload: Dict[str, Any]) -> str:
    payload = _unwrap_payload(payload)
    return str(payload.get("generationId") or payload.get("generation_id") or "")


def _status(payload: Dict[str, Any]) -> str:
    payload = _unwrap_payload(payload)
    return str(payload.get("status") or "unknown")


def _works(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = _unwrap_payload(payload)
    rows = payload.get("works") or []
    return [row for row in rows if isinstance(row, dict)]


def _best_video_url(payload: Dict[str, Any]) -> str:
    rows = _works(payload)
    for work in rows:
        ctype = str(
            work.get("contentType") or work.get("content_type") or ""
        ).lower()
        if ctype and "video" not in ctype:
            continue
        url = (
            work.get("urlWithoutWatermark")
            or work.get("url_without_watermark")
            or work.get("url")
        )
        if url:
            return str(url)
    for work in rows:
        url = (
            work.get("urlWithoutWatermark")
            or work.get("url_without_watermark")
            or work.get("url")
        )
        if url:
            return str(url)
    raise Kling4CLIError(
        "Kling generation completed without a video URL: "
        + json.dumps(payload)[:2500]
    )


def _finish_video(payload: Dict[str, Any]):
    gid = _generation_id(payload)
    status = _status(payload).lower()
    if status in _FAILED:
        raise Kling4CLIError(
            "Kling generation failed: " + json.dumps(payload)[:2500]
        )
    if status not in _SUCCESS and not _works(payload):
        raise Kling4CLIError(
            "Kling generation is not complete (status=%s, generationId=%s). "
            "Use Kling 4.0 Task Status to query it later."
            % (status, gid or "unknown")
        )
    url = _best_video_url(payload)
    path, name = download_to_output(url)
    video = load_video_to_tensor(path)
    audio = load_audio_to_tensor(path)
    return (
        video,
        name,
        audio,
        url,
        gid,
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


def _validate_model_basics(model: str, duration: int, resolution: str) -> None:
    duration = int(duration)
    if model == KLING4_FLASH:
        if not 3 <= duration <= 20:
            raise ValueError("Kling 4.0 Flash supports 3-20 seconds.")
        if resolution not in ("auto", "720p"):
            raise ValueError("Kling 4.0 Flash supports 720p only.")
    elif not 3 <= duration <= 30:
        raise ValueError("Kling 4.0 supports 3-30 seconds.")


def _add_common_video_args(
    cmd: List[str],
    spec: Dict[str, Any],
    duration: int,
    aspect_ratio: str,
    resolution: str,
    audio: str,
) -> None:
    _append_if_declared(
        cmd, spec, ["duration"], int(duration), require=True, label="duration"
    )
    if aspect_ratio != "auto":
        _append_if_declared(
            cmd,
            spec,
            ["aspect_ratio", "aspectRatio"],
            aspect_ratio,
            require=True,
            label="aspect ratio",
        )
    if resolution != "auto":
        _append_if_declared(
            cmd,
            spec,
            ["resolution", "video_resolution", "videoResolution"],
            resolution,
            require=True,
            label="resolution",
        )
    if audio != "auto":
        _append_if_declared(
            cmd,
            spec,
            ["enable_audio", "enableAudio"],
            "true" if audio == "on" else "false",
            require=True,
            label="audio toggle",
        )


def _element_count(elements_json: str) -> int:
    return len(_parse_elements_json(elements_json))


class Kling4CLI_Status:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("ready", "status", "capabilities_json")
    FUNCTION = "check"
    CATEGORY = "Kling 4.0/Config"

    def check(self):
        exe = shutil.which("kling")
        if not exe:
            return (
                False,
                "Official Kling CLI is not installed. Install/upgrade @klingai/cli-global (>= %s)." % MIN_KLING_CLI_VERSION,
                "{}",
            )
        try:
            who = _who_am_i()
            models = _available_models(who)
            found = sorted(
                {
                    row.get("model")
                    for rows in models.values()
                    if isinstance(rows, list)
                    for row in rows
                    if isinstance(row, dict) and row.get("model") in KLING4_MODELS
                }
            )
            ready = bool(found)
            message = (
                "Kling CLI authenticated. Available Kling 4 models: %s." % found
                if ready
                else "Kling CLI authenticated, but who_am_i does not advertise Kling 4.0 for this account yet."
            )
            return ready, message, json.dumps(who, indent=2, ensure_ascii=False)
        except Exception as exc:
            return (
                False,
                "Kling CLI exists at %s but OAuth/capability check failed: %s. "
                "Run kling login on the ComfyUI server." % (exe, exc),
                "{}",
            )


class Kling4CLI_TaskStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_id": (
                    "STRING",
                    {"default": "", "forceInput": True},
                ),
                "poll_seconds": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1800},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "result_json", "generation_id")
    FUNCTION = "check"
    CATEGORY = "Kling 4.0/Config"

    def check(self, generation_id, poll_seconds):
        gid = (generation_id or "").strip()
        if not gid:
            raise ValueError("generation_id is required.")
        cmd = ["query_tasks", gid]
        if int(poll_seconds) > 0:
            cmd += ["--poll", str(int(poll_seconds))]
        cmd.append("--quiet")
        result = _run_cli(
            cmd,
            timeout=max(60, int(poll_seconds) + 45),
        )
        return (
            _status(result),
            json.dumps(result, indent=2, ensure_ascii=False),
            gid,
        )


class Kling4CLI_TextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": (
                    KLING4_MODELS,
                    {"default": KLING4_STANDARD},
                ),
                "duration": (
                    "INT",
                    {"default": 10, "min": 3, "max": 30},
                ),
                "aspect_ratio": (
                    ASPECT_RATIOS,
                    {"default": "16:9"},
                ),
                "resolution": (
                    RESOLUTIONS,
                    {"default": "auto"},
                ),
                "audio": (
                    AUDIO_MODES,
                    {"default": "auto"},
                ),
                "poll_seconds": (
                    "INT",
                    {"default": 600, "min": 30, "max": 1800},
                ),
                "advanced_cli_args": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional argv from kling text_to_video --help. No shell is used.",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "AUDIO",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "video",
        "video_file",
        "audio",
        "url",
        "generation_id",
        "metadata_json",
    )
    FUNCTION = "generate"
    CATEGORY = "Kling 4.0/Video"

    def generate(
        self,
        prompt,
        model,
        duration,
        aspect_ratio,
        resolution,
        audio,
        poll_seconds,
        advanced_cli_args,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        _validate_model_basics(model, duration, resolution)
        spec = _model_spec("text_to_video", model, _who_am_i())
        cmd = ["text_to_video", "--model", model]
        _add_common_video_args(
            cmd, spec, duration, aspect_ratio, resolution, audio
        )
        _append_advanced_args(cmd, advanced_cli_args)
        cmd += [
            "--poll",
            str(int(poll_seconds)),
            "--quiet",
            prompt,
        ]
        result = _run_cli(cmd, timeout=int(poll_seconds) + 90)
        return _finish_video(result)


class Kling4CLI_ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"tail_frame": ("IMAGE",)}
        for i in range(1, 9):
            optional["keyframe_%d" % i] = ("IMAGE",)
        return {
            "required": {
                "first_frame": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": (
                    KLING4_MODELS,
                    {"default": KLING4_STANDARD},
                ),
                "duration": (
                    "INT",
                    {"default": 10, "min": 3, "max": 30},
                ),
                "resolution": (
                    RESOLUTIONS,
                    {"default": "auto"},
                ),
                "audio": (
                    AUDIO_MODES,
                    {"default": "auto"},
                ),
                "poll_seconds": (
                    "INT",
                    {"default": 600, "min": 30, "max": 1800},
                ),
                "advanced_cli_args": (
                    "STRING",
                    {"default": ""},
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "AUDIO",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "video",
        "video_file",
        "audio",
        "url",
        "generation_id",
        "metadata_json",
    )
    FUNCTION = "generate"
    CATEGORY = "Kling 4.0/Video"

    def generate(
        self,
        first_frame,
        prompt,
        model,
        duration,
        resolution,
        audio,
        poll_seconds,
        advanced_cli_args,
        tail_frame=None,
        keyframe_1=None,
        keyframe_2=None,
        keyframe_3=None,
        keyframe_4=None,
        keyframe_5=None,
        keyframe_6=None,
        keyframe_7=None,
        keyframe_8=None,
    ):
        _validate_model_basics(model, duration, resolution)
        keyframes = [
            keyframe_1,
            keyframe_2,
            keyframe_3,
            keyframe_4,
            keyframe_5,
            keyframe_6,
            keyframe_7,
            keyframe_8,
        ]
        if model == KLING4_FLASH:
            if tail_frame is not None:
                raise ValueError(
                    "Kling 4.0 Flash does not support a tail/end frame."
                )
            if any(x is not None for x in keyframes):
                raise ValueError(
                    "Kling 4.0 Flash does not support multi-keyframes."
                )
        image_count = (
            1
            + (1 if tail_frame is not None else 0)
            + sum(x is not None for x in keyframes)
        )
        if image_count > 10:
            raise ValueError(
                "Kling 4.0 supports at most 10 keyframe images in this workflow."
            )

        spec = _model_spec("image_to_video", model, _who_am_i())
        input_names = _declared_inputs(spec)
        first_name = _find_declared_name(
            input_names,
            ["first_image", "firstImage", "image", "input"],
        )
        if not first_name:
            raise Kling4CLIError(
                "Live schema for %s exposes no first-image slot: %s"
                % (model, input_names)
            )
        tail_name = _find_declared_name(
            input_names,
            ["tail_image", "tailImage", "last_image", "lastImage"],
        )

        with tempfile.TemporaryDirectory(prefix="kling4-") as tmp:
            root = Path(tmp)
            first_path = root / "first.png"
            _save_image(first_frame, first_path)
            cmd = [
                "image_to_video",
                "--model",
                model,
                "--input",
                "%s=%s" % (first_name, first_path),
            ]

            for i, image in enumerate(keyframes, 1):
                if image is None:
                    continue
                slot = _find_declared_name(
                    input_names,
                    [
                        "keyframe_%d" % i,
                        "keyframe%d" % i,
                        "keyFrame_%d" % i,
                    ],
                )
                if not slot:
                    raise Kling4CLIError(
                        "The live %s schema does not expose keyframe slot %d. "
                        "Declared inputs: %s. No paid request was submitted."
                        % (model, i, input_names)
                    )
                path = root / ("keyframe_%d.png" % i)
                _save_image(image, path)
                cmd += ["--input", "%s=%s" % (slot, path)]

            if tail_frame is not None:
                if not tail_name:
                    raise Kling4CLIError(
                        "The live %s schema does not expose a tail/end-frame slot. "
                        "Declared inputs: %s" % (model, input_names)
                    )
                tail_path = root / "tail.png"
                _save_image(tail_frame, tail_path)
                cmd += [
                    "--input",
                    "%s=%s" % (tail_name, tail_path),
                ]

            _add_common_video_args(
                cmd,
                spec,
                duration,
                "auto",
                resolution,
                audio,
            )
            _append_advanced_args(cmd, advanced_cli_args)
            cmd += [
                "--poll",
                str(int(poll_seconds)),
                "--quiet",
            ]
            if (prompt or "").strip():
                cmd.append(prompt.strip())
            result = _run_cli(
                cmd,
                timeout=int(poll_seconds) + 120,
            )
        return _finish_video(result)


class Kling4CLI_OmniReference:
    @classmethod
    def INPUT_TYPES(cls):
        optional: Dict[str, Any] = {}
        for i in range(1, 11):
            optional["refer_image_%d" % i] = ("IMAGE",)
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": (
                    KLING4_MODELS,
                    {"default": KLING4_STANDARD},
                ),
                "duration": (
                    "INT",
                    {"default": 10, "min": 3, "max": 30},
                ),
                "aspect_ratio": (
                    ASPECT_RATIOS,
                    {"default": "auto"},
                ),
                "resolution": (
                    RESOLUTIONS,
                    {"default": "auto"},
                ),
                "audio": (
                    AUDIO_MODES,
                    {"default": "auto"},
                ),
                "keep_original_sound": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "video_refs": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Up to five public video URLs or local paths, one per line.",
                    },
                ),
                "audio_ref": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional public URL or local audio path.",
                    },
                ),
                "elements_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional JSON array passed to the official CLI elements argument.",
                    },
                ),
                "poll_seconds": (
                    "INT",
                    {"default": 600, "min": 30, "max": 1800},
                ),
                "advanced_cli_args": (
                    "STRING",
                    {"default": ""},
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "AUDIO",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "video",
        "video_file",
        "audio",
        "url",
        "generation_id",
        "metadata_json",
    )
    FUNCTION = "generate"
    CATEGORY = "Kling 4.0/Video"

    def generate(
        self,
        prompt,
        model,
        duration,
        aspect_ratio,
        resolution,
        audio,
        keep_original_sound,
        video_refs,
        audio_ref,
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
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        _validate_model_basics(model, duration, resolution)
        if keep_original_sound and audio == "on":
            raise ValueError(
                "Kling does not allow enable_audio and keepOriginalSound at the same time."
            )

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
        videos = [
            item.strip()
            for item in (video_refs or "").splitlines()
            if item.strip()
        ]
        if len(videos) > 5:
            raise ValueError(
                "Kling 4.0 supports at most 5 reference videos."
            )
        elements = _element_count(elements_json)
        if elements > 7:
            raise ValueError(
                "Kling 4.0 supports at most 7 subjects/elements."
            )
        combined = (
            len(images)
            + len(videos)
            + elements
            + (1 if (audio_ref or "").strip() else 0)
        )
        if combined > 15:
            raise ValueError(
                "Kling 4.0 supports at most 15 combined reference assets."
            )

        spec = _model_spec("omni_ref_video", model, _who_am_i())
        with tempfile.TemporaryDirectory(
            prefix="kling4-omni-"
        ) as tmp:
            root = Path(tmp)
            cmd = ["omni_ref_video", "--model", model]
            for i, image in enumerate(images, 1):
                path = root / ("ref_%d.png" % i)
                _save_image(image, path)
                cmd += ["--image", str(path)]
            for ref in videos:
                cmd += ["--video", ref]
            if (audio_ref or "").strip():
                cmd += ["--audio", audio_ref.strip()]
            if (elements_json or "").strip():
                cmd += ["--elements", elements_json.strip()]

            _add_common_video_args(
                cmd,
                spec,
                duration,
                aspect_ratio,
                resolution,
                audio,
            )
            if keep_original_sound:
                _append_if_declared(
                    cmd,
                    spec,
                    [
                        "keepOriginalSound",
                        "keep_original_sound",
                    ],
                    "true",
                    require=True,
                    label="keep original sound",
                )
            _append_advanced_args(
                cmd,
                advanced_cli_args,
            )
            cmd += [
                "--poll",
                str(int(poll_seconds)),
                "--quiet",
                prompt,
            ]
            result = _run_cli(
                cmd,
                timeout=int(poll_seconds) + 120,
            )
        return _finish_video(result)


NODE_CLASS_MAPPINGS = {
    "Kling4CLI_Status": Kling4CLI_Status,
    "Kling4CLI_TaskStatus": Kling4CLI_TaskStatus,
    "Kling4CLI_TextToVideo": Kling4CLI_TextToVideo,
    "Kling4CLI_ImageToVideo": Kling4CLI_ImageToVideo,
    "Kling4CLI_OmniReference": Kling4CLI_OmniReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling4CLI_Status": "Kling 4.0 - CLI Status & Capabilities",
    "Kling4CLI_TaskStatus": "Kling 4.0 - Task Status",
    "Kling4CLI_TextToVideo": "Kling 4.0 - Text to Video (Official CLI)",
    "Kling4CLI_ImageToVideo": "Kling 4.0 - Image to Video + Keyframes (Official CLI)",
    "Kling4CLI_OmniReference": "Kling 4.0 - Omni Reference (Official CLI)",
}
