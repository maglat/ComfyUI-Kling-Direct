# ComfyUI-Kling-Direct

A direct Kling AI integration for ComfyUI with:

- **Kling Video 4.0 Early Access REST** through the direct single API key transport
- **schema-driven Kling video** through Kling's official OAuth CLI/MCP transport, including current Kling 4.x rollouts
- **Kling Image 3.0 / Image 3.0 Omni / Image O1**
- **dynamic image-model discovery** for future Kling image releases
- **Kling 3.0 Full REST** support using the current single API key
- the original Kling-Direct legacy nodes for existing workflows

Current fork version: **4.4.0**

Repository: https://github.com/maglat/ComfyUI-Kling-Direct

---

## Current model status

As of **2026-10-26** the public Kling product/API split is:

| Area | Model | This fork |
|---|---|---|
| Video | Kling 4.0 | Yes - direct Early Access REST + official CLI/MCP |
| Video | Kling 4.0 Flash | Yes - official CLI/MCP |
| Video | Kling 3.0 | Yes - direct REST |
| Video | Kling 3.0 Turbo | Yes - direct REST |
| Image | Kling Image 3.0 | Yes |
| Image | Kling Image 3.0 Omni | Yes |
| Image | Kling Image O1 | Yes |
| Image | Kling Image 4.0 | **Not officially released yet** |

Kling 4.0 is currently a **video-generation family**. Kling has not publicly released a production Image 4.0 model/API at the time of this update.

The new Image CLI nodes are deliberately **not hard-coded to Image 3.0**. They validate the model name against the live output of `kling who_am_i`. If Kling later exposes a model such as `kling-image-v4_0`, the same nodes can use it without waiting for another hard-coded dropdown update.

---

# Installation

## New install

```bash
cd /path/to/ComfyUI/custom_nodes

git clone https://github.com/maglat/ComfyUI-Kling-Direct.git

cd ComfyUI-Kling-Direct
pip install -r requirements.txt 2>/dev/null || true
```

Restart ComfyUI afterwards.

## Update an existing installation

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Kling-Direct

git remote set-url origin https://github.com/maglat/ComfyUI-Kling-Direct.git
git checkout main
git pull --ff-only origin main
```

Then restart ComfyUI.

---

# Authentication

This fork currently supports **three Kling authentication generations** because Kling exposes different product generations through different transports.

| Nodes | Authentication |
|---|---|
| Kling 4.0 Early Access REST | single `KLING_API_KEY` |
| Modern Kling Video CLI | official Kling CLI OAuth |
| Modern Kling Image CLI | official Kling CLI OAuth |
| Kling 3.0 Full REST | single `KLING_API_KEY` |
| Original legacy nodes | Access Key + Secret Key |

## Official Kling CLI / MCP

Required for the modern dynamic Kling Video and Kling Image nodes.

Node.js 18+ is required.

```bash
node --version

npm i -g @klingai/cli-global@latest --registry=https://registry.npmjs.org

kling --version   # v4.2.1 expects Kling CLI >= 0.2.3
kling login
kling who_am_i
```

The CLI uses Kling's official OAuth flow. This custom-node project does **not** request, store or print the OAuth token.

The most important diagnostic command is:

```bash
kling who_am_i
```

It returns the models, arguments, allowed values, input slots and combined-item limits currently enabled for the signed-in Kling account.

**Kling CLI 0.2.3 or newer is recommended/expected for the modern dynamic nodes.** The official 0.2.3 behavior distinguishes an omitted prompt from an explicitly empty prompt, exposes generation-command help from `who_am_i`, and documents standalone Omni audio references.

## Kling 3.0 Full REST API

The modern REST nodes use the current single Kling API key.

Recommended:

```bash
export KLING_API_KEY='YOUR_KLING_API_KEY'
```

Then leave the key field blank in:

`Kling 3.0 Full - API Key`

The environment variable avoids storing the key inside workflow JSON.

---


## Kling 4.0 local reference videos

Version 4.4.0 adds official Kling material upload for local reference videos.

Use the existing **Kling Video Loader** node and connect its `video_path` output directly to one of the optional Kling 4.0 Omni inputs:

- `base_video_path`
- `feature_video_path_1` ... `feature_video_path_5`

The file is uploaded to Kling's own `/v1/materials` endpoint with the same Bearer `KLING_API_KEY`, and the returned Kling-hosted URL is inserted into the Kling 4.0 Omni request automatically.

Public URL inputs remain supported through `base_video_url` and `feature_video_urls`.

A standalone **Kling 4.0 REST • Official Video Upload** node is also included for workflows that need the uploaded URL explicitly.

Supported local formats for Kling 4.0 Omni are `.mp4` and `.mov`; the local preflight rejects files larger than 300 MB.

# Kling Video 4.0 - Early Access REST

This branch adds direct API-key support for the temporary Kling Video 4.0 Early Access API.

Base URL:

```text
https://api-singapore.klingai.com
```

Authentication:

```text
Authorization: Bearer <KLING_API_KEY>
```

Implemented endpoints:

```text
POST /text-to-video/kling-4.0
POST /image-to-video/kling-4.0
POST /omni-video/kling-4.0
GET  /tasks?task_ids=<id>
```

Nodes:

- **Kling 4.0 REST • API Key**
- **Kling 4.0 REST • Access Check**
- **Kling 4.0 REST • Task Status**
- **Kling 4.0 REST • Text to Video**
- **Kling 4.0 REST • Image to Video / Multi-Keyframes**
- **Kling 4.0 REST • Omni Reference / Video Edit**

Highlights:

- 3-30 second output duration
- 720p / 1080p
- T2V aspect ratios: 21:9, 16:9, 9:16, 1:1
- I2V with first / middle / last keyframes and up to 10 images
- Omni with reference images, feature videos, one base video, Elements and Voice references
- Omni adaptive aspect ratio
- local preflight checks for documented image/reference-count limits
- original result video is downloaded to ComfyUI output and its audio is exposed as an AUDIO output
- existing Kling 3.0 REST and Kling 4.x CLI/MCP nodes remain unchanged

The temporary Early Access REST specification does not define a separate request-side audio toggle for the three Kling 4.0 endpoints. The REST nodes therefore do not invent one.


# Kling Video - dynamic CLI/MCP layer

Version **4.2.0** added the schema-driven video layer; **v4.2.1** synchronizes it with official Kling CLI 0.2.3 behavior.

The official Kling CLI/MCP backend is treated as the source of truth. The new nodes do **not** assume a fixed Kling 4.0 model ID such as `kling-video-v4_0`. Instead, they read the model names, arguments, allowed values, input slots and combined-item limits returned by:

```bash
kling who_am_i
```

This matters because Kling can roll out model names and capabilities per account/region before the public REST documentation is updated.

## Recommended dynamic video nodes

- **Kling Video - Models & Capabilities (Official CLI)**
- **Kling Video - Task Status (Official CLI)**
- **Kling Video - Text to Video (Dynamic CLI)**
- **Kling Video - Image to Video / Keyframes (Dynamic CLI)**
- **Kling Video - Omni / Video Edit (Dynamic CLI)**
- **Kling Video - Motion Control (Dynamic CLI)**

These nodes are the recommended path for Kling 4.x and future server-advertised video generations.

## How model selection works

The capability node filters the live `who_am_i` response for:

```text
text_to_video
image_to_video
omni_ref_video
motion_control
```

It returns the model names that are actually enabled for each command.

Generation nodes use a free-form model field. A model is accepted only when the live schema advertises it for that exact command.

That means the client itself does not block future names such as a hypothetical:

```text
kling-video-v4_1
kling-video-v5
```

but typing an invented model name does **not** bypass Kling's account entitlements: the node stops before a paid request if `who_am_i` does not list it.

## Live argument validation

For the dynamic nodes, these controls are schema-gated:

- duration
- aspect ratio
- resolution
- audio
- dynamic range / HDR
- keep-original-sound
- Elements
- named input slots

When a live argument publishes `allowedValues`, the selected value is checked locally before generation.

Set **duration = 0** or another control to **auto** when you want Kling's live/default behavior instead of forcing a value.

### HDR / dynamic range

The UI exposes:

```text
auto
sdr
hdr
```

but HDR is only submitted when the selected live model exposes a compatible `dynamicRange`/HDR argument. If it does not, an explicit HDR request fails before spending credits.

This avoids claiming HDR support merely because it appeared in a marketing/release document.

## Dynamic Image-to-Video / keyframes

The dynamic I2V node no longer requires a hard-coded first-frame model layout.

It supports:

- optional first frame
- optional tail/end frame
- up to eight additional ComfyUI keyframe sockets
- exact named live input slots through `named_inputs_json`
- prompt-only operation when the selected live model declares no required image input
- correct CLI 0.2.3 distinction between an omitted prompt and explicit `--prompt ""`

If the live model declares `prompt` as required and the ComfyUI prompt field is empty, v4.2.1 sends an explicit empty prompt. If prompt is optional and empty, it is omitted so the server can apply its default behavior.

Example named input:

```json
{
  "some_new_server_slot": "https://example.com/reference.png"
}
```

The slot name must exist in the selected model's live `inputs[]` declaration.

This provides an escape hatch for new keyframe/input layouts without a custom-node release.

## Dynamic Omni / Video Editing

**Kling Video - Omni / Video Edit (Dynamic CLI)** supports:

- up to 10 local ComfyUI reference-image sockets
- additional video references
- standalone audio references through repeated official CLI `--audio`
- Elements JSON using the current persistent-Element binding shape `id + bindName + type`
- a `primary_video` field for base/source/primary-video workflows
- exact server-defined inputs through `named_inputs_json`
- duration / aspect / resolution / audio / dynamic-range controls when declared
- keep-original-sound when declared
- live `combinedItemCap` validation

If the selected model declares a named primary/base/source-video slot, `primary_video` is mapped to that slot. Otherwise it is passed as a normal official CLI video reference.

Standalone audio references are now officially documented by Kling CLI 0.2.3. The node exposes an `audio_refs` multiline field and sends one `--audio` per entry. Before submission it verifies that the selected live model actually declares enough `audio_*` slots. Unsupported audio references therefore fail before upload/generation.

## Combined reference limits

The official capability schema may return:

```text
combinedItemCap:
  maxItems
  inputPrefixes
  arguments
```

The dynamic Omni/I2V path uses this information rather than maintaining a separate hard-coded "15 items forever" rule.

This is especially useful when future Kling generations change reference-count limits.

## Motion Control

The dynamic Motion Control node accepts any model advertised for `motion_control` and supports either:

- a reference motion-video URL/local path, or
- a saved Kling motion ID.

Exactly one must be supplied.

## Result handling

The CLI output parser now understands both direct tool results and the official CLI envelope:

```text
{ ok, status, body }
```

It also unwraps polled:

```text
body.generations[].result
```

before extracting `generationId`, `status` and `works[]`.

This fix is shared by the modern Image CLI nodes as well.

## Kling 4.0 compatibility nodes

The older nodes remain registered so existing v4.0/v4.1 workflows keep loading:

- **Kling 4.0 - CLI Status & Capabilities**
- **Kling 4.0 - Task Status**
- **Kling 4.0 - Text to Video (Official CLI)**
- **Kling 4.0 - Image to Video + Keyframes (Official CLI)**
- **Kling 4.0 - Omni Reference (Official CLI)**

Those compatibility nodes contain the earlier Kling-4-specific assumptions. For new workflows, prefer the **Kling Video - ... Dynamic CLI** nodes above.

---

# Kling Image

The image side now has **two modern transports**.

## 1. Direct REST Image nodes

These are useful when you want to use the normal Kling API key directly.

### Existing Image 3.0 node

- **Kling Image 3.0 Full - Generation**
- model: `kling-v3`
- endpoint family: `/v1/images/generations`
- Text to Image
- 1K / 2K
- up to 9 outputs

### Omni REST node

- **Kling Image - Omni REST (3.0 / O1)**

Selectable models:

```text
kling-v3-omni
kling-image-o1
```

The node supports:

- prompt generation
- reference images
- Element IDs
- 1K / 2K / 4K
- automatic or explicit aspect ratio
- single-image generation
- image series
- fixed series amount or automatic series amount
- up to 10 combined reference images + Elements

The older **Kling Image 3.0 Full - Omni** node remains registered unchanged so existing workflows continue to load.

---

# Dynamic Kling Image CLI/MCP support

Version 4.1 adds an image layer based on Kling's official CLI/MCP capability discovery.

## Nodes

- **Kling Image - Models & Capabilities (Official CLI)**
- **Kling Image - Text to Image (Official CLI)**
- **Kling Image - Image to Image (Official CLI)**

## Why these nodes are dynamic

The official Kling CLI does not require the client to maintain a permanent static list of all image models.

Instead:

```text
kling who_am_i
        |
        v
availableModels.text_to_image
availableModels.image_to_image
        |
        +--> model names
        +--> arguments
        +--> allowed values
        +--> input slots
```

Before a paid request, the nodes verify:

1. that the entered model exists for the selected command;
2. that requested parameters are exposed by that model;
3. that a requested value is allowed when the live schema publishes an allowed-value list.

This means a future image model can be used as soon as Kling advertises it to the account.

For example, if Kling eventually exposes:

```text
kling-image-v4_0
```

you can enter that model name in the Image CLI node. If `who_am_i` does not advertise it, the node stops **before generation**.

There is intentionally no fake Image 4.0 dropdown today.

## Current default CLI image model

```text
kling-image-v3_0
```

The exact live model list for your account always takes precedence.

## Text to Image CLI node

Exposes:

- prompt
- free-form live model name
- image count 1-9
- resolution selector
- aspect-ratio selector
- polling timeout
- advanced CLI arguments

Resolution and aspect-ratio values are only submitted when the selected live model declares those parameters.

## Image to Image CLI node

Supports:

- up to 10 ComfyUI reference-image inputs
- prompt
- dynamic model name
- image count
- resolution
- aspect ratio
- optional Elements JSON
- exact live input-slot mapping through `named_inputs_json`
- correct omitted-vs-explicit-empty prompt handling from CLI 0.2.3
- advanced CLI arguments

For Image-to-Image, `refer_image_*` uses the CLI convenience `--image` mapping. `named_inputs_json` uses exact `--input slot=value` mapping. CLI 0.2.3 does not allow mixing those two input modes in one request, so the node rejects that combination before submission.

Persistent Element bindings should use the official 0.2.3 structure:

```json
[
  {
    "id": "ELEMENT_ID",
    "bindName": "hero",
    "type": "character"
  }
]
```

If an Element object contains an `id`, the modern nodes now require `id`, `bindName` and `type` together.

Reference images are written to a temporary directory and passed to the official CLI as local files. The CLI performs its supported upload flow. The temporary files are deleted automatically after the operation.

---

# Preparing for Kling Image 4.0

There is currently **no officially released Image 4.0 API model** in this repository.

The preparation in v4.1.0 consists of:

- no fixed image-model whitelist in the CLI generation nodes;
- live model discovery through `who_am_i`;
- live argument discovery;
- allowed-value validation;
- generic result handling based on `works[]`;
- dynamic support for future model names;
- a regression test using a hypothetical `kling-image-v4_0` model to ensure the client itself does not block a future server-advertised model.

The server remains the source of truth. A model name entered manually is **not** accepted unless Kling actually advertises it for the logged-in account.

---

# Kling 3.0 Full REST

The Kling 3.0 Full nodes continue to use the direct API-key transport.

## Video nodes

- **Kling 3.0 Full - Text to Video**
- **Kling 3.0 Full - Image to Video**
- **Kling 3.0 Full - Omni Video**
- **Kling 3.0 Full - Motion Control**
- **Kling 3.0 Full - Storyboard Builder**
- **Kling 3.0 Full - Task Status**
- **Kling 3.0 Full - Health Check**

Main supported models:

```text
kling-3.0
kling-3.0-turbo
kling-3.0-omni
```

Full Kling 3.0 T2V/I2V supports the implemented 3-15 second flow, 720p/1080p/4K options, native audio controls, multi-shot and First/Last Frame where supported.

Turbo capability differences are validated before submission.

---

# Legacy Kling-Direct nodes

The original upstream nodes are intentionally retained for backwards compatibility.

This includes legacy functionality such as:

- Text/Image to Video
- Video Extend
- Lip Sync
- Advanced Lip Sync
- Avatar generation
- Effects
- Text to Speech / Audio
- Voice tools
- Virtual Try-On
- Image Extend
- Upscale
- loaders/savers
- utility nodes

Old saved workflows therefore do not need to be converted just because the fork gained new Kling generations.

---

# Architecture

```text
ComfyUI-Kling-Direct
|
+-- Original Kling-Direct
|   +-- legacy Access Key + Secret Key workflows
|
+-- Kling 3.0 Full REST
|   +-- single KLING_API_KEY
|   +-- Video 3.0 / Turbo / Omni
|   +-- Image 3.0 REST
|
+-- Kling Image REST
|   +-- kling-v3-omni
|   +-- kling-image-o1
|
+-- Official Kling CLI / MCP
    +-- OAuth login
    +-- live who_am_i capability discovery
    |
    +-- Kling Video dynamic layer (recommended)
    |   +-- live model discovery
    |   +-- Text to Video
    |   +-- Image to Video / dynamic keyframes
    |   +-- Omni / Video Edit
    |   +-- Motion Control
    |   +-- live arguments + combinedItemCap
    |
    +-- Kling 4.0 compatibility nodes
    |   +-- retained for existing workflows
    |
    +-- Kling Image
        +-- dynamic Text to Image
        +-- dynamic Image to Image
        +-- future server-advertised image models
```

---

# Safety around paid requests

The newer CLI nodes are intentionally conservative.

Before submission they validate as much as possible from the live model declaration. Invalid model names and unsupported explicitly requested parameters fail before generation.

The code also:

- invokes the CLI without a shell;
- tokenizes advanced arguments rather than shell-evaluating them;
- does not expose OAuth credentials;
- does not log the REST Authorization header;
- reuses the upstream URL/download security checks.

A successful model/schema check does not guarantee a generation will succeed; content moderation, account balance, transient service errors and other server-side conditions can still reject a task.

---

# Testing

Run:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Kling-Direct
python -m pytest tests/
```

Important test files added by this fork include:

```text
tests/test_kling3_full.py
tests/test_kling4_cli.py
tests/test_kling_video_dynamic.py
tests/test_kling_image_modern.py
```

The modern integration tests mock task submission and do not intentionally spend Kling credits.

---

# Troubleshooting

## Dynamic Video/Image CLI nodes say the CLI is missing or too old

```bash
npm i -g @klingai/cli-global@latest --registry=https://registry.npmjs.org
kling --version
```

Use **Kling CLI 0.2.3 or newer** for the v4.2.1 dynamic nodes.

## CLI exists but the node says authentication failed

```bash
kling login
kling who_am_i
```

Run these as the same Linux user that runs ComfyUI.

## A model is not available

Check:

```bash
kling who_am_i
```

Kling may roll out models per account/region/product. The ComfyUI nodes do not bypass those entitlements.

## Kling 3.0 REST authentication fails

Verify the single API key is available to the ComfyUI process:

```bash
export KLING_API_KEY='YOUR_KLING_API_KEY'
```

---

# Credits

Original project:

https://github.com/IxMxAMAR/ComfyUI-Kling-Direct

This fork and the Kling 3.0 / 4.0 / modern Image integration:

https://github.com/maglat/ComfyUI-Kling-Direct

---

# License

MIT - see `LICENSE`.
