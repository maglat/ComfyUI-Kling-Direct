# ComfyUI-Kling-Direct

A direct Kling AI integration for ComfyUI with:

- **Kling 4.0 video** through Kling's official OAuth CLI/MCP transport
- **Kling Image 3.0 / Image 3.0 Omni / Image O1**
- **dynamic image-model discovery** for future Kling image releases
- **Kling 3.0 Full REST** support using the current single API key
- the original Kling-Direct legacy nodes for existing workflows

Current fork version: **4.1.0**

Repository: https://github.com/maglat/ComfyUI-Kling-Direct

---

## Current model status

As of **2026-10-15** the public Kling product/API split is:

| Area | Model | This fork |
|---|---|---|
| Video | Kling 4.0 | Yes - official CLI/MCP |
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
| Kling 4.0 video | official Kling CLI OAuth |
| Modern Kling Image CLI | official Kling CLI OAuth |
| Kling 3.0 Full REST | single `KLING_API_KEY` |
| Original legacy nodes | Access Key + Secret Key |

## Official Kling CLI / MCP

Required for Kling 4.0 and the dynamic modern Image nodes.

Node.js 18+ is required.

```bash
node --version

npm i -g @klingai/cli-global --registry=https://registry.npmjs.org

kling --version
kling login
kling who_am_i
```

The CLI uses Kling's official OAuth flow. This custom-node project does **not** request, store or print the OAuth token.

The most important diagnostic command is:

```bash
kling who_am_i
```

It returns the models and arguments currently enabled for the signed-in Kling account.

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

# Kling 4.0 Video

Kling 4.0 support uses the **official Kling CLI/MCP transport**.

The public paid REST developer interface had not yet published a final Kling 4.0 REST contract when this integration was implemented, so this fork does not guess undocumented endpoints.

## Nodes

- **Kling 4.0 - CLI Status & Capabilities**
- **Kling 4.0 - Task Status**
- **Kling 4.0 - Text to Video (Official CLI)**
- **Kling 4.0 - Image to Video + Keyframes (Official CLI)**
- **Kling 4.0 - Omni Reference (Official CLI)**

## Model IDs

```text
kling-video-v4_0
kling-video-v4_0_flash
```

The selected model must also be advertised by `kling who_am_i` for the exact generation command before a paid task is submitted.

## Full Kling 4.0

The ComfyUI nodes handle the current published workflow envelope including:

- Text to Video
- Image to Video
- 3-30 second duration
- start/end-frame workflows
- multiple keyframes
- up to 10 total frame images in the I2V node
- 21:9 / 16:9 / 1:1 / 9:16 / automatic aspect handling when declared
- 720p / 1080p / 4K selection when declared by the live account schema
- audio enable/disable when declared
- Omni Reference
- up to 10 reference images
- up to 5 reference videos
- Element references
- optional audio reference
- combined Omni reference validation up to 15 assets
- result URL, local output file, video tensor, audio tensor and raw metadata

## Kling 4.0 Flash

The node validates known Flash limits before a paid request:

- 3-20 seconds
- 720p only
- no tail/end frame
- no multi-keyframe I2V

Any additional server-side restrictions are still verified from the live model schema.

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
- advanced CLI arguments

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
    +-- Kling 4.0 Video
    |   +-- Standard
    |   +-- Flash
    |   +-- Keyframes
    |   +-- Omni Reference
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
tests/test_kling_image_modern.py
```

The modern integration tests mock task submission and do not intentionally spend Kling credits.

---

# Troubleshooting

## Kling 4.0 or Image CLI nodes say the CLI is missing

```bash
npm i -g @klingai/cli-global --registry=https://registry.npmjs.org
kling --version
```

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
