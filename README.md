# ComfyUI-Kling-Direct

Direct Kling AI integration for ComfyUI with **Kling 4.0**, **Kling 3.0 Full**, and the original Kling-Direct legacy nodes.

This fork keeps the original legacy nodes, adds direct Kling 3.0 REST support, and adds Kling 4.0 through Kling's official OAuth CLI/MCP transport.


## Kling 4.0 support

Kling 4.0 support in this fork uses Kling's official CLI/MCP transport rather than guessed REST endpoints. As of 2026-10-15, the public paid REST developer catalog still labels Kling 4.0 as Preview/Coming Soon, while the official CLI can access models enabled for the signed-in Kling account.

### Kling 4.0 nodes

- **Kling 4.0 - CLI Status & Capabilities**
- **Kling 4.0 - Task Status**
- **Kling 4.0 - Text to Video (Official CLI)**
- **Kling 4.0 - Image to Video + Keyframes (Official CLI)**
- **Kling 4.0 - Omni Reference (Official CLI)**

The nodes call kling who_am_i before paid submissions. The live server declaration is used to verify that the selected model and requested parameter/input names are actually available for the current account.

Full Kling 4.0 model ID: kling-video-v4_0

Flash model ID: kling-video-v4_0_flash

The ComfyUI implementation supports the published Kling 4.0 workflow envelope: 3-30 second Standard generation, 3-20 second Flash validation, Standard keyframe workflows with up to 10 total frame images in the node, Omni references, up to 10 reference images, up to 5 reference videos, Elements, optional audio reference, and a combined reference limit of 15 assets. Resolution and audio controls are sent only when the live model schema exposes those parameters.

Flash is locally restricted to 720p and rejects tail/end-frame and multi-keyframe inputs before submission.

### Kling 4.0 CLI setup

The official Kling CLI requires Node.js 18 or newer.

    node --version
    npm i -g @klingai/cli-global --registry=https://registry.npmjs.org
    kling --version
    kling login
    kling who_am_i

Kling CLI authentication is OAuth-only. Complete the login flow on the ComfyUI server account. This project does not request or store the CLI OAuth token.

After login, run **Kling 4.0 - CLI Status & Capabilities** in ComfyUI. It reports which Kling 4 models are currently advertised by the account without submitting a generation.

### Kling 4.0 request safety

Before spending credits, the nodes check the live model schema and local capability limits. They stop before submission when the model is unavailable, a requested keyframe slot is absent, Flash receives an unsupported option, or a requested resolution/audio parameter is not exposed by the server.

The CLI is invoked without a shell. Advanced CLI arguments are tokenized as argv and cannot override the node's model, poll, or quiet transport controls.


## Highlights

### Kling 3.0 Full video
- Text to Video
- Image to Video
- First + Last Frame
- 3–15 second duration
- 720p / 1080p / 4K on Kling 3.0 T2V/I2V
- Native audio on/off
- Multi-shot
- Kling 3.0 Turbo support
- Kling 3.0 Omni Video
- Motion Control
- Unified new-standard `/tasks` polling

### Kling Image 3.0
- `kling-v3` image generation
- 1K / 2K
- `kling-v3-omni`
- 1K / 2K / 4K
- Multiple local reference images
- Single and series generation

### Authentication
Kling 3.0 Full uses the current **single Kling API Key**:

```http
Authorization: Bearer <KLING_API_KEY>
```

The recommended setup is an environment variable:

```bash
export KLING_API_KEY='YOUR_KLING_API_KEY'
```

Then leave the API-key field empty in the **Kling 3.0 Full • API Key** node.

The original legacy Kling-Direct nodes are still present for compatibility with older workflows and their legacy Access Key + Secret Key authentication.

## Install

Clone this fork into ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maglat/ComfyUI-Kling-Direct.git
```

Then restart ComfyUI.

If you already cloned another Kling-Direct repo:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Kling-Direct
git remote set-url origin https://github.com/maglat/ComfyUI-Kling-Direct.git
git pull
```

## Kling 3.0 Full nodes

### Config
- **Kling 3.0 Full • API Key**
- **Kling 3.0 Full • Health Check**
- **Kling 3.0 Full • Task Status**

### Prompt
- **Kling 3.0 Full • Storyboard Builder**

The Storyboard Builder supports up to six shots and produces a prompt plus total duration.

### Video
- **Kling 3.0 Full • Text to Video**
- **Kling 3.0 Full • Image to Video**
- **Kling 3.0 Full • Omni Video**
- **Kling 3.0 Full • Motion Control**

### Image
- **Kling Image 3.0 Full • Generation**
- **Kling Image 3.0 Full • Omni**

## Model differences

### kling-3.0
Recommended default for full feature access.

Supported by these nodes:
- 3–15 seconds
- 720p / 1080p / 4K for Text-to-Video and Image-to-Video
- Native audio / audio off
- Multi-shot
- First + Last Frame for I2V
- Element references
- Motion Control

### kling-3.0-turbo
Turbo is available for Text-to-Video and Image-to-Video, but has different API constraints:
- 720p / 1080p only
- Native audio is always active
- No explicit `settings.audio`
- No explicit `settings.multi_shot`
- Multi-shot is driven by shot syntax in the prompt
- No Last Frame / element references in the Full I2V node

The Full nodes validate these differences before submitting a paid task.

### kling-3.0-omni
Omni Video supports:
- First Frame
- Last Frame
- Reference images
- Feature-video URL
- Base-video URL
- Elements
- 720p / 1080p / 4K
- Native / original / off audio
- Multi-shot

Reference videos are URL inputs. Local images are sent directly as Base64.

## API architecture

Kling's current video API uses path-per-model endpoints such as:

```text
POST /text-to-video/kling-3.0
POST /image-to-video/kling-3.0
POST /omni-video/kling-3.0-omni
POST /motion-control/kling-3.0
GET  /tasks?task_ids=...
```

The current image API remains on the `/v1/images/...` family and uses the same single Bearer API key.

## Existing legacy nodes

All original Kling-Direct functionality remains available, including:
- Legacy Text/Image to Video
- Video Extend
- Lip Sync
- Avatar
- Effects
- TTS / Audio
- Virtual Try-On
- Image Extend
- Upscale
- Utility/configuration nodes

This keeps old workflows loadable while new workflows can use the Kling 3.0 Full nodes.

## Security

- API keys are password-masked in ComfyUI.
- Prefer `KLING_API_KEY` rather than serializing a key into workflow JSON.
- The Full client does not log the Authorization header.
- Existing Kling-Direct download/SSRF protections remain in use for media downloads.

## Tests

```bash
python -m pytest tests/
```

All tests are intended to be mock/local tests. They must not submit paid Kling generation requests.

## Credits

Original project:
- https://github.com/IxMxAMAR/ComfyUI-Kling-Direct

Kling 3.0 Full integration maintained in this fork:
- https://github.com/maglat/ComfyUI-Kling-Direct

## License

MIT — see `LICENSE`.
