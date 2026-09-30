# ComfyUI-Kling-Direct

Direct Kling AI integration for ComfyUI with **Kling 3.0 Full** support.

This fork keeps the original legacy nodes from IxMxAMAR/ComfyUI-Kling-Direct and adds a current API-key-first Kling 3.0 implementation for the 2026 Kling API.

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
