# Voice Studio GPU Worker

Serverless AMD64/CUDA worker for **VPS 3 Voice Studio**.

## Engines

- **Chatterbox Multilingual V3**: Arabic, English, Hindi and the official Chatterbox multilingual language set.
- **IndicF5**: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu.

The two engines intentionally run in **separate Python virtual environments** because their current upstream Transformer requirements conflict: Chatterbox pins Transformers 5.2.0 while IndicF5 requires `<4.50`. A lightweight gateway keeps one model process active at a time so a 24 GB RTX 3090 is not forced to keep both models resident simultaneously.

## VPS3 contract

- `GET /healthz`
- `GET /readyz`
- `POST /v1/clone/generate`
- optional bearer authentication via `WORKER_TOKEN`

The request contract matches the existing VPS3 `voice_clone_dispatcher.py`.

## Canary target

- linux/amd64
- RTX 3090 24 GB
- High priority
- 8 vCPU
- 16 GB host RAM
- 25 GB ephemeral disk
- 2 GB shared memory
- 1 replica for canary only

## Security boundary

The container does **not** contain Google Drive credentials or permanent Voice Studio state. Reference audio arrives per request, is stored only in temporary storage, and is deleted after generation.

## Reproducible upstream pins

- Chatterbox: `5de7a54aa4e5e2baadb0182dde554908b48b85c2`
- IndicF5: `13f7c4d627cc10111aea8fe9c0039462cacacdc7`

## Image

GitHub Actions publishes the canary image to:

`ghcr.io/dilshadshahsyed883-cmd/voice-studio-gpu-worker:canary`

Do not put Salad, Drive, Hugging Face, or worker secrets in this repository.
