# Voice Studio GPU Worker

Serverless AMD64/CUDA worker for **VPS 3 Voice Studio**.

## Engine

This image contains **IndicF5 only**.

Supported languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu.

Chatterbox has been removed from this image and from the worker API contract.

## VPS3 contract

- `GET /healthz`
- `GET /readyz`
- `POST /v1/clone/generate`
- optional bearer authentication via `WORKER_TOKEN`

The generation endpoint accepts only `engine: "indicf5"`.

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

## Reproducible upstream pin

- IndicF5: `13f7c4d627cc10111aea8fe9c0039462cacacdc7`

## Image

GitHub Actions publishes this branch separately as:

`ghcr.io/dilshadshahsyed883-cmd/voice-studio-gpu-worker:indicf5-only`

and an immutable commit-specific tag:

`ghcr.io/dilshadshahsyed883-cmd/voice-studio-gpu-worker:indicf5-only-<git-sha>`

The existing `:canary` image is intentionally left untouched for rollback.

Do not put Salad, Drive, Hugging Face, or worker secrets in this repository.
