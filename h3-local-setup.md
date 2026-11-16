# H3 Local GPU Integration via Custom Provider

Connect ArcReel to a local H3 GPU instance via the existing
`minimax-h3` declarative endpoint definition.

## Prerequisites

1. AMD host running `h3 serve` (port 8321):
   ```bash
   ssh amd 'nohup ~/.local/bin/h3 serve > /tmp/h3serve.log 2>&1 &'
   ```

2. Verify the API is reachable:
   ```bash
   curl http://192.168.1.123:8321/health
   # → {"status":"ok"}
   ```

## ArcReel Configuration

In ArcReel's Custom Provider UI:

- **Name**: H3 Local GPU
- **Base URL**: `http://192.168.1.123:8321`
- **API Key**: (leave blank — local network, no auth)
- **Endpoint Definition**: Select built-in `minimax-h3`
- **Model**: `MiniMax-H3`
- **Concurrency**: `1` (single GPU)
- **Poll Timeout**: `3600` seconds (balanced preset takes ~21 min)

## Limitations

- Text-to-video only (first_frame requires upload support in h3 serve)
- 5s clips at 768p (1344x768), 24 fps with audio
- One job at a time (FIFO queue)
- 16:9 aspect ratio only (other ratios unvalidated)

## API Protocol

The shim speaks the MiniMax H3 v2 API:

```
POST /v2/video_generation
  {"model":"MiniMax-H3","content":[{"type":"text","text":"..."}]}
  → {"task_id":"h3-000001"}

GET /v2/query/video_generation/h3-000001
  → {"task":{"status":"running","task_id":"h3-000001"}}
  → {"task":{"status":"succeeded","task_id":"h3-000001","content":{"url":"http://..."}}}

GET /runs/{work}/output.mp4
  → video/mp4 file
```
