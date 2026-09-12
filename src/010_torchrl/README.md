# Ray RLLib + DreamerV3

## Quickstart

```bash
podman-compose up -d --build
uv run ray job submit --working-dir . --address="http://127.0.0.1:8265" -- uv run tune # train
uv run infer # inference using checkpoint
```
