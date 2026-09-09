# Ray RLLib + DreamerV3

## Quickstart

```bash
podman-compose up -d --build
uv run ray job submit --working-dir . --address="http://127.0.0.1:8265" -- uv run trl

# Local (unsupported)
# uv run --env-file .env --active trl
```
