# Hello Dreamerv3

Wrapper around Dreamerv3 that allows for quick configuration without modifying source code.

## Get Started

1. Create two files to define your environment:

- `env.py`: Must only contain a subclass of `embodied.Env`. Keep `step` empty, it is implemented by this project.
- `configs.yaml`: Must be structured according to DreamerV3 `configs.yaml`.

2. Invoke the wrapper

```bash
uv run dreamer --env env.py --configs configs.yaml -- --jax.platform cpu
```

Extra dreamerv3 flags (like `--jax.platform`) are passed after `--`.

3. In your environment, create a portal client and listen to steps.

```py
import portal

client = portal.Client('localhost:2222')
client.connect()
while True:
    action = client.step(obs).result()
    obs = compute_obs(action)
```

## Methodology

- _Installation_
    - Containerfile has aged and is unusable, therefore implemented my own
    - In order to prevent code from aging further, added lockfile with uv
- _Architecture_
    - Isolated dreamer completely in container

## Containerfile

Abandoned for now
