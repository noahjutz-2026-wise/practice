# Hello Dreamerv3

Wrapper around Dreamerv3 that allows for quick configuration without modifying source code.

## Get Started

1. Create two files to define your environment:

- `env.py`: Must only contain a subclass of `embodied.Env`. Keep `step` empty, it is implemented by this project.
- `configs.yaml`: Must be structured according to DreamerV3 `configs.yaml`.

2. Invoke the wrapper

```bash
uv run dreamer --env env.py --configs configs.yaml
```

## Methodology

- _Installation_
    - Containerfile has aged and is unusable, therefore implemented my own
    - In order to prevent code from aging further, added lockfile with uv
- _Architecture_
    - Isolated dreamer completely in container

## Containerfile

Abandoned for now
