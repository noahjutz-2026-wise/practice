# Monkey-Patched dreamerv3.main:main in order to pass configs.yaml and env.py from outside

from pathlib import Path

import dreamerv3.main
import elements
import yaml

from hello_dreamerv3.env import MyEnv
from hello_dreamerv3.env_gym import MyGymEnv
from hello_dreamerv3.from_gymnasium import FromGymnasium


def custom_make_env(cfg, idx, **ow):
    return dreamerv3.main.wrap_env(FromGymnasium(env=MyGymEnv()), cfg)


dreamerv3.main.make_env = custom_make_env

_original_read = elements.Path.read


def _deep_merge(base, overrides):
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def custom_read(self, *args, **kw):
    if self.name == "configs.yaml":
        original = _original_read(self, *args, **kw)
        my_config_path = Path(__file__).parent / "my_config.yaml"
        overrides = yaml.safe_load(my_config_path.read_text()) or {}
        docs = list(yaml.safe_load_all(original))
        merged_docs = [_deep_merge(doc, overrides) for doc in docs]
        return "\n".join(yaml.dump(doc) for doc in merged_docs)
    return _original_read(self, *args, **kw)


elements.Path.read = custom_read


def main():
    dreamerv3.main.main()
