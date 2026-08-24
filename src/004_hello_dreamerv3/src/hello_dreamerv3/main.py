import importlib.util
import queue
import threading

import elements
import embodied
import portal
import typer

import dreamerv3.main

app = typer.Typer()
orig_read = elements.Path.read


def load_env_class(env_path: str):
    spec = importlib.util.spec_from_file_location("user_env", env_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for obj in vars(mod).values():
        if isinstance(obj, type) and issubclass(obj, embodied.Env) and obj is not embodied.Env:
            return obj
    raise ValueError(f"No embodied.Env subclass found in {env_path}")


class PortalEnv(embodied.Env):
    """Wraps a user env, forwarding step() over portal."""

    def __init__(self, env_class, port):
        self._inner = env_class()
        self._action_q = queue.Queue()
        self._obs_q = queue.Queue()
        self._server = portal.Server(port)
        self._server.bind("step", self._handle_step)
        threading.Thread(target=self._server.start, daemon=True).start()

    def _handle_step(self, obs):
        """Called by the remote client: sends obs, receives next action."""
        self._obs_q.put(obs)
        return self._action_q.get()

    @property
    def obs_space(self):
        return self._inner.obs_space

    @property
    def act_space(self):
        return self._inner.act_space

    def step(self, action):
        self._action_q.put(action)
        return self._obs_q.get()

    def close(self):
        self._server.close()
        self._inner.close()


@app.command()
def main(
    env: str = typer.Option(..., help="Path to env.py"),
    configs: str = typer.Option(..., help="Path to configs.yaml"),
    port: int = typer.Option(2222, help="Portal server port"),
    extra: list[str] = typer.Argument(None, help="Extra dreamerv3 flags"),
):
    env_class = load_env_class(env)

    def custom_make_env(cfg, idx, **ow):
        return dreamerv3.main.wrap_env(PortalEnv(env_class, port), cfg)

    dreamerv3.main.make_env = custom_make_env

    def custom_read(self, *args, **kw):
        if self.name == "configs.yaml":
            return open(configs).read()
        return orig_read(self, *args, **kw)

    elements.Path.read = custom_read
    dreamerv3.main.main(extra)
