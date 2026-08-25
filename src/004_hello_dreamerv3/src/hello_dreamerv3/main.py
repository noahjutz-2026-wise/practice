import dreamerv3.main
import elements

from hello_dreamerv3.myenv import MyEnv


def custom_make_env(cfg, idx, **ow):
    return dreamerv3.main.wrap_env(MyEnv(), cfg)


dreamerv3.main.make_env = custom_make_env


def custom_read(self, *args, **kw):
    if self.name == "configs.yaml":
        return open("/config/configs.yaml").read()
    return elements.Path.read(self, *args, **kw)


elements.Path.read = custom_read
main = dreamerv3.main.main
