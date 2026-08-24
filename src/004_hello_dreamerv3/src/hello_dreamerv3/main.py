import dreamerv3.main

from hello_dreamerv3.myenv import MyEnv


def main() -> None:
    def custom_make_env(config, index, **overrides):
        env = MyEnv()
        return dreamerv3.main.wrap_env(env, config)

    # Monkey patch make_env inside the dreamerv3.main module
    dreamerv3.main.make_env = custom_make_env

    # Run the main dreamerv3 routine
    dreamerv3.main.main()
