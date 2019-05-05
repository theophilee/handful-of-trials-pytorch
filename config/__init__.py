def get_config(env):
    if env == "cartpole":
        from .cartpole import Config
    elif env == "half_cheetah":
        from .half_cheetah import Config
    elif env == "hopper":
        from .hopper import Config
    elif env == "swimmer":
        from .swimmer import Config
    else:
        raise NotImplementedError

    return Config().get_config()