def get_config(env):
    if env == "cartpole":
        from .cartpole import Config
    elif env == "halfcheetah":
        from .half_cheetah import Config
    elif env == "reacher3D":
        from .reacher3D import Config
    elif env == "pusher":
        from .pusher import Config
    elif env == "hopper":
        from .hopper import Config
    else:
        raise NotImplementedError

    return Config().get_config()