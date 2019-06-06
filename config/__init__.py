def get_config(env):
    if env == "ant":
        from .ant import Config
    elif env == "cartpole":
        from .cartpole import Config
    elif env == "half_cheetah":
        from .half_cheetah import Config
    elif env == "humanoid":
        from .humanoid import Config
    elif env == "inverted_pendulum":
        from .inverted_pendulum import Config
    elif env == "pusher":
        from .pusher import Config
    elif env == "swimmer":
        from .swimmer import Config
    elif env == "walker2d":
        from .walker2d import Config
    else:
        raise NotImplementedError

    return Config().get_config()