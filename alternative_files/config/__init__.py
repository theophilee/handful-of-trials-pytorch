def get_config(env):
    if env == "cartpole":
        from .cartpole import Config

    elif env == "halfcheetah":
        from .halfcheetah import Config

    elif env == "reacher":
        from .reacher import Config

    elif env == "pusher":
        from .pusher import Config

    else:
        raise NotImplementedError

    return Config().get_config()