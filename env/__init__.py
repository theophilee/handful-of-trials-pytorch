from gym.envs.registration import register


register(
    id='MyCartpole-v0',
    entry_point='env.cartpole:CartpoleEnv',
    max_episode_steps=200,
)

register(
    id='MyHalfCheetah-v2',
    entry_point='env.half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
)

register(
    id='MySwimmer-v2',
    entry_point='env.swimmer:SwimmerEnv',
    max_episode_steps=1000,
)

register(
    id='MyPusher-v2',
    entry_point='env.pusher:PusherEnv',
    max_episode_steps=100,
)

register(
    id='MyInvertedPendulum-v2',
    entry_point='env.inverted_pendulum:InvertedPendulumEnv',
    max_episode_steps=1000,
)

register(
    id='MyAnt-v2',
    entry_point='env.ant:AntEnv',
    max_episode_steps=1000,
)

register(
    id='MyWalker2d-v2',
    entry_point='env.walker2d:Walker2dEnv',
    max_episode_steps=1000,
)

register(
    id='MyHumanoid-v2',
    entry_point='env.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)