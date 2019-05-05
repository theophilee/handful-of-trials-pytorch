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
    id='MyHopper-v2',
    entry_point='env.hopper:HopperEnv',
    max_episode_steps=1000,
)