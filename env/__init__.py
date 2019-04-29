from gym.envs.registration import register


register(
    id='MyCartpole-v0',
    entry_point='env.cartpole:CartpoleEnv'
)

register(
    id='MyHalfCheetah-v0',
    entry_point='env.half_cheetah:HalfCheetahEnv'
)

register(
    id='MySwimmer-v0',
    entry_point='env.swimmer:SwimmerEnv'
)

register(
    id='MyReacher3D-v0',
    entry_point='env.reacher3D:Reacher3DEnv'
)

register(
    id='MyPusher-v0',
    entry_point='env.pusher:PusherEnv'
)