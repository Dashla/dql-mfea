from gym.envs.registration import registry, register, make, spec

# ----------------------------------------
#                                 CARTPOLE
register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control.cartpole:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CartPole-v1',
    entry_point='gym.envs.classic_control.cartpole:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control.cartpole:CartPoleEnv3',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CartPole-v3',
    entry_point='gym.envs.classic_control.cartpole:CartPoleEnv4',
    max_episode_steps=500,
    reward_threshold=475.0,
)


# ----------------------------------------
#                                  ACROBOT

register(
    id='Acrobot-v0',
    entry_point='gym.envs.classic_control.acrobot:AcrobotEnv',
    reward_threshold=-100.0,
    max_episode_steps=500,
)

register(
    id='Acrobot-v1',
    entry_point='gym.envs.classic_control.acrobot:AcrobotEnv2',
    reward_threshold=-100.0,
    max_episode_steps=500,
)

register(
    id='Acrobot-v2',
    entry_point='gym.envs.classic_control.acrobot:AcrobotEnv3',
    reward_threshold=-100.0,
    max_episode_steps=500,
)

register(
    id='Acrobot-v3',
    entry_point='gym.envs.classic_control.acrobot:AcrobotEnv4',
    reward_threshold=-100.0,
    max_episode_steps=500,
)


# ----------------------------------------
#                                 PENDULUM

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control.pendulum:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='Pendulum-v1',
    entry_point='gym.envs.classic_control.pendulum:PendulumEnv2',
    max_episode_steps=200,
)

register(
    id='Pendulum-v2',
    entry_point='gym.envs.classic_control.pendulum:PendulumEnv3',
    max_episode_steps=200,
)

register(
    id='Pendulum-v3',
    entry_point='gym.envs.classic_control.pendulum:PendulumEnv4',
    max_episode_steps=200,
)
