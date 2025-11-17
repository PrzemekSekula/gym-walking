from gymnasium.envs.registration import register


# classics
register(
    id='Walking5-v0',
    entry_point='gym_walking.envs:WalkingEnv',
    kwargs={'n_states': 7},
    max_episode_steps=500,
)

register(
    id='Walking7-v0',
    entry_point='gym_walking.envs:WalkingEnv',
    kwargs={'n_states': 9},
    max_episode_steps=1000,
)

register(
    id='Walking9-v0',
    entry_point='gym_walking.envs:WalkingEnv',
    kwargs={'n_states': 11},
    max_episode_steps=1000,
)