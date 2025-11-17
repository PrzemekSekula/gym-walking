"""TD 0 implementation for Walking5-v0.
Author: Przemek Sekula
Parameters:
    -e, --episodes (int) Number of episodes. Default: 10
    -g, --gamma (float): Gamma parameter for td target formula. Default: 1.0
    -a, --alpha (float): Alpha parameter for td target formula. Default: 0.05
    -r, --render (bool): Whether to render or not. Default: True
"""

import argparse
import gymnasium as gym
import numpy as np
import gym_walking

parser = argparse.ArgumentParser(description='Runs TD(0) on Walking5-v0.')

parser.add_argument('-e', '--episodes',
                    type=int,
                    default=10,
                    help='Number of episodes. Default: 10'
                    )

parser.add_argument('-g', '--gamma',
                    type=float,
                    default=1.0,
                    help='Gamma parameter for td target formula. Default: 1.0'
                    )

parser.add_argument('-a', '--alpha',
                    type=float,
                    default=0.05,
                    help='Alpha parameter for td target formula. Default: 0.05'
                    )

parser.add_argument('-r', '--render',
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True,
                    help='Whether to render or not. Default: True'
                    )


env = gym.make('Walking5-v0')

def pi(state): 
    """Uniform random policy
    Args:
        state (any): state of the environment. Useful in a general case,
        here not used.
    Returns:
        int: Action to take (0 or 1)
    """
    return np.random.randint(2)  # uniform random policy


def td_0(pi, env, gamma=1.0, alpha=0.05, n_episodes=10, render=True):
    """TD(0) algorithm
    Args:
        pi (func): policy to follow
        env (gym.Env): Gym environment
        gamma (float, optional): gamma parameter for td target formula.
            Defaults to 1.0.
        alpha (float, optional): alpha parameter for td target formula.
            Defaults to 0.05.
        n_episodes (int, optional): number of episodes. Defaults to 10.
        render (bool, optional): whether to render or not. Defaults to True.

    Returns:
        list: list of state-value function values.
    """
    V = np.array([0] + [0.5] * (env.observation_space.n - 2) + [0])
    for t in range(n_episodes):
        state, info = env.reset()
        if render:
            env.unwrapped.render(V)
        terminal = False
        while not terminal:
            action = pi(state)
            next_state, reward, terminal, truncated, info = env.step(action)

            # V(terminal state) is zero by definition
            td_target = reward + gamma * (V[next_state] if not terminal else 0)
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            state = next_state
            if render:
                env.unwrapped.render(V)

    return V


if __name__ == "__main__":
    args = parser.parse_args()
    V = td_0(
        pi, env,
        gamma=args.gamma,
        alpha=args.alpha,
        n_episodes=args.episodes,
        render=args.render
    )
    print('Final state values: {}'.format(V))
