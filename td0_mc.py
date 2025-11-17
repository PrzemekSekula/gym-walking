"""TD 0 and Monte Carlo implementation for Walking5-v0.
Author: Przemek Sekula
Parameters:
    -e, --episodes (int) Number of episodes. Default: 10
    -g, --gamma (float): Gamma parameter for td target formula. Default: 1.0
    -a, --alpha (float): Alpha parameter for td target formula. Default: 0.05
    -r, --render (bool): Whether to render or not. Default: True
Note - the main purpose of this script is to show the difference between TD(0) 
and Monte Carlo methods.
"""

import argparse
import gymnasium as gym
import numpy as np
import gym_walking

parser = argparse.ArgumentParser(description='Runs TD(0) and Monte Carlo on Walking5-v0.')

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

parser.add_argument('-s', '--size',
                    type=int,
                    default=5,
                    help='Size of the environment (number of non-terminal '
                    'states). Default: 5. Possible values: 5, 7, 9'
                    )



def pi(state): 
    """Uniform random policy
    Args:
        state (any): state of the environment. Useful in a general case,
        here not used.
    Returns:
        int: Action to take (0 or 1)
    """
    return np.random.randint(2)  # uniform random policy


def td0_mc(pi, size, gamma=1.0, alpha=0.05, n_episodes=10, render=True):
    if size == 5:
        env = gym.make('Walking5-v0')
    elif size == 7:
        env = gym.make('Walking7-v0')
    else:
        env = gym.make('Walking9-v0')

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
    V_td0 = np.array([0] + [0.5] * (env.observation_space.n - 2) + [0])
    V_mc = np.array([0] + [0.5] * (env.observation_space.n - 2) + [0])

    n_visited = np.array([1] * env.observation_space.n)    
    for t in range(n_episodes):
        visited_states = []
        rewards = []
        state, info = env.reset()
        n_visited[state] += 1
        if render:
            env.unwrapped.render(V_td0, V_mc)
        terminal = False
        while not terminal:
            visited_states.append(state)
            action = pi(state)
            next_state, reward, terminal, truncated, info = env.step(action)
            rewards.append(reward)

            # V(terminal state) is zero by definition
            td_target = reward + gamma * (V_td0[next_state] if not terminal else 0)
            td_error = td_target - V_td0[state]
            V_td0[state] += alpha * td_error
            state = next_state
            if render:
                env.unwrapped.render(V_td0, V_mc)

        # Now the episode is done. Let's update the state values        
        G = 0
        for state, reward in zip (visited_states[::-1], rewards[::-1]):
            n_visited[state] += 1
            G = gamma * G + reward
            V_mc[state] += (G - V_mc[state]) / n_visited[state]

    return V_td0, V_mc


if __name__ == "__main__":
    args = parser.parse_args()
    V_td0, V_mc = td0_mc(
        pi, args.size,
        gamma=args.gamma,
        alpha=args.alpha,
        n_episodes=args.episodes,
        render=args.render
    )
    real_values = np.round(np.array([x / (args.size+1) for x in range(args.size+2)]), 3)
    real_values[len(real_values) - 1] = 0
    print(f'Real state values:                  {real_values}')
    print ('')
    print(f'Final state values for TD(0):       {np.round(V_td0, 3)}')
    print(f'Final state values for Monte Carlo: {np.round(V_mc, 3)}')
    print ('')
    print(f'TD(0) and real values difference:   {np.round(V_td0 - real_values, 3)}')
    print(f'MC and real values difference:      {np.round(V_mc - real_values, 3)}')
    print ('')
    print (f'TD(0) and real values avg diff.:   ' \
        f'{np.sum(np.abs(V_td0 - real_values))/(len(real_values)-2):.3f}')
    print (f'MC and real values avg diff.:      ' \
        f'{np.sum(np.abs(V_mc - real_values))/(len(real_values)-2):.3f}')