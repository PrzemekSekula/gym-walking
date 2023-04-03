import argparse


import gym
import gym_walking
import numpy as np

parser = argparse.ArgumentParser(description='Runs Monte Carlo state value estimate on Walking5-v0.')

parser.add_argument('-e', '--episodes',
                    type=int,
                    default=10,
                    help='Number of episodes. Default: 10'
                   )

parser.add_argument('-g', '--gamma',
                    type=float,
                    default=1.0,
                    help='Gamma parameter. Update is based on gamma * (R - V). Default: 1.0'
                   )

parser.add_argument('-r', '--render',
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True,
                    help='Whether to render or not. Default: True'
                   )


env = gym.make('Walking5-v0')
pi = lambda x: np.random.randint(2) # uniform random policy

def mc(pi, env, gamma=1.0, n_episodes=10, render=True):
    V = np.array([0] + [0.5] * (env.observation_space.n - 2) + [0])
    n_visited = np.array([0] * env.observation_space.n)

    for t in range(n_episodes):
        state, info = env.reset()
        visited_states = [state]
        rewards = [0]
        
        if render:
            env.render(V)
        terminal = False
        while not terminal:
            action = pi(state)
            next_state, reward, terminal, truncated, info = env.step(action)
            state = next_state
            visited_states.append(state)
            if render:
                env.render(V)
                
        for state in visited_states[::-1]:
            n_visited[state] += 1
            print (V[state], reward, n_visited[state])
            V[state] += gamma * (reward - V[state]) / n_visited[state]
            
    return V

if __name__ == "__main__":
    args = parser.parse_args()
    V = mc(
        pi, env, 
        gamma=args.gamma, 
        n_episodes=args.episodes,
        render=args.render
        )
    print ('Final state values: {}'.format(V))