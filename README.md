# gym-walking Environment


![Alt text](./img/gym-walking.jpg?raw=true "Gym Walking")

## Description

OpenAI Gym walking environment - for learning purposes. This environment contains pygame-based rendering.

## Actions
- 0: Left
- 1: Right

## Observations
State (position of the robot). Integer number. The left-most (terminal) state is 0, the right-most is len(states) - 1.

## Reward
+1 in the right terminal state, otherwise 0.

## Arguments
- `env = gym.make(Walking5-v0)` - 5 non-terminal (7 total) states
- `env = gym.make(Walking7-v0)` - 7 non-terminal (9 total) states
- `env = gym.make(Walking9-v0)` - 9 non-terminal (11 total) states

## Rendering
`env.render()` works as usual, but you may also use additional arguments to make rendering look more informative. `env.render(state_values)` will display state values on each non-terminal state. See `mc.py` or `td0.py` examples.

## Repository content 
- `gym_walking` - Environment
- `mc.py` - example. Learns state value functions using Monte Carlo method and random uniform policy.
- `td0.py` - example. Learns state value functions using TD(0) method and random uniform policy.

## Installation:

```bash
git clone https://github.com/PrzemekSekula/gym-walking.git
cd gym-walking
pip install .
```

or:

```bash
pip install git+https://github.com/PrzemekSekula/gym-walking.git
```

## Usage:
From gym-walking folder:

```bash
# Run with defaults
python td0.py

# Run for 20 episodes
python td0.py -e 20

# Run for 100 episodes, change gamma and alpha values, no rendering
python td0.py -e 100 --gamma 0.9 --alpha 0.4 --render False

# Help:
python td0.py --help
```


