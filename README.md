# gym-walking
OpenAI Gym walking environment - for learning purposes. This environment contains pygame-based rendering.


## Content
- Environment
- `mc.py` - example. Learns state value functions using Monte Carlo method and random uniform policy.
- `td0.py` - example. Learns state value functions using TD(0) method and random uniform policy.

## Installation:

```bash
git clone https://github.com/PrzemekSekula/gym-walking.git
cd gym-walk
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

## Registered environments
- `Walking5-v0` 5 non-terminal (7 total) states
- `Walking7-v0` 7 non-terminal (9 total) states
- `Walking9-v0` 9 non-terminal (11 total) states

## Additional info
`env.render()` works as usual, but you may also use additional arguments to make rendering look more informative. `env.render(state_values)` will also display state values on each non-terminal state. See `mc.py` or `td0.py` examples.

