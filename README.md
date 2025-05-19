# Mini Clash Royale RL Environment

A lightweight, 1-lane toy implementation of Clash Royale built as an OpenAI Gym environment, designed for reinforcement learning experimentation. This repository includes:

* **`envs/mini_clash.py`**: Custom `MiniClashEnv` Gym environment with:

  * 1-lane battlefield with two towers.
  * Discrete actions: spawn one of three unit types or pass.
  * Elixir resource management for both agent and opponent.
  * Collision and combat resolution.
  * Reward shaping and termination handling.

* **`scripts/run_env.py`**: Quick script to validate the environment:

  * Runs a random agent baseline.
  * Runs a simple scripted-policy baseline.

* **`notebooks/`**: Jupyter notebooks for prototyping, logging, and visualizing results.

* **`agents/`**: (Future) DQN and other RL agent implementations.

## 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SanketRt/mini-clash-royale.git
   cd mini-clash-royale
   ```

2. **Create and activate a Python virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

Minimum required packages:

```text
gym
numpy
matplotlib  
torch         
```

## 🚀 Usage

### 1. Test the environment

Run the baseline script to ensure everything is set up correctly:

```bash
python3 scripts/run_env.py
```

You should see average rewards for both a random agent and a simple scripted agent.

### 2. Integrate your own agent

In `scripts/train_dqn.py` , import and use `MiniClashEnv`:

```python
from envs.mini_clash import MiniClashEnv

env = MiniClashEnv()
# Your RL training loop here
i     s = env.reset()
# ...
```

## 🛠️ Environment Details

* **State (observation)**: 1D `numpy.int32` array of length `4 + lane_length`:
  `[your_hp, opp_hp, your_elixir, opp_elixir, lane[0], ..., lane[n-1]]`

  * Tower HP ranges from `0` to `tower_hp`.
  * Elixir ranges from `0` to `max_elixir`.
  * Lane cells codes: `0 = empty`, `1/2/3 = your units A/B/C`, `4/5/6 = opponent units A/B/C`.

* **Action space**: `Discrete(4)`:

  * `0`: Pass
  * `1/2/3`: Spawn unit types A/B/C at your tower (cost 2/4/6 elixir)

* **Reward**:

  * `+1.0` for damaging opponent tower.
  * `-0.1` for your tower taking damage.
  * `+0.1` per enemy unit killed, `-0.1` per your unit killed.

* **Episode termination**:

  * `terminated` = when either tower HP ≤ 0.
  * `truncated`  = when step count ≥ `max_steps`.

## 📁 Project Structure

```
mini-clash-royale/
├── envs/                
│   └── mini_clash.py
├── scripts/             
│   └── run_env.py
├── notebooks/          
├── agents/              
├── requirements.txt    
├── README.md            
└── .gitignore
```


Refer to the project roadmap in the main proposal for detailed deadlines.

---

*Happy coding & RL-ing!*
