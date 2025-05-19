import gym
from gym import spaces
import numpy as np

class MiniClashEnv(gym.Env):
    """
    A 1-lane Clash-Royale toy:
      - You vs scripted opponent  
      - Discrete time-steps  
      - Discrete actions: spawn A/B/C or pass
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, lane_length: int = 10, tower_hp: int = 10, elixir_regen: int = 1, max_elixir: int = 10, max_steps: int = 50):
        super().__init__()
        self.lane_length   = lane_length
        self.initial_tower_hp = tower_hp
        self.elixir_regen  = elixir_regen
        self.max_elixir    = max_elixir
        self.max_steps     = max_steps

        # Actions: 0 = pass, 1/2/3 = spawn your A/B/C
        self.action_space = spaces.Discrete(4)

        # Observation: [you_HP, opp_HP, you_elixir, opp_elixir] + lane cells
        # lane cells: 0=empty, 1/2/3=your units, 4/5/6=opp units
        obs_len = 4 + self.lane_length
        low  = np.zeros(obs_len, dtype=np.int32)
        high = np.concatenate([
            [tower_hp, tower_hp,  max_elixir, max_elixir], 
            np.full(self.lane_length, 6, dtype=np.int32)
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment.
        Follows Gym API: 
          - seed: for RNG 
          - options: unused
        Returns:
          obs, info
        """

        super().reset(seed=seed)
        self.your_hp      = self.initial_tower_hp
        self.opp_hp       = self.initial_tower_hp
        self.your_elixir  = self.max_elixir
        self.opp_elixir   = self.max_elixir
        self.lane         = np.zeros(self.lane_length, dtype=np.int32)
        self.step_count   = 0
        return self._get_obs(), {}

    def _get_obs(self):
        """Pack the full observation into a 1D array."""
        return np.concatenate([
            [self.your_hp, self.opp_hp, self.your_elixir, self.opp_elixir], self.lane
        ]).astype(np.int32)

    def step(self, action: int):
        """
        1) Your action (spawn if enough elixir)
        2) Opponent action (simple script + elixir)
        3) Move units + resolve combat
        4) Regen elixir, compute reward & done
        """
        reward = 0.0
        terminated = False
        truncated = False

        cost = {1: 2, 2: 4, 3: 6}.get(action, 0)
        if action != 0 and self.your_elixir >= cost and self.lane[0]==0:
            self.lane[0] = action
            self.your_elixir -= cost

        # if opp_elixir ≥ 4, spawn B (type 2)
        opp_cost = 4
        if self.opp_elixir >= opp_cost and (self.step_count % 2 == 0) and self.lane[-1]==0:
            opp_unit_code = 2 + 3  # 2→B plus +3→opponent side = code 5
            self.lane[-1] = opp_unit_code
            self.opp_elixir -= opp_cost

        new_lane = np.zeros_like(self.lane)
        for pos in range(self.lane_length):
            unit_np = self.lane[pos]
            unit = int(unit_np) 
            if unit == 0:
                continue

            is_mine = (unit <= 3)
            typ     = unit if is_mine else unit - 3
            direction =  1 if is_mine else -1
            next_pos  = pos + direction

            if next_pos < 0:
                self.your_hp -= typ
                reward -= 0.1  # penalty for your tower damage
            elif next_pos >= self.lane_length:
                self.opp_hp  -= typ
                reward += 1.0  # big reward for enemy tower hit
            else:
                other = self.lane[next_pos]
                if other != 0 and (other <= 3) != is_mine:
                    # both die
                    reward += 0.1 if is_mine else -0.1
                else:
                    # move forward
                    new_lane[next_pos] = unit

        self.lane = new_lane
        self.step_count += 1

        self.your_elixir = min(self.max_elixir, self.your_elixir + self.elixir_regen)
        self.opp_elixir  = min(self.max_elixir, self.opp_elixir  + self.elixir_regen)

        terminated = (self.your_hp <= 0) or (self.opp_hp <= 0)
        truncated  = (self.step_count >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, {"terminated": terminated, "truncated": truncated, "step": self.step_count
        }

    def render(self, mode='human'):
        """ASCII‐print the lane plus tower/elixir status."""
        symbols = {0:'.',1:'a',2:'b',3:'c',4:'A',5:'B',6:'C'}
        lane_str = ''.join(symbols[int(u)] for u in self.lane)
        print(f"HP (you,opp) = {self.your_hp:2d},{self.opp_hp:2d} │ "f"{lane_str} │ Elixir (you,opp) = {self.your_elixir:2d},{self.opp_elixir:2d}")
