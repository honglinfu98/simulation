"""RL market maker trained in the action-conditional world model.

A small Gaussian-MLP policy maps obs -> quoting action (half-spread, inventory
skew); trained by REINFORCE-with-baseline against the same informed-flow world
model used in world_model.py. The trained policy then plugs into
world_model.compare() via `make_rl_policy(net)` and is scored on the SAME metric
battery as the A-S inventory baseline and the heuristics -> a maker-comparison
table (RL vs A-S vs naive). Dynamics here mirror world_model.run exactly so
training and evaluation are consistent.

    obs    = [q / inv_limit, last_mid_move_ticks]
    action = (half_spread_ticks > 0, inv_skew_ticks >= 0)   via softplus
    reward = d(equity) - inv_pen * q^2      (capture spread, punish inventory risk)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class WMEnv:
    """Step-able action-conditional world model (matches world_model.run dynamics)."""
    def __init__(self, *, tick=0.01, quote_size=1.0, kappa=0.7, inv_limit=50.0,
                 informed_frac=0.3, fair_vol_ticks=0.3, impact_ticks=0.5, relax=0.2,
                 inv_pen=0.02, ep_len=256, seed=0):
        self.p = dict(tick=tick, quote_size=quote_size, kappa=kappa, inv_limit=inv_limit,
                      informed_frac=informed_frac, fair_vol_ticks=fair_vol_ticks,
                      impact_ticks=impact_ticks, relax=relax, inv_pen=inv_pen, ep_len=ep_len)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.fair = 100.0; self.mid = 100.0; self.q = 0.0; self.cash = 0.0
        self.prev_mid = 100.0; self.k = 0
        self.eq = self.cash + self.q * self.mid
        return np.array([0.0, 0.0], dtype=np.float32)

    def step(self, half_ticks, skew_ticks):
        p = self.p; t = p["tick"]
        self.fair += p["fair_vol_ticks"] * t * self.rng.standard_normal()
        self.mid += p["relax"] * (self.fair - self.mid)
        res = self.mid - self.q * skew_ticks * t
        bid, ask = res - 0.5 * half_ticks * t, res + 0.5 * half_ticks * t
        spread_ticks = (ask - bid) / t
        # aggressor side, conditioned on the maker's quotes vs fair
        if self.rng.random() < p["informed_frac"]:
            informed = True
            if ask < self.fair - 0.5 * spread_ticks * t:   side = +1
            elif bid > self.fair + 0.5 * spread_ticks * t:  side = -1
            else:                                            side = 0
        else:
            informed = False; side = int(self.rng.choice([-1, 1]))
        if side > 0 and self.q > -p["inv_limit"]:
            d = max((ask - self.mid) / t, 0.0)
            if informed or self.rng.random() < np.exp(-p["kappa"] * d):
                f = p["quote_size"]; self.q -= f; self.cash += f * ask
        elif side < 0 and self.q < p["inv_limit"]:
            d = max((self.mid - bid) / t, 0.0)
            if informed or self.rng.random() < np.exp(-p["kappa"] * d):
                f = p["quote_size"]; self.q += f; self.cash += -f * bid
        if side != 0:
            self.mid += p["impact_ticks"] * t * side * (0.5 + 0.5 * self.rng.random())
        new_eq = self.cash + self.q * self.mid
        reward = (new_eq - self.eq) - p["inv_pen"] * (self.q ** 2) * t
        self.eq = new_eq
        obs = np.array([self.q / p["inv_limit"], (self.mid - self.prev_mid) / t], dtype=np.float32)
        self.prev_mid = self.mid; self.k += 1
        return obs, float(reward), self.k >= p["ep_len"]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=2, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(self, obs):
        h = self.body(obs); return self.mean(h), self.log_std.exp()

    @staticmethod
    def to_action(raw):  # raw [.,2] -> (half_ticks>=0.5, skew_ticks>=0)
        a = torch.nn.functional.softplus(raw)
        return a[..., 0] + 0.5, a[..., 1]


def train(net, episodes=400, lr=3e-3, seed=0, **env_kw):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    env = WMEnv(seed=seed, **env_kw)
    base = 0.0; hist = []
    for ep in range(episodes):
        obs = env.reset(); logps = []; rews = []
        for _ in range(env.p["ep_len"]):
            mean, std = net(torch.tensor(obs))
            dist = torch.distributions.Normal(mean, std)
            raw = dist.sample(); logp = dist.log_prob(raw).sum()
            half, skew = PolicyNet.to_action(raw)
            obs, r, done = env.step(float(half), float(skew))
            logps.append(logp); rews.append(r)
            if done: break
        R = float(np.sum(rews)); hist.append(R)
        base = 0.95 * base + 0.05 * R                      # running baseline
        # REINFORCE with reward-to-go and baseline
        rtg = np.cumsum(rews[::-1])[::-1].copy()
        adv = torch.tensor(rtg - base / max(len(rews), 1), dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = -(torch.stack(logps) * adv).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return hist


def make_rl_policy(net, tick=0.01):
    """Adapter for world_model.compare(): stateful policy(mid,q)->(bid,ask)."""
    state = {"prev": None}
    @torch.no_grad()
    def pol(mid, q):
        dmid = 0.0 if state["prev"] is None else (mid - state["prev"]) / tick
        state["prev"] = mid
        obs = torch.tensor([q / 50.0, dmid], dtype=torch.float32)
        mean, _ = net(obs); half, skew = PolicyNet.to_action(mean)
        res = mid - q * float(skew) * tick
        return res - 0.5 * float(half) * tick, res + 0.5 * float(half) * tick
    return pol


if __name__ == "__main__":
    torch.manual_seed(0)
    net = PolicyNet()
    # tuned: enough episodes + a strong-enough inventory penalty that the policy
    # learns to widen + skew (i.e. discovers the A-S strategy) under adverse selection.
    hist = train(net, episodes=1200, lr=1e-3, inv_pen=0.08)
    print(f"RL training: avg episode reward first50={np.mean(hist[:50]):.2f} -> last50={np.mean(hist[-50:]):.2f}")
    learned = PolicyNet.to_action(net.mean(net.body(torch.zeros(2))).detach())
    print(f"learned quoting at flat inventory: half_spread={float(learned[0]):.2f} ticks, skew={float(learned[1]):.3f} ticks/unit")

    import world_model as wm
    pols = {
        "naive":         wm.make_naive(spread_ticks=2.0),
        "A-S inventory": wm.make_as_inventory(spread_ticks=2.0, inv_skew_ticks=0.3),
        "RL (trained)":  make_rl_policy(net),
    }
    print("\n### Maker comparison in the action-conditional world (informed_frac=0.3) ###")
    wm.compare(pols, informed_frac=0.3, seed=7)
