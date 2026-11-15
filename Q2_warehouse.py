"""
CS 351 - Assignment 03, Part B Q2
Starter skeleton: Warehouse robot planning with Value Iteration
(and Policy Iteration for the bonus).

Map characters:
    . = free, # = shelf, G = packing station (+100),
    X = spill zone (-100), S = start

Complete the TODO sections according to the assignment instructions.
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


ACTIONS = ["U", "D", "L", "R"]
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}
ARROW = {"U": (0, -0.35), "D": (0, 0.35), "L": (-0.35, 0), "R": (0.35, 0)}


# ----------------------------------------------------------------------
# Map loading
# ----------------------------------------------------------------------

def load_grid(path):
    """Load the warehouse map."""
    with open(path) as f:
        lines = [l.rstrip("\n\r") for l in f if l.strip()]
    w = max(len(l) for l in lines)
    return np.array([list(l.ljust(w, "#")) for l in lines])


# ----------------------------------------------------------------------
# Warehouse environment
# ----------------------------------------------------------------------

class Warehouse:
    def __init__(self, grid, goal_reward=100.0, spill_reward=-100.0,
                 p_intended=0.8):
        self.grid = grid
        self.H, self.W = grid.shape
        self.p = p_intended
        self.goal_reward = goal_reward
        self.spill_reward = spill_reward
        self.N = self.H * self.W

        self.wall2d = grid == "#"
        self.goal2d = grid == "G"
        self.spill2d = grid == "X"
        #cells where the robot actually picks an action
        self.states2d = ~(self.wall2d | self.goal2d | self.spill2d)

        starts = np.argwhere(grid == "S")
        if len(starts) == 0:
            raise ValueError("map has no S")
        self.start = (int(starts[0][0]), int(starts[0][1]))

        #where every cell ends up for each direction, as flat cell numbers
        self.nxt = {}
        for a in ACTIONS:
            arr = np.zeros(self.N, dtype=int)
            for r in range(self.H):
                for c in range(self.W):
                    nr, nc = self._move((r, c), a)
                    arr[r * self.W + c] = nr * self.W + nc
            self.nxt[a] = arr
        self._trans = {}

    def _move(self, c, action):
        r, col = c
        dr, dc = MOVES[action]
        nr, nc = r + dr, col + dc
        if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
            return c
        if self.grid[nr, nc] == "#":
            return c
        return (nr, nc)

    def _transitions(self, stochastic):
        #for each action: list of (probability, where you land) pairs
        if stochastic in self._trans:
            return self._trans[stochastic]
        side = (1 - self.p) / 2
        table = []
        for a in ACTIONS:
            if stochastic:
                parts = [(self.p, self.nxt[a]),
                         (side, self.nxt[PERP[a][0]]),
                         (side, self.nxt[PERP[a][1]])]
            else:
                parts = [(1.0, self.nxt[a])]
            table.append(parts)
        self._trans[stochastic] = table
        return table

    def q_values(self, V, stochastic, gamma, r):
        Vf = V.reshape(-1)
        #reward you get for landing on each cell
        land = np.full(self.N, r, dtype=float)
        land[self.goal2d.reshape(-1)] = self.goal_reward
        land[self.spill2d.reshape(-1)] = self.spill_reward
        Q = np.zeros((self.N, 4))
        for i, parts in enumerate(self._transitions(stochastic)):
            for prob, nxt in parts:
                Q[:, i] += prob * (land[nxt] + gamma * Vf[nxt])
        return Q.reshape(self.H, self.W, 4)

    def value_iteration(self, stochastic=False, gamma=0.9, r=-1.0,
                        theta=1e-4, max_iter=100000):
        V = np.zeros((self.H, self.W))
        iters = 0
        while iters < max_iter:
            iters += 1
            Q = self.q_values(V, stochastic, gamma, r)
            newV = np.where(self.states2d, Q.max(axis=2), 0.0)
            delta = np.abs(newV - V).max()
            V = newV
            if delta < theta:
                break
        return V, iters

    def greedy(self, V, stochastic, gamma, r):
        Q = self.q_values(V, stochastic, gamma, r)
        best = Q.argmax(axis=2)
        policy = {}
        for rr, cc in zip(*np.nonzero(self.states2d)):
            policy[(int(rr), int(cc))] = ACTIONS[best[rr, cc]]
        return policy

    def policy_evaluation(self, policy, stochastic, gamma, r):
        acts = np.zeros((self.H, self.W), dtype=int)
        for (rr, cc), a in policy.items():
            acts[rr, cc] = ACTIONS.index(a)
        V = np.zeros((self.H, self.W))
        for _ in range(100000):
            Q = self.q_values(V, stochastic, gamma, r)
            chosen = np.take_along_axis(Q, acts[:, :, None], axis=2)[:, :, 0]
            newV = np.where(self.states2d, chosen, 0.0)
            delta = np.abs(newV - V).max()
            V = newV
            if delta < 1e-8:
                break
        return V

    def policy_iteration(self, stochastic=True, gamma=0.9, r=-1.0,
                         max_iter=1000):
        #start with every cell going Up, then keep improving
        policy = {(int(rr), int(cc)): "U" for rr, cc in zip(*np.nonzero(self.states2d))}
        iters = 0
        while iters < max_iter:
            iters += 1
            V = self.policy_evaluation(policy, stochastic, gamma, r)
            Q = self.q_values(V, stochastic, gamma, r)
            changed = False
            for (rr, cc), a in policy.items():
                best = int(Q[rr, cc].argmax())
                #only switch if the new action is clearly better
                if Q[rr, cc, best] > Q[rr, cc, ACTIONS.index(a)] + 1e-9:
                    policy[(rr, cc)] = ACTIONS[best]
                    changed = True
            if not changed:
                break
        return V, policy, iters

    def simulate(self, policy, episodes=1000, r=-1.0,
                 max_steps=500, seed=0):
        """Simulate episodes and return performance statistics."""
        #always runs in the stochastic world, starting from S
        rng = np.random.default_rng(seed)
        side = (1 - self.p) / 2
        returns = []
        steps_all = []
        wins = 0
        for _ in range(episodes):
            cell = self.start
            total = 0.0
            steps = 0
            while steps < max_steps:
                a = policy[cell]
                x = rng.random()
                if x < self.p:
                    act = a
                elif x < self.p + side:
                    act = PERP[a][0]
                else:
                    act = PERP[a][1]
                cell = self._move(cell, act)
                steps += 1
                ch = self.grid[cell]
                if ch == "G":
                    total += self.goal_reward
                    wins += 1
                    break
                if ch == "X":
                    total += self.spill_reward
                    break
                total += r
            returns.append(total)
            steps_all.append(steps)
        return {"avg_return": float(np.mean(returns)),
                "avg_steps": float(np.mean(steps_all)),
                "success_rate": wins / episodes}

    def plot(self, V, policy, title, ax, mark=None):
        """Plot values and policy."""
        vals = V[self.states2d]
        lo = min(vals.min(), -1.0)
        hi = max(vals.max(), 1.0)
        show = V.copy()
        show[self.goal2d] = hi
        show[self.spill2d] = lo
        show = np.ma.masked_where(self.wall2d, show)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("dimgray")
        ax.imshow(show, cmap=cmap, norm=TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi))

        fs = 7 if self.H <= 12 else 4
        for r in range(self.H):
            for c in range(self.W):
                ch = self.grid[r, c]
                if ch in "GX":
                    ax.text(c, r, ch, ha="center", va="center", fontsize=12, fontweight="bold")
                    continue
                if ch == "#":
                    continue
                dx, dy = ARROW[policy[(r, c)]]
                ax.arrow(c - dx, r - dy, 2 * dx, 2 * dy, head_width=0.2,
                         length_includes_head=True, color="black")
                ax.text(c - 0.45, r - 0.45, f"{V[r, c]:.0f}", ha="left", va="top", fontsize=fs)

        sr, sc = self.start
        ax.add_patch(Rectangle((sc - 0.5, sr - 0.5), 1, 1, fill=False, ec="blue", lw=2))
        for (r, c) in (mark or []):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="red", lw=2))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])


# ----------------------------------------------------------------------
# Random warehouse map
# ----------------------------------------------------------------------

def random_map(n=20, shelf=0.22, spills=12, seed=3):
    """Generate a reachable random warehouse map."""
    rng = np.random.default_rng(seed)
    while True:
        grid = np.where(rng.random((n, n)) < shelf, "#", ".")
        grid[0, 0] = "S"
        grid[n - 1, n - 1] = "G"
        free = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == "."]
        for k in rng.choice(len(free), size=spills, replace=False):
            grid[free[k]] = "X"
        seen = reachable(grid)
        if not seen[n - 1, n - 1]:
            continue
        #shelve off free cells the robot can never reach
        grid[(grid == ".") & ~seen] = "#"
        return grid


def reachable(grid):
    #cells the robot can get to from S without stepping on a spill zone
    H, W = grid.shape
    start = tuple(np.argwhere(grid == "S")[0])
    seen = np.zeros((H, W), dtype=bool)
    seen[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        if grid[r, c] == "X":
            continue
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if seen[nr, nc] or grid[nr, nc] == "#":
                continue
            seen[nr, nc] = True
            stack.append((nr, nc))
    return seen


# ----------------------------------------------------------------------
# Main experiments
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="maps/warehouse.txt")
    parser.add_argument("--out", default="out/q2")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    grid = load_grid(args.map)
    env = Warehouse(grid)

    # Part (a): Deterministic Value Iteration
    Vd, it_d = env.value_iteration(stochastic=False, gamma=0.9, r=-1.0)
    pol_d = env.greedy(Vd, False, 0.9, -1.0)
    print("(a) deterministic: converged in", it_d, "iterations")
    fig, ax = plt.subplots(figsize=(7, 7))
    env.plot(Vd, pol_d, f"Deterministic V* ({it_d} iterations)", ax)
    fig.savefig(os.path.join(args.out, "a_deterministic.png"), dpi=130)
    plt.close(fig)

    # Part (b): Stochastic Value Iteration
    Vs, it_s = env.value_iteration(stochastic=True, gamma=0.9, r=-1.0)
    pol_s = env.greedy(Vs, True, 0.9, -1.0)
    print("(b) stochastic: converged in", it_s, "iterations")
    diff = [c for c in pol_d if pol_d[c] != pol_s[c]]
    print("cells where the policy changes (row, col): deterministic -> stochastic")
    for c in diff:
        print(" ", c, pol_d[c], "->", pol_s[c], " V det", round(Vd[c], 1), " V stoch", round(Vs[c], 1))
    fig, ax = plt.subplots(figsize=(7, 7))
    env.plot(Vs, pol_s, f"Stochastic V* ({it_s} iterations), red = differs from (a)", ax, mark=diff)
    fig.savefig(os.path.join(args.out, "b_stochastic.png"), dpi=130)
    plt.close(fig)

    # Part (c): Policy simulation
    res_d = env.simulate(pol_d, episodes=1000, r=-1.0)
    res_s = env.simulate(pol_s, episodes=1000, r=-1.0)
    lines = ["policy                  avg return   avg steps   success rate"]
    for name, res in (("deterministic policy", res_d), ("stochastic policy", res_s)):
        lines.append(f"{name:<22}  {res['avg_return']:>9.2f}   {res['avg_steps']:>9.2f}   {res['success_rate']:>11.3f}")
    print("(c) both policies run in the stochastic world")
    print("\n".join(lines))
    with open(os.path.join(args.out, "c_simulation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Part (d): Parameter sweep
    gammas = [0.5, 0.9, 0.99]
    rewards = [-5, -1, 0, 1]
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    lines = ["gamma   r    iterations   success rate   avg steps   avg return"]
    for i, g in enumerate(gammas):
        for j, rw in enumerate(rewards):
            V, it = env.value_iteration(stochastic=True, gamma=g, r=rw)
            pol = env.greedy(V, True, g, rw)
            res = env.simulate(pol, episodes=1000, r=rw)
            lines.append(f"{g:<6}  {rw:>2}   {it:>10}   {res['success_rate']:>12.3f}   {res['avg_steps']:>9.1f}   {res['avg_return']:>10.1f}")
            env.plot(V, pol, f"gamma={g}, r={rw}  ({it} it, success {res['success_rate']:.0%})", axes[i, j])
    print("(d) sweep")
    print("\n".join(lines))
    with open(os.path.join(args.out, "d_sweep.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "d_sweep_policies.png"), dpi=90)
    plt.close(fig)

    # Bonus: Policy Iteration vs Value Iteration
    big = Warehouse(random_map())
    t0 = time.perf_counter()
    Vv, it_v = big.value_iteration(stochastic=True, gamma=0.9, r=-1.0)
    t_v = time.perf_counter() - t0
    t0 = time.perf_counter()
    Vp, pol_p, it_p = big.policy_iteration(stochastic=True, gamma=0.9, r=-1.0)
    t_p = time.perf_counter() - t0
    same = np.allclose(Vv, Vp, atol=1e-2)
    lines = ["method             iterations   time (s)",
             f"value iteration    {it_v:>10}   {t_v:>8.4f}",
             f"policy iteration   {it_p:>10}   {t_p:>8.4f}",
             f"values agree: {same}"]
    print("bonus on a 20 x 20 map")
    print("\n".join(lines))
    with open(os.path.join(args.out, "bonus.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()