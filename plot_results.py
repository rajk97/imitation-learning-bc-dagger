"""Make Figure 1 (BC train-steps sweep) and Figure 2 (DAgger curves) from tensorboard logs in data/."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SWEEP_STEPS = [250, 500, 1000, 2000, 3000, 5000, 10000]


def latest_run(prefix: str) -> str:
    runs = sorted(glob.glob(os.path.join(DATA_DIR, prefix + "_*")), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError(f"No run directory matching {prefix}_* in {DATA_DIR}")
    return runs[-1]


def scalars(run_dir: str, tag: str) -> list[float]:
    acc = EventAccumulator(run_dir)
    acc.Reload()
    return [event.value for event in acc.Scalars(tag)]


def plot_bc_sweep() -> None:
    means, stds = [], []
    for steps in SWEEP_STEPS:
        run = latest_run(f"q1_bc_ant_steps{steps}_Ant-v2")
        means.append(scalars(run, "Eval_AverageReturn")[-1])
        stds.append(scalars(run, "Eval_StdReturn")[-1])
    expert = scalars(run, "Initial_DataCollection_AverageReturn")[-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(SWEEP_STEPS, means, yerr=stds, marker="o", capsize=4, label="BC (mean ± std)")
    ax.axhline(expert, color="green", linestyle="--", label="Expert")
    ax.set_xscale("log")
    ax.set_xlabel("num_agent_train_steps_per_iter")
    ax.set_ylabel("Eval average return")
    ax.set_title("BC on Ant-v2: effect of training steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_bc_sweep.png"), dpi=150)


def plot_dagger_panel(ax: plt.Axes, env: str, env_id: str) -> None:
    dagger_run = latest_run(f"q2_dagger_{env}_{env_id}")
    bc_run = latest_run(f"q1_bc_{env}_{env_id}")
    means = scalars(dagger_run, "Eval_AverageReturn")
    stds = scalars(dagger_run, "Eval_StdReturn")
    expert = scalars(dagger_run, "Initial_DataCollection_AverageReturn")[-1]
    bc = scalars(bc_run, "Eval_AverageReturn")[-1]

    iters = list(range(len(means)))
    ax.errorbar(iters, means, yerr=stds, marker="o", capsize=4, label="DAgger (mean ± std)")
    ax.axhline(expert, color="green", linestyle="--", label="Expert")
    ax.axhline(bc, color="red", linestyle=":", label="BC")
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("Eval average return")
    ax.set_title(env_id)
    ax.legend()


def plot_dagger() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    plot_dagger_panel(left, "ant", "Ant-v2")
    plot_dagger_panel(right, "humanoid", "Humanoid-v2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_dagger.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_bc_sweep()
    plot_dagger()
    print("Saved figures to", FIG_DIR)
