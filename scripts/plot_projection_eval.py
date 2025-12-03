import itertools
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_eval_dir(
    data_dir: str, personas_filter: Optional[set[str]] = None, stage: str = "base"
) -> pd.DataFrame:
    """Load persona eval CSVs into a per-checkpoint, per-persona DataFrame."""
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pattern = re.compile(r"checkpoint-(\d+)_([^.]+)\.csv$")
    rows = []

    for csv_path in base.glob("checkpoint-*_*.csv"):
        match = pattern.match(csv_path.name)
        if not match:
            continue

        checkpoint = int(match.group(1))
        persona_name = match.group(2)
        if personas_filter and persona_name not in personas_filter:
            continue

        df = pd.read_csv(csv_path)

        persona_col = persona_name if persona_name in df.columns else None
        if persona_col is None:
            persona_col = next((c for c in df.columns if c.lower() == persona_name.lower()), None)
        coherence_col = next((c for c in df.columns if c.lower() == "coherence"), None)

        # Prefer exact finetuning shift column; fall back to substring search.
        finetune_col_name = (
            f"checkpoint-{checkpoint}_{persona_name}_response_avg_diff_prompt_last_proj_layer20_finetuning_shift"
        )
        finetune_candidates = [c for c in df.columns if finetune_col_name in c]
        if not finetune_candidates:
            finetune_candidates = [c for c in df.columns if "finetuning_shift" in c]
        finetune_col = finetune_candidates[0] if finetune_candidates else None

        if persona_col is None:
            raise ValueError(f"Could not find persona column for {csv_path}")
        if coherence_col is None:
            raise ValueError(f"Missing coherence column in {csv_path}")

        if finetune_col is None:
            warnings.warn(f"No finetuning shift column in {csv_path}; using NaN.")
            finetune_value = float("nan")
        else:
            finetune_series = pd.to_numeric(df[finetune_col], errors="coerce")
            finetune_value = finetune_series.mean()

        persona_series = pd.to_numeric(df[persona_col], errors="coerce")
        coherence_series = pd.to_numeric(df[coherence_col], errors="coerce")

        rows.append(
            {
                "checkpoint": checkpoint,
                "persona": persona_name,
                "persona_score": persona_series.mean(),
                "coherence": coherence_series.mean(),
                "finetuning_shift": finetune_value,
                "stage": stage,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["checkpoint", "persona", "persona_score", "coherence", "finetuning_shift"])

    return pd.DataFrame(rows).sort_values(["checkpoint", "persona"]).reset_index(drop=True)


def summarize_projection_eval(
    data_dir: str, sequential_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Load persona eval CSVs and return per-checkpoint, per-persona averages.

    If sequential_dir is provided, treat data_dir as the first-stage (A) evals and
    sequential_dir as the second-stage (A_B) evals. Personas are restricted to those
    present in the sequential_dir, and checkpoints in the sequential_dir are offset by
    the max checkpoint in data_dir.

    Returns (df, offset), where df columns include:
        [checkpoint, persona, persona_score, coherence, finetuning_shift, coherence_avg, stage].
    Offset is the checkpoint shift applied to sequential_dir (0 if not used).
    """
    offset = 0
    if sequential_dir is None:
        df = _load_eval_dir(data_dir, stage="base")
    else:
        base_df = _load_eval_dir(data_dir, stage="base")
        seq_df = _load_eval_dir(sequential_dir, stage="sequential")
        if seq_df.empty:
            raise ValueError(f"No checkpoint CSV files found in sequential dir {sequential_dir}")

        personas = set(seq_df["persona"].unique().tolist())
        if not base_df.empty:
            base_df = base_df[base_df["persona"].isin(personas)]

        offset = int(base_df["checkpoint"].max()) if not base_df.empty else 0
        seq_df = seq_df[seq_df["persona"].isin(personas)].assign(
            checkpoint=lambda d: d["checkpoint"] + offset
        )

        df = pd.concat([base_df, seq_df], ignore_index=True)

    if df.empty:
        raise ValueError("No checkpoint CSV files found.")

    df = df.sort_values(["checkpoint", "persona"]).reset_index(drop=True)

    coherence_avg = (
        df.groupby("checkpoint")["coherence"]
        .mean()
        .rename("coherence_avg")
        .reset_index()
    )
    df = df.merge(coherence_avg, on="checkpoint", how="left")
    return df, offset


def plot_projection_eval(
    data_dir: str,
    save_path: Optional[str] = None,
    show: bool = False,
    combine: bool = False,
    sequential_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[plt.Axes]]:
    """
    Summarize projection eval metrics per checkpoint and plot them.

    Args:
        data_dir: Path to an eval directory.
        save_path: Optional base file path to write the plot image(s). If combine is False,
            two files are written with suffixes _scores.png and _finetuning.png.
        show: Whether to call plt.show() at the end.
        combine: When True, generate a single combined plot (previous behavior). When False,
            generate two plots: (1) persona score + coherence avg, (2) finetuning shift.
        sequential_dir: Optional path to second-stage (A_B) evals. When provided, checkpoints
            in sequential_dir are offset by the max checkpoint in data_dir, and personas are
            restricted to those appearing in sequential_dir.

    Returns:
        (summary_df, axes_list)
    """
    summary, offset = summarize_projection_eval(data_dir, sequential_dir=sequential_dir)

    personas = summary["persona"].unique().tolist()

    # Style: color by persona, line style by metric.
    colors = plt.cm.tab10.colors
    color_cycle = itertools.cycle(colors)
    persona_to_color = {p: next(color_cycle) for p in personas}
    metric_styles = {
        "persona_score": "-",
        "finetuning_shift": ":",
    }

    axes = []

    def _plot_scores(ax):
        for persona in personas:
            persona_df = summary[summary["persona"] == persona]
            ax.plot(
                persona_df["checkpoint"],
                persona_df["persona_score"],
                label=f"{persona} (persona_score)",
                color=persona_to_color[persona],
                linestyle=metric_styles["persona_score"],
                marker="o",
            )
        # Single coherence line averaged over personas.
        dedup = summary.drop_duplicates("checkpoint")
        ax.plot(
            dedup["checkpoint"],
            dedup["coherence_avg"],
            label="coherence (avg all personas)",
            color="black",
            linestyle="--",
            marker="s",
        )
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Average score")
        ax.set_title(f"{Path(data_dir).name} - scores")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
        if sequential_dir and offset:
            ax.axvline(offset, color="gray", linestyle="-.", alpha=0.6)
            ax.text(offset, ax.get_ylim()[1], " finetune switch", rotation=90, va="top", ha="left")

    def _plot_finetune(ax):
        for persona in personas:
            persona_df = summary[summary["persona"] == persona]
            ax.plot(
                persona_df["checkpoint"],
                persona_df["finetuning_shift"],
                label=f"{persona} (finetuning_shift)",
                color=persona_to_color[persona],
                linestyle=metric_styles["finetuning_shift"],
                marker="o",
            )
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Finetuning shift")
        ax.set_title(f"{Path(data_dir).name} - finetuning shift")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
        if sequential_dir and offset:
            ax.axvline(offset, color="gray", linestyle="-.", alpha=0.6)
            ax.text(offset, ax.get_ylim()[1], " finetune switch", rotation=90, va="top", ha="left")

    if combine:
        fig, ax = plt.subplots(figsize=(9, 5))
        _plot_scores(ax)
        for persona in personas:
            persona_df = summary[summary["persona"] == persona]
            ax.plot(
                persona_df["checkpoint"],
                persona_df["finetuning_shift"],
                label=f"{persona} (finetuning_shift)",
                color=persona_to_color[persona],
                linestyle=metric_styles["finetuning_shift"],
                marker="o",
            )
        axes.append(ax)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
    else:
        fig1, ax1 = plt.subplots(figsize=(9, 5))
        _plot_scores(ax1)
        fig1.tight_layout()
        axes.append(ax1)

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        _plot_finetune(ax2)
        fig2.tight_layout()
        axes.append(ax2)

        if save_path:
            base = Path(save_path)
            out_suffix = base.suffix or ".png"
            fig1.savefig(base.with_name(f"{base.stem}_scores{out_suffix}"), dpi=200)
            fig2.savefig(base.with_name(f"{base.stem}_finetuning{out_suffix}"), dpi=200)

    if show:
        plt.show()

    return summary, axes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot average persona/coherence/finetuning shift vs. checkpoint."
    )
    parser.add_argument("data_dir", help="Path to projection eval directory with checkpoint CSVs.")
    parser.add_argument(
        "--save",
        dest="save_path",
        default=None,
        help="Optional file path to save the plot image (e.g., plot.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() so the plot is displayed.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Use a single combined plot (scores + finetuning). Default generates two plots.",
    )
    parser.add_argument(
        "--sequential-dir",
        default=None,
        help="Optional path to second-stage (A_B) evals; checkpoints will be offset by the max in data_dir and personas filtered to those in this dir.",
    )
    args = parser.parse_args()

    plot_projection_eval(
        args.data_dir,
        save_path=args.save_path,
        show=args.show,
        combine=args.combine,
        sequential_dir=args.sequential_dir,
    )
