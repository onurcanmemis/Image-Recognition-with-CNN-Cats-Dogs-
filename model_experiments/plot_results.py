"""Plot val-accuracy curves across all tournament models.

Reads the most recent results_YYYY-MM-DD.json next to this script and emits
a single combined line plot to model_experiments/plots/val_accuracy_curves.png.
Models whose training_history is missing (e.g. failed builds) are skipped, so
the script works against a partial results JSON (some models failed) and a
full one without modification.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"


def main() -> None:
    json_files = sorted(RESULTS_DIR.glob("results_*.json"))
    if not json_files:
        raise SystemExit("No results_*.json files found.")
    latest = json_files[-1]
    data = json.loads(latest.read_text())

    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plotted = 0
    for entry in data.get("models", []):
        hist = entry.get("training_history") or {}
        ys = hist.get("val_accuracy")
        if not ys:
            continue
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, label=entry["name"], linewidth=2)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"{latest.name} contains no training_history entries.")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(
        f"Validation accuracy across epochs ({data.get('run_date', latest.stem)})"
    )
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = PLOTS_DIR / "val_accuracy_curves.png"
    fig.savefig(out, dpi=140)
    print(f"Wrote {out} ({plotted} model{'s' if plotted != 1 else ''})")


if __name__ == "__main__":
    main()
