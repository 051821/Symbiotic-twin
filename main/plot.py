from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_shared.model_comparison import (
    build_comparison_figures,
    load_model_summaries,
    print_metric_table,
)


def main() -> None:
    """
    Build fair comparison plots using the SAME metrics and SAME graph style
    for Symbiotic-Twin, FedAvg, and Centralized models.
    """
    summaries = load_model_summaries(ROOT)
    print_metric_table(summaries)

    fig_raw, fig_norm = build_comparison_figures(summaries)
    plt.show()

    # Keep references alive for interactive backends
    _ = (fig_raw, fig_norm)


if __name__ == "__main__":
    main()