"""Evaluation result visualizations via charts"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

def _require_matplotlib():
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )

def load_evaluation_json(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _latest_eval_file(results_dir: str | Path) -> Optional[Path]:
    rdir = Path(results_dir)
    files = sorted(rdir.glob("evaluation_*.json"), reverse=True)
    return files[0] if files else None

def plot_per_query_metrics(
    data: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
    title: str = "Evaluation Metrics by Query",
) -> Optional[Path]:
    _require_matplotlib()

    results = data.get("individual_results", [])
    if not results:
        logger.warning("No individual results to plot")
        return None

    if metrics is None:
        metrics = ["precision_at_5", "recall_at_5", "faithfulness", "groundedness", "answer_completeness"]

    available = [m for m in metrics if m in results[0]]
    if not available:
        logger.warning("None of the requested metrics found in results")
        return None

    queries = [r.get("query", f"Q{i+1}")[:50] for i, r in enumerate(results)]
    n_queries = len(queries)
    n_metrics = len(available)
    x = np.arange(n_queries)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, n_queries * 1.2), 6))

    for j, metric in enumerate(available):
        values = [r.get(metric, 0.0) for r in results]
        offset = (j - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title())
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(output_path) if output_path else Path("eval_per_query.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved per-query bar chart to %s", out)
    return out

def plot_summary_averages(
    data: Dict[str, Any],
    output_path: Optional[str | Path] = None,
    title: str = "Average Evaluation Metrics",
) -> Optional[Path]:
    _require_matplotlib()

    summary = data.get("summary", {})
    if not summary:
        logger.warning("No summary data to plot")
        return None

    avg_items = {
        k.replace("avg_", "").replace("_", " ").title(): v
        for k, v in summary.items()
        if k.startswith("avg_")
    }
    if not avg_items:
        return None

    labels = list(avg_items.keys())
    values = list(avg_items.values())

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(labels)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9,
        )
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(output_path) if output_path else Path("eval_summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved summary bar chart to %s", out)
    return out

def plot_category_distribution(
    data: Dict[str, Any],
    output_path: Optional[str | Path] = None,
    title: str = "Retrieved Document Categories",
) -> Optional[Path]:

    _require_matplotlib()

    results = data.get("individual_results", [])
    if not results:
        return None

    has_labels_field = any("has_labels" in r for r in results)
    labeled_count = sum(1 for r in results if r.get("has_labels"))
    unlabeled_count = len(results) - labeled_count

    fig, ax = plt.subplots(figsize=(6, 4))
    if has_labels_field:
        bars = ax.bar(
            ["Labeled", "Unlabeled"],
            [labeled_count, unlabeled_count],
            color=["#2196F3", "#FF9800"],
            edgecolor="white",
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(int(bar.get_height())), ha="center", va="bottom",
            )
        ax.set_ylabel("Query Count")
        ax.set_title("Labeled vs Unlabeled Evaluation Queries")
    else:
        ax.bar(["Total Queries"], [len(results)], color="#2196F3")
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(output_path) if output_path else Path("eval_categories.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved category distribution chart to %s", out)
    return out

def plot_run_comparison(
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
    output_path: Optional[str | Path] = None,
    label_a: str = "Run A",
    label_b: str = "Run B",
    title: str = "Evaluation Run Comparison",
) -> Optional[Path]:

    _require_matplotlib()

    summary_a = data_a.get("summary", {})
    summary_b = data_b.get("summary", {})

    common_keys = sorted(
        k for k in summary_a
        if k.startswith("avg_") and k in summary_b
    )
    if not common_keys:
        logger.warning("No common avg_ metrics to compare")
        return None

    labels = [k.replace("avg_", "").replace("_", " ").title() for k in common_keys]
    vals_a = [summary_a[k] for k in common_keys]
    vals_b = [summary_b[k] for k in common_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    bars_a = ax.bar(x - width / 2, vals_a, width, label=label_a, color="#2196F3")
    bars_b = ax.bar(x + width / 2, vals_b, width, label=label_b, color="#FF9800")

    for bars in [bars_a, bars_b]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(output_path) if output_path else Path("eval_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved comparison chart to %s", out)
    return out

def generate_all_charts(
    eval_json_path: str | Path,
    output_dir: str | Path,
    compare_with: Optional[str | Path] = None,
) -> List[Path]:
    """Generate all available charts for an evaluation result file.

    Returns a list of generated file paths.
    """
    _require_matplotlib()

    data = load_evaluation_json(eval_json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    p = plot_per_query_metrics(data, output_path=out_dir / "per_query_metrics.png")
    if p:
        generated.append(p)

    p = plot_summary_averages(data, output_path=out_dir / "summary_averages.png")
    if p:
        generated.append(p)

    p = plot_category_distribution(data, output_path=out_dir / "query_label_distribution.png")
    if p:
        generated.append(p)

    if compare_with:
        data_b = load_evaluation_json(compare_with)
        ts_a = data.get("timestamp", "Run A")
        ts_b = data_b.get("timestamp", "Run B")
        p = plot_run_comparison(
            data, data_b,
            output_path=out_dir / "run_comparison.png",
            label_a=f"Run {ts_a}",
            label_b=f"Run {ts_b}",
        )
        if p:
            generated.append(p)

    logger.info("Generated %d chart(s) in %s", len(generated), out_dir)
    return generated
