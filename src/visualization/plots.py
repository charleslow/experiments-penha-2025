"""Visualization utilities for experiment results."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


def plot_embedding_ablation(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Embedding Strategy Comparison",
):
    """
    Plot bar chart comparing embedding strategies (Search vs Rec vs Multi-task).

    Args:
        results: Dict mapping strategy name to metrics dict
            Example: {"search": {"NDCG@10": 0.5, "Recall@10": 0.6}, ...}
        output_path: Path to save the plot
        title: Plot title
    """
    # Prepare data
    strategies = list(results.keys())
    metrics = ["NDCG@10", "Recall@10"]

    data = []
    for strategy in strategies:
        for metric in metrics:
            value = results[strategy].get(metric, 0)
            data.append({
                "Strategy": strategy.replace("_", " ").title(),
                "Metric": metric,
                "Value": value,
            })

    df = pd.DataFrame(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(strategies))
    width = 0.35

    for i, metric in enumerate(metrics):
        metric_data = df[df["Metric"] == metric]
        offset = width * (i - 0.5)
        bars = ax.bar(
            x + offset,
            metric_data["Value"],
            width,
            label=metric,
            alpha=0.8,
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Embedding Strategy")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in strategies])
    ax.legend()
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved embedding ablation plot to {output_path}")


def plot_discretization_ablation(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Discretization Method Comparison",
):
    """
    Plot bar chart comparing discretization methods.

    Args:
        results: Dict mapping method name to metrics dict
        output_path: Path to save the plot
        title: Plot title
    """
    # Prepare data
    methods = list(results.keys())
    metrics = ["NDCG@10", "Recall@10"]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        values = [results[m].get(metric, 0) for m in methods]

        colors = sns.color_palette("husl", len(methods))
        bars = ax.bar(
            range(len(methods)),
            values,
            color=colors,
            alpha=0.8,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Discretization Method")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Method")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.upper().replace("_", "-") for m in methods], rotation=45)
        ax.set_ylim(0, max(values) * 1.2 if values else 1.0)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved discretization ablation plot to {output_path}")


def plot_tradeoff_scatter(
    results: List[Dict[str, float]],
    output_path: Path,
    title: str = "Search vs Recommendation Trade-off",
):
    """
    Plot scatter plot of Search NDCG vs Rec NDCG.

    Args:
        results: List of dicts with keys like:
            {"strategy": "multi_task", "method": "rq_kmeans",
             "search_ndcg": 0.5, "rec_ndcg": 0.4}
        output_path: Path to save the plot
        title: Plot title
    """
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by strategy, marker by method
    strategies = df["strategy"].unique()
    methods = df["method"].unique()

    colors = sns.color_palette("husl", len(strategies))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for s_idx, strategy in enumerate(strategies):
        for m_idx, method in enumerate(methods):
            subset = df[(df["strategy"] == strategy) & (df["method"] == method)]
            if len(subset) > 0:
                ax.scatter(
                    subset["search_ndcg"],
                    subset["rec_ndcg"],
                    c=[colors[s_idx]],
                    marker=markers[m_idx % len(markers)],
                    s=150,
                    alpha=0.8,
                    label=f"{strategy}/{method}",
                )

    ax.set_xlabel("Search NDCG@10")
    ax.set_ylabel("Recommendation NDCG@10")
    ax.set_title(title)

    # Add diagonal line
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_aspect("equal")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved trade-off scatter plot to {output_path}")


def plot_training_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Optional[Dict[str, List[float]]] = None,
    output_path: Path = None,
    title: str = "Training Curves",
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Dict mapping run name to list of training losses
        val_losses: Optional dict mapping run name to list of val losses
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(train_losses))

    for idx, (name, losses) in enumerate(train_losses.items()):
        epochs = range(1, len(losses) + 1)
        ax.plot(
            epochs,
            losses,
            color=colors[idx],
            linestyle="-",
            linewidth=2,
            label=f"{name} (train)",
        )

        if val_losses and name in val_losses:
            ax.plot(
                epochs,
                val_losses[name],
                color=colors[idx],
                linestyle="--",
                linewidth=2,
                label=f"{name} (val)",
                alpha=0.7,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved training curves to {output_path}")
    else:
        plt.show()


def plot_full_comparison(
    results: pd.DataFrame,
    output_path: Path,
    title: str = "Full Experiment Results",
):
    """
    Plot comprehensive comparison across all configurations.

    Args:
        results: DataFrame with columns:
            [embedding_strategy, discretization_method, search_ndcg_10,
             search_recall_10, rec_ndcg_10, rec_recall_10]
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metrics = [
        ("search_ndcg_10", "Search NDCG@10"),
        ("search_recall_10", "Search Recall@10"),
        ("rec_ndcg_10", "Rec NDCG@10"),
        ("rec_recall_10", "Rec Recall@10"),
    ]

    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Pivot for heatmap
        pivot = results.pivot(
            index="embedding_strategy",
            columns="discretization_method",
            values=metric_col,
        )

        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": metric_name},
        )
        ax.set_title(metric_name)
        ax.set_xlabel("Discretization Method")
        ax.set_ylabel("Embedding Strategy")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved full comparison plot to {output_path}")


def create_results_table(
    results: List[Dict],
    output_path: Path,
):
    """
    Create a results table in markdown format.

    Args:
        results: List of result dictionaries
        output_path: Path to save the markdown table
    """
    df = pd.DataFrame(results)

    # Format numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")

    # Save as markdown
    markdown = df.to_markdown(index=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Experiment Results\n\n")
        f.write(markdown)

    logger.info(f"Saved results table to {output_path}")
