"""Ad-hoc analysis: task-vs-task correlation heatmap + pairwise scatter matrix
across languages, computed from the aggregated `results/results.json` snapshot.

Run as a script from the repo root:

    uv run python evals/plots.py

Writes `task_correlation_matrix.png` and `task_scatter_matrix.png` to the
current working directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logit


def transform_classification_scores(row):
    if row["task"] == "classification":
        # Avoid division by zero and infinite values by clipping
        score = np.clip(row["score"], 0.001, 0.999)
        # Apply logit transformation (log(p/(1-p)))
        return logit(score)
    else:
        return row["score"]


def get_color_and_label(lang_code, highlighted_languages):
    if lang_code in highlighted_languages:
        color_map = {
            "en": "red",
            "zh": "blue",
            "hi": "green",
            "es": "orange",
            "ar": "purple",
        }
        return color_map[lang_code], lang_code
    else:
        return "lightgray", "Other"


def main():
    results_path = Path(__file__).resolve().parent.parent / "results" / "results.json"
    df = pd.read_json(results_path)

    df = df[df["metric"] != "chrf"]
    df = df.groupby(["task", "metric", "bcp_47"]).agg({"score": "mean"}).reset_index()

    df["score"] = df.apply(transform_classification_scores, axis=1)

    # Pivot: tasks as columns, languages as rows
    pivot_df = df.pivot_table(
        values="score", index="bcp_47", columns="task", aggfunc="mean"
    )

    ordered_tasks = [
        "translation_from",
        "translation_to",
        "classification",
        "mmlu",
        "arc",
        "mgsm",
    ]
    pivot_df = pivot_df[[task for task in ordered_tasks if task in pivot_df.columns]]

    correlation_matrix = pivot_df.corr()

    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="Blues",
        center=0,
        square=True,
        mask=mask,
        cbar_kws={"shrink": 0.8},
        fmt=".3f",
    )
    plt.xlabel("Tasks", fontsize=12)
    plt.ylabel("Tasks", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("task_correlation_matrix.png", dpi=300, bbox_inches="tight")

    print("Correlation Matrix:")
    print("Note: Classification scores have been logit-transformed to reduce skewness")
    print(correlation_matrix.round(3))

    # Scatter plot matrix for pairwise relationships with highlighted languages
    highlighted_languages = ["en", "zh", "hi", "es", "ar"]
    tasks = pivot_df.columns.tolist()
    n_tasks = len(tasks)

    fig, axes = plt.subplots(n_tasks, n_tasks, figsize=(15, 12))
    fig.suptitle("Pairwise Task Performance", fontsize=16, fontweight="bold")

    legend_elements = []
    for lang in highlighted_languages:
        color, _ = get_color_and_label(lang, highlighted_languages)
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=lang,
            )
        )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightgray",
            markersize=8,
            label="Other",
        )
    )

    for i, task_y in enumerate(tasks):
        for j, task_x in enumerate(tasks):
            ax = axes[i, j]

            if i == j:
                task_data = pivot_df[task_y].dropna()
                ax.hist(task_data, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
                ax.set_title(f"{task_y}", fontsize=10)
            else:
                for lang_code in pivot_df.index:
                    if pd.notna(pivot_df.loc[lang_code, task_x]) and pd.notna(
                        pivot_df.loc[lang_code, task_y]
                    ):
                        color, _ = get_color_and_label(lang_code, highlighted_languages)
                        alpha = 0.8 if lang_code in highlighted_languages else 0.3
                        size = 50 if lang_code in highlighted_languages else 20
                        ax.scatter(
                            pivot_df.loc[lang_code, task_x],
                            pivot_df.loc[lang_code, task_y],
                            c=color,
                            alpha=alpha,
                            s=size,
                        )

            if i == n_tasks - 1:
                ax.set_xlabel(task_x, fontsize=10)
            if j == 0:
                ax.set_ylabel(task_y, fontsize=10)
            if i != n_tasks - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    plt.tight_layout()
    plt.savefig("task_scatter_matrix.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
