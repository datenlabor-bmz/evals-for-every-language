import json

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

with open("results.json") as f:
    results = json.load(f)


def mean(lst):
    return sum(lst) / len(lst)


def create_leaderboard_df(results):
    # Sort languages by average BLEU to determine resource categories
    langs_with_bleu = [lang for lang in results if lang["bleu"] is not None]
    sorted_langs = sorted(langs_with_bleu, key=lambda x: x["bleu"], reverse=True)
    n_langs = len(sorted_langs)
    high_cutoff = n_langs // 4  # top 25%
    low_cutoff = n_langs - n_langs // 4  # bottom 25%

    # Create sets of languages for each category
    high_resource = {lang["language_name"] for lang in sorted_langs[:high_cutoff]}
    low_resource = {lang["language_name"] for lang in sorted_langs[low_cutoff:]}

    # Get all model scores with categorization
    model_scores = {}
    for lang in results:
        category = (
            "High-Resource"
            if lang["language_name"] in high_resource
            else "Low-Resource"
            if lang["language_name"] in low_resource
            else "Mid-Resource"
        )

        for score in lang["scores"]:
            model = score["model"]
            if model not in model_scores:
                model_scores[model] = {
                    "High-Resource": [],
                    "Mid-Resource": [],
                    "Low-Resource": [],
                }
            model_scores[model][category].append(score["bleu"])

    # Calculate average scores and create DataFrame
    leaderboard_data = []
    for model, categories in model_scores.items():
        # Calculate averages for each category
        high_avg = (
            round(mean(categories["High-Resource"]), 3)
            if categories["High-Resource"]
            else 0
        )
        mid_avg = (
            round(mean(categories["Mid-Resource"]), 3)
            if categories["Mid-Resource"]
            else 0
        )
        low_avg = (
            round(mean(categories["Low-Resource"]), 3)
            if categories["Low-Resource"]
            else 0
        )

        # Calculate overall average
        all_scores = (
            categories["High-Resource"]
            + categories["Mid-Resource"]
            + categories["Low-Resource"]
        )
        overall_avg = round(sum(all_scores) / len(all_scores), 3)

        model_name = model.split("/")[-1]
        leaderboard_data.append(
            {
                "Model": f"[{model_name}](https://openrouter.ai/{model})",
                "Overall BLEU": overall_avg,
                "High-Resource BLEU": high_avg,
                "Mid-Resource BLEU": mid_avg,
                "Low-Resource BLEU": low_avg,
                "Languages Tested": len(all_scores),
            }
        )

    # Sort by overall BLEU
    df = pd.DataFrame(leaderboard_data)
    df = df.sort_values("Overall BLEU", ascending=False)

    # Add rank and medals
    df["Rank"] = range(1, len(df) + 1)
    df["Rank"] = df["Rank"].apply(
        lambda x: "🥇" if x == 1 else "🥈" if x == 2 else "🥉" if x == 3 else str(x)
    )

    # Reorder columns
    df = df[
        [
            "Rank",
            "Model",
            "Overall BLEU",
            "High-Resource BLEU",
            "Mid-Resource BLEU",
            "Low-Resource BLEU",
            "Languages Tested",
        ]
    ]

    return gr.DataFrame(
        value=df,
        label="Model Leaderboard",
        show_search=False,
        datatype=[
            "number",
            "markdown",
            "number",
            "number",
            "number",
            "number",
            "number",
        ],
    )


def create_model_comparison_plot(results):
    # Extract all unique models
    models = set()
    for lang in results:
        for score in lang["scores"]:
            models.add(score["model"])
    models = list(models)

    # Create traces for each model
    traces = []
    for model in models:
        x_vals = []  # languages
        y_vals = []  # BLEU scores

        for lang in results:
            model_score = next(
                (s["bleu"] for s in lang["scores"] if s["model"] == model), None
            )
            if model_score is not None:
                x_vals.append(lang["language_name"])
                y_vals.append(model_score)

        traces.append(
            go.Bar(
                name=model.split("/")[-1],
                x=x_vals,
                y=y_vals,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="BLEU Scores by Model and Language",
        xaxis_title="Language",
        yaxis_title="BLEU Score",
        barmode="group",
        height=500,
    )
    return fig


def create_language_stats_df(results):
    # Create a list to store flattened data
    flat_data = []

    for lang in results:
        # Find the best model and its BLEU score
        best_score = max(
            lang["scores"] or [{"bleu": None, "model": None}], key=lambda x: x["bleu"]
        )

        model = best_score['model']
        model_name = model.split('/')[-1] if model else "N/A"
        model_link = f"<a href='https://openrouter.ai/{model}' style='text-decoration: none; color: inherit;'>{model_name}</a>" if model else "N/A"
        row = {
            "Language": f"**{lang['language_name']}**",
            "Speakers (M)": round(lang["speakers"] / 1_000_000, 1),
            "Models Tested": len(lang["scores"]),
            "Average BLEU": round(lang["bleu"], 3)
            if lang["bleu"] is not None
            else "N/A",
            "Best Model": model_link,
            "Best Model BLEU": round(best_score["bleu"], 3)
            if best_score["bleu"] is not None
            else "N/A",
            "CommonVoice Hours": lang["commonvoice_hours"],
        }
        flat_data.append(row)

    df = pd.DataFrame(flat_data)
    return gr.DataFrame(
        value=df,
        label="Language Results",
        show_search="search",
        datatype=["markdown", "number", "number", "number", "markdown", "number"],
    )


def create_scatter_plot(results):
    fig = go.Figure()

    x_vals = [lang["speakers"] / 1_000_000 for lang in results]  # Convert to millions
    y_vals = [lang["bleu"] for lang in results]
    labels = [lang["language_name"] for lang in results]

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>"
            + "Speakers: %{x:.1f}M<br>"
            + "BLEU Score: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Language Coverage: Speakers vs BLEU Score",
        xaxis_title="Number of Speakers (Millions)",
        yaxis_title="Average BLEU Score",
        height=500,
        showlegend=False,
    )

    # Use log scale for x-axis since speaker numbers vary widely
    fig.update_xaxes(type="log")

    return fig


# Create the visualization components
with gr.Blocks(title="AI Language Translation Benchmark") as demo:
    gr.Markdown("# AI Language Translation Benchmark")
    gr.Markdown(
        "Comparing translation performance across different AI models and languages"
    )

    bar_plot = create_model_comparison_plot(results)
    scatter_plot = create_scatter_plot(results)

    create_leaderboard_df(results)
    gr.Plot(value=bar_plot, label="Model Comparison")
    create_language_stats_df(results)
    gr.Plot(value=scatter_plot, label="Language Coverage")

    gr.Markdown(
        """
        ## Methodology
        ### Dataset
        - Using [FLORES-200](https://huggingface.co/datasets/openlanguagedata/flores_plus) evaluation set, a high-quality human-translated benchmark comprising 200 languages
        - Each language is tested with the same 100 sentences
        - All translations are from the evaluated language to a fixed set of representative languages sampled by number of speakers
        - Language statistics sourced from Ethnologue and Wikidata

        ### Models & Evaluation
        - Models accessed through [OpenRouter](https://openrouter.ai/), including fast models of all big labs, open and closed
        - **BLEU Score**: Translations are evaluated using the BLEU metric, which measures how similar the AI's translation is to a human reference translation -- higher is better
        
        ### Language Categories
        Languages are divided into three tiers based on translation difficulty:
        - High-Resource: Top 25% of languages by BLEU score (easiest to translate)
        - Mid-Resource: Middle 50% of languages
        - Low-Resource: Bottom 25% of languages (hardest to translate)
    """,
        container=True,
    )

demo.launch()
