import gradio as gr
import json
import pandas as pd
import plotly.graph_objects as go

# Load and process results
with open("results.json") as f:
    results = json.load(f)


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


def create_results_df(results):
    # Create a list to store flattened data
    flat_data = []

    for lang in results:
        # Find the best model and its BLEU score
        best_score = max(lang["scores"] or [{"bleu": None, "model": None}], key=lambda x: x["bleu"])
        
        row = {
            "Language": lang["language_name"],
            "Speakers (M)": round(lang["speakers"] / 1_000_000, 1),
            "Models Tested": len(lang["scores"]),
            "Average BLEU": round(lang["bleu"], 3) if lang["bleu"] is not None else "N/A",
            "Best Model": best_score["model"] if best_score["model"] is not None else "N/A",
            "Best Model BLEU": round(best_score["bleu"], 3) if best_score["bleu"] is not None else "N/A",
        }
        flat_data.append(row)

    return pd.DataFrame(flat_data)


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
        category = ("High-Resource" if lang["language_name"] in high_resource else
                   "Low-Resource" if lang["language_name"] in low_resource else
                   "Mid-Resource")
        
        for score in lang["scores"]:
            model_name = score["model"].split("/")[-1]
            if model_name not in model_scores:
                model_scores[model_name] = {
                    "High-Resource": [],
                    "Mid-Resource": [],
                    "Low-Resource": []
                }
            model_scores[model_name][category].append(score["bleu"])
    
    # Calculate average scores and create DataFrame
    leaderboard_data = []
    for model, categories in model_scores.items():
        # Calculate averages for each category
        high_avg = round(sum(categories["High-Resource"]) / len(categories["High-Resource"]), 3) if categories["High-Resource"] else 0
        mid_avg = round(sum(categories["Mid-Resource"]) / len(categories["Mid-Resource"]), 3) if categories["Mid-Resource"] else 0
        low_avg = round(sum(categories["Low-Resource"]) / len(categories["Low-Resource"]), 3) if categories["Low-Resource"] else 0
        
        # Calculate overall average
        all_scores = (categories["High-Resource"] + 
                     categories["Mid-Resource"] + 
                     categories["Low-Resource"])
        overall_avg = round(sum(all_scores) / len(all_scores), 3)
        
        leaderboard_data.append({
            "Model": model,
            "Overall BLEU": overall_avg,
            "High-Resource BLEU": high_avg,
            "Mid-Resource BLEU": mid_avg,
            "Low-Resource BLEU": low_avg,
            "Languages Tested": len(all_scores),
        })
    
    # Sort by overall BLEU
    df = pd.DataFrame(leaderboard_data)
    df = df.sort_values("Overall BLEU", ascending=False)
    
    # Add rank and medals
    df["Rank"] = range(1, len(df) + 1)
    df["Rank"] = df["Rank"].apply(
        lambda x: "🥇" if x == 1 else "🥈" if x == 2 else "🥉" if x == 3 else str(x)
    )
    
    # Reorder columns
    df = df[["Rank", "Model", "Overall BLEU", "High-Resource BLEU", 
             "Mid-Resource BLEU", "Low-Resource BLEU", "Languages Tested"]]
    
    return df


# Create the visualization components
with gr.Blocks(title="AI Language Translation Benchmark") as demo:
    gr.Markdown("# AI Language Translation Benchmark")
    gr.Markdown(
        "Comparing translation performance across different AI models and languages"
    )

    df = create_results_df(results)
    leaderboard_df = create_leaderboard_df(results)
    bar_plot = create_model_comparison_plot(results)
    scatter_plot = create_scatter_plot(results)

    gr.DataFrame(value=leaderboard_df, label="Model Leaderboard", show_search=False)
    gr.Plot(value=bar_plot, label="Model Comparison")
    gr.DataFrame(value=df, label="Language Results", show_search="search")
    gr.Plot(value=scatter_plot, label="Language Coverage")

demo.launch()
