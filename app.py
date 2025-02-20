import gradio as gr
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            model_score = next((s["bleu"] for s in lang["scores"] if s["model"] == model), None)
            if model_score is not None:
                x_vals.append(lang["language_name"])
                y_vals.append(model_score)
        
        traces.append(go.Bar(
            name=model.split('/')[-1],
            x=x_vals,
            y=y_vals,
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="BLEU Scores by Model and Language",
        xaxis_title="Language",
        yaxis_title="BLEU Score",
        barmode='group',
        height=500
    )
    return fig

def create_scatter_plot(results):
    fig = go.Figure()
    
    x_vals = [lang["speakers"] / 1_000_000 for lang in results]  # Convert to millions
    y_vals = [lang["bleu"] for lang in results]
    labels = [lang["language_name"] for lang in results]
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        text=labels,
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>" +
                      "Speakers: %{x:.1f}M<br>" +
                      "BLEU Score: %{y:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Language Coverage: Speakers vs BLEU Score",
        xaxis_title="Number of Speakers (Millions)",
        yaxis_title="Average BLEU Score",
        height=500,
        showlegend=False
    )
    
    # Use log scale for x-axis since speaker numbers vary widely
    fig.update_xaxes(type="log")
    
    return fig

def create_results_df(results):
    # Create a list to store flattened data
    flat_data = []
    
    for lang in results:
        row = {
            "Language": lang["language_name"],
            "Speakers (M)": round(lang["speakers"] / 1_000_000, 1),
            "Average BLEU": round(lang["bleu"], 3),
        }
        # Add individual model scores
        for score in lang["scores"]:
            model_name = score["model"].split('/')[-1]
            row[f"{model_name} BLEU"] = round(score["bleu"], 3)
        
        flat_data.append(row)
    
    return pd.DataFrame(flat_data)

# Create the visualization components
with gr.Blocks(title="AI Language Translation Benchmark") as demo:
    gr.Markdown("# AI Language Translation Benchmark")
    gr.Markdown("Comparing translation performance across different AI models and languages")
    
    df = create_results_df(results)
    bar_plot = create_model_comparison_plot(results)
    scatter_plot = create_scatter_plot(results)
    
    gr.DataFrame(value=df, label="Translation Results", show_search="search")
    gr.Plot(value=bar_plot, label="Model Comparison")
    gr.Plot(value=scatter_plot, label="Language Coverage")

demo.launch()