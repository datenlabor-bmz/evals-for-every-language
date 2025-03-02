import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pycountry

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
    top_languages = sorted(results, key=lambda x: x["speakers"], reverse=True)[:10]
    scores_flat = [
        {"language": lang["language_name"], "model": score["model"], "bleu": score["bleu"]}
        for lang in top_languages
        for score in lang["scores"]
    ]
    df = pd.DataFrame(scores_flat)
    fig = px.bar(df, x="language", y="bleu", color="model", barmode="group")
    fig.update_layout(
        title="BLEU Scores by Model and Language",
        xaxis_title=None,
        yaxis_title="BLEU Score",
        barmode="group",
        height=500,
        legend=dict(
            orientation="h",  # horizontal orientation
            yanchor="bottom",
            y=-0.3,  # position below plot
            xanchor="center",
            x=0.5,  # center horizontally
        ),
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

        model = best_score["model"]
        model_name = model.split("/")[-1] if model else "N/A"
        model_link = (
            f"<a href='https://openrouter.ai/{model}' style='text-decoration: none; color: inherit;'>{model_name}</a>"
            if model
            else "N/A"
        )
        commonvoice_link = (
            f"<!--{lang['commonvoice_hours']:07} (for sorting)--> <a href='https://commonvoice.mozilla.org/{lang['commonvoice_locale']}/speak' style='text-decoration: none; color: inherit;'>🎙️ {lang['commonvoice_hours']}</a>"
            if lang["commonvoice_hours"]
            else "N/A"
        )
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
            "CommonVoice Hours": commonvoice_link,
        }
        flat_data.append(row)

    df = pd.DataFrame(flat_data)
    return gr.DataFrame(
        value=df,
        label="Language Results",
        show_search="search",
        datatype=[
            "markdown",
            "number",
            "number",
            "number",
            "markdown",
            "number",
            "markdown",
        ],
    )


def create_scatter_plot(results):
    fig = go.Figure()

    x_vals = [
        lang["speakers"] / 1_000_000 for lang in results if lang["speakers"] >= 10_000
    ]  # Convert to millions
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
        title=None,
        xaxis_title="Number of Speakers (Millions)",
        yaxis_title="Average BLEU Score",
        height=500,
        showlegend=False,
    )

    # Use log scale for x-axis since speaker numbers vary widely
    fig.update_xaxes(type="log")

    return gr.Plot(value=fig, label="Speaker population vs BLEU")


def format_number(n):
    """Format number with K/M suffix"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

def get_population_data():
    import xml.etree.ElementTree as ET
    from language_data.util import data_filename

    filename = data_filename("supplementalData.xml")
    root = ET.fromstring(open(filename).read())
    territories = root.findall("./territoryInfo/territory")

    data = {}
    for territory in territories:
        t_code = territory.attrib['type']
        t_population = float(territory.attrib['population'])
        data[t_code] = t_population
    return data

def create_world_map(results):
    # Collect all country data
    population_data = get_population_data()
    country_data = {}
    for lang in results:
        if "population" not in lang or lang["bleu"] is None:
            continue

        for country_code, speakers in lang["population"].items():
            try:
                # Convert alpha_2 (2-letter) to alpha_3 (3-letter) code
                country = pycountry.countries.get(alpha_2=country_code)
                if country is None:
                    continue

                iso3_code = country.alpha_3
                if iso3_code not in country_data:
                    country_data[iso3_code] = {
                        "total_speakers": 0,
                        "population": population_data.get(country_code, 0),
                        "weighted_bleu_sum": 0,
                        "languages": [],
                    }

                country_data[iso3_code]["total_speakers"] += speakers
                country_data[iso3_code]["weighted_bleu_sum"] += speakers * lang["bleu"]
                country_data[iso3_code]["languages"].append(
                    {
                        "name": lang["language_name"],
                        "speakers": speakers,
                        "bleu": lang["bleu"],
                    }
                )
            except (KeyError, AttributeError):
                # Skip invalid or unrecognized country codes
                continue

    # Calculate final weighted averages and prepare hover text
    countries = []
    bleu_scores = []
    hover_texts = []

    def make_black_bar(value, max_width=10):
        filled = int(value * max_width)
        return "⬛️" * filled + "⬜️" * (max_width - filled)

    def make_colored_bar(value, max_width=10):
        """Create a colored bar using Unicode blocks
        🟦 for high values (>0.35)
        🟨 for medium values (0.25-0.35)
        🟥 for low values (<0.25)
        ⬜ for empty space
        """
        filled = int(value * max_width)
        filled = max(0, min(filled, max_width))
        empty = max_width - filled

        if value > 0.35:
            return "🟦" * filled + "⬜" * empty
        elif value > 0.25:
            return "🟨" * filled + "⬜" * empty
        else:
            return "🟥" * filled + "⬜" * empty

    for country_code, data in country_data.items():
        weighted_avg = data["weighted_bleu_sum"] / data["total_speakers"]

        try:
            country_name = pycountry.countries.get(alpha_3=country_code).name
        except AttributeError:
            country_name = country_code

        # Sort languages by number of speakers
        langs = sorted(data["languages"], key=lambda x: x["speakers"], reverse=True)

        # Take top 5 languages and summarize the rest
        main_langs = langs[:5]
        other_langs = langs[5:]

        # Create language rows with bars
        lang_rows = []
        for lang in main_langs:
            percentage = (lang["speakers"] / data["population"]) * 100
            speaker_bar = make_black_bar(percentage / 100)
            bleu_bar = make_colored_bar((lang["bleu"] - 0.2) / 0.2)

            lang_rows.append(
                f"<b>{lang['name']}</b><br>"
                f"{speaker_bar} {format_number(lang['speakers'])} speakers<br>"
                f"{bleu_bar} {lang['bleu']:.3f} BLEU<br>"
            )

        # Add summary for other languages if any
        if other_langs:
            other_speakers = sum(lang["speakers"] for lang in other_langs)
            other_percentage = (other_speakers / data["population"]) * 100
            other_avg_bleu = sum(lang["bleu"] for lang in other_langs) / len(
                other_langs
            )

            speaker_bar = make_black_bar(other_percentage / 100)
            bleu_bar = make_colored_bar((other_avg_bleu - 0.2) / 0.2)

            lang_rows.append(
                f"<b>+{len(other_langs)} other languages</b><br>"
                f"{speaker_bar} {format_number(other_speakers)} speakers<br>"
                f"{bleu_bar} {other_avg_bleu:.3f} BLEU<br>"
            )

        hover_text = (
            f"<b>{country_name}</b><br><br>"
            f"{'<br>'.join(lang_rows)}"
        )

        countries.append(country_code)
        bleu_scores.append(weighted_avg)
        hover_texts.append(hover_text)

    # Create the choropleth map
    fig = go.Figure(
        data=go.Choropleth(
            locations=countries,
            locationmode="ISO-3",
            z=bleu_scores,
            text=hover_texts,
            hoverinfo="text",
            colorscale=[[0, "#ff9999"], [1, "#99ccff"]],
            colorbar=dict(
                title="BLEU Score",
                orientation="h",  # horizontal orientation
                y=-0.2,  # position below map
                yanchor="bottom",
                len=0.5,  # length of colorbar
                x=0.5,  # center horizontally
                xanchor="center",
                thickness=20,  # make it a bit thicker when horizontal
            ),
            zmin=0.1,
            zmax=0.5,
        )
    )

    fig.update_layout(
        title=dict(text="BLEU Score by Country", x=0.5, xanchor="center"),
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type="equal earth",
            showland=True,
            landcolor="#f8f9fa",
            coastlinecolor="#e0e0e0",
            countrycolor="#e0e0e0",
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="white",
        hoverlabel=dict(
            bgcolor="beige",
            font_size=12,
        ),
    )

    return fig


# Create the visualization components
with gr.Blocks(title="AI Language Translation Benchmark") as demo:
    gr.Markdown("# AI Language Translation Benchmark")
    gr.Markdown(
        "Comparing translation performance across different AI models and languages"
    )

    bar_plot = create_model_comparison_plot(results)
    world_map = create_world_map(results)

    create_leaderboard_df(results)
    gr.Plot(value=bar_plot, label="Model Comparison")
    create_language_stats_df(results)
    create_scatter_plot(results)
    gr.Plot(value=world_map, container=False, elem_classes="fullwidth-plot")

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
