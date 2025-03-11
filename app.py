import json
from functools import partial

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
from gradio_rangeslider import RangeSlider

with open("results.json") as f:
    languages = json.load(f) 

languages_with_scores = [lang for lang in languages if lang["t2t_score"] is not None]

# Global constants for metric mappings
METRICS = {
    "t2t": [
        {
            "display_name": "Overall Text-to-Text Performance",
            "field_name": "t2t_score",
            "label": "Overall Score",
            "explanation": """
    **Overall Score for Text-to-Text Performance**: A weighted combination of all metrics, providing a holistic view of model performance across different language tasks. 
    Higher scores indicate better overall language capabilities.
    """,
        },
        {
            "display_name": "Translation (BLEU)",
            "field_name": "mt_bleu",
            "label": "BLEU Score",
            "explanation": """
    **Translation BLEU**: BiLingual Evaluation Understudy (BLEU) measures how similar AI-generated translations are to human reference translations.
    It calculates n-gram precision and applies a brevity penalty. Scores range from 0 to 1, with higher values indicating better translation quality.
    """,
        },
        {
            "display_name": "Translation (ChrF)",
            "field_name": "mt_chrf",
            "label": "ChrF Score",
            "explanation": """
    **Translation ChrF**: Character n-gram F-score evaluates translations at the character level rather than word level.
    This metric is particularly valuable for morphologically rich languages and can better capture partial word matches.
    Higher scores (0-1) indicate better translations.
    """,
        },
        {
            "display_name": "Classification (Accuracy)",
            "field_name": "cls_acc",
            "label": "Classification Accuracy",
            "explanation": """
    **Classification Accuracy**: Measures how accurately models can classify text into predefined categories.
    This evaluates a model's understanding of content and context across different languages.
    Reported as a percentage where higher values indicate better classification performance.
    """,
        },
        {
            "display_name": "Masked Language Modeling (ChrF)",
            "field_name": "mlm_chrf",
            "label": "MLM ChrF Score",
            "explanation": """
    **Masked Language Modeling ChrF**: Evaluates how well models can predict masked (hidden) portions of text.
    This tests a model's understanding of language structure and semantics by measuring the character-level similarity
    between predicted and actual text. Higher scores indicate better language understanding.
    """,
        },
    ],
    "s2t": [
        {
            "display_name": "Overall Speech-to-Text Performance",
            "field_name": "s2t_score",
            "label": "Overall Score",
            "explanation": """
    **Overall Score for Speech-to-Text Performance**: A weighted combination of all metrics, providing a holistic view of model performance across different language tasks. 
    Higher scores indicate better overall language capabilities.
    """,
        },
        {
            "display_name": "Automatic Speech Recognition (WER)",
            "field_name": "asr_wer",
            "label": "WER",
            "explanation": """ 
    **Automatic Speech Recognition Word Error Rate**: Measures the accuracy of speech-to-text transcription.
    It calculates the minimum number of word edits (insertions, deletions, substitutions) needed to transform the 
    transcription into the reference text, divided by the number of words in the reference.
    Lower scores indicate better performance, with 0 being perfect transcription.
    """,
        },
        {
            "display_name": "Automatic Speech Recognition (ChrF)",
            "field_name": "asr_chrf",
            "label": "ChrF",
            "explanation": """
    **Automatic Speech Recognition ChrF**: Character n-gram F-score evaluates translations at the character level rather than word level.
    This metric is particularly valuable for morphologically rich languages and can better capture partial word matches.
    Higher scores (0-1) indicate better translations.
    """,
        },
    ],
}


def mean(lst):
    return sum(lst) / len(lst)


def create_leaderboard_df(model_type, metric=None):
    metric = metric or METRICS[model_type][0]
    _model_type = {"t2t": "text-to-text", "s2t": "speech-to-text"}[model_type]
    models = {
        score["model"]
        for lang in languages_with_scores
        for score in lang["scores"]
        if score["model_type"] == _model_type
    }
    model_scores = [
        {"model": score["model"], metric["field_name"]: score[metric["field_name"]]}
        for lang in languages_with_scores
        for score in lang["scores"]
        for model in models
        if score["model"] == model
    ]
    df = (
        pd.DataFrame(model_scores)
        .groupby("model")
        .agg({metric["field_name"]: ["mean", "count"]})
        .reset_index()
    )
    # Flatten the multi-level column names
    df.columns = df.columns.map(
        lambda x: f"{x[0]}_{x[1]}" if isinstance(x, tuple) else x
    )
    df = df.rename(
        columns={
            f"{metric['field_name']}_mean": metric["label"],
            f"{metric['field_name']}_count": "Languages Tested",
            "model_": "Model",
        }
    )
    df[metric["label"]] = df[metric["label"]].round(3)
    df = df.sort_values(metric["label"], ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    df["Rank"] = df["Rank"].apply(
        lambda x: "🥇" if x == 1 else "🥈" if x == 2 else "🥉" if x == 3 else str(x)
    )
    df = df[["Rank", "Model", metric["label"]]]
    return gr.DataFrame(
        value=df,
        label="Model Leaderboard",
        show_search=False,
        datatype=["number", "markdown", "number"],
    )


def create_model_comparison_plot(metric):
    top_languages = sorted(
        languages_with_scores, key=lambda x: x["speakers"], reverse=True
    )[:10]

    # Create appropriate title and y-axis label based on metric
    title = f"{metric['display_name']} by Model and Language"
    y_label = metric["label"]

    # Flatten the data for the selected metric
    scores_flat = []
    for lang in top_languages:
        for score in lang["scores"]:
            # Get the value directly using the field name
            if metric["field_name"] not in score:
                continue
            value = score[metric["field_name"]]
            if value is not None:
                scores_flat.append(
                    {
                        "language": lang["language_name"],
                        "model": score["model"],
                        "value": value,
                    }
                )

    df = pd.DataFrame(scores_flat)
    fig = px.bar(df, x="language", y="value", color="model", barmode="group")
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_label,
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


def create_language_stats_df(metric):
    # Create a list to store flattened data
    flat_data = []

    for lang in languages:
        # Find the best model and its BLEU score
        best_model = (
            max(
                lang["scores"] or [{"t2t_score": None, "model": None}],
                key=lambda x: x.get("t2t_score", 0),
            )
            if lang["t2t_score"] is not None
            else None
        )

        model = best_model["model"] if best_model else None
        model_name = model.split("/")[-1] if model else "N/A"
        model_link = (
            f"<a href='https://openrouter.ai/{model}' style='text-decoration: none; color: inherit;'>{model_name}</a>"
            if model
            else "N/A"
        )
        commonvoice_link = (
            f"<!--{lang['commonvoice_hours']:07} (for sorting)--> <a href='https://commonvoice.mozilla.org/{lang['commonvoice_locale']}/speak' style='text-decoration: none; color: inherit;'>🎙️ {round(lang['commonvoice_hours'])}h</a>"
            if lang["commonvoice_hours"]
            else "N/A"
        )
        row = {
            "Language": f"**{lang['language_name']}**",
            "Speakers (M)": round(lang["speakers"] / 1_000_000, 1),
            # "Models Tested": len(lang["scores"]),
            # "Overall": round(lang["overall_score"], 3)
            # if lang["overall_score"] is not None
            # else "N/A",
            "Best Model": model_link,
            "Trans­la­ti­on": round(lang["mt_chrf"], 3)
            if lang["mt_chrf"] is not None
            else "N/A",
            "Classi­fi­ca­ti­on": round(lang["cls_acc"], 3)
            if lang["cls_acc"] is not None
            else "N/A",
            "Masked Language Modeling": round(lang["mlm_chrf"], 3)
            if lang["mlm_chrf"] is not None
            else "N/A",
            "Speech Reco­gni­ti­on": round(lang["asr_chrf"], 3) if lang["asr_wer"] is not None else "N/A",
            "Common­Voice": commonvoice_link,
        }
        flat_data.append(row)

    df = pd.DataFrame(flat_data)
    return gr.DataFrame(
        value=df,
        label="Language Results",
        show_search="search",
        pinned_columns=1,
        column_widths=[
            "100px",
            "100px",
            # "100px",
            # "100px",
            "200px",  # Best Model
            "100px",  # MT
            "100px",  # CLS
            "100px",  # MLM
            "100px",  # ASR
            "100px",  # Common Voice
        ],
        datatype=[
            "markdown",  # Language
            "number",  # Speakers
            # "number", # Models Tested
            # "number",  # Overall
            "markdown",  # Best Model
            "number",  # Translation
            "number",  # Classification
            "number",  # MLM
            "number",  # ASR
            "markdown",  # CommonVoice Hours
        ],
    )


def create_scatter_plot(metric):
    # Create a list to store data for the scatter plot
    scatter_data = []
    for lang in languages_with_scores:
        if lang["speakers"] < 100_000:
            continue
        # Calculate average score for this metric across all models
        scores = [
            score[metric["field_name"]]
            for score in lang["scores"]
            if metric["field_name"] in score and score[metric["field_name"]] is not None
        ]
        if scores:  # Only include if we have valid scores
            avg_score = sum(scores) / len(scores)
            scatter_data.append(
                {
                    "language": lang["language_name"],
                    "speakers": lang["speakers"],
                    "score": avg_score,
                    "family": lang["language_family"],
                }
            )

    fig = go.Figure()
    x_vals = [data["speakers"] / 1_000_000 for data in scatter_data]
    y_vals = [data["score"] for data in scatter_data]
    s_vals = [data["speakers"] / 20_000_000 for data in scatter_data]
    color_pallette = [
        "LightSkyBlue",
        "LightGreen",
        "LightCoral",
        "LightPink",
        "LightGoldenRodYellow",
        "LightGray",
        "LightSalmon",
        "LightSeaGreen",
    ]
    color_mapping = {
        family: color
        for family, color in zip(
            sorted(set(data["family"] for data in scatter_data)), color_pallette
        )
    }
    c_vals = [color_mapping.get(data["family"], "LightGray") for data in scatter_data]
    labels = [data["language"] for data in scatter_data]
    hover_template = f"<b>%{{text}}</b><br>Speakers: %{{x:.1f}}M<br>{metric['label']}: %{{y:.3f}}<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            marker=dict(size=s_vals, color=c_vals),
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertemplate=hover_template,
        )
    )
    fig.update_layout(
        title=None,
        xaxis_title="Number of Speakers (Millions)",
        yaxis_title=metric["label"],
        height=500,
        showlegend=False,
    )
    fig.update_xaxes(type="log")
    return fig


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
        t_code = territory.attrib["type"]
        t_population = float(territory.attrib["population"])
        data[t_code] = t_population
    return data


# Helper functions for visualization
def make_black_bar(value, max_width=10):
    filled = int(value * max_width)
    return "⬛️" * filled + "⬜️" * (max_width - filled)


def make_colored_bar(score, max_width=10):
    """Create a colored bar using Unicode blocks based on normalized score
    🟦 for high values (>0.35)
    🟨 for medium values (0.25-0.35)
    🟥 for low values (<0.25)
    ⬜ for empty space

    This function handles both normalization and bar creation.
    """

    # Create the bar based on normalized value
    filled = int(score * max_width)
    filled = max(0, min(filled, max_width))
    empty = max_width - filled

    if score > 0.35:
        return "🟦" * filled + "⬜" * empty
    elif score > 0.25:
        return "🟨" * filled + "⬜" * empty
    else:
        return "🟥" * filled + "⬜" * empty


def create_world_map(metric):
    # Collect all country data
    population_data = get_population_data()
    country_data = {}
    for lang in languages:
        # Skip languages without the required data
        if "population" not in lang or lang[metric["field_name"]] is None:
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
                        "weighted_score_sum": 0,
                        "languages": [],
                    }

                country_data[iso3_code]["total_speakers"] += speakers
                country_data[iso3_code]["weighted_score_sum"] += (
                    speakers * lang[metric["field_name"]]
                )
                country_data[iso3_code]["languages"].append(
                    {
                        "name": lang["language_name"],
                        "speakers": speakers,
                        "score": lang[metric["field_name"]],
                    }
                )
            except (KeyError, AttributeError):
                # Skip invalid or unrecognized country codes
                continue

    # Calculate final weighted averages and prepare hover text
    countries = []
    scores = []
    hover_texts = []

    for country_code, data in country_data.items():
        weighted_avg = data["weighted_score_sum"] / data["total_speakers"] if data["total_speakers"] > 0 else None

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

            # Use the integrated make_colored_bar function directly
            score_bar = make_colored_bar(lang["score"])

            lang_rows.append(
                f"<b>{lang['name']}</b><br>"
                f"{speaker_bar} {format_number(lang['speakers'])} speakers<br>"
                f"{score_bar} {lang['score']:.3f} {metric['label']}<br>"
            )

        # Add summary for other languages if any
        if other_langs:
            other_speakers = sum(lang["speakers"] for lang in other_langs)
            other_percentage = (other_speakers / data["population"]) * 100
            other_avg_score = sum(lang["score"] for lang in other_langs) / len(
                other_langs
            )

            speaker_bar = make_black_bar(other_percentage / 100)

            # Use the integrated make_colored_bar function directly
            score_bar = make_colored_bar(other_avg_score)

            lang_rows.append(
                f"<b>+{len(other_langs)} other languages</b><br>"
                f"{speaker_bar} {format_number(other_speakers)} speakers<br>"
                f"{score_bar} {other_avg_score:.3f} {metric['label']}<br>"
            )

        hover_text = f"<b>{country_name}</b><br><br>" f"{'<br>'.join(lang_rows)}"

        countries.append(country_code)
        scores.append(weighted_avg)
        hover_texts.append(hover_text)

    fig = go.Figure(
        data=go.Choropleth(
            locations=countries,
            locationmode="ISO-3",
            z=scores,
            text=hover_texts,
            hoverinfo="text",
            colorscale=[[0, "#ff9999"], [1, "#99ccff"]],
            colorbar=dict(
                title=metric["label"],
                orientation="h",  # horizontal orientation
                y=-0.2,  # position below map
                yanchor="bottom",
                len=0.5,  # length of colorbar
                x=0.5,  # center horizontally
                xanchor="center",
                thickness=20,  # make it a bit thicker when horizontal
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{metric['display_name']} by Country", x=0.5, xanchor="center"
        ),
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


def create_metric_selector(model_type):
    match model_type:
        case "t2t":
            choices = [m["display_name"] for m in METRICS["t2t"]]
        case "s2t":
            choices = [m["display_name"] for m in METRICS["s2t"]]
    return gr.Dropdown(
        choices=choices, value=choices[0], label="Select Metric", interactive=True
    )


def create_metric_explanation(metric):
    return gr.Markdown(metric["explanation"], container=True)

css="""
.radio-group .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
}
"""

# Create the visualization components
with gr.Blocks(title="AI Language Proficiency Benchmark", css=css) as demo:
    gr.Markdown("# AI Language Proficiency Benchmark")
    gr.Markdown("Comparing language proficiency across different models and languages.")

    language_choices = [
        f"{lang['language_name']} ({lang['bcp_47']})" for lang in languages
    ]
    models = {score["model"] for lang in languages for score in lang["scores"]}
    search = gr.Dropdown(
        choices=list(models) + language_choices,
        value=None,
        label="Search for Language or Model",
        interactive=True,
    )
    with gr.Row():
        start_model_type = "Text-to-Text"
        model_type = gr.Radio(
            choices=["Text-to-Text", "Speech-to-Text"],
            value=start_model_type,
            label="Select Model Type",
            interactive=True,
            elem_classes="radio-group",
        )
        start_metric = METRICS["t2t"][0]
        metric = gr.Dropdown(
            choices=[metric["display_name"] for metric in METRICS["t2t"]],
            value=start_metric["display_name"],
            label="Main task and metric to display in figures and map",
            interactive=True,
        )
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Model Filters", open=False):
                model_licenses = gr.CheckboxGroup(
                    choices=["open source", "commercial"],
                    value=["open source", "commercial"],
                    label="Filter by Model License",
                    interactive=True,
                )
                model_sizes = RangeSlider(
                    minimum=0,
                    maximum=1000,
                    value=(0, 1000),
                    label="Filter by Model Size (in Billion Parameters)",
                    interactive=True,
                )
        with gr.Column():
            with gr.Accordion("Language Filters", open=False):
                unit_of_analysis = gr.Radio(
                    choices=["Languages", "Language Families", "Regions"],
                    value="Languages",
                    label="Select Unit of Analysis",
                    interactive=True,
                )
                family_filter = gr.CheckboxGroup(
                    choices=[
                        "Indo-European",
                        "Sino-Tibetan",
                        "Afro-Asiatic",
                        "Dravidian",
                        "Uralic",
                        "Austronesian",
                        "Other",
                    ],
                    value=[
                        "Indo-European",
                        "Sino-Tibetan",
                        "Afro-Asiatic",
                        "Dravidian",
                        "Uralic",
                        "Austronesian",
                        "Other",
                    ],
                    label="Filter by Language Family",
                    interactive=True,
                )
                speakers_filter = RangeSlider(
                    minimum=0,
                    maximum=100_000_000,
                    value=(0, 100_000_000),
                    label="Filter by Number of Speakers",
                    interactive=True,
                )

    gr.Markdown("## Model Comparison")
    leaderboard_df = create_leaderboard_df("t2t", start_metric)

    model_comparison_plot = gr.Plot(
        value=create_model_comparison_plot(start_metric),
        label="Model Comparison",
    )

    gr.Markdown("## Language Stats")
    create_language_stats_df(start_metric)
    scatter_plot = gr.Plot(
        value=create_scatter_plot(start_metric),
        label="Speaker Population vs. Metric",
    )
    world_map = gr.Plot(
        value=create_world_map(start_metric),
        label="World Map",
        container=False,
        elem_classes="fullwidth-plot",
    )

    def update_model_type(model_type_choice):
        model_type = {"Text-to-Text": "t2t", "Speech-to-Text": "s2t"}[model_type_choice]
        return create_metric_selector(model_type), create_leaderboard_df(model_type)

    model_type.change(
        fn=update_model_type,
        inputs=model_type,
        outputs=[metric, leaderboard_df],
    )

    def update_component(fn, model_type_choice, metric_choice):
        model_type = {"Text-to-Text": "t2t", "Speech-to-Text": "s2t"}[model_type_choice]
        metric = [m for m in METRICS[model_type] if m["display_name"] == metric_choice][
            0
        ]
        return fn(metric)
    metric.change(
        fn=partial(update_component, create_model_comparison_plot),
        inputs=[model_type, metric],
        outputs=model_comparison_plot,
    )
    metric.change(
        fn=partial(update_component, create_scatter_plot),
        inputs=[model_type, metric],
        outputs=scatter_plot,
    )
    metric.change(
        fn=partial(update_component, create_world_map),
        inputs=[model_type, metric],
        outputs=world_map,
    )

    with gr.Accordion("Methodology", open=False):
        gr.Markdown(
            """
            ### Benchmark Data
            We use the [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) dataset for evaluation, which contains parallel text in over 200 languages, as well as topic labels for each sentence. Where FLORES+ includes multiple scripts for one language, we use only the most common one.

            Population and speaker data and language code resolution are from Unicode [CLDR](https://github.com/unicode-org/cldr) via the [langcodes](https://github.com/rspeer/langcodes) package.

            ### AI Models
            We use [OpenRouter](https://openrouter.ai/) to access all relevant AI models via a unified API.

            ### Evaluation Tasks
            Our benchmark includes three core tasks to assess different aspects of language understanding:

            1. **Machine Translation**: Models translate text _from_ the evaluated language _to_ a fixed set of target languages. The set of target languages is representative of global speaker populations. Performance is measured using:
            - [BLEU Score](https://huggingface.co/metrics/bleu): Measures n-gram precision with a brevity penalty
            - [ChrF Score](https://huggingface.co/metrics/chrf): Character-level F-score that better captures morphological variations

            2. **Text Classification**: Models classify text into predefined topics after being shown examples. We:
            - Group sentences by URL into paragraphs with the same topic
            - Use the 5 most common topics, encoded as numbers rather than English labels
            - Provide 5 examples of each topic as few-shot examples
            - Test the model's ability to classify new text
            - Report accuracy as the primary metric

            3. **Masked Language Modeling**: Models predict missing portions of text (marked with `<mask>`). We:
            - Mask approximately 5% of each sentence at a random position
            - Provide 10 examples of complete sentences paired with masked versions in a few-shot setting
            - Evaluate predictions using ChrF score against the original text

            The overall performance score combines metrics from all tasks to provide a holistic assessment of model capabilities across languages.
        """
        )

demo.launch()
