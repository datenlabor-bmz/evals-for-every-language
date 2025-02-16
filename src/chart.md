---
theme: dashboard
title: AI Language Monitor
toc: false
head: <link rel="icon"
  href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22 fill=%22black%22>🌍</text></svg>">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  <!--<link rel="stylesheet" href="styles.css">-->
  <script src="https://cdn.tailwindcss.com"></script>
---

# AI Language Monitor

```js
const data = FileAttachment("data/languagebench.json").json();
```

```js
const scoreKey = "bleu"
const scoreName = "BLEU Score"

// Format captions
const formatScore = (score) => score > 0 ? score.toFixed(2) : "No benchmark available!"
const formatTitle = d => (d.language_name + "\n" + parseInt(d.speakers / 1_000_00) / 10 + "M speakers\n" + scoreName + ": " + formatScore(d[scoreKey]))

// Create summary plot
const chart = Plot.plot({
    width: 600,
    height: 400,
    marginBottom: 100,
    x: { label: "Number of speakers", axis: null },
    y: { label: `${scoreName} (average across models)` },
    // color: { scheme: "BrBG" },
    marks: [
        Plot.rectY(data, Plot.stackX({
            x: "speakers",
            order: scoreKey,
            reverse: true,
            y2: scoreKey, // y2 to avoid stacking by y
            title: formatTitle,
            tip: true,
            fill: d => d[scoreKey] > 0 ? "black" : "pink"
        })),
        Plot.rectY(data, Plot.pointerX(Plot.stackX({
            x: "speakers",
            order: scoreKey,
            reverse: true,
            y2: scoreKey, // y2 to avoid stacking by y
            fill: "grey",
        }))),
        Plot.text(data, Plot.stackX({
            x: "speakers",
            y2: scoreKey,
            order: scoreKey,
            reverse: true,
            text: "language_name",
            frameAnchor: "bottom",
            textAnchor: "end",
            dy: 10,
            rotate: 270,
            opacity: (d) => d.speakers > 50_000_000 ? 1 : 0,
        }))
    ]
});
display(chart);
```

<style>
    body {
        margin: 0 auto;
        padding: 20px;
        font-family: sans-serif;
    }

    .language-header {
        margin-bottom: 10px;
    }

    .speaker-count {
        font-size: 0.8em;
        color: #666;
        font-weight: normal;
        margin: 0;
    }
</style>
