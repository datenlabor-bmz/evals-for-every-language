---
theme: dashboard
title: List
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

// Get unique languages with their speaker counts
const languageMap = new Map();
data.forEach(r => {
    if (!languageMap.has(r.language_name)) {
        languageMap.set(r.language_name, r.speakers);
    }
});

// Sort languages by speaker count (descending)
const languages = [...languageMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([lang]) => lang);

// Section for each language
languages.forEach(language => {
    display(html`<h2 class="language-header">${language}</h2>`)

    const speakerCount = (languageMap.get(language) / 1_000_000).toFixed(1);
    display(html`${speakerCount}M speakers`);

    const languageData = data.filter(r => r.language_name === language)[0]["scores"];
    console.log(languageData)

    const descriptor = code => {
        let [org, model] = code.split("/")
        return model.split("-")[0]
    }

    // Plot for how well the models perform on this language
    if (languageData && languageData.length >= 1) {
        console.log("yes")
        const chart = Plot.plot({
            width: 400,
            height: 200,
            margin: 30,
            y: {
                domain: [0, 1],
                label: scoreName
            },
            marks: [
                Plot.barY(languageData, {
                    x: d => descriptor(d.model),
                    y: scoreKey
                })
            ]
        });
        display(chart)
    }
});
```