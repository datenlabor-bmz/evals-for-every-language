---
theme: dashboard
title: Compare languages
---

# Compare languages

```js
import { languageChart } from "./components/language-chart.js";

const data = FileAttachment("data/languagebench.json").json();
```

```js
const scoreKey = "bleu"
const scoreName = "BLEU Score"

// Create summary plot
display(languageChart(data, {width: 1000, height: 400, scoreKey: scoreKey, scoreName: scoreName}))
```
