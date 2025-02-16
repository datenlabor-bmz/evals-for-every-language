---
toc: false
---

<div class="hero">
  <h1>AI Language Monitor</h1>
  <h2>Benchmarking all big AI models on all benchmarkable languages.</h2>
</div>

```js
import { languageChart } from "./components/language-chart.js";

const data = FileAttachment("data/languagebench.json").json();
```


<div class="grid grid-cols-2" style="grid-auto-rows: 504px;">
  <div class="card">
    <h2 class="hero">Compare languages</h2>
    ${resize((width) => languageChart(data, {width: 1000, height: 400, scoreKey: "bleu", scoreName: "BLEU Score"}))}
  </div>
  <div class="card">
    <h2 class="hero">Compare AI models</h2>
    ...
  </div>
</div>

<style>

.hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: var(--sans-serif);
  margin: 4rem 0 8rem;
  text-wrap: balance;
  text-align: center;
}

.hero h1 {
  margin: 1rem 0;
  padding: 1rem 0;
  max-width: none;
  font-size: 90px;
  font-weight: 900;
  line-height: 1;
  background: linear-gradient(30deg, var(--theme-foreground-focus), currentColor);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

</style>
