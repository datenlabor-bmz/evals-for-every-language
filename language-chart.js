import * as Plot from "npm:@observablehq/plot";

export function languageChart(
  languageData,
  { width, height, scoreKey, scoreName } = {}
) {
  // Format captions
  const formatScore = (score) =>
    score > 0 ? score.toFixed(2) : "No benchmark available!";
  const formatTitle = (d) =>
    d.language_name +
    "\n" +
    parseInt(d.speakers / 1_000_00) / 10 +
    "M speakers\n" +
    scoreName +
    ": " +
    formatScore(d[scoreKey]);

  return Plot.plot({
    width: width,
    height: height,
    marginBottom: 100,
    x: { label: "Number of speakers", axis: null },
    y: { label: `${scoreName} (average across models)` },
    // color: { scheme: "BrBG" },
    marks: [
      Plot.rectY(
        languageData,
        Plot.stackX({
          x: "speakers",
          order: scoreKey,
          reverse: true,
          y2: scoreKey, // y2 to avoid stacking by y
          title: formatTitle,
          tip: true,
          fill: (d) => (d[scoreKey] > 0 ? "black" : "pink"),
        })
      ),
      Plot.rectY(
        languageData,
        Plot.pointerX(
          Plot.stackX({
            x: "speakers",
            order: scoreKey,
            reverse: true,
            y2: scoreKey, // y2 to avoid stacking by y
            fill: "grey",
          })
        )
      ),
      Plot.text(
        languageData,
        Plot.stackX({
          x: "speakers",
          y2: scoreKey,
          order: scoreKey,
          reverse: true,
          text: "language_name",
          frameAnchor: "bottom",
          textAnchor: "end",
          dy: 10,
          rotate: 270,
          opacity: (d) => (d.speakers > 50_000_000 ? 1 : 0),
        })
      ),
    ],
  });
}
