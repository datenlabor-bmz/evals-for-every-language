import { useRef, useEffect } from 'react'
import * as Plot from '@observablehq/plot'

const SpeakerPlot = ({ data }) => {
  const containerRef = useRef()
  const allSpeakers = data.language_table.reduce((sum, curr) => sum + curr.speakers, 0)
  const languages = data.language_table.sort((a, b) => b.speakers - a.speakers).slice(0, 100).reduce((acc, d) => {
    acc.push({
      ...d,
      rank: acc.length + 1,
      cumSpeakers: acc.reduce((sum, curr) => sum + curr.speakers, 0) + d.speakers,
      cumSpeakersPercent: (acc.reduce((sum, curr) => sum + curr.speakers, 0) + d.speakers) / allSpeakers
    })
    return acc
  }, [])

  useEffect(() => {
    const plot = Plot.plot({
      width: 750,
      height: 500,
      // title: 'Proficiency of Languages by Number of Speakers',
      x: {
        label: 'Languages',
        ticks: [],
      },
      y: {
        label: 'Number of Speakers (millions)',
      },
      color: {
        legend: true,
        domain: ["Speakers", "Cumulative Speakers"],
        range: ["green", "lightgrey"],
      },
      marks: [
        Plot.barY(languages,
          {
          x: "rank",
          y: d => d.cumSpeakers / 1e6,
          fill: d => "Cumulative Speakers",
          sort: { x: 'y' },
          title: d => `The ${d.rank} most spoken languages cover\n${d.cumSpeakersPercent.toLocaleString("en-US", { style: 'percent'})} of all speakers`,
          tip: true // {y: d => d.cumSpeakers / 1e6 * 2}
        }),
        Plot.barY(languages,
          {
          x: "rank",
          y: d => d.speakers / 1e6,
          title: d => `${d.language_name}\n(${d.speakers.toLocaleString("en-US", {notation: 'compact', compactDisplay: 'long'})} speakers)`,
          tip: true,
          fill: d => "Speakers",
          sort: { x: '-y' }
        }),
        Plot.crosshairX(languages, {x: "rank", y: d => d.cumSpeakers / 1e6, textStrokeOpacity: 0, textFillOpacity: 0}),
        Plot.tip(["The 41 most spoken languages cover 80% of all speakers."], {x: 41, y: languages[40].cumSpeakers / 1e6})
      ],
    })
    containerRef.current.append(plot)
    return () => plot.remove()
  }, [])

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    />
  )
}

export default SpeakerPlot
