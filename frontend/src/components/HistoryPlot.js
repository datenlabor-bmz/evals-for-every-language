import { useRef, useEffect } from 'react'
import * as Plot from '@observablehq/plot'

const HistoryPlot = ({ data }) => {
  const containerRef = useRef()
  const models = [...data.model_table] // sort copy, not in place
    .filter(d => d.average !== null)
    .sort((a, b) => new Date(a.creation_date) - new Date(b.creation_date))
    .reduce((acc, curr) => {
      const last = acc[acc.length - 1]?.maxAverage || 0
      acc.push({
        ...curr,
        maxAverage: Math.max(last, curr.average),
        newRecord: curr.average > last
      })
      return acc
    }, [])
  console.log(models)
  useEffect(() => {
    const plot = Plot.plot({
      width: 750,
      height: 500,
      subtitle: 'Model performance over time',
      x: {
        label: 'Date',
        type: 'time',
        tickFormat: '%Y-%m'
      },
      y: {
        label: 'Language Proficiency Score'
      },
      symbol: {
        legend: true
      },
      marks: [
        Plot.dot(models, {
          x: d => d.creation_date,
          y: d => d.average,
          symbol: "provider_name",
          stroke: "provider_name",
          title: d =>
            `${d.provider_name} - ${d.name} (${
              d.size?.toLocaleString('en-US', { notation: 'compact' }) || '?B'
            })\nPublished: ${d.creation_date}\nScore: ${d.average.toFixed(2)}`,
          tip: true
        }),
        Plot.line(
          [
            ...models.filter(d => d.newRecord),
            {
              creation_date: new Date(),
              maxAverage: models[models.length - 1].maxAverage
            }
          ],
          {
            x: d => d.creation_date,
            y: d => d.maxAverage,
            curve: 'step-after',
            strokeOpacity: 0.3
          }
        )
      ]
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
        justifyContent: 'center'
      }}
    />
  )
}

export default HistoryPlot
