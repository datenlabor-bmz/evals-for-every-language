import { useRef, useEffect } from 'react'
import * as Plot from '@observablehq/plot'

const CostPlot = ({ data, width = 750, height = 500 }) => {
  const containerRef = useRef()
  useEffect(() => {
    const models = [...data.model_table] // sort copy, not in place
      .filter(d => d.average !== null && d.cost > 0)
      .sort((a, b) => a.cost - b.cost)
      .reduce((acc, curr) => {
        const last = acc[acc.length - 1]?.maxAverage || 0
        acc.push({
          ...curr,
          maxAverage: Math.max(last, curr.average),
          newRecord: curr.average > last
        })
        return acc
      }, [])
    let USDollar = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    })
    const plot = Plot.plot({
      width: width,
      height: height,
      subtitle: 'Cost vs Performance',
      x: {
        label: 'Cost (USD)',
        type: 'log',
        // format dollar / ct
        tickFormat: d => USDollar.format(d)
      },
      y: {
        label: 'Language Proficiency Score'
      },
      symbol: {
        legend: true
      },
      marks: [
        Plot.dot(models, {
          x: d => d.cost,
          y: d => d.average,
          symbol: 'provider_name',
          stroke: 'provider_name',
          title: d =>
            `${d.provider_name} - ${d.name} (${
              d.size?.toLocaleString('en-US', { notation: 'compact' }) || '?B'
            })\nCost: ${USDollar.format(d.cost)}\nScore: ${d.average.toFixed(
              2
            )}`,
          tip: true
        }),
        Plot.line(
          [
            ...models.filter(d => d.newRecord),
            {
              cost: models.map(d => d.cost).reduce((a, b) => Math.max(a, b), 0),
              maxAverage: models[models.length - 1]?.maxAverage || 0
            }
          ],
          {
            x: d => d.cost,
            y: d => d?.maxAverage || 0,
            curve: 'catmull-rom',
            strokeOpacity: 0.3
          }
        )
      ]
    })
    containerRef.current.append(plot)
    return () => plot.remove()
  }, [data, width, height])

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

export default CostPlot
