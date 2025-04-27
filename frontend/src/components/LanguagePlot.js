import { useRef, useEffect } from 'react'
import * as Plot from '@observablehq/plot'

const LanguagePlot = ({ data }) => {
  const containerRef = useRef()
  const languages = data.language_table.filter(a => a.average > 0)
  const families = [...new Set(languages.map(a => a.family))]

  useEffect(() => {
    const plot = Plot.plot({
      width: 750,
      height: 500,
      subtitle: 'Proficiency scores by language',
      x: {
        label: 'Number of Speakers',
        type: 'log'
      },
      y: {
        label: 'Language Proficiency Score'
      },
      marks: [
        Plot.dot(languages, {
          x: 'speakers',
          y: d => d.average,
          r: 'speakers',
          fill: 'family',
          title: d =>
            `${d.language_name}\n${d.speakers.toLocaleString('en-US', {
              notation: 'compact'
            })} speakers\nScore: ${d.average.toFixed(2)}`,
          tip: true
        }),
        Plot.text(
          languages.filter(a => a.speakers > 1e8),
          {
            x: 'speakers',
            y: d => d.average,
            text: d => d.language_name,
            fill: 'black',
            frameAnchor: 'left',
            dx: 10,
            marginRight: 100
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

export default LanguagePlot
