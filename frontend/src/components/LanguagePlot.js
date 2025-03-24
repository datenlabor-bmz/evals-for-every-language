import { useRef, useEffect } from 'react'
import * as Plot from '@observablehq/plot'

const LanguagePlot = ({ data }) => {
  const containerRef = useRef()
  const languages = data.language_table.filter (a => a.average > 0)
  const families = [...new Set(languages.map(a => a.family))]

  useEffect(() => {
    const plot = Plot.plot({
      width: 750,
      height: 500,
      // title: 'Proficiency of Languages by Number of Speakers',
      x: {
        label: 'Number of Speakers',
        type: 'log'
      },
      y: {
        label: 'Language Proficiency Score',
      },
      marks: [
        Plot.dot(languages, {
          x: 'speakers',
          y: d => d.average,
          r: "speakers",
          fill: 'family',
          fillOpacity: 0.5,
          title: d => d.language_name,
          tip: true
        }),
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

export default LanguagePlot
