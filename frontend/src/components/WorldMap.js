import { useRef, useEffect, useState } from 'react'
import * as Plot from '@observablehq/plot'

const smoothProgressBar = fraction => {
  const blocks = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
  const width = 10
  const totalUnits = width * 8
  const filledUnits = Math.round(fraction * totalUnits)
  const fullBlocks = Math.floor(filledUnits / 8)
  const remainder = filledUnits % 8
  return (
    '█'.repeat(fullBlocks) + (remainder > 0 ? blocks[remainder - 1] : '') || '▏'
  )
}

const makeTitle = data => d => {
  const cData = data[d.properties?.ISO_A2_EH]
  const languages = cData?.languages.toSorted(
    (a, b) => b.population - a.population
  )
  const pop = languages?.map(a => a.population).reduce((prev, a) => prev + a, 0)
  const langstring =
    languages
      ?.slice(0, 10)
      .map(
        a =>
          `${smoothProgressBar(a.population / pop)} ${
            a.name
          } – ${a.score === null || a.score === undefined ? "n/a" : a.score.toFixed(2)}`
      )
      .join('\n\n') + (languages?.length > 10 ? `\n\n...` : '')
  return `${d.properties.ADMIN} – ${cData?.score === null || cData?.score === undefined ? "n/a" : cData.score.toFixed(2)}\n\n${langstring}`
}

const WorldMap = ({ data, width = 750, height = 500, allLanguages = [] }) => {
  const containerRef = useRef()
  const [mapData, setMapData] = useState()

  useEffect(() => {
    fetch('/world.geo.json')
      .then(res => res.json())
      .then(setMapData)
  }, [])

  useEffect(() => {
    if (mapData === undefined || data === undefined) return
    const countriesDict = data.reduce((acc, country) => {
      acc[country.iso2] = country
      return acc
    }, {})
    // Count languages that have any evaluation data
    const evaluatedLanguagesCount = allLanguages.filter(lang => {
      const hasAnyScores = [
        'translation_from_bleu',
        'translation_to_bleu', 
        'classification_accuracy',
        'mmlu_accuracy',
        'arc_accuracy',
        'truthfulqa_accuracy',
        'mgsm_accuracy'
      ].some(metric => lang[metric] !== null && lang[metric] !== undefined)
      return hasAnyScores
    }).length

    const plot = Plot.plot({
      subtitle: `Language Proficiency Score by Country (Coverage: ~${evaluatedLanguagesCount} languages evaluated)`,
      width: width,
      height: height,
      projection: 'equal-earth',
      marks: [
        Plot.geo(mapData, {
          fill: d => countriesDict[d.properties?.ISO_A2_EH]?.score,
          title: makeTitle(countriesDict),
          tip: true
        })
      ],
      color: {
        scheme: 'RdYlGn',
        unknown: '#d0d0d0',
        label: 'Score',
        legend: true,
        domain: [0, 1],
        pivot: 0.5
      },
      style: {
        fontFamily: 'monospace'
      }
    })
    containerRef.current.append(plot)
    return () => plot.remove()
  }, [mapData, data, width, height])

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

export default WorldMap
