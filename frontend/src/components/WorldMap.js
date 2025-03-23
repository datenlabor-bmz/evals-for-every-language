import { useRef, useEffect, useState } from 'react'
import * as topojson from 'topojson-client'
import * as Plot from '@observablehq/plot'

const WorldMap = ({ data }) => {
  const containerRef = useRef()
  const [mapData, setMapData] = useState()

  useEffect(() => {
    fetch('/world.geo.json')
      .then(res => res.json())
      .then(setMapData)
  }, [])

  useEffect(() => {
    if (mapData === undefined) return
    const countries = mapData
    // const countries = topojson.feature(mapData, mapData.objects["world.geo"])
    console.log(countries)
    const codes = countries.features.map(d => d.properties?.ISO_A2_EH)
    console.log(codes.toSorted().join(', '))
    const plot = Plot.plot({
      width: 750,
      height: 400,
      projection: 'equal-earth',
      marks: [
        Plot.geo(countries, {
          fill: d => {
            const score = data.countries[d.properties?.ISO_A2_EH]?.score
            return score
          },
          title: d => `<b>${d.properties?.ISO_A2_EH}</b> (${d.properties?.NAME_EN})`,
          tip: true
        })
      ],
      color: {
        range: ["red", "blue"],
        unknown: 'gray',
        // type: 'linear',
        label: 'Score',
        legend: true,
        // percent: true,
        domain: [0, 0.5]
      }
    })
    containerRef.current.append(plot)
    return () => plot.remove()
  }, [mapData])

  return <div ref={containerRef} style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }} />
}

export default WorldMap
