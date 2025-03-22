import { useRef, useEffect, useState } from 'react'
import * as topojson from 'topojson-client'
import * as Plot from '@observablehq/plot'

const WorldMap = () => {
  const containerRef = useRef()
  const [data, setData] = useState()

  useEffect(() => {
    fetch('/un.topo.json')
      .then(res => res.json())
      .then(setData)
  }, [])

  useEffect(() => {
    if (data === undefined) return
    // const plot = Plot.plot({
    //   y: {grid: true},
    //   color: {scheme: "burd"},
    //   marks: [
    //     Plot.ruleY([0]),
    //     Plot.dot(data, {x: "Date", y: "Anomaly", stroke: "Anomaly"})
    //   ]
    // });
    const countries = topojson.feature(data, data.objects.un)
    const plot = Plot.plot({
      width: 750,
      height: 400,
      projection: 'equal-earth',
      marks: [
        Plot.geo(countries, {
          // fill: d => console.log(d.properties?.iso2cd),
          // title: d => d.properties?.iso2cd
          // tip: true
        })
      ]
    })
    containerRef.current.append(plot)
    return () => plot.remove()
  }, [data])

  return <svg ref={containerRef} style={{ width: '100%', height: '100%' }} />
}

export default WorldMap
