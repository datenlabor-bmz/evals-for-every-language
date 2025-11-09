import { useRef, useEffect } from 'react'
import * as Plot from '@observablehq/plot'

const LicenseHistoryPlot = ({ data, width = 750, height = 500 }) => {
  const containerRef = useRef()
  
  const licenseHistory = [...(data.license_history || [])]
    .filter(d => d.proficiency_score !== null && d.creation_date !== null)
    .sort((a, b) => new Date(a.creation_date) - new Date(b.creation_date))
  
  const licenseTypes = ['Commercial', 'Open-source']
  const licenseRecords = {}
  
  licenseTypes.forEach(type => {
    const typeData = licenseHistory.filter(d => d.license_type === type)
    const records = []
    let maxScore = 0
    
    typeData.forEach(curr => {
      if (curr.proficiency_score > maxScore) {
        maxScore = curr.proficiency_score
        records.push({
          ...curr,
          maxScore: maxScore,
          newRecord: true
        })
      } else {
        records.push({
          ...curr,
          maxScore: maxScore,
          newRecord: false
        })
      }
    })
    
    licenseRecords[type] = records
  })
  
  // Only show dots for new records
  const recordBreakingDots = Object.values(licenseRecords).flat().filter(d => d.newRecord)
  
  // Create step function data
  const stepData = licenseTypes.flatMap(type => {
    const records = licenseRecords[type].filter(d => d.newRecord)
    if (records.length === 0) return []
    
    return [
      ...records,
      {
        license_type: type,
        creation_date: new Date(),
        maxScore: records[records.length - 1]?.maxScore || 0
      }
    ]
  })
  
  useEffect(() => {
    const plot = Plot.plot({
      width: width,
      height: height,
      subtitle: 'Commercial vs Open-source models over time',
      x: {
        label: 'Date',
        type: 'time',
        tickFormat: '%Y-%m'
      },
      y: {
        label: 'Language Proficiency Score'
      },
      color: {
        legend: true,
        domain: licenseTypes
      },
      marks: [
        Plot.dot(recordBreakingDots, {
          x: d => new Date(d.creation_date),
          y: d => d.proficiency_score,
          fill: 'license_type',
          stroke: 'license_type',
          title: d =>
            `${d.provider_name} - ${d.name} (${
              d.size?.toLocaleString('en-US', { notation: 'compact' }) || '?B'
            })\nType: ${d.license_type}\nPublished: ${new Date(
              d.creation_date
            ).toLocaleDateString()}\nScore: ${d.proficiency_score.toFixed(2)}`,
          tip: true
        }),
        Plot.line(stepData, {
          x: d => new Date(d.creation_date),
          y: d => d.maxScore || 0,
          stroke: 'license_type',
          curve: 'step-after',
          strokeOpacity: 0.5,
          strokeWidth: 2
        })
      ]
    })
    containerRef.current.append(plot)
    return () => plot.remove()
  }, [recordBreakingDots, stepData, width, height])

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

export default LicenseHistoryPlot

