import { useRef, useEffect } from 'react'
import * as Plot from '@observablehq/plot'

const LanguageTierHistoryPlot = ({ data, width = 750, height = 500 }) => {
  const containerRef = useRef()
  
  const tierHistory = [...(data.language_tier_history || [])]
    .filter(d => d.proficiency_score !== null && d.creation_date !== null)
    .sort((a, b) => new Date(a.creation_date) - new Date(b.creation_date))
  
  // Get unique tiers from data, dynamically
  const tiers = [...new Set(tierHistory.map(d => d.tier))]
  
  // Add " languages" suffix for legend display
  const tierWithSuffix = (tier) => `${tier} languages`
  
  // Calculate max proficiency over time for each tier
  const tierRecords = {}
  
  tiers.forEach(tier => {
    const tierData = tierHistory.filter(d => d.tier === tier)
    const records = []
    let maxScore = 0
    
    tierData.forEach(curr => {
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
    
    tierRecords[tier] = records
  })
  
  // Flatten for plotting - only show dots for new records
  // Add " languages" suffix to tier for display
  const recordBreakingDots = Object.values(tierRecords)
    .flat()
    .filter(d => d.newRecord)
    .map(d => ({ ...d, tierDisplay: tierWithSuffix(d.tier) }))
  
  // Create step function data for each tier
  const stepData = tiers.flatMap(tier => {
    const records = tierRecords[tier].filter(d => d.newRecord)
    if (records.length === 0) return []
    
    return [
      ...records.map(d => ({ ...d, tierDisplay: tierWithSuffix(d.tier) })),
      {
        tier: tier,
        tierDisplay: tierWithSuffix(tier),
        creation_date: new Date(),
        maxScore: records[records.length - 1]?.maxScore || 0
      }
    ]
  })
  
  useEffect(() => {
    const plot = Plot.plot({
      width: width,
      height: height,
      subtitle: 'Model performance on language tiers over time',
      x: {
        label: 'Date',
        type: 'time',
        tickFormat: '%Y-%m'
      },
      y: {
        label: 'Overall Score by Language Tier'
      },
      color: {
        legend: true,
        domain: tiers.map(tierWithSuffix)
      },
      marks: [
        Plot.dot(recordBreakingDots, {
          x: d => new Date(d.creation_date),
          y: d => d.proficiency_score,
          fill: 'tierDisplay',
          stroke: 'tierDisplay',
          title: d =>
            `${d.provider_name} - ${d.name} (${
              d.size?.toLocaleString('en-US', { notation: 'compact' }) || '?B'
            })\nTier: ${d.tier}\nPublished: ${new Date(
              d.creation_date
            ).toLocaleDateString()}\nScore: ${d.proficiency_score.toFixed(2)}`,
          tip: true
        }),
        Plot.line(stepData, {
          x: d => new Date(d.creation_date),
          y: d => d.maxScore || 0,
          stroke: 'tierDisplay',
          curve: 'step-after',
          strokeOpacity: 0.5,
          strokeWidth: 2
        })
      ]
    })
    containerRef.current.append(plot)
    return () => plot.remove()
  }, [recordBreakingDots, stepData, width, height, tiers])

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

export default LanguageTierHistoryPlot

