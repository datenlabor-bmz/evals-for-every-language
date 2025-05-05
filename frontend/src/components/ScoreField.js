const ScoreField = (score, minScore, maxScore) => {
  let percentage = 100
  let barColor = "rgba(210, 106, 255, 0.1)" // light violet for missing data
  if (score !== null) {
  // Calculate percentage based on the provided min and max scores
  // This normalizes the score to a 0-100 range for visualization
  const normalizedScore = Math.min(Math.max(score, minScore), maxScore)
  percentage =
    ((normalizedScore - minScore) / (maxScore - minScore)) * 100

  // Continuous color gradient from red to green based on score
  // For a smooth transition, calculate the RGB values directly

  // Red component decreases as score increases
  const red = Math.round(255 * (1 - percentage / 100))
  // Green component increases as score increases
  const green = Math.round(255 * (percentage / 100))
  // Use a low opacity for subtlety (0.1-0.2 range)
  const opacity = 0.1 + (percentage / 100) * 0.1

  barColor = `rgba(${red}, ${green}, 0, ${opacity.toFixed(2)})`
  }

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        padding: '0.5rem'
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          height: '100%',
          width: `${percentage}%`,
          backgroundColor: barColor,
          zIndex: 0,
          transition: 'width 0.3s, background-color 0.3s'
        }}
      />

      <span
        style={{
          position: 'relative',
          zIndex: 1
        }}
      >
        {score !== null ? (score * 100).toFixed(1)+"%" : '–'}
      </span>
    </div>
  )
}

export default ScoreField