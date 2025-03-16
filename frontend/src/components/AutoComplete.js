import { AutoComplete as PrimeAutoComplete } from 'primereact/autocomplete'
import { useState } from 'react'
const AutoComplete = ({ allSuggestions, onComplete }) => {
  const [autoComplete, setAutoComplete] = useState('')
  const [suggestions, setSuggestions] = useState([])

  const search = e => {
    console.log(allSuggestions)
    const matches = allSuggestions.filter(suggestion =>
      suggestion.searchText.includes(e.query.toLowerCase())
    )
    setSuggestions(matches)
  }

  const itemTemplate = item => {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between'
        }}
      >
        <div>{item.value}</div>
        <div style={{ color: 'gray' }}>{item.type}</div>
      </div>
    )
  }

  return (
    <PrimeAutoComplete
      placeholder='Search for model, language, or dataset'
      value={autoComplete}
      onChange={e => setAutoComplete(e.value)}
      onSelect={e => {
        setAutoComplete(e.value.value)
        onComplete(e.value.value)
      }}
      suggestions={suggestions}
      completeMethod={search}
      virtualScrollerOptions={{ itemSize: 50 }}
      delay={500}
      autoHighlight
      autoFocus
      itemTemplate={itemTemplate}
    />
  )
}

export default AutoComplete
