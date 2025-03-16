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
    let detail
    if (item.type === "Dataset") {
        detail = <span>
            {item.detail.map(task => <span key={task} style={{ color: "gray", marginLeft: '1rem', backgroundColor: 'lightgray', padding: '0.2rem', borderRadius: '0.2rem' }}>{task}</span>)}
        </span>
    } else if (item.detail) {
        detail = <span style={{ color: 'gray', marginLeft: '1rem' }}>{item.detail}</span>
    } else {
        detail = null
    }
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
        }}
      >
        <span>{item.value}{detail}</span>
        <span style={{ color: 'gray' }}>{item.type}</span>
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
