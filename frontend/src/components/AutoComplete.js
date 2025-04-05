import { AutoComplete as PrimeAutoComplete } from 'primereact/autocomplete'
import { useState } from 'react'
const AutoComplete = ({ languages, onComplete }) => {
  const [autoComplete, setAutoComplete] = useState('')
  const [suggestions, setSuggestions] = useState([])

  const exampleCodes = ['de', 'fr', 'ar', 'hi', 'sw', 'fa']
  const exampleLanguages = languages?.filter(item =>
    exampleCodes.includes(item.bcp_47)
  )

  const search = e => {
    const matches = languages.filter(language => {
      const query = e.query.toLowerCase()
      return (
        language.language_name.toLowerCase().includes(query) ||
        language.autonym.toLowerCase().includes(query) ||
        language.bcp_47.toLowerCase().includes(query)
      )
    })
    setSuggestions(matches)
  }

  const itemTemplate = item => (
    <div
      style={{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between'
      }}
    >
      <span>
        {item.autonym}
        <span style={{ color: 'gray', marginLeft: '1rem' }}>
          {item.language_name}
        </span>
      </span>
      <span style={{ color: 'gray' }}>{item.bcp_47}</span>
    </div>
  )

  return (
    <>
      <PrimeAutoComplete
        placeholder='Search for language-specific leaderboards...'
        value={autoComplete}
        onChange={e => setAutoComplete(e.value)}
        onClick={() => {
          setAutoComplete('')
          setSuggestions(languages)
        }}
        onSelect={e => {
          setAutoComplete(e.value.language_name)
          onComplete([e.value])
        }}
        suggestions={suggestions}
        completeMethod={search}
        virtualScrollerOptions={{ itemSize: 50 }} // smaller values give layout problems
        delay={500}
        autoHighlight
        autoFocus
        itemTemplate={itemTemplate}
        field='language_name'
        minLength={0}
      />
      <span
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          rowGap: '0.3rem',
          marginTop: '1rem',
          maxWidth: '600px',
          justifyContent: 'center',
          color: '#555'
        }}
      >
        <span>Examples:</span>
        {exampleLanguages?.map(language => (
          <a
            onClick={() => {
              onComplete([language])
              setAutoComplete(language.language_name)
            }}
            style={{ textDecoration: 'underline', cursor: 'pointer' }}
          >
            {language.language_name} Leaderboard
          </a>
        ))}
        {/* <li>African Leaderboard</li>
              <li>Indic Leaderboard</li>
              <li>Transcription Leaderboard</li>
              <li>Dataset Availability for African Languages</li>
              <li>GPT 4.5 Evaluation</li>
              <li>MMLU Evaluation of Open Models</li> */}
      </span>
    </>
  )
}

export default AutoComplete
