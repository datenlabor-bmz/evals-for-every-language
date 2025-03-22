import { useState, useEffect } from 'react'
import { PrimeReactProvider } from 'primereact/api'
import 'primereact/resources/themes/lara-light-cyan/theme.css'
import ModelTable from './components/ModelTable'
import LanguageTable from './components/LanguageTable'
import DatasetTable from './components/DatasetTable'
import WorldMap from './components/WorldMap'
import AutoComplete from './components/AutoComplete'

function App () {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [allSuggestions, setAllSuggestions] = useState([])
  const [filters, setFilters] = useState([])
  useEffect(() => {
    fetch('/results.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok')
        }
        return response.json()
      })
      .then(jsonData => {
        setData(jsonData)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (data) {
      const models = data.model_table.map(item => ({ type: 'Model', value: item.model, detail: item.provider, searchText: item.provider.toLowerCase() + ' ' + item.model.toLowerCase() }))
      const languages = data.language_table.map(item => ({ type: 'Language', value: item.autonym, detail: item.language_name, searchText: item.language_name.toLowerCase() + ' ' + item.autonym.toLowerCase() }))
      const datasets = data.dataset_table.map(item => ({ type: 'Dataset', value: item.name, detail: item.tasks, searchText: item.author.toLowerCase() + ' ' + item.name.toLowerCase() + ' ' + item.tasks.map(task => task.toLowerCase()).join(' ') }))
      const allSuggestions = [...models, ...languages, ...datasets]
      setAllSuggestions(allSuggestions)
    }
  }, [data])

  return (
    <PrimeReactProvider>
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <header
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '5vh 0'
        }}
      >
        <div>
          <span
            role='img'
            aria-label='Globe Emoji'
            style={{ fontSize: '70px' }}
          >
            🌍
          </span>
        </div>
        <h1 style={{ fontSize: '2.5rem', fontWeight: '700' }}>
          Global AI Language Monitor
        </h1>
        <p style={{ fontSize: '1.15rem', color: '#555', marginTop: '0' }}>
          Tracking language proficiency of AI models for every language
        </p>
        <AutoComplete allSuggestions={allSuggestions} onComplete={item => setFilters(prevFilters => [item])} />
      </header>
      <main
            style={{
              display: 'flex',
              flexDirection: 'row',
              flexWrap: 'wrap',
              gap: '2rem',
              alignItems: 'center',
              width: '100%',
              height: '100%',
              justifyContent: 'center',
              paddingBottom: '5vh'
            }}
          >
        {loading && <i className='pi pi-spinner pi-spin' style={{ fontSize: '4rem' }} />}
        {error && <p>Error: {error}</p>}
        {data && (
          <>
            <div
              id='figure'
              style={{
                flex: '100vw 100vw 100vw',
                maxWidth: 'min(100vw, 800px)',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
                <WorldMap data={data} />
            </div>
            <div
              style={{
                flex: '60vw 100vw 40vw',
                maxWidth: 'min(100vw, 800px)',
              }}
            >
              <ModelTable data={data} />
            </div>
            <div
              style={{
                flex: '60vw 100vw 40vw',
                maxWidth: 'min(100vw, 800px)'
              }}
            >
              <LanguageTable data={data} />
            </div>
            <div
              style={{
                flex: '60vw 100vw 40vw',
                maxWidth: 'min(100vw, 800px)'
              }}
            >
              <DatasetTable data={data} />
            </div>
            </>
        )}
      </main>
    </div>
    </PrimeReactProvider>
  )
}

export default App
