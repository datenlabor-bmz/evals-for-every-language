import { useState, useEffect } from 'react'
import { PrimeReactProvider } from 'primereact/api'
import 'primereact/resources/themes/lara-light-cyan/theme.css'
import ModelTable from './components/ModelTable'
import LanguageTable from './components/LanguageTable'
import DatasetTable from './components/DatasetTable'
function App () {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

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

  return (
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
      </header>
      <PrimeReactProvider>
        {loading && <p>...</p>}
        {error && <p>Error: {error}</p>}
        {data && (
          <div
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
          </div>
        )}
      </PrimeReactProvider>
    </div>
  )
}

export default App
