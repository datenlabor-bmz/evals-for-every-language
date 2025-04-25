import { useState, useEffect } from 'react'
import { PrimeReactProvider } from 'primereact/api'
import 'primereact/resources/themes/lara-light-cyan/theme.css'
import ModelTable from './components/ModelTable'
import LanguageTable from './components/LanguageTable'
import DatasetTable from './components/DatasetTable'
import WorldMap from './components/WorldMap'
import AutoComplete from './components/AutoComplete'
import LanguagePlot from './components/LanguagePlot'
import SpeakerPlot from './components/SpeakerPlot'
import HistoryPlot from './components/HistoryPlot'
import { Carousel } from 'primereact/carousel'

function App () {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedLanguages, setSelectedLanguages] = useState([])
  useEffect(() => {
    fetch('/api/data', {
      method: 'POST',
      body: JSON.stringify({ selectedLanguages })
    })
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
  }, [selectedLanguages])

  return (
    <PrimeReactProvider>
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <div style={{backgroundColor: '#fff3cd', color: '#856404', padding: '0.75rem 1.25rem', marginBottom: '1rem', border: '1px solid #ffeeba', borderRadius: '0.25rem', textAlign: 'center'}}>
          <strong>Work in Progress:</strong> This dashboard is currently under active development. Data and features may change.
        </div>
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
              AI Language Proficiency Monitor
          </h1>
          <p style={{ fontSize: '1.15rem', color: '#555', marginTop: '0' }}>
            Tracking language proficiency of AI models for every language
          </p>
          {data && (
            <AutoComplete
              languages={data?.language_table}
              onComplete={items => setSelectedLanguages(items)}
            />
          )}
          <div style={{maxWidth: '600px', textAlign: 'center', marginTop: '2rem'}}>
            <p>The <i>AI Language Proficiency Monitor</i> presents comprehensive multilingual evaluation results of AI language models.</p>
            <ul style={{textAlign: 'left'}}>
              <li><b>Developers and researchers</b> can identify the best-performing model for any given language or region.</li>
              <li><b>Policymakers and funders</b> can identify languages that are neglected by current AI systems and prioritize investment accordingly.</li>
              <li><b>Model developers</b> gain a global metric of multilinguality that incentivizes broad language coverage.</li>
            </ul>
            <p>We also want to celebrate the community's effort towards multilingual datasets and models. We invite you to integrate additional benchmark datasets, and to add your custom language models to the monitor.</p>
            <p>Benchmark results automatically refresh every night for the most popular AI models, as well as community-listed models. For a more detailed methodlogy, see <a href="#">our non-existing paper</a>.</p>
            <p>The language monitor is a collaboration between BMZ's <a href="https://www.bmz-digital.global/en/overview-of-initiatives/the-bmz-data-lab/" target="_blank" rel="noopener noreferrer">Data Lab</a>, GIZ's <a href="https://www.giz.de/expertise/html/61982.html" target="_blank" rel="noopener noreferrer">FairForward</a> initiative, and the <a href="https://www.dfki.de/en/web/research/research-departments/multilinguality-and-language-technology/ee-team" target="_blank" rel="noopener noreferrer">E&E group</a> of DFKI's Multilinguality and Language Technology Lab.</p>
          </div>
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
          {loading && (
            <i className='pi pi-spinner pi-spin' style={{ fontSize: '4rem' }} />
          )}
          {error && <p>Error: {error}</p>}
          {data && (
            <>
              <div
                style={{
                  flex: '60vw 100vw 40vw',
                  maxWidth: 'min(100vw, 800px)'
                }}
              >
                <ModelTable data={data.model_table} />
              </div>
              <div
                style={{
                  flex: '60vw 100vw 40vw',
                  maxWidth: 'min(100vw, 800px)'
                }}
              >
                <LanguageTable
                  data={data.language_table}
                  selectedLanguages={selectedLanguages}
                  setSelectedLanguages={setSelectedLanguages}
                />
              </div>
              <div
                style={{
                  flex: '60vw 100vw 40vw',
                  maxWidth: 'min(100vw, 800px)'
                }}
              >
                <DatasetTable data={data} />
              </div>
              <div
                id='figure'
                style={{
                  flex: '100vw 100vw 100vw',
                  maxWidth: 'min(100vw, 800px)',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '100%'
                }}
              >
                <Carousel
                  value={[
                    <WorldMap data={data.countries} />,
                    <LanguagePlot data={data} />,
                    <SpeakerPlot data={data} />,
                    <HistoryPlot data={data} />,
                  ]}
                  numScroll={1}
                  numVisible={1}
                  itemTemplate={item => item}
                  circular
                  style={{ width: '100%', minHeight: '650px' }}
                />
              </div>
            </>
          )}
        </main>
      </div>
    </PrimeReactProvider>
  )
}

export default App
