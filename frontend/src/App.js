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
import { Dialog } from 'primereact/dialog'
import { Button } from 'primereact/button'

function App () {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedLanguages, setSelectedLanguages] = useState([])
  const [dialogVisible, setDialogVisible] = useState(false)

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

  const [windowWidth, setWindowWidth] = useState(window.innerWidth)
  const [windowHeight, setWindowHeight] = useState(window.innerHeight)
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth)
      setWindowHeight(window.innerHeight)
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <PrimeReactProvider>
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <div style={{backgroundColor: '#fff3cd', color: '#856404', padding: '0.75rem 1.25rem', marginBottom: '1rem', border: '1px solid #ffeeba', borderRadius: '0.25rem', textAlign: 'center'}}>
          <strong>Work in Progress:</strong> This dashboard is currently under active development. Evaluation results are not yet final.
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
              <li><b>Practitioners</b> can pick the best model for a given language.</li>
              <li><b>Policymakers and funders</b> can identify and prioritize neglected languages.</li>
              <li><b>Model developers</b> can compete on our <i>AI Language Proficiency</i> metric.</li>
            </ul>
            <p>We invite the community to <a href="https://forms.gle/ckvY9pS7XLcHYnaV8" target="_blank" rel="noopener noreferrer">submit</a> their custom finetuned models, and to <a href="https://github.com/datenlabor-bmz/ai-language-monitor/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">integrate</a> benchmarks for more languages and tasks.</p>
            <p>Benchmark results automatically refresh every night and include the most popular models on <a href="https://openrouter.ai" target="_blank" rel="noopener noreferrer">OpenRouter</a>, plus community-listed models.</p>
            {/* <p>For a detailed methodlogy, see <a href="#">XXX</a>.</p> */}
            <p>The AI Language Proficiency Monitor is a collaboration between BMZ's <a href="https://www.bmz-digital.global/en/overview-of-initiatives/the-bmz-data-lab/" target="_blank" rel="noopener noreferrer">Data Lab</a>, GIZ's <a href="https://www.giz.de/expertise/html/61982.html" target="_blank" rel="noopener noreferrer">FairForward</a> initiative, and the <a href="https://www.dfki.de/en/web/research/research-departments/multilinguality-and-language-technology/ee-team" target="_blank" rel="noopener noreferrer">E&E group</a> of DFKI's Multilinguality and Language Technology Lab.</p>
          </div>
          <a href="https://github.com/datenlabor-bmz/ai-language-monitor" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', color: 'inherit', marginTop: '2rem' }}>
            <i className="pi pi-github" style={{ fontSize: '1.5rem' }} />
          </a>
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
                  width: '100%',
                  position: 'relative'
                }}
              >
                <Button
                  icon="pi pi-external-link"
                  className="p-button-text p-button-plain"
                  onClick={() => setDialogVisible(true)}
                  tooltip="Open in larger view"
                  style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    zIndex: 1,
                    color: '#666'
                  }}
                />
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

        <Dialog
          visible={dialogVisible}
          onHide={() => setDialogVisible(false)}
          style={{ width: '90vw', height: '90vh' }}
          maximizable
          modal
          header="Charts"
        >
          {data && (
            <div style={{ width: '100%', height: '100%' }}>
              <Carousel
                value={[
                  <WorldMap data={data.countries} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                  <LanguagePlot data={data} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                  <SpeakerPlot data={data} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                  <HistoryPlot data={data} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                ]}
                numScroll={1}
                numVisible={1}
                itemTemplate={item => item}
                circular
                style={{ width: '100%', height: 'calc(90vh - 120px)' }}
              />
            </div>
          )}
        </Dialog>
      </div>
    </PrimeReactProvider>
  )
}

export default App
