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
import CostPlot from './components/CostPlot'
import { Carousel } from 'primereact/carousel'
import { Dialog } from 'primereact/dialog'
import { Button } from 'primereact/button'

function App () {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedLanguages, setSelectedLanguages] = useState([])
  const [dialogVisible, setDialogVisible] = useState(false)
  const [aboutVisible, setAboutVisible] = useState(false)
  const [contributeVisible, setContributeVisible] = useState(false)

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
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', width: '100vw' }}>
        <div style={{backgroundColor: '#fff3cd', color: '#856404', padding: '0.75rem 1.25rem', marginBottom: '1rem', border: '1px solid #ffeeba', borderRadius: '0.25rem', textAlign: 'center'}}>
          <strong>Work in Progress:</strong> This dashboard is currently under active development. Evaluation results are not yet final.
          <a href="https://github.com/datenlabor-bmz/ai-language-monitor" target="_blank" rel="noopener noreferrer" style={{ 
            textDecoration: 'none', 
            color: '#856404', 
            float: 'right', 
            fontSize: '1.2rem',
            fontWeight: 'bold',
            padding: '0 0.5rem',
            borderRadius: '3px',
            backgroundColor: 'rgba(255,255,255,0.3)'
          }}>
            <i className="pi pi-github" title="View on GitHub" style={{ marginRight: '0.3rem' }} />
            GitHub
          </a>
        </div>
        <header
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '5vh 5vw',
            width: '100%',
            maxWidth: '1400px',
            margin: '0 auto'
          }}
        >
          <div>
            <span
              role='img'
              aria-label='Globe Emoji'
              style={{ fontSize: '40px' }}
            >
              🌍
            </span>
          </div>
          <h1 style={{ 
            fontSize: '2.5rem', 
            fontWeight: '600',
            margin: '1rem 0 0.5rem 0',
            color: '#333',
            letterSpacing: '-0.01em'
          }}>
              AI Language Proficiency Monitor
          </h1>
          <p style={{ 
            fontSize: '1.1rem', 
            color: '#666', 
            margin: '0 0 2.5rem 0',
            fontWeight: '400',
            maxWidth: '700px',
            lineHeight: '1.5'
          }}>
            Comprehensive multilingual evaluation results for AI language models
          </p>
          
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
            <Button 
              label="📚 About this tool" 
              className="p-button-text"
              onClick={() => setAboutVisible(true)}
              style={{
                color: '#666',
                border: '1px solid #ddd',
                padding: '0.5rem 1rem',
                borderRadius: '4px',
                fontSize: '0.9rem'
              }}
            />
            
            <Button 
              label="🚀 Add your model" 
              className="p-button-text"
              onClick={() => setContributeVisible(true)}
              style={{
                color: '#666',
                border: '1px solid #ddd',
                padding: '0.5rem 1rem',
                borderRadius: '4px',
                fontSize: '0.9rem'
              }}
            />
          </div>
          
          {data && (
            <AutoComplete
              languages={data?.language_table}
              onComplete={items => setSelectedLanguages(items)}
            />
          )}
        </header>
        <main
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '3rem',
            width: '100%',
            paddingBottom: '5vh',
            padding: '1rem 15vw 5vh 15vw'
          }}
        >
          {loading && (
            <div style={{ width: '100%', textAlign: 'center' }}>
              <i className='pi pi-spinner pi-spin' style={{ fontSize: '4rem' }} />
            </div>
          )}
          {error && <div style={{ width: '100%', textAlign: 'center' }}><p>Error: {error}</p></div>}
          {data && (
            <>
              <div style={{ width: '100%' }}>
                <ModelTable 
                  data={data.model_table} 
                  selectedLanguages={selectedLanguages}
                  allLanguages={data.language_table || []}
                />
              </div>
              <div style={{ width: '100%' }}>
                <LanguageTable
                  data={data.language_table}
                  selectedLanguages={selectedLanguages}
                  setSelectedLanguages={setSelectedLanguages}
                  totalModels={data.model_table?.length || 0}
                />
              </div>
              <div style={{ width: '100%' }}>
                <DatasetTable data={data} />
              </div>
              <div
                id='figure'
                style={{
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
                    <CostPlot data={data} />,
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

        {/* About Dialog */}
        <Dialog
          visible={aboutVisible}
          onHide={() => setAboutVisible(false)}
          style={{ width: '600px' }}
          modal
          header="About this tool"
        >
          <div>
            <p>The <i>AI Language Proficiency Monitor</i> presents comprehensive multilingual evaluation results of AI language models.</p>
            <h4>Who is this for?</h4>
            <ul>
              <li><b>Practitioners</b> can pick the best model for a given language.</li>
              <li><b>Policymakers and funders</b> can identify and prioritize neglected languages.</li>
              <li><b>Model developers</b> can compete on our <i>AI Language Proficiency</i> metric.</li>
            </ul>
            <h4>⚡ Live Updates</h4>
            <p>Benchmark results automatically refresh every night and include the most popular models from <a href="https://openrouter.ai" target="_blank" rel="noopener noreferrer">OpenRouter</a>, plus community-submitted models.</p>
            <h4>Authors</h4>
            <p>The AI Language Proficiency Monitor is a collaboration between BMZ's <a href="https://www.bmz-digital.global/en/overview-of-initiatives/the-bmz-data-lab/" target="_blank" rel="noopener noreferrer">Data Lab</a>, the BMZ-Initiative <a href="https://www.bmz-digital.global/en/overview-of-initiatives/fair-forward/" target="_blank" rel="noopener noreferrer">Fair Forward</a> (implemented by GIZ), and the <a href="https://www.dfki.de/en/web/research/research-departments/multilinguality-and-language-technology/ee-team" target="_blank" rel="noopener noreferrer">E&E group</a> of DFKI's Multilinguality and Language Technology Lab.</p>
            <h4>🔗 Links</h4>
            <p>
              <a 
                href="https://github.com/datenlabor-bmz/ai-language-monitor" 
                target="_blank" 
                rel="noopener noreferrer"
                style={{ 
                  color: '#666', 
                  textDecoration: 'none',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <i className="pi pi-github" style={{ fontSize: '1.2rem' }} />
                View source code on GitHub
              </a>
            </p>
          </div>
        </Dialog>

        {/* Contribute Dialog */}
        <Dialog
          visible={contributeVisible}
          onHide={() => setContributeVisible(false)}
          style={{ width: '600px' }}
          modal
          header="Add your model & Contribute"
        >
          <div>
            <h4>🚀 Submit Your Model</h4>
            <p>Have a custom fine-tuned model you'd like to see on the leaderboard?</p>
            <p><a href="https://forms.gle/ckvY9pS7XLcHYnaV8" target="_blank" rel="noopener noreferrer" style={{ color: '#28a745', fontWeight: 'bold' }}>→ Submit your model here</a></p>
            
            <h4>🔧 Contribute to Development</h4>
            <p>Help us expand language coverage and add new evaluation tasks:</p>
            <p><a href="https://github.com/datenlabor-bmz/ai-language-monitor/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer" style={{ color: '#007bff', fontWeight: 'bold' }}>→ Contribution guidelines</a></p>
          </div>
        </Dialog>

        {/* Full-screen Dialog for Charts */}
        <Dialog
          visible={dialogVisible}
          onHide={() => setDialogVisible(false)}
          style={{ width: '90vw', height: '90vh' }}
          maximizable
          modal
          header={null}
        >
          {data && (
            <div style={{ width: '100%', height: '100%' }}>
              <Carousel
                value={[
                  <WorldMap data={data.countries} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                  <LanguagePlot data={data} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                  <SpeakerPlot data={data} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                  <HistoryPlot data={data} width={windowWidth * 0.7} height={windowHeight * 0.6} />,
                  <CostPlot data={data} />,
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
