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
import LanguageTierHistoryPlot from './components/LanguageTierHistoryPlot'
import LicenseHistoryPlot from './components/LicenseHistoryPlot'
import CostPlot from './components/CostPlot'
import { Carousel } from 'primereact/carousel'
import { Dialog } from 'primereact/dialog'
import { Button } from 'primereact/button'

function App () {
  const [data, setData] = useState(null)
  const [baseData, setBaseData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [modelTableLoading, setModelTableLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedLanguages, setSelectedLanguages] = useState([])
  const [machineTranslatedMetrics, setMachineTranslatedMetrics] = useState([])
  const [dialogVisible, setDialogVisible] = useState(false)
  const [aboutVisible, setAboutVisible] = useState(false)
  const [contributeVisible, setContributeVisible] = useState(false)
  
  // Add state for carousel items
  const [carouselItems, setCarouselItems] = useState([])
  const [fullScreenCarouselItems, setFullScreenCarouselItems] = useState([])

  useEffect(() => {
    // For initial load, use main loading state; for language changes, use model table loading
    if (!data) {
      setLoading(true)
    } else {
      setModelTableLoading(true)
    }
    
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
        setMachineTranslatedMetrics(jsonData.machine_translated_metrics || [])
        if (!baseData) setBaseData(jsonData)
        setLoading(false)
        setModelTableLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
        setModelTableLoading(false)
      })
  }, [selectedLanguages])

  // Create carousel items when data is loaded
  useEffect(() => {
    if (data) {
      // Add a small delay to ensure components are ready
      const timer = setTimeout(() => {
        setCarouselItems([
          <WorldMap key="worldmap-0" data={(baseData || data).countries} allLanguages={(baseData || data).language_table} width={750} height={500} />,
          <LanguagePlot key="langplot-1" data={data} width={750} height={500} />,
          <SpeakerPlot key="speakerplot-2" data={data} width={750} height={500} />,
          <HistoryPlot key="histplot-3" data={data} width={750} height={500} />,
          <LanguageTierHistoryPlot key="tierhistplot-4" data={data} width={750} height={500} />,
          <LicenseHistoryPlot key="licensehistplot-5" data={data} width={750} height={500} />,
          <CostPlot key="costplot-6" data={data} width={750} height={500} />
        ]);
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [data, baseData])

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

  // Create full-screen carousel items when data or window size changes
  useEffect(() => {
    if (data) {
      const timer = setTimeout(() => {
        setFullScreenCarouselItems([
          <WorldMap
            key="fs-worldmap-0"
            data={(baseData || data).countries}
            allLanguages={(baseData || data).language_table}
            width={windowWidth * 0.7}
            height={windowHeight * 0.6}
          />,
          <LanguagePlot
            key="fs-langplot-1"
            data={data}
            width={windowWidth * 0.7}
            height={windowHeight * 0.6}
          />,
          <SpeakerPlot
            key="fs-speakerplot-2"
            data={data}
            width={windowWidth * 0.7}
            height={windowHeight * 0.6}
          />,
          <HistoryPlot
            key="fs-histplot-3"
            data={data}
            width={windowWidth * 0.7}
            height={windowHeight * 0.6}
          />,
          <LanguageTierHistoryPlot
            key="fs-tierhistplot-4"
            data={data}
            width={windowWidth * 0.7}
            height={windowHeight * 0.6}
          />,
          <LicenseHistoryPlot
            key="fs-licensehistplot-5"
            data={data}
            width={windowWidth * 0.7}
            height={windowHeight * 0.6}
          />,
          <CostPlot key="fs-costplot-6" data={data} width={windowWidth * 0.7} height={windowHeight * 0.6} />
        ]);
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [data, baseData, windowWidth, windowHeight])

  return (
    <PrimeReactProvider>
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          width: '100vw'
        }}
      >
        {/* <div
          style={{
            backgroundColor: '#fff3cd',
            color: '#856404',
            padding: '1rem 1.5rem',
            marginBottom: '1rem',
            border: '1px solid #ffeeba',
            borderRadius: '0.25rem',
            textAlign: 'center',
            lineHeight: '1.5',
            position: 'relative'
          }}
        >
          <strong>Work in Progress:</strong> This dashboard is currently under
          active development. Evaluation results are not yet final. More extensive evaluation runs will be released later this year.
        </div> */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            padding: '0 1.5rem',
            marginBottom: '1rem'
          }}
        >
          <a
            href='https://github.com/datenlabor-bmz/ai-language-monitor'
            target='_blank'
            rel='noopener noreferrer'
            style={{
              textDecoration: 'none',
              color: '#6c757d',
              fontSize: '1rem',
              fontWeight: '500',
              padding: '0.5rem 1rem',
              borderRadius: '0.375rem',
              backgroundColor: '#f8f9fa',
              border: '1px solid #e9ecef',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
              ':hover': {
                backgroundColor: '#e9ecef',
                color: '#495057'
              }
            }}
          >
            <i className='pi pi-github' title='View on GitHub' />
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
          <h1
            style={{
              fontSize: '2.5rem',
              fontWeight: '600',
              margin: '1rem 0 0.5rem 0',
              color: '#333',
              letterSpacing: '-0.01em'
            }}
          >
            AI Language Benchmarks
          </h1>
          <p
            style={{
              fontSize: '1.1rem',
              color: '#666',
              margin: '0 0 2.5rem 0',
              fontWeight: '400',
              maxWidth: '700px',
              lineHeight: '1.5'
            }}
          >
            AI model evaluations for every language in the world
          </p>

          <div
            style={{
              display: 'flex',
              gap: '0.75rem',
              marginBottom: '2rem',
              flexWrap: 'wrap',
              justifyContent: 'center'
            }}
          >
            <button
              onClick={() => setAboutVisible(true)}
              style={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                padding: '0.75rem 1.5rem',
                borderRadius: '12px',
                fontSize: '0.95rem',
                fontWeight: '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                boxShadow: '0 4px 15px rgba(102, 126, 234, 0.25)',
                transition: 'all 0.3s ease',
                ':hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 8px 25px rgba(102, 126, 234, 0.35)'
                }
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 8px 25px rgba(102, 126, 234, 0.35)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 4px 15px rgba(102, 126, 234, 0.25)';
              }}
            >
              <span style={{ fontSize: '1.1rem' }}>📚</span>
              About this tool
            </button>

            <button
              onClick={() => setContributeVisible(true)}
              title='This feature is on our roadmap and will be available soon.'
              style={{
                background: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%)',
                color: '#6b46c1',
                border: 'none',
                padding: '0.75rem 1.5rem',
                borderRadius: '12px',
                fontSize: '0.95rem',
                fontWeight: '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                boxShadow: '0 4px 15px rgba(255, 154, 158, 0.25)',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 8px 25px rgba(255, 154, 158, 0.35)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 4px 15px rgba(255, 154, 158, 0.25)';
              }}
            >
              <span style={{ fontSize: '1.1rem' }}>🚀</span>
              Add your model/benchmark
            </button>
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
              <i
                className='pi pi-spinner pi-spin'
                style={{ fontSize: '4rem' }}
              />
            </div>
          )}
          {error && (
            <div style={{ width: '100%', textAlign: 'center' }}>
              <p>Error: {error}</p>
            </div>
          )}
          {data && (
            <>
              <div style={{ position: 'relative' }}>
                {modelTableLoading && (
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(255, 255, 255, 0.8)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1000
                  }}>
                    <i className='pi pi-spinner pi-spin' style={{ fontSize: '3rem' }} />
                  </div>
                )}
                <ModelTable
                  data={data.model_table}
                  selectedLanguages={selectedLanguages}
                  allLanguages={data.language_table || []}
                  machineTranslatedMetrics={machineTranslatedMetrics}
                />
              </div>
              <LanguageTable
                data={data.language_table}
                selectedLanguages={selectedLanguages}
                setSelectedLanguages={setSelectedLanguages}
                totalModels={data.model_table?.length || 0}
              />
              <DatasetTable data={data} />
              <div
                id='figure'
                style={{
                  width: '100%',
                  position: 'relative'
                }}
              >
                <Button
                  icon='pi pi-external-link'
                  className='p-button-text p-button-plain'
                  onClick={() => setDialogVisible(true)}
                  tooltip='Open in larger view'
                  style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    zIndex: 1,
                    color: '#666'
                  }}
                />
                {carouselItems.length > 0 && (
                  <Carousel
                    key={`main-carousel-${carouselItems.length}-${Date.now()}`}
                    value={carouselItems}
                    numScroll={1}
                    numVisible={1}
                    itemTemplate={item => item}
                    circular={false}
                    activeIndex={0}
                    style={{ width: '100%', minHeight: '650px' }}
                  />
                )}
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
          header='About this tool'
        >
          <div>
            <p>
              <i>languagebench</i> provides AI model evaluations for every language in the world.
            </p>
            <h4>👥 Who is this for?</h4>
            <ul>
              <li>
                <b>Practitioners</b> can pick the best model for a given
                language.
              </li>
              <li>
                <b>Policymakers and funders</b> can identify and prioritize
                neglected languages.
              </li>
              <li>
                <b>Model developers</b> can compete on our benchmarks.
              </li>
            </ul>
            <h4>⚡ Live Updates</h4>
            <p>
              Benchmark results automatically refresh every night and include
              the most popular models from{' '}
              <a
                href='https://openrouter.ai'
                target='_blank'
                rel='noopener noreferrer'
              >
                OpenRouter
              </a>
              , plus community-submitted models.
            </p>
            <h4>⚠️ Note on interpretation</h4>
            <p>
              Results are currently based on a sample of 10 sentences per language and task to keep computation affordable. For this reason, we report confidence intervals and recommend treating small differences between models with caution. In future iterations, we plan to add more benchmark datasets and richer visualisations, with large-scale evaluations across many more prompts and tasks as a longer-term goal.
            </p>
            <h4>✍️ Authors</h4>
            <p>
              languagebench is a collaboration between
              BMZ's{' '}
              <a
                href='https://www.bmz-digital.global/en/overview-of-initiatives/the-bmz-data-lab/'
                target='_blank'
                rel='noopener noreferrer'
              >
                Data Lab
              </a>
              (<a href='https://www.linkedin.com/in/davidpomerenke/'>David Pomerenke</a>), the BMZ-Initiative{' '}
              <a
                href='https://www.bmz-digital.global/en/overview-of-initiatives/fair-forward/'
                target='_blank'
                rel='noopener noreferrer'
              >
                GIZ Fair Forward
              </a>{' '}
              (<a href='https://www.linkedin.com/in/jonas-nothnagel-bb42b114b/'>Jonas Nothnagel</a>), and the{' '}
              <a
                href='https://www.dfki.de/en/web/research/research-departments/multilinguality-and-language-technology/ee-team'
                target='_blank'
                rel='noopener noreferrer'
              >
                E&E group
              </a>{' '}
              of DFKI's Multilinguality and Language Technology Lab.
            </p>
            <h4>🔗 Links</h4>
            <p>
              <a
                href='https://github.com/datenlabor-bmz/ai-language-monitor'
                target='_blank'
                rel='noopener noreferrer'
                style={{
                  color: '#666',
                  textDecoration: 'none',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <i className='pi pi-github' style={{ fontSize: '1.2rem' }} />
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
          header='Contribute your Model/Benchmark'
        >
          <div>
            <h4>🚀 Submit Your Model</h4>
            <p>
              Have a custom fine-tuned model you'd like to see on the
              leaderboard or a new benchmark you think should be added?
            </p>
            <p>
              <a
                href='https://forms.gle/ckvY9pS7XLcHYnaV8'
                target='_blank'
                rel='noopener noreferrer'
                style={{ color: '#28a745', fontWeight: 'bold' }}
              >
                → Submit your model here
              </a>
            </p>

            <h4>🔧 Contribute to Development</h4>
            <p>
              Help us expand language coverage and add new evaluation tasks:
            </p>
            <p>
              <a
                href='https://github.com/datenlabor-bmz/ai-language-monitor/blob/main/CONTRIBUTING.md'
                target='_blank'
                rel='noopener noreferrer'
                style={{ color: '#007bff', fontWeight: 'bold' }}
              >
                → Contribution guidelines
              </a>
            </p>
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
          {fullScreenCarouselItems.length > 0 && (
            <div style={{ width: '100%', height: '100%' }}>
              <Carousel
                key={`fs-carousel-${fullScreenCarouselItems.length}-${Date.now()}`}
                value={fullScreenCarouselItems}
                numScroll={1}
                numVisible={1}
                itemTemplate={item => item}
                circular={false}
                activeIndex={0}
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