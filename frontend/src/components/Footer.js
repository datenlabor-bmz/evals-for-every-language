const linkStyle = {
  color: 'var(--accent)',
  textDecoration: 'none',
  fontWeight: 500
}

const Footer = () => {
  return (
    <footer
      style={{
        borderTop: '1px solid var(--border)',
        background: 'var(--surface-warm)',
        padding: '3rem 15vw'
      }}
    >
      <div
        style={{
          maxWidth: '900px',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem'
        }}
      >
        <div>
          <div
            style={{
              fontWeight: 600,
              fontSize: '1.2rem',
              letterSpacing: '-0.01em',
              color: 'var(--ink)'
            }}
          >
            AI Language Benchmarks
          </div>
          <div
            style={{
              color: 'var(--ink-muted)',
              fontSize: '0.95rem',
              marginTop: '0.25rem'
            }}
          >
            AI model evaluations for every language in the world.
          </div>
        </div>

        <p
          style={{
            color: 'var(--ink-muted)',
            fontSize: '0.9rem',
            lineHeight: 1.6,
            margin: 0
          }}
        >
          Pomerenke, D., Nothnagel, J., &amp; Ostermann, S. (2025).{' '}
          <a
            href='https://arxiv.org/abs/2507.08538'
            target='_blank'
            rel='noopener noreferrer'
            style={{ ...linkStyle, fontStyle: 'italic' }}
          >
            The AI Language Proficiency Monitor – Tracking the Progress of LLMs
            on Multilingual Benchmarks
          </a>
          . arXiv:2507.08538.
        </p>

        <p
          style={{
            color: 'var(--ink-muted)',
            fontSize: '0.9rem',
            lineHeight: 1.6,
            margin: 0
          }}
        >
          A collaboration of{' '}
          <a
            href='https://www.bmz-digital.global/en/overview-of-initiatives/the-bmz-data-lab/'
            target='_blank'
            rel='noopener noreferrer'
            style={linkStyle}
          >
            BMZ Data Lab
          </a>
          ,{' '}
          <a
            href='https://www.bmz-digital.global/en/overview-of-initiatives/fair-forward/'
            target='_blank'
            rel='noopener noreferrer'
            style={linkStyle}
          >
            GIZ Fair Forward
          </a>
          , and the{' '}
          <a
            href='https://www.dfki.de/en/web/research/research-departments/multilinguality-and-language-technology/ee-team'
            target='_blank'
            rel='noopener noreferrer'
            style={linkStyle}
          >
            DFKI
          </a>{' '}
          Multilinguality and Language Technology Lab.
        </p>

        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '1.75rem',
            fontSize: '0.9rem',
            paddingTop: '1rem',
            borderTop: '1px solid var(--border)'
          }}
        >
          <a
            href='https://github.com/datenlabor-bmz/evals-for-every-language'
            target='_blank'
            rel='noopener noreferrer'
            style={linkStyle}
          >
            <i className='pi pi-github' style={{ marginRight: '0.4rem' }} />
            GitHub
          </a>
          <a
            href='https://huggingface.co/spaces/fair-forward/languagebench'
            target='_blank'
            rel='noopener noreferrer'
            style={linkStyle}
          >
            Hugging Face Space
          </a>
          <a
            href='https://arxiv.org/abs/2507.08538'
            target='_blank'
            rel='noopener noreferrer'
            style={linkStyle}
          >
            Read the paper
          </a>
        </div>
      </div>
    </footer>
  )
}

export default Footer
