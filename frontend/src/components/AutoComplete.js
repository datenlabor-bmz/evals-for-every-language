import { useState, useEffect, useRef } from 'react'

const AutoComplete = ({ languages, onComplete }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedLanguage, setSelectedLanguage] = useState(null)
  const [filteredLanguages, setFilteredLanguages] = useState([])
  const dropdownRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (!languages) return

    // Most spoken languages (by number of speakers) - you can adjust this list
    const mostSpokenCodes = [
      'en',
      'zh',
      'hi',
      'es',
      'ar',
      'bn',
      'pt',
      'ru',
      'ja',
      'pa',
      'de',
      'jv',
      'ko',
      'fr',
      'te',
      'mr',
      'tr',
      'ta',
      'vi',
      'ur'
    ]

    if (searchTerm.trim() === '') {
      // Show most spoken languages first, then others
      const mostSpoken = mostSpokenCodes
        .map(code => languages.find(lang => lang.bcp_47 === code))
        .filter(Boolean)

      const others = languages
        .filter(lang => !mostSpokenCodes.includes(lang.bcp_47))
        .sort((a, b) => a.language_name.localeCompare(b.language_name))

      setFilteredLanguages([...mostSpoken, ...others])
    } else {
      const query = searchTerm.toLowerCase()
      const matches = languages.filter(
        language =>
          language.language_name.toLowerCase().includes(query) ||
          language.autonym.toLowerCase().includes(query) ||
          language.bcp_47.toLowerCase().includes(query)
      )
      setFilteredLanguages(matches)
    }
  }, [searchTerm, languages])

  useEffect(() => {
    const handleClickOutside = event => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
        setSearchTerm('')
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSelect = language => {
    setSelectedLanguage(language)
    setIsOpen(false)
    setSearchTerm('')
    onComplete([language])
  }

  const handleClear = e => {
    e.stopPropagation()
    setSelectedLanguage(null)
    onComplete([])
  }

  const handleContainerClick = () => {
    setIsOpen(true)
    if (!selectedLanguage) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }

  const handleInputChange = e => {
    // If user starts typing while a language is selected, clear the selection to enable search
    if (selectedLanguage && e.target.value.length > 0) {
      setSelectedLanguage(null)
      onComplete([])
    }
    setSearchTerm(e.target.value)
  }

  const handleKeyDown = e => {
    if (e.key === 'Escape') {
      setIsOpen(false)
      setSearchTerm('')
    }
  }

  const containerStyle = {
    position: 'relative',
    display: 'inline-block',
    minWidth: '400px',
    maxWidth: '600px'
  }

  const buttonStyle = {
    color: selectedLanguage ? '#333' : '#666',
    border: '1px solid #ddd',
    padding: '0.75rem 1rem',
    borderRadius: '4px',
    fontSize: '0.95rem',
    backgroundColor: '#fff',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    minHeight: '44px',
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease'
  }

  const inputStyle = {
    border: 'none',
    outline: 'none',
    fontSize: '0.95rem',
    width: '100%',
    backgroundColor: 'transparent',
    color: '#333'
  }

  const dropdownStyle = {
    position: 'absolute',
    top: '100%',
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    border: '1px solid #ddd',
    borderTop: 'none',
    borderRadius: '0 0 4px 4px',
    maxHeight: '300px',
    overflowY: 'auto',
    zIndex: 1000,
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
  }

  const itemStyle = {
    padding: '0.75rem 1rem',
    cursor: 'pointer',
    borderBottom: '1px solid #f0f0f0',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    transition: 'background-color 0.2s ease'
  }

  const clearButtonStyle = {
    background: 'none',
    border: 'none',
    color: '#999',
    cursor: 'pointer',
    padding: '0.25rem',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '20px',
    height: '20px',
    fontSize: '14px',
    marginLeft: '0.5rem'
  }

  return (
    <div style={containerStyle} ref={dropdownRef}>
      <div
        style={buttonStyle}
        onClick={handleContainerClick}
        onMouseEnter={e => {
          if (!selectedLanguage) {
            e.target.style.borderColor = '#bbb'
          }
        }}
        onMouseLeave={e => {
          e.target.style.borderColor = '#ddd'
        }}
      >
        {selectedLanguage && !isOpen ? (
          <>
            <span style={{ fontWeight: '500' }}>
              {selectedLanguage.language_name} Leaderboard
            </span>
            <button
              style={clearButtonStyle}
              onClick={handleClear}
              onMouseEnter={e => {
                e.target.style.backgroundColor = '#f0f0f0'
              }}
              onMouseLeave={e => {
                e.target.style.backgroundColor = 'transparent'
              }}
              title='View overall leaderboard'
            >
              ×
            </button>
          </>
        ) : isOpen ? (
          <input
            ref={inputRef}
            style={inputStyle}
            placeholder={
              selectedLanguage
                ? 'Type to search other languages...'
                : 'Type to search languages...'
            }
            value={searchTerm}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
          />
        ) : (
          <span>Go to leaderboard for specific language...</span>
        )}
        {(!selectedLanguage || isOpen) && (
          <span style={{ color: '#999', fontSize: '12px' }}>
            {isOpen ? '▲' : '▼'}
          </span>
        )}
      </div>

      {isOpen && (
        <div style={dropdownStyle}>
          {filteredLanguages.length === 0 ? (
            <div style={{ ...itemStyle, color: '#999', cursor: 'default' }}>
              No languages found
            </div>
          ) : (
            filteredLanguages.slice(0, 20).map((language, index) => (
              <div
                key={language.bcp_47}
                style={itemStyle}
                onClick={() => handleSelect(language)}
                onMouseEnter={e => {
                  e.target.style.backgroundColor = '#f8f9fa'
                }}
                onMouseLeave={e => {
                  e.target.style.backgroundColor = 'transparent'
                }}
              >
                <div>
                  <div style={{ fontWeight: '500', marginBottom: '2px' }}>
                    {language.language_name} Leaderboard
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#666' }}>
                    {language.autonym}
                  </div>
                </div>
                <div style={{ color: '#999', fontSize: '0.8rem' }}>
                  {language.bcp_47}
                </div>
              </div>
            ))
          )}
          {filteredLanguages.length > 20 && (
            <div
              style={{
                ...itemStyle,
                color: '#999',
                cursor: 'default',
                fontStyle: 'italic'
              }}
            >
              Type to search for more languages...
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default AutoComplete
