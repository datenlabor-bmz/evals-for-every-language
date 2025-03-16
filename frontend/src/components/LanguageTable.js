import { DataTable } from 'primereact/datatable'
import { Column } from 'primereact/column'
import { FilterMatchMode } from 'primereact/api'
import { MultiSelect } from 'primereact/multiselect'
import { useState, useEffect } from 'react'
import { Slider } from 'primereact/slider'
import ScoreField from './ScoreField'

const LanguageTable = ({ data }) => {
  const [filters, setFilters] = useState({
    language_name: { value: null, matchMode: FilterMatchMode.CONTAINS },
    family: { value: null, matchMode: FilterMatchMode.IN },
    speakers: { value: null, matchMode: FilterMatchMode.BETWEEN },
  })
  const table = data.language_table

  const families = [...new Set(table.map(item => item.family))]
  const familyRowFilterTemplate = options => {
    return (
      <MultiSelect
        value={options.value}
        options={families}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            family: { value: e.value, matchMode: FilterMatchMode.IN }
          }))
        }}
        placeholder='All families'
      />
    )
  }

  const formatPopulation = population => {
    if (population === null) {
      return ''
    } else if (population < 1000) {
      return population.toFixed(0) + ''
    } else if (population < 1000 * 1000) {
      return (population / 1000).toFixed(1) + 'K'
    } else if (population < 1000 * 1000 * 1000) {
      return (population / 1000 / 1000).toFixed(1) + 'M'
    } else {
      return (population / 1000 / 1000 / 1000).toFixed(1) + 'B'
    }
  }

  const SliderWithLabel = ({ value, onChange }) => {
    const p = 10
    const min = 2
    const max = 12
    const start = value === null ? min : Math.log(value[0]) / Math.log(p)
    const stop = value === null ? max : Math.log(value[1]) / Math.log(p)
    const [_value, _setValue] = useState([start, stop])
    useEffect(() => {
      const timer = setTimeout(() => {
        onChange({
          value:
            _value[0] <= min + 0.1 && _value[1] >= max - 0.1
              ? null
              : [p ** _value[0], p ** _value[1]]
        })
      }, 1000)
      return () => clearTimeout(timer)
    }, [_value, onChange])
    return (
      <div style={{ minWidth: '20rem' }}>
        <div>{formatPopulation(p ** _value[0])}</div>
        <div>{formatPopulation(p ** _value[1])}</div>
        <Slider
          value={_value}
          onChange={e => _setValue(e.value)}
          placeholder='All sizes'
          min={min}
          max={max}
          step={0.01}
          range
          style={{ marginTop: '5rem' }}
        />
      </div>
    )
  }

  const speakerFilterTemplate = options => {
    return (
      <SliderWithLabel
        value={options.value}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            speakers: { value: e.value, matchMode: FilterMatchMode.BETWEEN }
          }))
        }}
      />
    )
  }

  const speakerBodyTemplate = rowData => {
    const populationStr = formatPopulation(rowData.speakers)
    return <div>{populationStr}</div>
  }

  const languageBodyTemplate = rowData => {
    return <div style={{ fontWeight: 'bold' }}>{rowData.language_name}</div>
  }

  const scoreBodyTemplate = (field, options = {}) => {
    const { minScore = 0, maxScore = 1 } = options

    return rowData => {
      const score = rowData[field]
      return ScoreField(score, minScore, maxScore)
    }
  }

  return (
    <DataTable
      value={table}
      header={<>Languages</>}
      sortField='speakers'
      removableSort
      filters={filters}
      filterDisplay='menu'
      scrollable
      scrollHeight='600px'
      id='language-table'
    >
      <Column
        field='language_name'
        header='Language'
        body={languageBodyTemplate}
        filter
        showFilterMatchModes={false}
        style={{ minWidth: '5rem' }}
        frozen
      />
      <Column
        field='speakers'
        header='Speakers'
        body={speakerBodyTemplate}
        filter
        filterElement={speakerFilterTemplate}
        showFilterMatchModes={false}
        style={{ minWidth: '5rem' }}
      />
      <Column
        field='family'
        header='Family'
        filter
        showFilterMatchModes={false}
        filterElement={familyRowFilterTemplate}
        style={{ minWidth: '10rem' }}
      />
      <Column
        field='average'
        header='Average'
        sortable
        body={scoreBodyTemplate('average', { minScore: 0.2, maxScore: 0.5 })}
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
      />
      <Column
        field='translation_chrf'
        header='Translation'
        sortable
        body={scoreBodyTemplate('translation_chrf', {
          minScore: 0.3,
          maxScore: 0.6
        })}
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
      />
      <Column
        field='classification_accuracy'
        header='Classification'
        sortable
        body={scoreBodyTemplate('classification_accuracy', {
          minScore: 0.3,
          maxScore: 0.7
        })}
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
      />
      <Column
        field='language_modeling_chrf'
        header='Language Modeling'
        sortable
        body={scoreBodyTemplate('language_modeling_chrf', {
          minScore: 0.8,
          maxScore: 1
        })}
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
      />
    </DataTable>
  )
}

export default LanguageTable
