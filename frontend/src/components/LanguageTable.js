import { DataTable } from 'primereact/datatable'
import { Column } from 'primereact/column'
import { FilterMatchMode } from 'primereact/api'
import { MultiSelect } from 'primereact/multiselect'
import { useState, useEffect } from 'react'
import { Slider } from 'primereact/slider'
import ScoreColumns from './ScoreColumns'

const LanguageTable = ({ data, selectedLanguages, setSelectedLanguages, totalModels = 0 }) => {
  const [filters, setFilters] = useState({
    language_name: { value: null, matchMode: FilterMatchMode.CONTAINS },
    family: { value: null, matchMode: FilterMatchMode.IN },
    speakers: { value: null, matchMode: FilterMatchMode.BETWEEN }
  })

  const families = [...new Set(data.map(item => item.family))].slice(0, 10)
  families.push('Other')
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
    } else if (population < 10 * 1000) {
      return (population / 1000).toFixed(1) + 'K'
    } else if (population < 1000 * 1000) {
      return (population / 1000).toFixed(0) + 'K'
    } else if (population < 10 * 1000 * 1000) {
      return (population / 1000 / 1000).toFixed(1) + 'M'
    } else if (population < 1000 * 1000 * 1000) {
      return (population / 1000 / 1000).toFixed(0) + 'M'
    } else {
      return (population / 1000 / 1000 / 1000).toFixed(1) + 'B'
    }
  }

  const SliderWithLabel = ({ value, onChange }) => {
    const p = 10
    const min = 4
    const max = 9.3
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
    return <div style={{ textAlign: 'center' }}>{populationStr}</div>
  }

  const languageBodyTemplate = rowData => {
    return (
      <div>
        <div style={{ fontWeight: 'bold' }}>{rowData.autonym}</div>
        <div style={{ fontSize: '0.8rem', color: 'gray' }}>
          {rowData.language_name}
        </div>
      </div>
    )
  }

  return (
    <DataTable
      value={data.filter(
        item => !selectedLanguages.some(l => l.bcp_47 === item.bcp_47)
      )}
      header={
        <span>
          <span style={{ fontWeight: 'bold', fontSize: '1.1em' }}>Languages</span>
          <span style={{ fontSize: '0.85em', marginLeft: '0.5rem' }}>
            Average performance of {totalModels} evaluated AI models
          </span>
        </span>
      }
      sortField='speakers'
      removableSort
      filters={filters}
      filterDisplay='menu'
      selectionMode='checkbox'
      selection={selectedLanguages}
      onSelectionChange={e => setSelectedLanguages(e.value)}
      frozenValue={selectedLanguages}
      virtualScrollerOptions={{ itemSize: 60 }}
      scrollable
      scrollHeight='600px'
      id='language-table'
      style={{ width: '100%', minHeight: '650px' }}
    >
      <Column selectionMode='multiple' headerStyle={{ width: '3rem' }} />
      <Column
        field='language_name'
        header='Language'
        body={languageBodyTemplate}
        style={{ minWidth: '10rem' }}
        filter
        showFilterMatchModes={false}
        frozen
      />
      <Column
        field='speakers'
        header={<i className='pi pi-users' title='Speakers' />}
        headerTooltip='Number of speakers of the language (according to CLDR 2018)'
        body={speakerBodyTemplate}
        filter
        filterElement={speakerFilterTemplate}
        showFilterMatchModes={false}
        sortable
        style={{ minWidth: '5rem' }}
      />
      <Column
        field='family'
        header='Family'
        headerTooltip='Language family'
        filter
        showFilterMatchModes={false}
        filterElement={familyRowFilterTemplate}
        style={{ minWidth: '10rem' }}
      />
      {ScoreColumns()}
    </DataTable>
  )
}

export default LanguageTable
