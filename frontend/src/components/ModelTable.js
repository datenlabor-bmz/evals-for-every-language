import { DataTable } from 'primereact/datatable'
import { Column } from 'primereact/column'
import { FilterMatchMode } from 'primereact/api'
import { MultiSelect } from 'primereact/multiselect'
import { useState, useEffect } from 'react'
import Medal from './Medal'
import { Slider } from 'primereact/slider'
import ScoreField from './ScoreField'

const ModelTable = ({ data }) => {
  const [filters, setFilters] = useState({
    provider: { value: null, matchMode: FilterMatchMode.IN },
    model: { value: null, matchMode: FilterMatchMode.CONTAINS },
    type: { value: null, matchMode: FilterMatchMode.IN },
    size: { value: null, matchMode: FilterMatchMode.BETWEEN }
  })
  const table = data.model_table
  const rankBodyTemplate = rowData => {
    return <Medal rank={rowData.rank} />
  }

  const providers = [...new Set(table.map(item => item.provider))]
  const providerRowFilterTemplate = options => {
    return (
      <MultiSelect
        value={options.value}
        options={providers}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            provider: { value: e.value, matchMode: FilterMatchMode.IN }
          }))
        }}
        placeholder='All providers'
      />
    )
  }
  const typeRowFilterTemplate = options => {
    return (
      <MultiSelect
        value={options.value}
        options={['Open', 'Commercial']}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            type: { value: e.value, matchMode: FilterMatchMode.IN }
          }))
        }}
        placeholder='All types'
      />
    )
  }

  const formatSize = size => {
    if (size === null) {
      return ''
    } else if (size < 1000) {
      return size.toFixed(0) + ''
    } else if (size < 1000 * 1000) {
      return (size / 1000).toFixed(0) + 'K'
    } else if (size < 1000 * 1000 * 1000) {
      return (size / 1000 / 1000).toFixed(0) + 'M'
    } else {
      return (size / 1000 / 1000 / 1000).toFixed(0) + 'B'
    }
  }

  const SliderWithLabel = ({ value, onChange }) => {
    const p = 10
    const min = 8
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
        <div>{formatSize(p ** _value[0])}</div>
        <div>{formatSize(p ** _value[1])}</div>
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

  const sizeFilterTemplate = options => {
    return (
      <SliderWithLabel
        value={options.value}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            size: { value: e.value, matchMode: FilterMatchMode.BETWEEN }
          }))
        }}
      />
    )
  }

  const sizeBodyTemplate = rowData => {
    const sizeStr = formatSize(rowData.size)
    return <div>{sizeStr}</div>
  }

  const modelBodyTemplate = rowData => {
    return <div style={{ fontWeight: 'bold' }}>{rowData.model}</div>
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
      header={<>AI Models</>}
      sortField='average'
      removableSort
      filters={filters}
      filterDisplay='menu'
      scrollable
      scrollHeight='500px'
      style={{ minWidth: '200px' }}
    >
      <Column field='rank' body={rankBodyTemplate} />
      <Column
        field='provider'
        header='Provider'
        filter
        filterElement={providerRowFilterTemplate}
        showFilterMatchModes={false}
        style={{ minWidth: '5rem' }}
      />
      <Column
        field='model'
        header='Model'
        filter
        showFilterMatchModes={false}
        style={{ minWidth: '15rem' }}
        body={modelBodyTemplate}
        frozen
      />
      <Column
        field='type'
        header='Type'
        filter
        filterElement={typeRowFilterTemplate}
        showFilterMatchModes={false}
        style={{ minWidth: '10rem' }}
      />
      <Column
        field='size'
        header='Size'
        filter
        filterElement={sizeFilterTemplate}
        showFilterMatchModes={false}
        sortable
        body={sizeBodyTemplate}
        style={{ minWidth: '5rem' }}
      />
      <Column
        field='average'
        header='Average'
        sortable
        body={scoreBodyTemplate('average', { minScore: 0.4, maxScore: 0.8 })}
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
      />
      <Column
        field='translation_chrf'
        header='Translation'
        sortable
        body={scoreBodyTemplate('translation_chrf', {
          minScore: 0.4,
          maxScore: 0.7
        })}
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
      />
      <Column
        field='classification_accuracy'
        header='Classification'
        sortable
        body={scoreBodyTemplate('classification_accuracy', {
          minScore: 0.4,
          maxScore: 1
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

export default ModelTable
