import { DataTable } from 'primereact/datatable'
import { Column } from 'primereact/column'
import { FilterMatchMode } from 'primereact/api'
import { MultiSelect } from 'primereact/multiselect'
import { useState, useEffect } from 'react'
import Medal from './Medal'
import { Slider } from 'primereact/slider'
import ScoreColumns from './ScoreColumns'
const ModelTable = ({ data, selectedLanguages = [], allLanguages = [] }) => {
  const [filters, setFilters] = useState({
    type: { value: null, matchMode: FilterMatchMode.IN },
    size: { value: null, matchMode: FilterMatchMode.BETWEEN },
    cost: { value: null, matchMode: FilterMatchMode.BETWEEN }
  })
  const rankBodyTemplate = rowData => {
    return <Medal rank={rowData.rank} />
  }

  const typeRowFilterTemplate = options => {
    return (
      <MultiSelect
        value={options.value}
        options={['open-source', 'closed-source']}
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
    } else if (size >= 0 && size <= 1) {
      return size.toFixed(2) + ''
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

  const SliderWithLabel = ({ value, onChange, min, max }) => {
    const p = 10;
    const start = value === null || value[0] === null ? min : Math.log(value[0]) / Math.log(p);
    const stop = value === null || value[1] === null ? max : Math.log(value[1]) / Math.log(p);
    const [_value, _setValue] = useState([start, stop]);
    useEffect(() => {
      const timer = setTimeout(() => {
        onChange({
          value:
            // set to "no filter" when (almost) the whole range is selected
            _value[0] <= min + 0.1 && _value[1] >= max - 0.1
              ? null
              : [p ** _value[0], p ** _value[1]],
        });
      }, 1000);
      return () => clearTimeout(timer);
    }, [_value, onChange, min, max]);
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
        min={8}
        max={12}
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

  const costFilterTemplate = options => {
    return (
      <SliderWithLabel
        value={options.value}
        min={-2}
        max={2}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            cost: { value: e.value, matchMode: FilterMatchMode.BETWEEN }
          }))
        }}
      />
    )
  }

  const sizeBodyTemplate = rowData => {
    const sizeStr = formatSize(rowData.size)
    return (
      <div style={{ textAlign: 'center' }}>
        <a
          href={`https://huggingface.co/${rowData.hf_id}`}
          target='_blank'
          rel='noopener noreferrer'
          style={{ textDecoration: 'none', color: 'inherit' }}
        >
          {sizeStr}
        </a>
      </div>
    )
  }

  const modelBodyTemplate = rowData => (
    <div style={{ fontWeight: 'bold', height: '100%' }}>{rowData.name}</div>
  )

  const typeBodyTemplate = rowData => {
    return rowData.type === 'open-source' ? (
      <i className='pi pi-lock-open' title='Open-source model' />
    ) : (
      <i className='pi pi-lock' title='Closed-source model' />
    )
  }

  const costBodyTemplate = rowData => {
    return (
      <div style={{ textAlign: 'center' }}>
        {rowData.cost === null ? 'n/a' : `$${rowData.cost.toFixed(2)}`}
      </div>
    )
  }

  const getHeaderText = () => {
    // Count languages that have evaluation data (average score available)
    const evaluatedLanguagesCount = allLanguages.filter(lang => 
      lang.average !== null && lang.average !== undefined
    ).length

    if (selectedLanguages.length === 0) {
      return (
        <span>
          <span style={{ fontWeight: 'bold', fontSize: '1.1em' }}>AI Models</span>
          <span style={{ fontSize: '0.85em', marginLeft: '0.5rem' }}>
            Average performance across {evaluatedLanguagesCount} evaluated languages
          </span>
        </span>
      )
    } else if (selectedLanguages.length === 1) {
      return (
        <span>
          <span style={{ fontWeight: 'bold', fontSize: '1.1em' }}>AI Models</span>
          <span style={{ fontSize: '0.85em', marginLeft: '0.5rem' }}>
            {selectedLanguages[0].language_name} performance
          </span>
        </span>
      )
    } else {
      const languageNames = selectedLanguages.map(lang => lang.language_name).join(', ')
      return (
        <span>
          <span style={{ fontWeight: 'bold', fontSize: '1.1em' }}>AI Models</span>
          <span style={{ fontSize: '0.85em', marginLeft: '0.5rem' }}>
            Performance for {languageNames}
          </span>
        </span>
      )
    }
  }

  return (
    <DataTable
      value={data}
      header={<>{getHeaderText()}</>}
      sortField='average'
      removableSort
      filters={filters}
      filterDisplay='menu'
      scrollable
      scrollHeight='600px'
      id='model-table'
      style={{ width: '100%', minHeight: '650px' }}
      emptyMessage='No models have been evaluated for the selected languages.'
    >
      <Column field='rank' body={rankBodyTemplate} headerTooltip='Rank' />
      <Column
        field='provider_name'
        header='Provider'
        style={{ minWidth: '7rem' }}
      />
      <Column
        field='name'
        header='Model'
        style={{ minWidth: '10rem' }}
        body={modelBodyTemplate}
        frozen
      />
      <Column
        field='type'
        header={<i className='pi pi-unlock' title='Open-source / Closed-source' />}
        headerTooltip='Open-source / Closed-source'
        filter
        filterElement={typeRowFilterTemplate}
        showFilterMatchModes={false}
        body={typeBodyTemplate}
      />
      <Column
        field='size'
        header='Size'
        headerTooltip='Number of parameters'
        filter
        filterElement={sizeFilterTemplate}
        showFilterMatchModes={false}
        sortable
        body={sizeBodyTemplate}
        style={{ minWidth: '5rem' }}
      />
      <Column
        field='cost'
        header='Cost'
        headerTooltip='Cost in USD per million completion tokens'
        filter
        filterElement={costFilterTemplate}
        showFilterMatchModes={false}
        sortable
        body={costBodyTemplate}
        style={{ minWidth: '5rem' }}
      />
      {ScoreColumns}
    </DataTable>
  )
}

export default ModelTable
