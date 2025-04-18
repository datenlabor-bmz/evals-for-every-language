import { DataTable } from 'primereact/datatable'
import { Column } from 'primereact/column'
import { FilterMatchMode } from 'primereact/api'
import { useState } from 'react'
import { MultiSelect } from 'primereact/multiselect'
import 'primeicons/primeicons.css'

const DatasetTable = ({ data }) => {
  const [filters, setFilters] = useState({
    tasks: { value: null, matchMode: FilterMatchMode.IN },
    translation: { value: null, matchMode: FilterMatchMode.IN },
    n_languages: { value: null, matchMode: FilterMatchMode.BETWEEN },
    parallel: { value: null, matchMode: FilterMatchMode.EQUALS },
    base: { value: null, matchMode: FilterMatchMode.IN },
  })
  const table = data.dataset_table

  const implementedBodyTemplate = rowData => {
    return <div style={{ display: 'flex', alignItems: 'center' }}>
      <div style={{ width: '16px', height: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{rowData.implemented ? <i className='pi pi-check' title='This dataset has been used for evaluation in this benchmark.' /> : <></>}</div>
    </div>
  }

  const authorBodyTemplate = rowData => {
    const url = rowData.author_url?.replace('https://', '')
    const img = url ? <img src={`https://favicone.com/${url}`} style={{borderRadius: '50%'}}/> : <></>
    return <div style={{ display: 'flex', alignItems: 'center' }}>
      <div style={{ width: '16px', height: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{img}</div>
      <div style={{ marginLeft: '0.5rem' }}>{rowData.author}</div>
    </div>
  }

  const nameBodyTemplate = rowData => {
    return <div style={{ fontWeight: 'bold' }}>{rowData.name}</div>
  }

  const tasksBodyTemplate = rowData => {
    return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
      {rowData.tasks.map(task => <div key={task} style={{ backgroundColor: '#f0f0f0', padding: '0.25rem 0.5rem', borderRadius: '0.25rem' }}>{task}</div>)}
    </div>
  }

  const linkBodyTemplate = rowData => {
    return <a href={rowData.url} target='_blank' style={{ textDecoration: 'none', color: 'inherit' }}><i className='pi pi-external-link' style={{ fontSize: '0.8rem' }} /></a>
  }

  const translationBodyTemplate = rowData => {
    const translationIcons = {
      human: <i className='pi pi-user' title='Human-translated' />,
      machine: <i className='pi pi-cog' title='Machine-translated' />,
      mixed: <><i className='pi pi-user' title='Partially human-translated' /> <i className='pi pi-cog' title='Partially machine-translated' /></>,
    }
    const icon = translationIcons[rowData.translation] ?? <></>
    return <div style={{ textAlign: 'center' }}>{icon}</div>
  }

  const nLanguagesBodyTemplate = rowData => {
    return <div style={{ textAlign: 'center' }}>
      {rowData.n_languages}
    </div>
  }

  const tasks = [...new Set(table.flatMap(item => item.tasks))].sort()
  const tasksRowFilterTemplate = options => {
    return (
      <MultiSelect
        value={options.value}
        options={tasks}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            tasks: { value: e.value, matchMode: FilterMatchMode.IN }
          }))
        }}
        placeholder='All tasks'
      />
    )
  }

  const translationRowFilterTemplate = options => {
    return (
      <MultiSelect
        value={options.value}
        options={['human', 'mixed', 'machine']}
        onChange={e => {
          options.filterApplyCallback(e.value)
          setFilters(prevFilters => ({
            ...prevFilters,
            translation: { value: e.value, matchMode: FilterMatchMode.IN }
          }))
        }}
        placeholder='All translation modes'
      />
    )
  }

  return (
    <DataTable
      value={table}
      rowGroupMode='subheader'
      rowGroupHeaderTemplate={rowData => {
        return <div style={{ fontWeight: 'bold' }}>{rowData.group}</div>
      }}
      groupRowsBy='group'
      header={<>Datasets</>}
      removableSort
      filters={filters}
      filterDisplay='menu'
      sortField='implemented'
      scrollable
      scrollHeight='600px'
      id='dataset-table'
      style={{ width: '800px', minHeight: '650px' }}
    >
      <Column
        field='implemented'
        header={null}
        headerTooltip='Whether the dataset has been integrated into this benchmark'
        style={{ maxWidth: '5rem' }}
        body={implementedBodyTemplate}
      />
      <Column
        field='author'
        header='Author'
        showFilterMatchModes={false}
        style={{ minWidth: '5rem' }}
        body={authorBodyTemplate}
      />
      <Column
        field='name'
        header='Name'
        body={nameBodyTemplate}
        style={{ minWidth: '10rem' }}
        frozen
      />
      <Column
        field='link'
        header={null}
        body={linkBodyTemplate}
      />
      <Column
        field='tasks'
        header='Tasks'
        filter
        filterElement={tasksRowFilterTemplate}
        showFilterMatchModes={false}
        style={{ minWidth: '10rem', maxWidth: '15rem' }}
        body={tasksBodyTemplate}
      />
      <Column
        field='translation'
        header={<i className='pi pi-language' />}
        headerTooltip='Whether the dataset has been translated by humans or machines'
        filter
        filterElement={translationRowFilterTemplate}
        showFilterMatchModes={false}
        body={translationBodyTemplate}
      />
      <Column
        field='n_languages'
        header='Languages'
        headerTooltip='Number of languages in the dataset'
        filter
        sortable
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
        body={nLanguagesBodyTemplate}
      />
    </DataTable>
  )
}

export default DatasetTable
