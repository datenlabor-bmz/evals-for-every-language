import { DataTable } from 'primereact/datatable'
import { Column } from 'primereact/column'
import { FilterMatchMode } from 'primereact/api'
import { useState } from 'react'

const DatasetTable = ({ data }) => {
  const [filters, setFilters] = useState({
    name: { value: null, matchMode: FilterMatchMode.CONTAINS },
    author: { value: null, matchMode: FilterMatchMode.IN },
    n_languages: { value: null, matchMode: FilterMatchMode.BETWEEN },
    tasks: { value: null, matchMode: FilterMatchMode.IN },
    parallel: { value: null, matchMode: FilterMatchMode.EQUALS },
    base: { value: null, matchMode: FilterMatchMode.IN },
    implemented: { value: null, matchMode: FilterMatchMode.EQUALS },
  })
  const table = data.dataset_table

  const nameBodyTemplate = rowData => {
    return <div style={{ fontWeight: 'bold' }}>{rowData.name}</div>
  }


  return (
    <DataTable
      value={table}
      header={<>Datasets</>}
      removableSort
      filters={filters}
      filterDisplay='menu'
      scrollable
      scrollHeight='500px'
      style={{ minWidth: '200px', width: "50%" }}
    >
      {/* <Column
        field='implemented'
        header='Implemented'
        filter
        style={{ minWidth: '5rem' }}
      /> */}
      <Column
        field='author'
        header='Author'
        filter
        showFilterMatchModes={false}
        style={{ minWidth: '5rem' }}
      />
      <Column
        field='name'
        header='Name'
        body={nameBodyTemplate}
        filter
        style={{ minWidth: '5rem' }}
        frozen
      />
      <Column
        field='tasks'
        header='Tasks'
        filter
        style={{ minWidth: '5rem', maxWidth: '10rem' }}
      />
      <Column
        field='n_languages'
        header='#Languages'
        filter
        sortable
        style={{ minWidth: '10rem' }}
      />
    </DataTable>
  )
}

export default DatasetTable
