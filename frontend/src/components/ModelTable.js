import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { FilterMatchMode } from 'primereact/api';
import { MultiSelect } from 'primereact/multiselect';
import { useState } from 'react';
import Medal from './Medal';
const ModelTable = ({ data }) => {
    const [filters, setFilters] = useState({
        "provider": { value: null, matchMode: FilterMatchMode.IN },
        "model": { value: null, matchMode: FilterMatchMode.CONTAINS }
    });
    const table = data.model_table;
    const rankBodyTemplate = (rowData) => {
        return <Medal rank={rowData.rank} />;
    };

    const providers = [...new Set(table.map(item => item.provider))];
    const providerRowFilterTemplate = (options) => {
        return (
            <MultiSelect
                value={options.value}
                options={providers}
                onChange={(e) => {
                    options.filterApplyCallback(e.value);
                    setFilters(prevFilters => ({
                        ...prevFilters,
                        provider: { value: e.value, matchMode: FilterMatchMode.IN }
                    }));
                }}
                placeholder="All providers"
            />
        );
    };

  return (
    <DataTable value={table} header={<>AI Models</>} sortField="average" removableSort filters={filters} filterDisplay="menu">
      <Column field="rank" body={rankBodyTemplate} />
      <Column field="provider" header="Provider" filter filterElement={providerRowFilterTemplate} showFilterMatchModes={false} />
      <Column field="model" header="Model" filter showFilterMatchModes={false} />
      <Column field="average" header="Average" sortable />
      <Column field="translation_chrf" header="Translation" sortable />
      <Column field="classification_accuracy" header="Classification" sortable />
      <Column field="language_modeling_chrf" header="Language Modeling" sortable />
    </DataTable>
    );
};

export default ModelTable;