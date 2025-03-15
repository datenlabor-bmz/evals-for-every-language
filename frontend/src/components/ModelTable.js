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

    const sizeBodyTemplate = (rowData) => {
        const size = rowData.size;
        if (size === null) {
            return <div>N/A</div>;
        }
        let sizeStr;
        if (size < 1000) {
            sizeStr = size.toFixed(0) + "";
        } else if (size < 1000 * 1000) {
            sizeStr = (size / 1000).toFixed(0) + "K";
        } else if (size < 1000 * 1000 * 1000) {
            sizeStr = (size / 1000 / 1000).toFixed(0) + "M";
        } else {
            sizeStr = (size / 1000 / 1000 / 1000).toFixed(0) + "B";
        }
        return <div>{sizeStr}</div>;
    };

    const modelBodyTemplate = (rowData) => {
        // bold
        return <div style={{ fontWeight: 'bold' }}>{rowData.model}</div>;
    };

  return (
    <DataTable value={table} header={<>AI Models</>} sortField="average" removableSort filters={filters} filterDisplay="menu" scrollable scrollHeight="500px">
      <Column field="rank" body={rankBodyTemplate} />
      <Column field="provider" header="Provider" filter filterElement={providerRowFilterTemplate} showFilterMatchModes={false} style={{ minWidth: '5rem' }} />
      <Column field="model" header="Model" filter showFilterMatchModes={false} style={{ minWidth: '15rem' }} body={modelBodyTemplate} />
      <Column field="type" header="Type" style={{ minWidth: '10rem' }} />
      <Column field="size" header="Size" sortable body={sizeBodyTemplate} style={{ minWidth: '5rem' }} />
      <Column field="average" header="Average" sortable style={{ minWidth: '5rem' }} />
      <Column field="translation_chrf" header="Translation" sortable style={{ minWidth: '5rem' }} />
      <Column field="classification_accuracy" header="Classification" sortable style={{ minWidth: '5rem' }} />
      <Column field="language_modeling_chrf" header="Language Modeling" sortable style={{ minWidth: '5rem' }} />
    </DataTable>
    );
};

export default ModelTable;