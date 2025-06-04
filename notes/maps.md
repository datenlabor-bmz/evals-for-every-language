Possible sources for maps:

- [Natural Earth](https://www.naturalearthdata.com/): Their main version is not politically correct, and they do provide additional "orld view" data, including for Germany, but not for UN or other international organizations, and it's not very straightforward to use. Also has some issues with ISO2 codes, one can use [`ISO_A2_EH`](https://github.com/nvkelso/natural-earth-vector/issues/284) to work around that; still lacking Somalia though.
- [UN](https://geoportal.un.org/arcgis/apps/sites/#/geohub/datasets/d7caaff3ef4b4f7c82689b7c4694ad92/about): Has some countries inverted, we can mostly [correct for that](https://observablehq.com/@bumbeishvili/rewind-geojson), but it still leaves some artifacts in Norway and the Gulf of Amerxico.
- [World Bank](https://datacatalog.worldbank.org/search/dataset/0038272): Has missing ISO2 country codes for France and Norway.
- [EU](https://ec.europa.eu/eurostat/web/gisco/geodata/administrative-units/countries): Displays very weirdly, haven't looked into the details.
