# Loader

Loader for **Global Plastic Watch** data. This takes the model
outputs and loads them into the API.

### Transformations

- Computes centroid with Shapely
- Gets address information from Nominatim

### Installation

Requires shapely, geopy, and requests. Install them globally or with:

```
pip install -r requirements.txt
```

### Running the script

The script refers to handcoded source data in this repository and
doesn't require any arguments - if the location of the data changes,
please update the `source_data` variable to point to it.

```
python load.py
```
