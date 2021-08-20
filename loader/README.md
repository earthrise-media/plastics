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
