# Detecting Plastics from Satellite Imagery
[Overview of project](http://minderoo.earthrise.media/)

![Temesi Contours through Time](./assets/Temesi%202019-2020%20Preds%20Transparent.png)

## Project Structure
This repo contains the tools for detecting plastics and/or landfills using Sentinel 2 imagery. The pipeline is mostly run through a set of Jupyter notebooks.

### Notebooks
There are two parallel pipelines. One for the pixelwise spectral classifier, and one for the patch-based spatial classifier. The pixel classifier is responsible for generating candidate sites, while the patch classifier is responsible for validating candidates.

*Pixel Classifier*

- `create_pixel_dataset`: Given a set of sampling locations, generate a set of pixel vectors and labels. These pixel vectors and labels are pickled and saved to `data/training_data/pixel_vectors/`.
- `train_pixel_classifier`: Train a neural network off of data and labels generated in `create_pixel_dataset`. Model is exported to `models/`.
- `run_pixel_classifier`: This functionality is duplicated in the script `nn_predict.py` for a single site, and also within the `train_pixel_classifier` notebook. This is a standalone notebook for evaluating a pixel classifier.

*Patch Classifier*

- `create_patch_dataset`: Given a set of sampling locations, generate a stack of multispectral images and labels. These are pickled and saved to `data/training_data/patches/`.
- `train_patch_classifier`: Train a neural network off of data and labels generated in `create_patch_dataset`. Model is exported to `models/`.
- `run_patch_classifier`: Given a set of candidate sites, run the patch classifier to verify whether the detection is valid. A csv and GeoJSON of confirmed candidate site locations is exported to `data/model_outputs/candidate_sites/`.


*Postprocessing and Monitoring*

- `detect_candidates`: The pixel classifier is run across a larger region on Descartes Labs. It creates a GeoTIFF of per-pixel plastic detection probabilities. `detect_candidates` takes in the heatmap GeoTIFF and isolates "blobs" of high probability predictions. The locations of these blobs are then exported as candidate sites to `data/model_outputs/candidate_sites/`.
- `generate_contours`: Given a list of sites, monitor how they change through time. Generate a site boundary contour as well as compute surface area. Contours and areas are exported as a GeoJSON to `data/model_outputs/site_contours` and images are site-specific contours are exported to `data/model_outputs/site_contours/monthly_contours`. These monthly contours are not tracked on github.

*Other*

- Notebooks that are not yet part of the core pipeline, but may still be useful are stored in the `notebooks/explorations/` directory.
- Notebooks used for data visualization are stored in the `notebooks/visualization/` directory.

### Scripts
Some core functions that are shared across notebooks are defined in `get_s2_data_ee.py` and `viz_tools.py`. 
Other useful scripts include:

- `nn_predict.py`: Run a pixel classifier on a region of interest.
- `deploy_nn_v0.py`: Run the pixel classifier on Descartes Labs.

### Data
- `data/sampling_locations/`: Lists of sampling sites for confirmed positive and negative classes.
- `data/training_data/patch_histories`: Raw export of data downloaded from GEE. These `patch_histories` are the base data structure for most of the pipeline (e.g. they can be turned into pixel vector datasets or patch datasets). They are structured as a dictionary of arrays in the format `[date][site_name][band][band_img]`. Not tracked on github because of file size.
- `data/training_data/pixel_vectors/`: Pickled pixel vector files. These are used to train the pixel classifier. Ideally, each new dataset should have a separate pixel vector. These are then combined when training a network. Not tracked on github because of file size.
- `data/training_data/patches/`: Pickled patch datasets. They are an analogue to the `pixel_vectors` data. Not tracked on github because of file size.
- `data/model_outputs/`: These are derived products created from models. They include output heatmaps, candidate sites, and site monitoring contours.
- `data/misc/`: Files that are useful to track on git, but don't have a core function in the pipeline. Oftentimes one-off dataset creations.

### Models
Pixel and patch classifier models are saved as keras .h5 files to the models directory. Given their small size, they are tracked on github.

### Web Visualizations
This repo works with github pages. Any `.html` file uploaded to the `/docs/` folder are viewable at `https://earthrise-media.github.io/plastics/page_name.html`.

### Assets
Miscellaneous files that are useful to track over time, or need to be referenced in the repo. Generally useful visualizations are often saved here.
