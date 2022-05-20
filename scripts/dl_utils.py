import datetime
import json
import os

import descarteslabs as dl
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import numpy as np
from scipy.stats import mode
import shapely
from tensorflow import keras

SENTINEL_BANDS = ['coastal-aerosol',
                  'blue',
                  'green',
                  'red',
                  'red-edge',
                  'red-edge-2',
                  'red-edge-3',
                  'nir',
                  'red-edge-4',
                  'water-vapor',
                  'swir1',
                  'swir2']

NORMALIZATION = 3000

def rect_from_point(coord, rect_height):
    """
    Create a geojson polygon from a coordinate pair.
    Inputs:
        - coord: coordinates in the form [lon, lat]
        - rect_height: the height of the rectangle in degrees latitude
    Returns: Geojson formatted polygon
    """
    lon, lat = coord
    lat_w = rect_height / 2
    lon_w = lat_w / np.cos(np.deg2rad(lat))
    rect = shapely.geometry.mapping(shapely.geometry.box(
        lon - lon_w, lat - lat_w, lon + lon_w, lat + lat_w))
    return rect

def get_tiles_from_roi(roi_file, tilesize, pad):
    """Retrieve tile keys covering ROI."""
    with open(roi_file, 'r') as f:
        fc = json.load(f)
        try:
            features = fc['features']
        except KeyError:
            features = fc['geometries']

    all_keys = list()
    ctr =0
    resolution = 10.0
    for feature in features:
        keys_iter = dl.scenes.DLTile.iter_from_shape(feature['geometry'], 
                                                     resolution, 
                                                     tilesize, 
                                                     pad, 
                                                     keys_only=True)
        keys = [k for k in keys_iter]
        all_keys.extend(keys)
        ctr += len(keys)
        print(ctr, end='\r')

    print('Split ROI into {} tiles'.format(len(all_keys)))
    return all_keys

def write_dlkeys(dlkeys, dlkeys_file):
    with open(dlkeys_file, 'w') as f:
        for dlkey in dlkeys:
            f.write(f"{dlkey}\n")

    print(f"Wrote dlkeys to {dlkeys_file}")
    return True

def read_dlkeys(dlkeys_file):
    with open(dlkeys_file, 'r') as f:
        dlkeys = [l.rstrip() for l in f]

    print(f"Read {len(dlkeys)} dlkeys from {dlkeys_file}")
    return dlkeys

def filt_tiles_by_pop(all_keys, pop_thresh):
    """Filter keys by a population threshold."""
    if pop_thresh == 0:
        return all_keys

    pop_id = "d15c019579fa0985f7006094bba7c7288f830e1f:GPW_Population_Density_V4_0"
    pop_bands = ['population']
    start_datetime = '2019-01-01'
    end_datetime = '2021-01-01'
    resolution = 1000 # resolution to rasterize population data at

    keys_filt = list()
    ctr = 0
    print(f"Filtering keys by population threshold of {pop_thresh}")
    for dlkey in all_keys:
        tile = dl.scenes.DLTile.from_key(dlkey)

        # make this an AOI so we can operate at a lower resolution
        aoi = dl.scenes.AOI(geometry=tile.geometry,
                            resolution=resolution,
                            crs=tile.crs)

        scenes, ctx = dl.scenes.search(products=pop_id,
                                       aoi=aoi,
                                       start_datetime=start_datetime,
                                       end_datetime=end_datetime)

        if not scenes:
            continue

        pop = scenes.mosaic(bands=pop_bands, ctx=ctx)
        if np.any(pop > pop_thresh):
            keys_filt.append(dlkey)
            ctr += 1
            print(ctr, end='\r')

    print(f"Filtered to {len(keys_filt)}")
    return keys_filt

def pad_patch(patch, height, width=None):
    """
    Depending on how a polygon falls across pixel boundaries, it can be slightly
    bigger or smaller than intended.
    pad_patch trims pixels extending beyond the desired number of pixels if the
    patch is larger than desired. If the patch is smaller, it will fill the
    edge by reflecting the values.
    Note: this function does not support padding along one axis, and trimming on another (or visa versa)
    """

    h, w, c = patch.shape
    if width:
        if h < height or w < width:
            padding_amount = ((0, height - h), (0, width - w), (0,0))
            patch = np.pad(patch, padding_amount, mode='reflect')
        else:
            margin_h = int(np.floor((h - height) / 2))
            margin_w = int(np.floor((w - width) / 2))
            patch = patch[margin_h:margin_h+height, margin_w:margin_w+width]

    else:
        if h < height or w < height:
            padding_amount = ((0, height - h), (0, height - w), (0, 0))
            patch = np.pad(patch, padding_amount, mode='reflect')
        else:
            margin = int(np.floor((h - height) / 2))
            patch = patch[margin:margin+height, margin:margin+height]
    return patch

def download_patch(polygon, start_date, end_date, s2_id='sentinel-2:L1C',
                   s2cloud_id='sentinel-2:L1C:dlcloud:v1'):
    """
    Download a stack of cloud-masked Sentinel data
    Inputs:
        - polygon: Geojson polygon enclosing the region of data to be extracted
        - start_date/end_date: The time bounds of the search
    Returns:
        - A list of images of shape (height, width, channels)
    """
    cloud_scenes, _ = dl.scenes.search(
        polygon,
        products=[s2cloud_id],
        start_datetime=start_date,
        end_datetime=end_date,
        limit=None
    )
    #print([s.properties.cloud_fraction for s in cloud_scenes])
    #cloud_scenes = cloud_scenes.filter(lambda s: s.properties.cloud_fraction  < 0.5)
    #print([s.properties.file_sizes[0] for s in cloud_scenes])
    #cloud_scenes = cloud_scenes.filter(lambda s: s.properties.file_sizes[0] / 2140200 > 0.8)
    #print(len(cloud_scenes), "scenes after filter for", start_date, end_date)
    scenes, geoctx = dl.scenes.search(
        polygon,
        products=[s2_id],
        start_datetime=start_date,
        end_datetime=end_date,
        limit=None,
        cloud_fraction=0.3
    )
    scenes = scenes.filter(lambda s: s.coverage(geoctx) > 0.9)

    cloud_keys = [scene.properties.key for scene in cloud_scenes]
    data_keys = [scene.properties.key for scene in scenes]
    shared_keys = set(cloud_keys) & set(data_keys)
    scenes = scenes.filter(
        lambda x: x.properties.key in shared_keys)
    cloud_scenes = cloud_scenes.filter(
        lambda x: x.properties.key in shared_keys)

    if len(scenes) > 0 and len(cloud_scenes) > 0:
        print("Downloading", len(scenes), "scenes at", datetime.datetime.isoformat(datetime.datetime.now())[:19], end='\r')
        # A cloud stack is an array with shape (num_img, data_band, height, width)
        # A value of 255 means that the pixel is cloud free, 0 means cloudy
        cloud_stack = cloud_scenes.stack(bands=['valid_cloudfree'], ctx=geoctx)

        img_stack, raster_info = scenes.stack(
            bands=SENTINEL_BANDS, ctx=geoctx, raster_info=True)
        # add date to metadata
        dates = [scene.properties.acquired[:10] for scene in scenes]
        for i, date in enumerate(dates):
            raster_info[i]['date'] = date
        cloud_masks = np.repeat(cloud_stack, repeats = 12, axis=1)
        # Add cloud masked pixels to the image mask
        img_stack.mask[cloud_masks.data == 0] = True

        # Remove fully masked images and reorder to channels last
        metadata = []
        for img, info in zip(img_stack, raster_info):
            if np.sum(img) > 0:
                metadata.append(info)

        img_stack = [np.moveaxis(img, 0, -1) for img in img_stack
                        if np.sum(img) > 0]

        return img_stack, metadata
    
    else:
        return [], []

def download_batches(polygon, start_date, end_date, batch_months):
    """Download cloud-masked Sentinel imagery in time-interval batches.

    Args:
        polygon: A GeoJSON-like polygon
        start_date: Isoformat start date
        end_date: Isoformat end start
        batch_months: Batch length in integer number of months

    Returns: List of lists of images, one list per batch
    """
    batches, raster_infos = [], []
    delta = relativedelta(months=batch_months)
    start = datetime.date.fromisoformat(start_date)
    end = start + delta
    while end <= datetime.date.fromisoformat(end_date):
        try:
            print("Downloading batch from", start.isoformat(), "to", end.isoformat(), end='\r')
            batch, raster_info = download_patch(polygon, start.isoformat(),
                                                end.isoformat())
        except IndexError as e:
            print(f'Failed to retrieve month {start.isoformat()}: {repr(e)}')
            batch, raster_info = [], []
        # Sometimes there are patches with no data. Ignore those
        if len(np.shape(batch)) > 1:
            batches.append(batch)
            raster_infos.append(raster_info)
        start += delta
        end += delta
    return batches, raster_infos

def get_starts(start_date, end_date, mosaic_period, spectrogram_length):
    """Get spectrogram start dates."""
    starts = []
    delta = relativedelta(months=mosaic_period)
    length = relativedelta(months=spectrogram_length)
    start = datetime.date.fromisoformat(start_date)
    while start + length <= datetime.date.fromisoformat(end_date):
        starts.append(start.isoformat())
        start += delta
    return starts

def download_mosaics(polygon, start_date, end_date, mosaic_period=1,
                     method='median'):
    """Download cloud-masked Sentinel image mosaics

    Args:
        polygon: A GeoJSON-like polygon
        start_date: Isoformat start date
        end_date: Isoformat end start
        mosaic_period: Integer months over which to mosaic image data
        method: String method to pass to mosaic()

    Returns: List of image mosaics and list of meta-datas
    """
    batches, raster_infos = download_batches(polygon, start_date, end_date,
                                                 mosaic_period)
    mosaics = [mosaic(batch, method) for batch in batches]
    # There are cases where some patches are sized differently
    # If that is the case, pad/clip them to the same shape
    heights = [np.shape(img)[0] for img in mosaics]
    widths = [np.shape(img)[1] for img in mosaics]
    if len(np.unique(heights)) > 1 or len(np.unique(widths)) > 1:
        h = mode(heights).mode[0]
        w = mode(widths).mode[0]
        mosaics = [np.ma.masked_array(pad_patch(img.data, h, w),
                                        pad_patch(img.mask, h, w)) for img in mosaics]
    mosaic_info = [r for r in raster_infos]
    return mosaics, mosaic_info

def mosaic(arrays, method):
    """Mosaic masked arrays.

    Args:
        arrays: A list of masked arrays
        method:
            'median': return the median of valid pixel values
            'min': return the minimum of valid pixel values
            'min_masked': return the array with fewest masked pixels

    Returns: A masked array or None if arrays is an empty list
    """
    if not arrays:
        return

    if method == 'median':
        stack = np.ma.stack(arrays)
        reduced = np.ma.median(stack, axis=0)
    elif method == 'min':
        stack = np.ma.stack(arrays)
        reduced = np.ma.min(stack, axis=0)
    elif method == 'min_masked':
        mask_sorted = sorted(arrays, key=lambda p:np.sum(p.mask))
        reduced = next(iter(mask_sorted))
    else:
        raise ValueError(f'Method {method} not recognized.')

    return reduced

def pair(mosaics, interval=6, dates=None):
    """Pair image mosaics from a list.

    Args:
        mosaics: A list of masked arrays
        interval: Integer interval between mosaics, in number of mosaic periods
        dates: Optional arg to return the start dates of the pairs

    Returns: A list of lists of images.
    """
    pairs = [[a, b] for a, b in zip(mosaics, mosaics[interval:])
            if a is not None and b is not None]
    if dates:
        date_list = []
        for date, a,b in zip(dates, mosaics, mosaics[interval:]):
            if a is not None and b is not None:
                date_list.append(date)
        return pairs, date_list
    else:
        return pairs

# WIP: Eventually we want to generalize from pairs to n-grams.
# This is a placeholder in the name-space for an eventual maker of n_grams.
def n_gram(mosaics, interval=6, n=2):
    return pair(mosaics, interval=interval)

# WIP: This needs to be generalized from pair to gram.
def masks_match(pair):
    """Check whether arrays in a pair share the same mask.

    This enforces identical cloud masking on an image pair. Any
    residual mask is expected to define a polygon within the raster.
    """
    return (pair[0].mask == pair[1].mask).all()

def shape_gram_as_pixels(gram):
    """Convert a sequence of images into a pixel-wise array of data samples.

    Returns: Array of pixel elements, each having shape (channels, len(gram))
    """
    height, width, channels = next(iter(gram)).shape
    pixels = np.moveaxis(np.array(gram), 0, -1)
    pixels = pixels.reshape(height * width, channels, len(gram))
    return pixels

def normalize(x):
    return np.array(x) / NORMALIZATION

def unit_norm_pixel(samples):
    """
    Channel-wise normalization of pixels in a vector.
    Means and deviations are constants generated from an earlier dataset.
    If changed, models will need to be retrained
    Input: (12) numpy array or list.
    Returns: normalized numpy array
    """
    means = [1367.8407, 1104.4116, 1026.8099, 856.1295, 1072.1476, 1880.3287, 2288.875, 2104.5999, 2508.7764, 305.3795, 1686.0194, 946.1319]
    deviations = [249.14418, 317.69983, 340.8048, 467.8019, 390.11594, 529.972, 699.90826, 680.56006, 798.34937, 108.10846, 651.8683, 568.5347]
    normalized_samples = ((samples - np.reshape(means, (1, 12, 1))) / (np.reshape(deviations, (1, 12, 1))))
    return normalized_samples

def unit_norm(samples):
    """
    Channel-wise normalization of pixels in a patch.
    Means and deviations are constants generated from an earlier dataset.
    If changed, models will need to be retrained
    Input: (n,n,12) numpy array or list.
    Returns: normalized numpy array
    """
    means = [1367.8407, 1104.4116, 1026.8099, 856.1295, 1072.1476, 1880.3287, 2288.875, 2104.5999, 2508.7764, 305.3795, 1686.0194, 946.1319]
    deviations = [249.14418, 317.69983, 340.8048, 467.8019, 390.11594, 529.972, 699.90826, 680.56006, 798.34937, 108.10846, 651.8683, 568.5347]
    normalized_samples = np.zeros_like(samples).astype('float32')
    for i in range(0, 12):
        #normalize each channel to global unit norm
        normalized_samples[:,:,i] = (np.array(samples)[:,:,i] - means[i]) / deviations[i]
    return normalized_samples

# WIP: needs to be generalized pairs -> grams
def preds_to_image(preds, input_pair):
    """Reshape and mask spectrogram model predictions."""
    channel00 = input_pair[0][:,:,0]
    channel10 = input_pair[1][:,:,0]
    img = preds.reshape(channel00.shape)
    mask = channel00.mask | channel10.mask | np.isnan(img)
    return np.ma.array(img, mask=mask)

def predict_spectrogram(image_gram, model, unit_norm=False):
    """Run a spectrogram model on a pair of images."""
    pixels = shape_gram_as_pixels(image_gram)
    if unit_norm:
        input_array = np.expand_dims(unit_norm_pixel(pixels), -1)
    else:
        input_array = np.expand_dims(normalize(pixels), -1)
    preds = model.predict(input_array)[:,1]
    output_img = preds_to_image(preds, image_gram)
    return output_img

def load_ensemble(folder_path):
    """Load all models in a directory. Outputs a list of models"""
    model_files = [file for file in os.listdir(folder_path) if '.h5' in file]
    model_list = []
    for file in model_files:
        model_list.append(keras.models.load_model(os.path.join(folder_path,file)))
    return model_list

def predict_ensemble(pairs, model_list, method='median'):
    """Given a list of models and list of pairs, output a combined prediction output"""
    ensemble_preds = []
    for pair in pairs:
        pred_stack = []
        for ensemble_model in model_list:
            pred_stack.append(predict_spectrogram(pair, ensemble_model, unit_norm=True))
        if method == 'median':
            ensemble_preds.append(np.median(pred_stack, axis=0))
        if method == 'mean':
            ensemble_preds.append(np.mean(pred_stack, axis=0))
    return ensemble_preds


def patches_from_tile(tile, raster_info, width, stride):
    """
    Break a larger tile of Sentinel data into a set of patches that
    a model can process.
    Inputs:
        - tile: Sentinel data. Typically a numpy masked array
        - raster_info: Descartes metadata for the tile
        - model: keras model
        - stride: number of pixels between each patch
    Outputs:
        - patches: A list of numpy arrays of the shape the model requires
        - patch_coords: A list of shapely polygon features describing the patch bounds
    """
    patch_coords = raster_info[0]['wgs84Extent']['coordinates'][0]
    delta_lon = patch_coords[2][0] - patch_coords[0][0]
    delta_lat = patch_coords[1][1] - patch_coords[0][1]
    lon_degrees_per_pixel = delta_lon / np.shape(tile)[0]
    lat_degrees_per_pixel = delta_lat / np.shape(tile)[1]
    top_left = patch_coords[0]

    # The tile is broken into the number of whole patches
    # Regions extending beyond will not be padded and processed
    patch_coords = []
    patches = []

    # Extract patches and create a shapely polygon for each patch
    for i in range(0, np.shape(tile)[0] - width, stride):
        for j in range(0, np.shape(tile)[1] - width, stride):
            patch = tile[j : j + width,
                         i : i + width]
            patches.append(patch)

            nw_coord = [top_left[0] + i * lon_degrees_per_pixel,
                        top_left[1] + j * lat_degrees_per_pixel]
            ne_coord = [top_left[0] + (i + width) * lon_degrees_per_pixel,
                        top_left[1] + j * lat_degrees_per_pixel]
            sw_coord = [top_left[0] + i * lon_degrees_per_pixel,
                        top_left[1] + (j + width) * lat_degrees_per_pixel]
            se_coord = [top_left[0] + (i + width) * lon_degrees_per_pixel,
                        top_left[1] + (j + width) * lat_degrees_per_pixel]
            tile_geometry = [nw_coord, sw_coord, se_coord, ne_coord, nw_coord]
            patch_coords.append(shapely.geometry.Polygon(tile_geometry))
    return patches, patch_coords

def compute_cost(roi, pop_limited=True):
    """
    Estimate the cost of a Descartes run. 
    These numbers are rough, based on values that DL reported for two runs.
    """
    tile_cost = 21888 / 73112
    data_fraction = 0.934
    area_cost = 21888 / 2278075879720
    if pop_limited:
        gdf = gpd.read_file(f'../data/boundaries/{roi}_pop_0.5_dltiles.geojson').to_crs('epsg:3857')
        #print(f"The DL cost per tile is ${tile_cost:.2f}")
        print(f"The total cost for {roi.title()} is ${tile_cost * len(gdf):,.0f}")
        print(f"The data cost is ${tile_cost * len(gdf) * data_fraction:,.0f} and the compute cost is ${tile_cost * len(gdf) * (1 - data_fraction):,.0f}")
    else:
        gdf = gpd.read_file(f'../data/boundaries/{roi}.geojson').to_crs('epsg:3857')
        area = gdf.area.sum()
        print(f"The total cost for {roi.title()} is ${area_cost * area:,.0f}")
        print(f"The data cost is ${area_cost * area * data_fraction:,.0f} and the compute cost is ${area_cost * area * (1 - data_fraction):,.0f}")


class SentinelData():
    """
    A class to search for and download Sentinel data.
    """

    def __init__(self, polygon, start_date, end_date, mosaic_period, spectrogram_interval, method='min'):
        """
        Inputs:
            - polygon: A shapely polygon feature
            - start_date: A date string
            - end_date: A date string
            - mosaic_period: A string describing the period of time to composite
            - spectrogram_interval: A string describing the time interval between composites
            - method: The method to use when compositing images
        """
        self.polygon = polygon
        self.start_date = start_date
        self.end_date = end_date
        self.spectrogram_interval = spectrogram_interval
        self.mosaic_period = mosaic_period
        self.method = method
    
    def search_scenes(self):
        """
        Search for Sentinel scenes that cover a polygon for the given date range.
        Scenes are filtered to limit clouds
        Returns a list of scenes, cloud_scenes, and a geoctx
        """

        cloud_scenes, _ = dl.scenes.search(
            self.polygon,
            products=['sentinel-2:L1C:dlcloud:v1'],
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            limit=None
        )
        
        scenes, geoctx = dl.scenes.search(
            self.polygon,
            products=['sentinel-2:L1C'],
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            limit=None,
            cloud_fraction=0.3
        )
        scenes = scenes.filter(lambda s: s.coverage(geoctx) > 0.9)

        cloud_keys = [scene.properties.key for scene in cloud_scenes]
        data_keys = [scene.properties.key for scene in scenes]
        shared_keys = set(cloud_keys) & set(data_keys)
        scenes = scenes.filter(
            lambda x: x.properties.key in shared_keys)
        cloud_scenes = cloud_scenes.filter(
            lambda x: x.properties.key in shared_keys)

        self.scenes = scenes
        self.cloud_scenes = cloud_scenes
        self.geoctx = geoctx

        return scenes, cloud_scenes, geoctx

    def download_scenes(self):
        """
        Download the scenes from the search results.
        Returns a list of scenes and a associated metadata
        """
        print("Downloading", len(self.scenes), "scenes at", datetime.datetime.isoformat(datetime.datetime.now())[:19], end='\r')
        # A cloud stack is an array with shape (num_img, data_band, height, width)
        # A value of 255 means that the pixel is cloud free, 0 means cloudy
        cloud_stack = self.cloud_scenes.stack(bands=['valid_cloudfree'], ctx=self.geoctx)

        img_stack, raster_info = self.scenes.stack(
            bands=SENTINEL_BANDS, ctx=self.geoctx, raster_info=True)
        # add date to metadata
        dates = [scene.properties.acquired[:10] for scene in self.scenes]
        for i, date in enumerate(dates):
            raster_info[i]['date'] = date
        cloud_masks = np.repeat(cloud_stack, repeats = 12, axis=1)
        # Add cloud masked pixels to the image mask
        img_stack.mask[cloud_masks.data == 0] = True

        # Remove fully masked images and reorder to channels last
        metadata = []
        for img, info in zip(img_stack, raster_info):
            if np.sum(img) > 0:
                metadata.append(info)

        img_stack = [np.moveaxis(img, 0, -1) for img in img_stack
                        if np.sum(img) > 0]

        self.img_stack = img_stack
        self.metadata = metadata

        return img_stack, metadata

    def create_composites(self):
        """
        Create composites from the downloaded images over the mosaic period.
        Returns a list of composites, a list of composite start dates, and composite metadata
        """
        delta = relativedelta(months=self.mosaic_period)
        start = datetime.date.fromisoformat(self.start_date)
        end = start + delta
        img_dates = [datetime.date.fromisoformat(d['date']) for d in self.metadata]

        self.composites = []
        self.composite_dates = []
        self.composite_metadata = []

        while end <= datetime.date.fromisoformat(self.end_date) + delta:
            # find indices where date is within start and end
            indices = [i for i, x in enumerate(img_dates) if x >= start and x <= end]
            if len(indices) > 0:
                self.composites.append(mosaic([self.img_stack[i] for i in indices], self.method))
                self.composite_dates.append(start.isoformat()[:10])
                self.composite_metadata.append(self.metadata[indices[0]])
            start += delta
            end += delta
        
    def create_pairs(self, enforce_match=False):
        """
        Create pairs of composites that are offset by the spectrogram interval.
        If no pair exists for a pair of months, that pair is skipped.
        Returns a list of pairs and a list of pair start dates
        If enforce_match is True, then only pairs with matching cloud masks are returned
        """
        self.pairs = []
        self.pair_starts = []
        period = relativedelta(months=self.mosaic_period)
        pair_delta = relativedelta(months=self.spectrogram_interval * self.mosaic_period) - relativedelta(days=1)
        start = datetime.date.fromisoformat(self.start_date)
        end = start + pair_delta
        img_dates = [datetime.date.fromisoformat(date) for date in self.composite_dates]
        while end <= datetime.date.fromisoformat(self.end_date):
            #print(f"Searching for images that start between {start} and {start + period - relativedelta(days=1)} and end between {end} and {end + period - relativedelta(days=1)}")
            start_index = [i for i, date in enumerate(img_dates) if date >= start and date < start + period - relativedelta(days=1)]
            end_index = [i for i, date in enumerate(img_dates) if date >= end and date < end + period - relativedelta(days=1)]
            if len(start_index) > 0 and len(end_index) > 0:
                start_composite = self.composites[start_index[0]]
                end_composite = self.composites[end_index[0]]
                pair = [start_composite, end_composite]
                if enforce_match==True:
                    if masks_match(pair):
                        self.pairs.append(pair)
                        self.pair_starts.append(start.isoformat()[:10])
                else:
                    self.pairs.append(pair)
                    self.pair_starts.append(start.isoformat()[:10])
            start += period
            end += period

class DescartesRun(object):
    """Class to manage bulk model prediction on the Descartes Labs platform.

    Attributes:
        product_id: DL id for output rasters
        product_name: String identifer for output rasters
        nodata: No-data value for output rasters
        band_names: String labels for output raster channels
        product: Instantiated dl.catalog.Product
        model_name: String identifier for learned Keras model
        model: Instantiated Keras model
        mosaic_period: Integer number of months worth of data to mosaic
        mosaic_method: Compositing method for the mosaic() function
        spectrogram_interval: Integer number of mosaic periods between mosaics
            input to spectrogram
        spectrogram_length: Total duration of a spectrogram in months
        input_bands: List of DL names identifying Sentinel bands

    External methods:
        init_product: Create or get DL catalog product with specified bands.
        reset_bands: Delete existing output bands.
        upload_model: Upload model to DL storage.
        init_model: Instantiate model from DL storage.
        __call__: Run model on a geographic tile.
        predict: Predict on image-mosaic spectrograms.
        add_band: Create a band in the DL product.
        upload_raster: Upload a raster to DL storage.
    """
    def __init__(self,
                 product_id,
                 patch_product_id,
                 model_name,
                 product_name='',
                 model_file='',
                 patch_model_file='',
                 patch_model_name='',
                 patch_stride=None,
                 mosaic_period=1,
                 mosaic_method='min',
                 spectrogram_interval=6,
                 nodata=-1,
                 input_bands=SENTINEL_BANDS,
                 **kwargs):
        if product_id.startswith('earthrise:'):
            self.product_id = product_id
            self.patch_product_id = patch_product_id
        else:
            self.product_id = f'earthrise:{product_id}'
        self.product_name = product_name if product_name else self.product_id
        self.nodata = nodata
        self.product = self.init_product()

        self.model_name = model_name
        if model_file:
            self.upload_model(model_file)
        self.model = self.init_model()

        if patch_model_file:
            self.patch_model_name = patch_model_name
            self.upload_patch_model(patch_model_file)
            self.patch_model = self.init_patch_model()
            self.patch_product = self.init_patch_product()
            if patch_stride:
                self.patch_stride = patch_stride
            else:
                self.patch_stride = self.patch_model.input_shape[2]

        self.mosaic_period = mosaic_period
        self.mosaic_method = mosaic_method
        self.spectrogram_interval = spectrogram_interval
        self.spectrogram_length = self._get_gram_length()

        self.input_bands = input_bands

    def init_product(self):
        """Create or get DL catalog product."""
        product = dl.catalog.Product.get_or_create(id=self.product_id,
                                                   name=self.product_name)
        product.save()
        print(f'Got patch product {self.product_id}')
        return product

    def init_patch_product(self):
        """Create or get DL catalog product."""
        fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list()]
        product_id = None
        for fc in fc_ids:
            if self.patch_product_id in fc:
                product_id = fc

        if not product_id:
            print("Creating product", self.patch_product_id + '_patches')
            product = dl.vectors.FeatureCollection.create(product_id=self.patch_product_id + '_patches',
                                                          title=self.product_name + '_patches',
                                                          description=self.patch_model_name)
        else:
            print(f"Product {self.patch_product_id}_patches already exists...")
            product = dl.vectors.FeatureCollection(product_id)
        return product

    def reset_bands(self):
        """Delete existing output bands.
        It is probably best to avoid reusing product_ids with different
        input parameters. Calling this function manually would avoid confusion
        in that case.
        """
        for band in self.product.bands():
            band.delete()

    def upload_model(self, model_file):
        """Upload model to DL storage."""
        if dl.Storage().exists(self.model_name):
            print(f'Model {self.model_name} found in DLStorage.')
        else:
            dl.Storage().set_file(self.model_name, model_file)
            print(f'Model {model_file} uploaded with key {self.model_name}.')

    def upload_patch_model(self, patch_model_file):
        """Upload model to DL storage."""
        if dl.Storage().exists(self.patch_model_name):
            print(f'Model {self.patch_model_name} found in DLStorage.')
        else:
            dl.Storage().set_file(self.patch_model_name, patch_model_file)
            print(f'Model {patch_model_file} uploaded with key {self.patch_model_name}.')

    def init_model(self):
        """Instantiate model from DL storage."""
        temp_file = 'tmp-' + self.model_name
        dl.Storage().get_file(self.model_name, temp_file)
        model = keras.models.load_model(temp_file, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                                         'ELU': keras.layers.ELU,
                                                                         'ReLU': keras.layers.ReLU
                                                                         })
        os.remove(temp_file)
        return model

    def init_patch_model(self):
        """Instantiate model from DL storage."""
        temp_file = 'tmp-' + self.patch_model_name
        dl.Storage().get_file(self.patch_model_name, temp_file)
        patch_model = keras.models.load_model(temp_file, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                                         'ELU': keras.layers.ELU,
                                                                         'ReLU': keras.layers.ReLU
                                                                         })
        os.remove(temp_file)
        return patch_model

    def _get_gram_length(self):
        """Compute the length of the spectrogram in months."""
        interval_months = self.mosaic_period * self.spectrogram_interval
        n_intervals = self.model.input_shape[2]
        last_interval_months = self.mosaic_period
        return interval_months * (n_intervals - 1) + last_interval_months

    def __call__(self, dlkey, start_date, end_date):
        """Run model on a geographic tile.

        Args:
            dlkey: Key idenifying a DL tile.
            start_date: Isoformat begin date for prediction window.
            end_date: Isoformat end date for prediction window.

        Returns: None. (Uploads raster output to DL storage.)
        """
        tile = dl.scenes.DLTile.from_key(dlkey)
        self.data = SentinelData(tile,
                                 start_date,
                                 end_date,
                                 self.mosaic_period,
                                 self.spectrogram_interval,
                                 self.mosaic_method
                                 )

        self.data.search_scenes()
        self.data.download_scenes()
        self.data.create_composites()
        self.data.create_pairs()
        self.pairs = self.data.pairs
        self.dates = self.data.pair_starts
        self.bounds = self.data.metadata[0]["wgs84Extent"]["coordinates"][0][:-1]
        self.preds = [predict_spectrogram(pair, self.model) for pair in self.pairs]
        self.preds.append(mosaic(self.preds, 'median'))
        self.preds = np.ma.stack(self.preds)
        self.band_names = self.data.pair_starts
        self.band_names.append('median')
        for band_name in self.band_names:
            self.add_band(band_name)
        self.upload_raster(
            self.preds, next(iter(self.data.metadata)), dlkey.replace(':', '_'))

        # Spatial patch classifier prediction

        # Generate a list of coordinates for the patches within the tile
        raster_info = self.data.metadata
        _, patch_coords = patches_from_tile(self.data.composites[0], raster_info, self.patch_model.input_shape[2], self.patch_stride)

        # Initialize a dictionary where the patch coordinate boundaries are the keys
        # Each value is an empty list where predictions will be appended
        pred_dict = {tuple(coord.bounds): [] for coord in patch_coords}

        # Set a threshold for acceptable cloudiness within a patch for a prediction to be valid
        patch_cloud_threshold = 0.1

        for pair in self.data.pairs:
            # generate patches for first image in pair
            patches_0, _ = patches_from_tile(pair[0], raster_info, self.patch_model.input_shape[2], self.patch_stride)
            # generate patches for second image in pair
            patches_1, _ = patches_from_tile(pair[1], raster_info, self.patch_model.input_shape[2], self.patch_stride)

            patch_pairs = []
            cloud_free = []
            for patch_0, patch_1 in zip(patches_0, patches_1):

                model_input = np.zeros((self.patch_model.input_shape[2],
                                        self.patch_model.input_shape[2],
                                        24))
                model_input[:,:,:12] = pad_patch(unit_norm(patch_0.filled(0)), 28)
                model_input[:,:,12:] = pad_patch(unit_norm(patch_1.filled(0)), 28)
                # Create a list of patch pairs.
                # Shape is (model_input_shape, model_input_shape, 24)
                patch_pairs.append(model_input)

                # Evaluate whether both patches in a sample are below cloud limit
                cloudiness_0 = np.sum(patch_0.mask) / np.size(patch_0.mask)
                cloudiness_1 = np.sum(patch_1.mask) / np.size(patch_1.mask)
                if cloudiness_0 < patch_cloud_threshold and cloudiness_1 < patch_cloud_threshold:
                    cloud_free.append(True)
                else:
                    cloud_free.append(False)

            pair_preds = self.patch_model.predict(np.array(patch_pairs))[:,1]

            # If patches were cloud free, append the prediction to the dictionary
            for coord, pred, cloud_bool in zip(patch_coords, pair_preds, cloud_free):
                if  cloud_bool == True:
                    pred_dict[tuple(coord.bounds)].append(pred)

        # Create dictionary for outputs.
        # In the future, patch classifier outputs should be a geotiff instead of geojson
        feature_list = []
        for coords, key in zip(patch_coords, pred_dict):
            preds = [round(pred, 4) for pred in pred_dict[key]]
            geometry = shapely.geometry.mapping(coords)
            if len(preds) > 0:
                properties = {
                    'mean': np.mean(preds, axis=0).astype('float'),
                    'median': np.median(preds, axis=0).astype('float'),
                    'min': np.min(preds, axis=0).astype('float'),
                    'max': np.max(preds, axis=0).astype('float'),
                    'std': np.std(preds, axis=0).astype('float'),
                    'count': np.shape(preds)[0],
                }
            else:
                properties = {
                    'mean': -1,
                    'median': -1,
                    'min': -1,
                    'max': -1,
                    'std': -1,
                    'count': np.shape(preds)[0],
                }
            # only save outputs that are above a threshold
            if properties['mean'] > 0.3:
                feature_list.append(dl.vectors.Feature(geometry = geometry, properties = properties))
        print(len(feature_list), 'features generated')
        if len(feature_list) > 0:
            self.patch_product.add(feature_list)

    def predict(self, image_gram):
        """Predict on image-mosaic spectrograms."""
        return predict_spectrogram(image_gram, self.model, unit_norm=False)

    def add_band(self, band_name):
        """Create a band in the DL product."""
        if self.product.get_band(band_name):
            return

        band = dl.catalog.SpectralBand(name=band_name, product=self.product)
        band.data_type = dl.catalog.DataType.FLOAT32
        band.data_range = (0, 1)
        band.display_range = (0, 1)
        band.nodata = self.nodata
        num_existing = self.product.bands().count()
        band.band_index = num_existing
        band.save()

    def upload_raster(self, img, raster_meta, name):
        """Upload a raster to DL storage."""
        image_upload = dl.catalog.Image(product=self.product, name=name)
        image_upload.acquired = datetime.date.today().isoformat()
        image_upload.upload_ndarray(
            img.filled(self.nodata), raster_meta=raster_meta, overwrite=True,
            overviews=[2**n for n in range(1, 10)],
            overview_resampler=dl.catalog.OverviewResampler.AVERAGE)
        print(f'Uploaded bands {self.band_names} to {self.product}.')
