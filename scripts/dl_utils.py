
import datetime
import json
import os

import descarteslabs as dl
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
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
    for feature in features:
        tiles = dl.Raster().iter_dltiles_from_shape(10.0, tilesize, pad,
                                                    feature)
        for tile in tiles:
            all_keys.append(tile['properties']['key'])
            ctr +=1
            print(ctr, end='\r')

    print('Split ROI into {} tiles'.format(len(all_keys)))
    return all_keys

def download_patch(polygon, start_date, end_date, s2_id='sentinel-2:L1C',
                   s2cloud_id='sentinel-2:L1C:dlcloud:v1',):
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
    )

    scenes, geoctx = dl.scenes.search(
        polygon,
        products=[s2_id],
        start_datetime=start_date,
        end_datetime=end_date
    )

    # select only scenes that have a cloud mask
    cloud_dates = [scene.properties.acquired for scene in cloud_scenes]
    dates = [scene.properties.acquired for scene in scenes]
    shared_dates = set(cloud_dates) & set(dates)
    scenes = scenes.filter(
        lambda x: x.properties.acquired in shared_dates)
    cloud_scenes = cloud_scenes.filter(
        lambda x: x.properties.acquired in shared_dates)

    # A cloud stack is an array with shape (num_img, data_band, height, width)
    # A value of 255 means that the pixel is cloud free, 0 means cloudy
    cloud_stack = cloud_scenes.stack(bands=['valid_cloudfree'], ctx=geoctx)

    img_stack, raster_info = scenes.stack(
        bands=SENTINEL_BANDS, ctx=geoctx, raster_info=True)
    cloud_masks = np.repeat(cloud_stack, repeats = 12, axis=1)

    # Add cloud masked pixels to the image mask
    img_stack.mask[cloud_masks.data == 0] = True

    # Remove fully masked images and reorder to channels last
    img_stack = [np.moveaxis(img, 0, -1) for img in img_stack
                     if np.sum(img) > 0]

    return img_stack, raster_info

def pad_patch(patch, width):
    """
    Depending on how a polygon falls across pixel boundaries, it can be slightly
    bigger or smaller than intended.
    pad_patch trims pixels extending beyond the desired number of pixels if the
    patch is larger than desired. If the patch is smaller, it will fill the
    edge by reflecting the values.
    """
    h, w, c = patch.shape
    if h < width or w < width:
        patch = np.pad(patch, width - np.min([h, w]), mode='reflect')
    patch = patch[int(np.floor((h - width) / 2)) : h - int(np.ceil((h - width) / 2)),
                  int(np.floor((w - width) / 2)) : w - int(np.ceil((w - width) / 2)), :12]
    return patch

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
            batch, raster_info = download_patch(polygon, start.isoformat(),
                                                end.isoformat())
        except IndexError as e:
            print(f'Failed to retreive month {start.isoformat()}: {repr(e)}')
            batch, raster_info = [], []
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
    mosaic_info = [next(iter(r)) for r in raster_infos]
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

def pair(mosaics, interval=6):
    """Pair image mosaics from a list.

    Args:
        mosaics: A list of masked arrays
        interval: Integer interval between mosaics, in number of mosaic periods

    Returns: A list of lists of images.
    """
    pairs = [[a, b] for a, b in zip(mosaics, mosaics[interval:])
                  if a is not None and b is not None]
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

# WIP: needs to be generalized pairs -> grams
def preds_to_image(preds, input_pair):
    """Reshape and mask spectrogram model predictions."""
    channel00 = input_pair[0][:,:,0]
    channel10 = input_pair[1][:,:,0]
    img = preds.reshape(channel00.shape)
    mask = channel00.mask | channel10.mask | np.isnan(img)
    return np.ma.array(img, mask=mask)

def predict_spectrogram(image_gram, model):
    """Run a spectrogram model on a pair of images."""
    pixels = shape_gram_as_pixels(image_gram)
    input_array = np.expand_dims(normalize(pixels), -1)
    preds = model.predict(input_array)[:,1]
    output_img = preds_to_image(preds, image_gram)
    return output_img

def patches_from_tile(tile, raster_info, width):
    """
    Break a larger tile of Sentinel data into a set of patches that
    a model can process.
    Inputs:
        - tile: Sentinel data. Typically a numpy masked array
        - raster_info: Descartes metadata for the tile
        - model: keras model
    Outputs:
        - patches: A list of numpy arrays of the shape the model requires
        - patch_coords: A list of shapely polygon features describing the patch bounds
    """
    patch_coords = raster_info[0]['wgs84Extent']['coordinates'][0]
    delta_lon = patch_coords[2][0] - patch_coords[0][0]
    delta_lat = patch_coords[1][1] - patch_coords[0][1]
    top_left = patch_coords[0]

    # The tile is broken into the number of whole patches
    # Regions extending beyond will not be padded and processed
    num_steps_lon = np.shape(tile)[0] // width
    num_steps_lat = np.shape(tile)[1] // width
    patch_coords = []
    patches = []
    for i in range(num_steps_lon):
        for j in range(num_steps_lat):
            patch = tile[j * width : width + j * width,
                         i * width : width + i * width]
            patches.append(patch)
            nw_coord = [top_left[0] + i * delta_lon / num_steps_lon,
                        top_left[1] + j * delta_lat / num_steps_lat]

            tile_geometry = [nw_coord,
                             [nw_coord[0], nw_coord[1] + delta_lat / num_steps_lat],
                             [nw_coord[0] + delta_lon / num_steps_lon,
                              nw_coord[1] + delta_lat / num_steps_lat],
                             [nw_coord[0] + delta_lon / num_steps_lon, nw_coord[1]],
                             nw_coord
                            ]
            patch_coords.append(shapely.geometry.Polygon(tile_geometry))
    return patches, patch_coords

def unit_norm(samples):
    means = [1298.7330617049158, 1046.728074661139, 968.1679384402669, 805.7169469984433, 1001.5350089770019, 1721.8117738966116, 2074.660173162641, 1906.273653848785, 2253.9860009889144, 295.4803929281597, 1544.334239964913, 900.2133779561017]
    deviations = [366.08860308693147, 378.157880474536, 389.7569162648193, 478.7348213604553, 435.20976171883257, 631.4499370533496, 798.0576089639353, 766.4725319841734, 892.286826414747, 131.88545750591763, 726.3113441205015, 601.8741576997301]
    normalized_samples = np.zeros_like(samples).astype('float32')
    for i in range(0, 12):
        #normalize each channel to global unit norm
        normalized_samples[:,:,:,:,i] = (np.array(samples)[:,:,:,:,i] - means[i]) / deviations[i]
    return normalized_samples

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
        #dl.vectors.FeatureCollection(fc).delete()
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
        model = keras.models.load_model(temp_file)
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
        mosaics, raster_info = download_mosaics(
            tile, start_date, end_date, self.mosaic_period, self.mosaic_method)
        # Spectrogram pixel classifier prediction
        image_grams = n_gram(mosaics, self.spectrogram_interval)
        preds = [self.predict(gram) for gram in image_grams]
        preds.append(mosaic(preds, 'median'))
        preds = np.ma.stack(preds)

        self.band_names = get_starts(
            start_date, end_date, self.mosaic_period, self.spectrogram_length)
        self.band_names.append('median')
        for band_name in self.band_names:
            self.add_band(band_name)
        self.upload_raster(
            preds, next(iter(raster_info)), dlkey.replace(':', '_'))

        # Spatial patch classifier prediction
        _, patch_coords = patches_from_tile(mosaics[0], raster_info, self.patch_model.input_shape[2])
        pred_dict = {tuple(coord.bounds): [] for coord in patch_coords}
        patch_cloud_threshold = 0.1
        for pair in image_grams:
            patches_0, _ = patches_from_tile(pair[0], raster_info, self.patch_model.input_shape[2])
            patches_1, _ = patches_from_tile(pair[1], raster_info, self.patch_model.input_shape[2])

            patch_pairs = []
            cloud_free = []
            for patch_0, patch_1 in zip(patches_0, patches_1):
                patch_pairs.append([patch_0.filled(0), patch_1.filled(0)])
                cloudiness_0 = np.sum(patch_0.mask) / np.size(patch_0.mask)
                cloudiness_1 = np.sum(patch_1.mask) / np.size(patch_1.mask)
                if cloudiness_0 < patch_cloud_threshold and cloudiness_1 < patch_cloud_threshold:
                    cloud_free.append(True)
                else:
                    cloud_free.append(False)

            patch_pairs = [[patch_0.filled(0), patch_1.filled(0)] for patch_0, patch_1 in zip(patches_0, patches_1)]
            pairs_norm = unit_norm(patch_pairs)
            pair_preds = self.patch_model.predict(pairs_norm)[:,1]
            for coord, pred, cloud_bool in zip(patch_coords, pair_preds, cloud_free):
                if  cloud_bool == True:
                    pred_dict[tuple(coord.bounds)].append(pred)

        feature_list = []
        for coords, preds in zip(patch_coords, pred_dict.values()):
            geometry = shapely.geometry.mapping(coords)
            if len(preds) > 0:
                properties = {
                    'mean': round(np.mean(preds, axis=0).astype('float'), 4),
                    'median': round(np.median(preds, axis=0).astype('float'), 4),
                    'min': round(np.min(preds, axis=0).astype('float'), 4),
                    'max': round(np.max(preds, axis=0).astype('float'), 4),
                    'std': round(np.std(preds, axis=0).astype('float'), 4),
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
            #if properties['mean'] > 0.01:
            feature_list.append(dl.vectors.Feature(geometry = geometry, properties = properties))
        print(len(feature_list), 'features generated')
        if len(feature_list) > 0:
            self.patch_product.add(feature_list)

    def predict(self, image_gram):
        """Predict on image-mosaic spectrograms."""
        return predict_spectrogram(image_gram, self.model)

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
