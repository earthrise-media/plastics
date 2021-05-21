
import datetime

import descarteslabs as dl
from dateutil.relativedelta import relativedelta
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
    rect = shapely.geometry.mapping(shapely.geometry.box(lon - lon_w, lat - lat_w, lon + lon_w, lat + lat_w))
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
    scenes = scenes.filter(lambda x: x.properties.acquired in shared_dates)
    cloud_scenes = cloud_scenes.filter(lambda x: x.properties.acquired in shared_dates)

    # A cloud stack is an array with shape (num_img, data_band, height, width)
    # A value of 255 means that the pixel is cloud free, 0 means cloudy
    cloud_stack = cloud_scenes.stack(bands=['valid_cloudfree'],
                                         ctx=geoctx)

    img_stack, raster_info = scenes.stack(
        bands=SENTINEL_BANDS, ctx=geoctx, raster_info=True)
    cloud_masks = np.repeat(cloud_stack, repeats = 12, axis=1)

    # Add cloud masked pixels to the image mask
    img_stack.mask[cloud_masks.data == 0] = True

    # Remove fully masked img from the stack and rearrange order to channels last
    img_stack = [np.moveaxis(img, 0, -1) for img in img_stack if np.sum(img) > 0]

    return img_stack, raster_info

def pad_patch(patch, width):
    """
    Depending on how a polygon falls across pixel boundaries, it can be slightly
    bigger or smaller than intended.
    pad_patch trims pixels extending beyond the desired number of pixels if the
    patch is larger than desired. If the patch is smaller, it will fill the edge by
    reflecting the values.
    """
    h, w, c = patch.shape
    if h < width or w < width:
        patch = np.pad(patch, width - np.min([h, w]), mode='reflect')
    patch = patch[:width, :width, :12]
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

def get_starts(start_date, end_date, mosaic_period):
    """Determine spectrogram start dates."""
    starts = []
    delta = relativedelta(months=mosaic_period)
    start = datetime.date.fromisoformat(start_date)
    while start + delta <= datetime.date.fromisoformat(end_date):
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
    
    Returns: List of image mosaics
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
            'min_masked': return the array with fewest masked pixels 
        
    Returns: A masked array or None if arrays is an empty list
    """
    if not arrays:
        return

    if method == 'median':
        stack = np.ma.stack(arrays)
        reduced = np.ma.median(stack, axis=0)
    elif method == 'min_masked': 
        mask_sorted = sorted(arrays, key=lambda p:np.sum(p.mask))
        reduced = next(iter(mask_sorted))
    else:
        raise ValueError(f'Method {method} not recognized.')
    
    return reduced

def pair(mosaics, gap=6):
    """Pair image mosaics from a list.

    Args: 
        mosaics: A list of masked arrays
        gap: Integer gap between mosaics in number of mosaic periods

    Returns: A list of lists of images.
    """
    pairs = [[a, b] for a, b in zip(mosaics, mosaics[gap:])
                  if a is not None and b is not None]
    return pairs

# WIP: Eventually we want to generalize from pairs to n-grams.
# This is a placeholder in the name-space for an eventual maker of n_grams.
def n_gram(mosaics, gap=6, n=2):
    return pair(mosaics, gap=gap)

def masks_match(pair):
    """Check whether arrays in a pair share the same mask.

    This enforces identical cloud masking on an image pair. Any 
    residual mask is expected to define a polygon within the raster. 
    """
    return (pair[0].mask == pair[1].mask).all()

def shape_pair_as_pixels(pair):
    """Convert a pair of images into a pixel-wise array of data samples.

    Returns: Array of pixel elements, each having shape (channels, len(pair))
    """
    height, width, channels = next(iter(pair)).shape
    pixels = np.moveaxis(np.array(pair), 0, -1)
    pixels = pixels.reshape(height * width, channels, len(pair))
    return pixels

def preds_to_image(preds, input_pair):
    """Reshape and mask spectrogram model predictions."""
    channel00 = input_pair[0][:,:,0]
    channel10 = input_pair[1][:,:,0]
    img = preds.reshape(channel00.shape)
    img = np.ma.masked_where(channel00.mask | channel10.mask, img)
    return img

def predict_spectrogram(image_pair, model):
    """Run a spectrogram model on a pair of images."""
    pixels = shape_pair_as_pixels(image_pair)
    input_array = np.expand_dims(normalize(pixels), -1)
    preds = model.predict(input_array)[:,1]
    output_img = preds_to_image(preds, image_pair)
    return output_img

class DescartesRun(object):

    def __init__(self,
                 product_id,
                 model_name,
                 product_name='',
                 output_band_names=[],
                 model_file='',
                 mosaic_period=1,
                 spectrogram_interval=6,
                 nodata=-1,
                 input_bands=SENTINEL_BANDS,
                 **kwargs):
        import pdb; pdb.set_trace()
        self.product_id = product_id
        self.product_name = product_name
        self.nodata = nodata
        self.output_band_names = output_band_names
        self.product = self.init_product()
        
        self.model_name = model_name
        if model_file:
            self.upload_model(model_file)
        self.model = self.init_model()
        self.spectrogram_interval = spectrogram_interval
        self.spectrogram_steps = self.model.input_shape[2]
        
        self.mosaic_period = mosaic_period
        self.input_bands = input_bands

    def init_product(self):
        """Create or get DL catalog product with specified bands."""
        product = dl.catalog.Product.get_or_create(id=self.product_id,
                                                   name=self.product_name)
        product.save()

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

        for band_name in self.output_band_names:
            add_band(band_name)

        print(f'Got product {product_id} and bands {band_names}')
        return product
        
    def upload_model(self, model_file):
        """Upload model to storage."""
        if dl.Storage().exists(self.model_name):
            print(f'Model {model_name} found in DLStorage.')
        else:
            dl.Storage().set_file(self.model_name, model_file)
            print(f'Model {model_file} uploaded with key {model_name}.')

    def init_model(self):
        dl.Storage().get_file(self.model_name, tmp_file)
        return keras.models.load_model(tmp_file)
    
    def __call__(self, dlkey, start_date, end_date):
        tile = dl.scenes.DLTile.from_key(dlkey)
        mosaics, raster_info = download_mosaics(
            tile, start_date, end_date, self.mosaic_period)
        image_grams = n_gram(mosaics, self.spectrogram_interval)

        preds = [self.predict(gram) for gram in image_grams]
        preds.append(mosaic(preds, 'median'))
        preds = np.ma.stack(preds)
        self.upload_image(
            preds, next(iter(raster_info)), dlkey.replace(':', '_'))

    def predict():
        return predict_spectrogram(image_gram, self.model)
    
    def upload_image(img, raster_meta, name):
        """Upload an image."""
        image_upload = dl.catalog.Image(product=self.product, name=name)
        image_upload.acquired = datetime.date.today().isoformat()
        upload = image_upload.upload_ndarray(
            img.filled(self.nodata), raster_meta=raster_meta, overwrite=True,
            overviews=[2**n for n in range(1, 10)],
            overview_resampler=dl.catalog.OverviewResampler.AVERAGE)

