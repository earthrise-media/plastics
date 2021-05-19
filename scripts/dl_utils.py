
import datetime

import descarteslabs as dl
from dateutil.relativedelta import relativedelta
import numpy as np
import shapely

sentinel_bands = ['coastal-aerosol',
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

def download_patch(polygon, start_date, end_date):
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
        products=["sentinel-2:L1C:dlcloud:v1"],
        start_datetime=start_date,
        end_datetime=end_date,
    )

    scenes, geoctx = dl.scenes.search(
        polygon,
        products=["sentinel-2:L1C"],
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

    img_stack = scenes.stack(bands=sentinel_bands,
                             ctx=geoctx)
    cloud_masks = np.repeat(cloud_stack, repeats = 12, axis=1)

    # Add cloud masked pixels to the image mask
    img_stack.mask[cloud_masks.data == 0] = True

    # Remove fully masked img from the stack and rearrange order to channels last
    img_stack = [np.moveaxis(img, 0, -1) for img in img_stack if np.sum(img) > 0]

    return img_stack

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
    batches = []
    delta = relativedelta(months=batch_months)
    start = datetime.date.fromisoformat(start_date)
    end = start + delta
    while end <= datetime.date.fromisoformat(end_date):
        try:
            batch = download_patch(polygon, start.isoformat(), end.isoformat())
        except IndexError as e:
            print(f'Failed to retreive month {start.isoformat()}: {repr(e)}')
            patch = []
        batches.append(batch)
        start += delta
        end += delta
    return batches

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
    batches = download_batches(polygon, start_date, end_date, mosaic_period)
    mosaics = [mosaic(batch, method) for batch in batches]
    return mosaics

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
