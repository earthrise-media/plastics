import descarteslabs as dl
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

def rect_from_point(coord, rect_width):
    """
    Create a geojson polygon from a coordinate pair.
    Inputs:
        - coord: coordinates in the form [lon, lat]
        - rect_width: the total width/height of the rectangle
    Returns: Geojson formatted polygon
    """
    w = rect_width / 2
    rect = shapely.geometry.mapping(shapely.geometry.box(coord[0] - w, coord[1] - w, coord[0] + w, coord[1] + w))
    return rect

def download_patch(polygon, start_date, end_date):
    """
    Download a stack of cloud-masked Sentinel data
    Inputs:
        - polygon: Geojson polygon enclosing the region of data to be extracted
        - start_date/end_date: The time bounds of the search
    Returns:
        - A numpy array of image patches of format (num_img, height, width, channels)
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

    cloud_stack = cloud_scenes.stack(bands=['valid_cloudfree'],
                                     ctx=geoctx)

    img_stack = scenes.stack(bands=sentinel_bands,
                             ctx=geoctx)
    cloud_masks = np.repeat(cloud_stack, repeats=12, axis=1)
    masked_img_stack = np.ma.masked_where(cloud_masks == 0, img_stack)
    # Rearrange order to (num_img, height, width, channels)
    masked_img_stack = np.moveaxis(masked_img_stack, 1, -1)
    return masked_img_stack

def flatten_stack(img_stack):
    return np.ma.median(img_stack, axis=0)
