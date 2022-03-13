import descarteslabs as dl
import functools
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio as rs
import shapely
from rasterio.windows import Window
from skimage.feature import blob_doh
from skimage.feature.peak import peak_local_max
from sklearn.neighbors import KDTree
from tqdm.contrib.concurrent import process_map

def merge_similar_sites(candidate_sites, search_radius=0.0025, plot=True):
    """
    This process iteratively moves points closer together by taking the mean position of all
    matched points. It then searches the KD tree using the unique clusters in these new points.
    The algorithm stops once the number of unique sites is the same as in the previous round.
    
    search_radius is given in degrees
    """
    coords = np.array(candidate_sites)
    
    # Create a KD tree for efficient lookup of points within a radius
    tree = KDTree(coords, leaf_size=2)
    
    # Initialize a mean_coords array for the search
    mean_coords = []
    for elem in tree.query_radius(coords, search_radius):
        mean_coords.append(np.mean(coords[elem], axis=0))
    mean_coords = np.array(mean_coords)
    
    num_coords = len(mean_coords)
    while True:
        search = tree.query_radius(mean_coords, search_radius)
        uniques = [list(x) for x in set(tuple(elem) for elem in search)]
        mean_coords = []
        for elem in uniques:
            mean_coords.append(np.mean(coords[elem], axis=0))
        if len(mean_coords) == num_coords:
            print(len(mean_coords), "unique sites detected")
            mean_coords = np.array(mean_coords)
            break
        num_coords = len(mean_coords)

    ids = [h3.geo_to_h3(coord[1], coord[0], 15) for coord in mean_coords]
    unique_sites = gpd.GeoDataFrame({'id': ids}, geometry=gpd.points_from_xy(*mean_coords.T))
    if plot:
        plt.figure(figsize=(10,10), dpi=150, facecolor=(1,1,1))
        plt.scatter(coords[:,0], coords[:,1], s=5, label='Original')
        plt.scatter(mean_coords[:,0], mean_coords[:,1], s=3, c='r', label='Unique')
        plt.axis('equal')
        plt.show()
    
    return unique_sites

def detect_blobs(source, name, pred_threshold=0.75, min_sigma=3.5, max_sigma=100, area_threshold=0.0025, window_size=5000, save=True):
    """
    Identify candidates using blob detection on the heatmap.
    prediction_threshold masks any prediction below a 0-1 threshold.
    min_sigma and area_threshold control the size sensitivity of the blob detection.
    Keep min_sigma low to detect smaller blobs
    area_threshold establishes a lower bound on candidate blob size. Reduce to detect smaller blobs
    """
    candidate_sites = []
    max_val = source.read(1).max()
    for x in range(0, source.shape[0], window_size):
        for y in range(0, source.shape[1], window_size):
            # print(f"Processing row {(x // window_size) + 1} of {int(source.shape[0] / window_size) + 1}, column {(y // window_size) + 1} of {int(source.shape[1] / window_size) + 1}")
            # Set min and max to analyze a subset of the image
            window = Window.from_slices((x,x + window_size), (y, y + window_size))
            window_median = (source.read(1, window=window) / max_val).astype('float')
            # mask predictions below a threshold
            mask = np.ma.masked_where(window_median < pred_threshold, window_median).mask
            window_median[mask] = 0

            blobs = blob_doh(window_median, min_sigma=min_sigma, max_sigma=max_sigma, threshold=area_threshold)
            # print(len(blobs), "candidates detected in window")
            
            overlap_threshold = 0.01
            transform = source.window_transform(window)
            for candidate in blobs:
                lon, lat = (transform * [candidate[1], candidate[0]])
                # Size doesn't mean anything at the moment. Should look into this later
                #size = candidate[2]
                candidate_sites.append([lon, lat])
    
    print(len(candidate_sites), "candidate sites detected in total")
    
    candidate_gdf = merge_similar_sites(candidate_sites, search_radius=0.01)
    display(candidate_gdf)
    
    if save:
        file_path = f"../data/model_outputs/candidate_sites/{name}_blobs_thresh_{pred_threshold}_min-sigma_{min_sigma}_area-thresh_{area_threshold}"
        # candidate_gdf.loc[:, ['lon', 'lat', 'name']].to_csv(file_path + '.csv', index=False)
        candidate_gdf.to_file(file_path + '.geojson', driver='GeoJSON')
    
    return candidate_gdf

def blob_detect(file, pred_threshold=0.75, min_sigma=3.5, max_sigma=100, area_threshold=0.0025, save=True):
    """
    Identify candidates using blob detection on the heatmap.
    prediction_threshold masks any prediction below a 0-1 threshold.
    min_sigma and area_threshold control the size sensitivity of the blob detection.
    Keep min_sigma low to detect smaller blobs
    area_threshold establishes a lower bound on candidate blob size. Reduce to detect smaller blobs
    """
    
    source = rs.open(file)
    data = source.read(1).astype('float')
    # mask predictions below a threshold
    mask = np.ma.masked_where(data < pred_threshold, data).mask
    data[mask] = 0
    
    candidate_sites = []

    blobs = blob_doh(data, min_sigma=min_sigma, max_sigma=max_sigma, threshold=area_threshold)
    if len(blobs) > 0:
        # print(len(blobs), "candidate(s) detected in window")
        transform = source.transform
        for candidate in blobs:
            lon, lat = (transform * [candidate[1], candidate[0]])
            # Size doesn't mean anything at the moment. Should look into this later
            #size = candidate[2]
            candidate_sites.append([lon, lat])
    
        return candidate_sites
    
    else:
        return None

def detect_blobs_tiled(source_dir, name, model_name, pred_threshold=0.75, min_sigma=3.5, max_sigma=100, area_threshold=0.0025, save=True, merge_radius=0.005):
    """
    Identify candidates using blob detection on the heatmap.
    prediction_threshold masks any prediction below a 0-1 threshold.
    min_sigma and area_threshold control the size sensitivity of the blob detection.
    Keep min_sigma low to detect smaller blobs
    area_threshold establishes a lower bound on candidate blob size. Reduce to detect smaller blobs
    """
    files = os.listdir(source_dir)
    file_paths = [os.path.join(source_dir, file) for file in files]
    
    blob_detect_partial = functools.partial(blob_detect, 
                                            pred_threshold=pred_threshold, 
                                            min_sigma=min_sigma, 
                                            max_sigma=max_sigma, 
                                            area_threshold=area_threshold, 
                                            save=save)
    
    site_list = process_map(blob_detect_partial, file_paths, chunksize=20)
    
    candidate_sites = []
    for tile_sites in site_list:
        if tile_sites:
            for site in tile_sites:
                candidate_sites.append(site)
    print(len(candidate_sites), "sites detected overall")
    candidate_gdf = merge_similar_sites(candidate_sites, name, search_radius=merge_radius)
    
    print(len(candidate_sites) - len(candidate_gdf), "sites merged")
    if save:
        directory = f"../data/model_outputs/candidate_sites/{model_name}"
        file_name = f"{name}_blobs_thresh_{pred_threshold}_min-sigma_{min_sigma}_area-thresh_{area_threshold}"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        candidate_gdf.to_file(file_path + '.geojson', driver='GeoJSON')\
    
    return candidate_gdf



def detect_peaks(source, name, threshold_abs=0.85, min_distance=100, window_size=5000, save=True):
    """
    Identify candidates using heatmap peak detection.
    Inputs:
      source: rasterio geotiff object
      name: file name
      threshold_abs: threshold for minimum prediction value
      min_distance: candidates within this distance will be merged by default. Distance in pixel space
      window_size: chunk the image into windows to reduce memory load
      save: boolean to write outputs to disk
    """
    candidate_peaks = []
    for x in range(0, source.shape[0], window_size):
        for y in range(0, source.shape[1], window_size):
            window = Window.from_slices((x,x + window_size), (y, y + window_size))
            transform = source.window_transform(window)
            subset = source.read(1, window=window)
            peaks = peak_local_max(subset, threshold_abs=threshold_abs, min_distance=min_distance)
            for candidate in peaks:
                lon, lat = (transform * [candidate[1], candidate[0]])
                candidate_peaks.append([lon, lat])
    print(len(candidate_peaks), "peaks detected")
    candidate_peaks = np.array(candidate_peaks)
    
    candidate_gdf = gpd.GeoDataFrame(candidate_peaks, columns=['lon', 'lat'], 
                                     geometry=gpd.points_from_xy(*candidate_peaks.T))
    candidate_gdf['name'] = [f"{name}_{i+1}" for i in candidate_gdf.index]
    
    if save:
        file_path = f"../data/model_outputs/candidate_sites/{name}_peaks_thresh_{threshold_abs}_min_dist_{min_distance}"
        # candidate_gdf.loc[:, ['lon', 'lat', 'name']].to_csv(file_path + '.csv', index=False)
        candidate_gdf.to_file(file_path + '.geojson', driver='GeoJSON')
    
    return candidate_gdf


class DescartesDetectRun(object):

    def __init__(self,
                 product_name,
                 pred_threshold,
                 min_sigma,
                 **kwargs):

        self.product_name = product_name
        self.pred_threshold = pred_threshold
        self.min_sigma = min_sigma
        if self.product_name.startswith('earthrise:'):
            self.product_id = f"{self.product_name}_blobs_thresh_{self.pred_threshold}_min-sigma_{self.min_sigma}_area-thresh_0.0025"
        else:
            self.product_id = f'earthrise:{f"{self.product_name}_blobs_thresh_{self.pred_threshold}_min-sigma_{self.min_sigma}_area-thresh_0.0025"}'
        self.product = self.init_product()
        self.raster_client = dl.Raster()

    def init_product(self):
        """Create or get DL catalog product."""
        fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list()]
        product_id = None
        for fc in fc_ids:
            if self.product_id in fc:
                product_id = fc

        if not product_id:
            print("Creating product", self.product_id)
            product = dl.vectors.FeatureCollection.create(product_id=self.product_id,
                                                          title=self.product_name,
                                                          description=self.product_name)
        else:
            print(f"Product {self.product_id} already exists...")
            product = dl.vectors.FeatureCollection(product_id)
        return product

    def download_tile(self, image, band='median'):
        outfile_name = os.path.join('./', image)
        try:
            self.raster_client.raster(inputs = image,
                                bands = [band],
                                save=True,
                                outfile_basename = outfile_name,
                                srs='WGS84')
            return outfile_name
        except dl.client.exceptions.BadRequestError as e:
            print(f'Warning: {repr(e)}\nContinuing...')
        except dl.client.exceptions.ServerError as e:
            print(f'Warning: {repr(e)}\nContinuing...')
            
    def detect_candidates(self, tile_path):
        # Detect candidates from directory of tiles. Multiprocessed
        blobs = blob_detect(
            tile_path+'.tif',
            self.pred_threshold,
            self.min_sigma
        )
        return blobs

    def __call__(self, image):
        """Detect candidates within a tile"""
        tile_path = self.download_tile(image)
        blobs = self.detect_candidates(tile_path)
        feature_list = []
        if blobs:
            for blob in blobs:
                feature = dl.vectors.Feature(
                    geometry = shapely.geometry.Point(blob[0], blob[1]),
                    properties = {'id': h3.geo_to_h3(blob[1], blob[0], 15)}
                )
                feature_list.append(feature)
            self.product.add(feature_list)