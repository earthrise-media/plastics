import geopandas as gpd
# pygeos is required to use geopandas spatial indexing
gpd.options.use_pygeos = True
import descarteslabs as dl
import json
from tqdm import tqdm

class DescartesQueryRun(object):

    def __init__(self,
                 roi,
                 pixel_product_name,
                 pixel_version,
                 patch_product_name,
                 patch_version,
                 pred_threshold,
                 **kwargs):

        self.roi = roi
        self.pixel_product_name = pixel_product_name
        self.pixel_version = pixel_version
        self.patch_product_name = patch_product_name
        self.patch_version = patch_version
        self.pred_threshold = pred_threshold

    def query_patch(self):
        output_id = f"earthrise:intersect_{self.pixel_product_name.split('earthrise:')[-1]}-{self.patch_product_name.split('earthrise:')[-1]}_threshold_{self.pred_threshold}.geojson"
        
        # Get all patch classifier features
        print("Downloading", self.patch_product_name)
        patch_fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if self.patch_product_name in fc.id][0]
        patch_fc = dl.vectors.FeatureCollection(patch_fc_id)
        region = gpd.GeoDataFrame.from_features(json.loads(self.roi))['geometry']
        features = []
        for i, r in enumerate(region):
            print("region", i + 1, "of", len(region))
            for elem in tqdm(patch_fc.filter(r).features()):
                features.append(elem.geojson)
        patch = gpd.GeoDataFrame.from_features(features)
        print(len(patch), 'patch features found')

        patch_threshold = patch[patch['mean'] > self.pred_threshold]
        patch_index = patch_threshold['geometry'].sindex
        
        # Get all pixel classifier features
        pixel_fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if self.pixel_product_name in fc.id][0]
        pixel_fc = dl.vectors.FeatureCollection(pixel_fc_id)
        features = []
        for i, r in enumerate(region):
            print("region", i)
            candidates = pixel_fc.filter(r).features()
            for elem in tqdm(candidates):
                features.append(elem.geojson)
        coords = [feat['geometry']['coordinates'] for feat in features]
        merged = self.merge_similar_sites(coords, search_radius=0.01)

        # Find pixel features that intersect with patch features
        overlap = []
        for candidate in merged['geometry']:
            if len(patch_index.query(candidate)) > 0:
                overlap.append(True)
            else:
                overlap.append(False)
        union = merged[overlap]

        # Create or get DL catalog product.
        fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list()]
        product_id = None
        for fc in fc_ids:
            if output_id.split('earthrise:')[-1] in fc:
                product_id = fc

        if not product_id:
            print("Creating product", output_id)
            product = dl.vectors.FeatureCollection.create(product_id=output_id,
                                                            title=output_id,
                                                            description=f"intersection of {output_id}")
        else:
            print(f"Product {output_id}_patches already exists...")
            product = dl.vectors.FeatureCollection(product_id)

        # Add features to product.
        features = json.loads(union.to_json(drop_id=True))['features']
        feature_list = []
        for feat in features:
            feature_list.append(dl.vectors.Feature(geometry=feat['geometry'], properties=feat['properties']))
        product.add(feature_list)
        
    def merge_similar_sites(self, candidate_sites, search_radius=0.0025):
        import numpy as np
        from sklearn.neighbors import KDTree
        import h3
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
        
        return unique_sites

    def __call__(self):
        """Detect candidates within a tile"""
        self.query_patch()