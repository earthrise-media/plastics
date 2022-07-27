import descarteslabs as dl
import ee
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import shapely.geometry

from scripts import dl_utils

import warnings
warnings.filterwarnings('ignore')

class GetROI(object):
    def __init__(self, name, region=False, pop_threshold=0.25, plot=True, save_polygons=True):
        self.location_name = name
        self.region = region
        self.pop_threshold = pop_threshold
        self.plot = plot
        self.save_polygons = save_polygons
        ee.Initialize()
    
    def get_country_boundary(self):
        if self.region:
            self.bounds_fc = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('wld_rgn', self.location_name))
        else:
            self.bounds_fc = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_na', self.location_name))
        region_bounds = gpd.GeoDataFrame.from_features(self.bounds_fc.getInfo()['features'], crs='WGS84')
        if len(region_bounds) > 0:
            self.region_bounds = gpd.GeoDataFrame.from_features(self.bounds_fc.getInfo()['features'], crs='WGS84')
            self.region_file_name = f'{self.location_name.lower()}.geojson'
            self.region_bounds.to_file(f'../data/boundaries/{self.region_file_name}')
        else: self.region_bounds = False


    def get_dlkeys(self, tilesize=900, pad=20, resolution=10.0):
        tiles = dl.scenes.DLTile.from_shape(self.region_bounds, resolution, tilesize, pad)
        dlkey_features = list()
        for tile in tiles:
            dlkey_feature = dict()
            dlkey_feature['geometry'] = shapely.geometry.mapping(tile.geometry)
            dlkey_feature['properties'] = dict()
            dlkey_feature['properties']['key'] = tile.key
            dlkey_features.append(dlkey_feature)
        self.dlkey_features = gpd.GeoDataFrame.from_features(dlkey_features)
        self.dlkey_index = self.dlkey_features['geometry'].sindex

    def get_country_roi(self):
        dataset = ee.ImageCollection("WorldPop/GP/100m/pop");
        data = ee.Image(dataset.select('population').reduce(ee.Reducer.median())).clipToCollection(self.bounds_fc)
        zones = data.gt(self.pop_threshold)
        zones = zones.updateMask(zones.neq(0));

        vectors = zones.addBands(data).reduceToVectors(
            geometry = self.bounds_fc,
            crs = data.projection(),
            scale = 2000,
            geometryType = 'polygon',
            eightConnected = True,
            labelProperty = 'zone',
            reducer = ee.Reducer.mean())

        url = vectors.getDownloadURL()
        df = pd.read_csv(url)
        geoms = [shapely.geometry.Polygon(json.loads(g)['coordinates'][0]) for g in df['.geo']]
        self.pop_polygons = gpd.GeoDataFrame(geometry=geoms, crs='EPSG:4326')

        if self.plot:
            fig, ax = plt.subplots(figsize=(8,5), dpi=150)
            self.region_bounds.plot(ax=ax)
            self.pop_polygons.plot(ax=ax, color='r')
            plt.xticks([])
            plt.yticks([])
            plt.title(f"{self.location_name}, Pop Threshold {self.pop_threshold}")
            plt.show()
        if self.save_polygons:
            self.pop_polygons.to_file(f'../data/boundaries/{self.location_name.lower()}_pop_{self.pop_threshold}.geojson', driver='GeoJSON')

    def intersect_roi(self):
        overlap = []
        for candidate in self.pop_polygons['geometry']:
            indices = self.dlkey_index.query(candidate, predicate='intersects')
            overlap += list(indices)
        union = self.dlkey_features.iloc[np.unique(overlap)]
        print(f"Approx DL Time: {len(union) / 500:.2f} hours")
        print(f"Number of DL Tiles to process: {len(union)}")
        self.tile_count = len(union)

        if self.plot:
            fig, ax = plt.subplots(figsize=(8,5), dpi=150)
            union.plot(ax=ax)
            plt.axis('equal')
            plt.title(f"{len(union):,} tiles ({len(union) / len(self.dlkey_features):.1%}) in {self.location_name} are populated at a threshold of {self.pop_threshold}")
            plt.xticks([])
            plt.yticks([])
            plt.show()

        self.dl_key_name = f'{self.location_name.lower()}_pop_{self.pop_threshold}_dlkeys.txt'
        dl_utils.write_dlkeys(union['key'], f'../data/boundaries/dlkeys/{self.dl_key_name}')
        self.dl_union = union[['geometry', 'key']]
        self.populated_dltiles_name = f'{self.location_name.lower()}_pop_{self.pop_threshold}_dltiles.geojson'
        self.dl_union.to_file(f'../data/boundaries/{self.populated_dltiles_name}', driver='GeoJSON')
            
    def run(self):
        self.get_country_boundary()
        if type(self.region_bounds) == gpd.geodataframe.GeoDataFrame:
            self.get_dlkeys()
            self.get_country_roi()
            self.intersect_roi()
        else:
            print(f"No boundaries found for {self.location_name}")
        