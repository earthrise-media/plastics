import argparse
import descarteslabs as dl
import geopandas as gpd
import json
import os

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    roi_name = config['roi']['name']
    candidate_product_id = config['candidate_detect']['product_id']
    intersect_product_id = config['intersect']['product_id']
    pixel_version = config['pixel']['version']

    roi = gpd.read_file(f'../data/boundaries/{roi_name}.geojson')
    
    # Download pixel candidates
    fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if candidate_product_id in fc.id][0]
    fc = dl.vectors.FeatureCollection(fc_id)
    features = []
    candidates = fc.filter(roi).features()
    for elem in candidates:
        features.append(elem.geojson)
    if len(features) > 0:
        candidate_gdf = gpd.GeoDataFrame.from_features(features)
        directory = f"../data/model_outputs/candidate_sites/{pixel_version}"
        candidate_gdf.to_file(os.path.join(directory, candidate_product_id + '.geojson'))
    else:
        print(f"No candidates found for {pixel_version}")

    # Download intersection candidates
    fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if intersect_product_id[:150] in fc.id][0]
    fc = dl.vectors.FeatureCollection(fc_id)
    features = []
    candidates = fc.filter(roi).features()
    for elem in candidates:
        features.append(elem.geojson)
    if len(features) > 0:
        candidate_gdf = gpd.GeoDataFrame.from_features(features)
        directory = f"../data/model_outputs/candidate_sites/{pixel_version}"
        candidate_gdf.to_file(os.path.join(directory, intersect_product_id.split(':')[-1] + '.geojson'), driver='GeoJSON')
    else:
        print(f"No intersection candidates found for {intersect_product_id}")

if __name__ == "__main__":
    # parse default arguments
    parser = argparse.ArgumentParser(description='Select config file')
    parser.add_argument('--config_path', type=str, default='../pipeline/configs/config.json', help="Path to run's config file")
    args = parser.parse_args()

    main(
        config_path=args.config_path,
    )