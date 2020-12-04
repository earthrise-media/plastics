'''
fires_to_geojson.py

Convert JSON fires data from NASA FIRMS into GeoJSON format
This makes it a lot easier to work with and visualize
'''


import glob
import json
import os


input_dir = './fires_v2'
output_file = './fires_v2/bali_fires.geojson'

input_files = glob.glob(os.path.join(input_dir, '*.json'))
print('Found {} input files to process'.format(len(input_files)))

output_features = list()
for input_file in input_files:
    with open(input_file, 'r') as f:
        input_features = json.load(f)

    for input_feature in input_features:
        lat, lon = input_feature['latitude'], input_feature['longitude']
        
        geometry = {
            'type': 'Point',
            'coordinates': [lon, lat]
        }

        input_feature.pop('latitude', None)
        input_feature.pop('longitude', None)

        output_feature = {
            'type': 'Feature',
            'geometry': geometry,
            'properties': input_feature
        }

        output_features.append(output_feature)

    print('Processed {}'.format(input_file))

fc = {
    'type': 'FeatureCollection',
    'features': output_features
}

with open(output_file, 'w') as f:
    json.dump(fc, f)

