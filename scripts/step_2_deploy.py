import argparse

import json
import os
import sys

import descarteslabs as dl
from descarteslabs.catalog import Image, properties
import geopandas as gpd
import rasterio as rs
from rasterio.merge import merge
from tensorflow.keras.models import load_model
from tensorflow import keras
from tqdm.notebook import tqdm


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from scripts import deploy_nn_v2

def setup_args(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # ROI info
    roi = config['roi']['name']
    roi_file = f'../data/boundaries/{roi}.geojson'
    dlkeys_file = f"../data/boundaries/dlkeys/{config['roi']['dlkey_file']}"

    # Time info
    start_date = config['dates']['start']
    end_date = config['dates']['end']

    # Pixel classifier info
    model_version = config['pixel']['version']
    model_name = config['pixel']['name']
    model_file = '../models/' + model_name + '.h5'
    product_id = f'earthrise:{roi}_v{model_version}_{start_date}_{end_date}' 
    config['pixel']['product_id'] = product_id
    product_name = product_id.split(':')[-1]  # Arbitrary string - optionally set this to something more human readable.
    config['pixel']['product_name'] = product_name

    # Patch classifier info
    patch_model_version = config['patch']['version']
    patch_model_name = config['patch']['name']
    patch_model_file = '../models/' + patch_model_name + '.h5'
    patch_model = load_model(patch_model_file, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                           'ELU': keras.layers.ELU,
                                                           'ReLU': keras.layers.ReLU})
    patch_input_shape = patch_model.input_shape[2]
    patch_stride = config['patch']['stride']
    patch_product_id = f'earthrise:{roi}_patch_{patch_model_version}_{start_date}_{end_date}_stride_{patch_stride}'
    config['patch']['product_id'] = patch_product_id

    run_local = bool(config['run_local'])

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # If running locally, get results faster by setting smalle tilesize (100?)
    # If running on Descartes, use tilesize 900

    if run_local:
        tilesize = 100
    else:
        tilesize = 900

    # Generally, leave padding at 0
    padding = patch_input_shape - patch_stride

    args = [
        '--roi_file',
        roi_file,
        '--dlkeys_file',
        dlkeys_file,
        '--product_id',
        product_id,
        '--patch_product_id',
        patch_product_id,
        '--product_name',
        product_name,
        '--model_file',
        model_file,
        '--model_name',
        model_name,
        '--patch_model_name',
        patch_model_name,
        '--patch_model_file',
        patch_model_file,
        '--patch_stride',
        str(patch_stride),
        '--mosaic_period',
        str(config['data']['mosaic_period']),
        '--mosaic_method',
        config['data']['mosaic_method'],
        '--spectrogram_interval',
        str(config['data']['spectrogram_interval']),
        '--start_date',
        start_date,
        '--end_date',
        end_date,
        '--pad',
        str(padding),
        '--tilesize',
        str((tilesize // patch_input_shape) * patch_input_shape - padding),
        # TODO: Pop_thresh should be removed
        '--pop_thresh',
        str(10)
    ]
    if run_local:
        args.append('--run_local')

    return config, args

def create_patch_collection(product_id):
    # Check if patch feature collection exists. If it does, delete the FC
    fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list() if product_id in fc.id]
    if len(fc_ids) > 0:
        fc_id = fc_ids[0]
        print("Existing product found.\nDeleting", fc_id)
        dl.vectors.FeatureCollection(fc_id).delete()


def main(config_path):
    config, args = setup_args(config_path)
    create_patch_collection(config['patch']['product_id'])
    deploy_nn_v2.main(args)

if __name__ == "__main__":
    # parse default arguments
    parser = argparse.ArgumentParser(description='Select config file')
    parser.add_argument('--config_path', type=str, default='../pipeline/configs/config.json', help="Path to run's config file")
    args = parser.parse_args()

    main(
        config_path=args.config_path,
    )