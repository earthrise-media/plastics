import tensorflow.keras.models
from flytekit import task, workflow, map_task, TaskMetadata
from flytekit.types.file import HDF5EncodedFile, FlyteFile

import uuid

import os, sys
import shutil

import descarteslabs as dl
from descarteslabs.catalog import Image, properties
import geopandas as gpd
import rasterio as rs
from rasterio.merge import merge

from tensorflow.keras.models import load_model
from tensorflow import keras

# hack warning!
SCRIPT_DIR = os.path.dirname(os.path.abspath('../scripts'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts import deploy_nn_v1


@task()
def upload_model(model_file: HDF5EncodedFile,
                 patch_model_file: HDF5EncodedFile,
               roi_file: FlyteFile,
                 run_id: str
                 ) -> bool:
    # load the model from S3
    model_file.download()
    patch_model = load_model(model_file.path, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                                  'ELU': keras.layers.ELU,
                                                                  'ReLU': keras.layers.ReLU})
    # load geojsons from S3
    roi_file.download()
    df = gpd.read_file(roi_file.path)['geometry']

    # run

    product_id = f'earthrise:product_{run_id}'
    patch_product_id = f'patch_product_{run_id}'
    patch_model_version = 'weak_labels_1.1'
    patch_model_name = 'v1.1_weak_labels_28x28x24'
    model_name = 'spectrogram_v0.0.11_2021-07-13'

    tilesize = 900
    patch_stride = 8
    patch_input_shape = patch_model.input_shape[2]

    # Note on dates: The date range should be longer than the spectrogram length.
    # Starting on successive mosaic periods (typically: monthly), as many
    # spectrograms are created as fit in the date range.
    start_date = '2019-01-01'
    end_date = '2021-06-01'

    mosaic_period = 3
    mosaic_method = 'min'
    spectrogram_interval = 2
    padding = patch_input_shape - patch_stride

    args = [
        '--roi_file',
        roi_file.path,
        '--product_id',
        product_id,
        '--patch_product_id',
        patch_product_id,
        '--product_name',
        run_id,
        '--model_file',
        model_file.path,
        '--model_name',
        model_name,
        '--patch_model_name',
        patch_model_name,
        '--patch_model_file',
        patch_model_file.path,
        '--mosaic_period',
        str(mosaic_period),
        '--mosaic_method',
        mosaic_method,
        '--spectrogram_interval',
        str(spectrogram_interval),
        '--start_date',
        start_date,
        '--end_date',
        end_date,
        '--pad',
        '0',
        '--tilesize',
        str((tilesize // patch_input_shape) * patch_input_shape - padding)
    ]
    # if run_local:
    args.append('--run_local')

    deploy_nn_v1.main(args)
    return True



@workflow()
def detect(model_file: str="s3://flyte-plastic-artifacts/models/spectrogram_v0.0.11_2021-07-13.h5",
           patch_model_file: str="s3://flyte-plastic-artifacts/models/v1.1_weak_labels_28x28x24.h5",
           roi_file: str="s3://flyte-plastic-artifacts/boundaries/test_patch.geojson"
           ) -> str:

    unique_id = str(uuid.uuid4())

    ok = upload_model(model_file="s3://flyte-plastic-artifacts/models/spectrogram_v0.0.11_2021-07-13.h5",
                      patch_model_file="s3://flyte-plastic-artifacts/models/v1.1_weak_labels_28x28x24.h5",
                      roi_file="s3://flyte-plastic-artifacts/boundaries/test_patch.geojson",
                      run_id=unique_id)

    return "Success"


def setup():
    # User inputs
    model_version = '0.0.11'
    model_name = 'spectrogram_v0.0.11_2021-07-13'
    model_file = '../models/' + model_name + '.h5'

    patch_model_version = 'weak_labels_1.1'
    patch_model_name = 'v1.1_weak_labels_28x28x24'
    patch_model_file = '../models/' + patch_model_name + '.h5'
    patch_model = load_model(patch_model_file, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                               'ELU': keras.layers.ELU,
                                                               'ReLU': keras.layers.ReLU})
    patch_stride = 8
    patch_input_shape = patch_model.input_shape[2]

    # Note on dates: The date range should be longer than the spectrogram length.
    # Starting on successive mosaic periods (typically: monthly), as many
    # spectrograms are created as fit in the date range.
    start_date = '2019-01-01'
    end_date = '2021-06-01'

    mosaic_period = 3
    mosaic_method = 'min'
    spectrogram_interval = 2

    roi = 'bali_foot'
    roi_file = f'../data/boundaries/{roi}.geojson'
    product_id = f'earthrise:{roi}_v{model_version}_{start_date}_{end_date}'
    patch_product_id = f'earthrise:{roi}_patch_{patch_model_version}_{start_date}_{end_date}_stride_{patch_stride}'
    product_name = product_id.split(':')[-1]  # Arbitrary string - optionally set this to something more human readable.

    run_local = False  # If False, the model prediction tasks are async queued and sent to DL for processing.

    # If running locally, get results faster by setting small tile size (100?)
    # If running on Descartes, use tile size 900

    if run_local:
        tilesize = 100
    else:
        tilesize = 900


if __name__ == "__main__":
    print(f"Running detect()... {detect()}")
