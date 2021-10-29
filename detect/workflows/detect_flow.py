import tensorflow.keras.models
from flytekit import task, workflow, conditional
from flytekit.types.file import HDF5EncodedFile, FlyteFile

import time, uuid
import boto3
from botocore.exceptions import ClientError

import descarteslabs as dl
from descarteslabs.catalog import Image, properties
import geopandas as gpd
import rasterio as rs
from rasterio.merge import merge

from tensorflow.keras.models import load_model
from tensorflow import keras

# hack warning!
# SCRIPT_DIR = os.path.dirname(os.path.abspath('../scripts'))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts import deploy_nn_v1


@task()
def launch_dl(model_file: HDF5EncodedFile,
                 patch_model_file: HDF5EncodedFile,
                 patch_product_id: str,
                 product_id: str,
                 roi_file: FlyteFile,
                 run_id: str
                 ) :
    # load the model from S3
    model_file.download()
    patch_model = load_model(model_file.path, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                                  'ELU': keras.layers.ELU,
                                                                  'ReLU': keras.layers.ReLU})
    # load geojsons from S3
    roi_file.download()
    df = gpd.read_file(roi_file.path)['geometry']

    # run


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
    # args.append('--run_local')

    deploy_nn_v1.main(args)
    # task = None
    # while task == None:
    #     time.sleep(5)
    #     task = dl.tasks.get_group_by_name(name=run_id)
    #
    # print(task.keys())

@task()
def find_task_id(task_name: str) -> str:

    groups = dl.tasks.list_groups(sort_order="desc", sort_field="created", limit=10)
    group_list = groups.get("groups")

    for group in group_list:
        if group.get("name") == task_name:
            print((f'Found task id: {group.get("id")}'))
            return str(group.get("id"))
    print("Waiting 10 seconds for task to appear...")
    time.sleep(10)
    return find_task_id(task_name=task_name)

@task()
def wait_for_results(task_id: str) -> bool:

    while True:
        results = dl.tasks.get_task_results(task_id)
        if len(results["results"]) != 2:
            # we're waiting for 2 files
            time.sleep(100)
            continue

        for res in results["results"]:
            print(f"Result status: {res['status']}")
            if res["status"] == 'FAILURE':
                # task failed, error out
                raise Exception("At least 1 Descartes Task Failed")

        return True

@task()
def download_patch_geojson(patch_product_id: str, roi_file: FlyteFile, unique_id: str) -> FlyteFile:
    #get the first file that contains our patch_product_id -- naming really needs to be cleaned up
    fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if patch_product_id in fc.id][0]
    print(f"feature collection id: {fc_id}")
    fc = dl.vectors.FeatureCollection(fc_id)
    # use our ROI as a geo filter -- this seems unnecessary
    roi_file.download()
    region = gpd.read_file(roi_file.path)['geometry']
    filtered_features = []
    # add matching features to temp collection
    for f in fc.filter(region).features():
        filtered_features.append(f.geojson)

    results = gpd.GeoDataFrame.from_features(filtered_features)
    tmp_file = f"file:///tmp/{patch_product_id}.geojson"
    results.to_file(tmp_file, driver='GeoJSON')

    return FlyteFile(tmp_file)


@task()
def upload_s3(local_file: FlyteFile, remote_name: str) -> FlyteFile:

    local_file.download()
    s3 = boto3.client('s3')

    with open(local_file, "rb") as f:
        try:
            response = s3.upload_fileobj(f, "flyte-plastic-artifacts", remote_name)
        except ClientError as e:
            print(e)
            raise Exception(f"Upload of {local_file} failed")
    return f"s3://flyte-plastic-artifacts{remote_name}"


@workflow()
def detect(model_file: str="s3://flyte-plastic-artifacts/models/spectrogram_v0.0.11_2021-07-13.h5",
           patch_model_file: str="s3://flyte-plastic-artifacts/models/v1.1_weak_labels_28x28x24.h5",
           roi_file: str="s3://flyte-plastic-artifacts/boundaries/test_patch.geojson"
           ) -> str:
    # unique_id = str(uuid.uuid4())
    unique_id = "dcf5856c-0c92-4412-be8c-3a51b177f05d"
    #setup_s3(f"s3://flyte-plastic-artifacts/runs/{unique_id}")
    patch_product_id = f'patch_product_{unique_id}'
    product_id = f'earthrise:product_{unique_id}'

    # ok = launch_dl(model_file="s3://flyte-plastic-artifacts/models/spectrogram_v0.0.11_2021-07-13.h5",
    #                   patch_model_file="s3://flyte-plastic-artifacts/models/v1.1_weak_labels_28x28x24.h5",
    #                   roi_file="s3://flyte-plastic-artifacts/boundaries/test_patch.geojson",
    #                   run_id=unique_id,
    #                   product_id=product_id,
    #                   patch_product_id=patch_product_id)

    dl_task_id = find_task_id(task_name=unique_id)
    ready = wait_for_results(task_id=dl_task_id)
    tmp_file = download_patch_geojson(patch_product_id=patch_product_id, roi_file=roi_file, unique_id=unique_id)
    remote_file = upload_s3(local_file=tmp_file, remote_name=f"/runs/{unique_id}/patch.geojson")

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
    print("Running Detect Workflow....")
    detect(model_file="s3://flyte-plastic-artifacts/models/spectrogram_v0.0.11_2021-07-13.h5",
           patch_model_file="s3://flyte-plastic-artifacts/models/v1.1_weak_labels_28x28x24.h5",
           roi_file="s3://flyte-plastic-artifacts/boundaries/test_patch.geojson")
