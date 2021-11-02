import os
from hashlib import sha256
from urllib.parse import urlparse
from datetime import date
import json, time
from io import BytesIO
import numpy
import skimage.io
from dateutil.relativedelta import relativedelta
import numpy as np
import luigi
from luigi.contrib.s3 import S3Target, S3Client, AtomicS3File
from skimage.feature import blob_doh
from skimage.feature.peak import peak_local_max
from sklearn.neighbors import KDTree
import descarteslabs as dl
from descarteslabs.catalog import Image, properties
import geopandas as gpd
import rasterio as rs
import boto3

from tensorflow.keras.models import load_model
from tensorflow import keras
import h3
from scripts import deploy_nn_v1, dl_utils

## config and globals

SENTINEL_BANDS = ['coastal-aerosol',
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

NORMALIZATION = 3000

class GlobalConfig(luigi.Config):
    s3_base_url = luigi.Parameter(default="s3://flyte-plastic-artifacts/runs")
    s3_bucket = luigi.Parameter(default="flyte-plastic-artifacts")
    s3_base_folder = luigi.Parameter(default="runs")

    local_temp_dir = luigi.Parameter(default="/tmp")
    # we should expect run_id to be set by the pipeline entrypoint


# Helper methods

def numpy_to_s3(data: np.array, dest: str):
    # s3_uri looks like f"s3://{BUCKET_NAME}/{KEY}"
    bytes_ = BytesIO()
    np.save(bytes_, data, allow_pickle=True)
    bytes_.seek(0)
    parsed_s3 = urlparse(dest)
    boto3.s3
    boto3.client("s3").upload_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )

def s3_to_numpy(s3_uri: str) -> np.array:

    bytes_ = BytesIO()
    parsed_s3 = urlparse(s3_uri)
    boto3.client("s3").download_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    bytes_.seek(0)
    return np.load(bytes_, allow_pickle=True)

def merge_similar_sites(candidate_sites, search_radius=0.0025):
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

    unique_sites = gpd.GeoDataFrame(mean_coords, columns=['lon', 'lat'], geometry=gpd.points_from_xy(*mean_coords.T))
    unique_sites['name'] = [f"site_{i + 1}" for i in unique_sites.index]
    return unique_sites

# Tasks

class ModelFile(luigi.Task):

    model: str = luigi.Parameter()

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(f'{GlobalConfig().local_temp_dir}/{self.model}')

    def run(self):
        s3file = f's3://{GlobalConfig().s3_bucket}/models/{self.model}'
        dest = f'{GlobalConfig().local_temp_dir}/{self.model}'
        S3Client().get(s3_path=s3file, destination_local_path=dest)


class ROIFile(luigi.Task):
    # test_patch
    roi: str = luigi.Parameter()

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(f'{GlobalConfig().local_temp_dir}/{self.roi}.geojson')

    def run(self):
        s3file = f's3://{GlobalConfig().s3_bucket}/boundaries/{self.roi}.geojson'
        dest = f'{GlobalConfig().local_temp_dir}/{self.roi}.geojson'
        S3Client().get(s3_path=s3file, destination_local_path=dest)


class PatchModelFile(luigi.Task):
    model: str = luigi.Parameter()

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(f'{GlobalConfig().local_temp_dir}/{self.model}')

    def run(self):
        s3file = f's3://{GlobalConfig().s3_bucket}/models/{self.model}'
        dest = f'{GlobalConfig().local_temp_dir}/{self.model}'
        S3Client().get(s3_path=s3file, destination_local_path=dest)


class LaunchDescartes(luigi.Task):
    # parameters
    run_id = luigi.Parameter()
    patch_product_id = luigi.Parameter()
    product_id = luigi.Parameter()
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    model = luigi.Parameter()
    roi = luigi.Parameter()
    patch_model = luigi.Parameter()
    # output
    dl_task_id: str

    def output(self):
        return S3Target(f'{GlobalConfig().s3_base_url}/{self.run_id}/{self.run_id}.json')

    def requires(self):
        return {'mf': ModelFile(self.model), 'rf': ROIFile(self.roi), 'pmf': PatchModelFile(self.patch_model), }

    def run(self):

        patch_model = load_model(self.input()['mf'].path, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                                          'ELU': keras.layers.ELU,
                                                                          'ReLU': keras.layers.ReLU})
        df = gpd.read_file(self.input()['rf'].path)['geometry']

        # todo: figure out how to generate this from file name -- if it matters...
        patch_model_version = 'weak_labels_1.1'

        patch_model_name = str(self.patch_model).replace(".h5", "")
        model_name = str(self.model).replace(".h5", "")

        tilesize = 900
        patch_stride = 8
        patch_input_shape = patch_model.input_shape[2]

        mosaic_period = 3
        mosaic_method = 'min'
        spectrogram_interval = 2
        padding = patch_input_shape - patch_stride

        args = [
            '--roi_file',
            self.input()['rf'].path,
            '--product_id',
            self.product_id,
            '--patch_product_id',
            self.patch_product_id,
            '--product_name',
            self.run_id,
            '--model_file',
            self.input()['mf'].path,
            '--model_name',
            model_name,
            '--patch_model_name',
            patch_model_name,
            '--patch_model_file',
            self.input()['pmf'].path,
            '--mosaic_period',
            str(mosaic_period),
            '--mosaic_method',
            mosaic_method,
            '--spectrogram_interval',
            str(spectrogram_interval),
            '--start_date',
            self.start_date.strftime("%Y-%m-%d"),
            '--end_date',
            self.end_date.strftime("%Y-%m-%d"),
            '--pad',
            '0',
            '--tilesize',
            str((tilesize // patch_input_shape) * patch_input_shape - padding)
        ]
        deploy_nn_v1.main(args)

        while True:
            groups = dl.tasks.list_groups(sort_order="desc", sort_field="created", limit=10)
            group_list = groups.get("groups")

            for group in group_list:
                if group.get("name") == self.run_id:
                    print(f'Found task id: {group.get("id")}')
                    # add run args to metadata file so we have it
                    group['run_args'] = args
                    with self.output().open('w') as outfile:
                        json.dump(group, outfile)
                    return
            print("Waiting 10 seconds for task to appear...")
            time.sleep(10)


class DownloadPatchGeojson(luigi.Task):
    run_id = luigi.Parameter()
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    model = luigi.Parameter()
    roi = luigi.Parameter()
    patch_model = luigi.Parameter()
    patch_product_id = luigi.Parameter(default=f'patch_product_{run_id}')
    product_id = luigi.Parameter(default=f'earthrise:product_{run_id}')

    dl_run = {}

    def requires(self):
        return {'ld': LaunchDescartes(run_id=self.run_id,
                                      patch_product_id=self.patch_product_id,
                                      product_id=self.product_id,
                                      start_date=self.start_date,
                                      end_date=self.end_date,
                                      model=self.model,
                                      patch_model=self.patch_model,
                                      roi=self.roi), 'rf': ROIFile(roi=self.roi)}

    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/{self.run_id}_patch.geojson")

    def run(self):

        # load json from previous step
        with self.input()["ld"].open('r') as infile:
            self.dl_run = json.load(infile)

        while True:
            try:
                group = dl.tasks.get_group(self.dl_run['id'])
                # task not running yet
                if group['status'] == 'pending':
                    print("Tasking is pending...")
                    time.sleep(100)
                    continue
                    # think "-1" means no more jobs will will be scheduled
                if group['queue']['pending'] <= 0:
                    if group['queue']['successes'] > 0:
                        # at least partial success (I think)
                        print('Task appears to be complete:')
                        print(
                            f"{group['queue']['successes']} successful jobs and "
                            f"{group['queue']['failures']} failed jobs")
                        break
                    if group['queue']['successes'] == 0 and group['queue']['failures'] > 0:
                        print("All Descartes Labs jobs appear to have failed")
                        raise Exception("All Descartes Labs jobs appear to have failed")
                elif group['status'] == 'running':
                    print(
                        f"Descartes task still running with {group['queue']['pending']} pending jobs, "
                        f"{group['queue']['successes']} successful jobs and {group['queue']['failures']} failed jobs")
                    time.sleep(100)
            except dl.client.exceptions.BadRequestError as e:
                print("API error....waiting to try again")
                time.sleep(100)
            except dl.client.exceptions.ServerError as e:
                print("API error....waiting to try again")
                time.sleep(100)

        # get the first file that contains our patch_product_id -- naming really needs to be cleaned up
        fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if self.patch_product_id in fc.id][0]
        fc = dl.vectors.FeatureCollection(fc_id)
        # use our ROI as a geo filter -- this seems unnecessary
        region = gpd.read_file(self.input()['rf'].path)['geometry']
        filtered_features = []
        # add matching features to temp collection
        for f in fc.filter(region).features():
            filtered_features.append(f.geojson)
        print(f"Found {len(filtered_features)} after filtering")
        results = gpd.GeoDataFrame.from_features(filtered_features)
        tmp = AtomicS3File(self.output().path, S3Client())

        results.to_file(tmp, driver='GeoJSON')
        # I don't think this should be required but was getting 0 length files in S3 without it
        tmp.flush()
        tmp.move_to_final_destination()


class DownloadHeatmaps(luigi.Task):
    run_id = luigi.Parameter()
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    model = luigi.Parameter()
    roi = luigi.Parameter()
    patch_model = luigi.Parameter()
    patch_product_id = luigi.Parameter(default=f'patch_product_{run_id}')
    product_id = luigi.Parameter(default=f'earthrise:product_{run_id}')
    dl_run = {}


    def requires(self):
        return {'dg': DownloadPatchGeojson(run_id=self.run_id,
                                           patch_product_id=self.patch_product_id,
                                           product_id=self.product_id,
                                           start_date=self.start_date,
                                           end_date=self.end_date,
                                           model=self.model,
                                           patch_model=self.patch_model,
                                           roi=self.roi),
                'rf': ROIFile(roi=self.roi)}

    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/heatmaps/")

    def run(self):

        search = Image.search().filter(properties.product_id == self.product_id)

        band = 'median'

        image_list = [image.id for image in search]

        dl_tasks = []
        for image in image_list:

            dl_tasks.append(DownloadHeatmap(
                image=image,
                run_id=self.run_id,
                band=band,
            ))
        yield dl_tasks


class DownloadHeatmap(luigi.Task):

    """
    This downloads heatmaps from Descartes and saves them on S3 (not locally)
    """

    image = luigi.Parameter()

    run_id = luigi.Parameter()
    band = luigi.Parameter()

    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/heatmaps/{self.image}_{self.band}.tif")

    def run(self):
        tmp = AtomicS3File(f"{GlobalConfig().s3_base_url}/{self.run_id}/heatmaps/{self.image}_{self.band}.tif", S3Client())
        print(f"image name is {self.image}")
        response = dl.Raster().raster(inputs=self.image,
                           bands=[self.band],
                           save=False,
                           srs='WGS84')

        tmp.write(response['files'][f"{self.image}_{self.band}.tif"])
        tmp.flush()
        tmp.move_to_final_destination()


class BlobDetect(luigi.Task):
    """
        Identify candidates using blob detection on the heatmap.
        prediction_threshold masks any prediction below a 0-1 threshold.
        min_sigma and area_threshold control the size sensitivity of the blob detection.
        Keep min_sigma low to detect smaller blobs
        area_threshold establishes a lower bound on candidate blob size. Reduce to detect smaller blobs
        """

    file = luigi.Parameter()
    pred_threshold = luigi.FloatParameter(default=0.75)
    min_sigma = luigi.FloatParameter(default=3.5)
    max_sigma = luigi.IntParameter(default=100)
    area_threshold = luigi.FloatParameter(default=0.0025)

    def requires(self):
        pass

    def output(self):

        return S3Target(self.file+".json")

    def run(self):

        source = rs.open(self.file)
        data = source.read(1).astype('float')
        # mask predictions below a threshold
        mask = np.ma.masked_where(data < self.pred_threshold, data).mask
        data[mask] = 0

        candidate_sites = []

        blobs = blob_doh(data, min_sigma=self.min_sigma, max_sigma=self.max_sigma, threshold=self.area_threshold)
        if len(blobs) > 0:
            print(len(blobs), "candidate(s) detected in window")
            transform = source.transform

            for candidate in blobs:
                lon, lat = (transform * [candidate[1], candidate[0]])
                    # Size doesn't mean anything at the moment. Should look into this later
                    # size = candidate[2]
                candidate_sites.append([lon, lat])

        # write this to file even if empty so luigi knows it's complete
        with self.output().open('w') as outfile:
            json.dump(candidate_sites, outfile)


class DetectBlobsTiled(luigi.Task):
    """
        Identify candidates using blob detection on the heatmap.
        prediction_threshold masks any prediction below a 0-1 threshold.
        min_sigma and area_threshold control the size sensitivity of the blob detection.
        Keep min_sigma low to detect smaller blobs
        area_threshold establishes a lower bound on candidate blob size. Reduce to detect smaller blobs
    """

    # parameters

    pred_threshold = luigi.FloatParameter(default=0.75)
    min_sigma = luigi.FloatParameter(default=3.5)
    max_sigma = luigi.IntParameter(default=100)
    area_threshold = luigi.FloatParameter(0.0025)
    merge_radius = luigi.FloatParameter(default=0.005)
    run_id = luigi.Parameter()

    def requires(self):
        pass

    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/candidates/candidates.geojson")

    def run(self):

        # this should produce absolute paths
        print(f"{GlobalConfig().s3_base_url}/{self.run_id}/heatmaps/")
        s3_file_list = S3Client().listdir(f"{GlobalConfig().s3_base_url}/{self.run_id}/heatmaps/")

        # instead of file paths, these are s3 paths
        files = []

        for s3_file in s3_file_list:
            if str(s3_file).endswith('.tif'):
                files.append(s3_file)

        # instead of the above we'll yield a list of tasks
        detection_tasks = [BlobDetect(file=file,
                                      pred_threshold=self.pred_threshold,
                                      min_sigma=self.min_sigma,
                                      max_sigma=self.max_sigma,
                                      area_threshold=self.area_threshold
                                      ) for file in files]
        # this will run all the detection tasks in luigi
        task_result = yield detection_tasks

        # look for json files this time
        s3_file_list = S3Client().listdir(f"{GlobalConfig().s3_base_url}/{self.run_id}/heatmaps/")

        json_files = []

        for s3_file in s3_file_list:
            if str(s3_file).endswith('.json'):
                # add the absolute s3 path as an S3 Target
                json_files.append(S3Target(s3_file))

        candidate_sites = []

        for jf in json_files:
            with jf.open('r') as json_file:
                more_sites = json.load(json_file)
                if isinstance(more_sites, list):
                    candidate_sites.extend(more_sites)
                else:
                    candidate_sites.append(more_sites)

        print(len(candidate_sites), "sites detected overall")

        candidate_gdf = merge_similar_sites(candidate_sites, search_radius=self.merge_radius)

        print(len(candidate_sites) - len(candidate_gdf), "sites merged")

        tmp = AtomicS3File(self.output().path, S3Client())
        candidate_gdf.to_file(tmp, driver='GeoJSON')
        tmp.flush()
        tmp.move_to_final_destination()


class SyncToS3(luigi.Task):

    source_dir = luigi.Parameter()
    dest_dir = luigi.Parameter()
    clobber = luigi.BoolParameter(default=False)

    def requires(self):
        pass

    def output(self):
        return S3Target(self.dest_dir)

    def run(self):
        if not os.path.exists(self.source_dir):
            raise Exception("source directory doesn't exist")
        import glob
        for filename in glob.iglob(self.source_dir+'**/**', recursive=True):
            if os.path.isdir(filename):
                continue
            s3file = self.dest_dir+os.path.relpath(filename, self.source_dir)

            # check if exists
            if S3Client().exists(s3file):
                # only overwrite if asked
                if not self.clobber:
                    print(f"Skipping existing file {s3file}")
                    continue
            # upload file
            S3Client().put(filename, s3file)


class SpectrogramRun(luigi.Task):
    # input params
    start_date = luigi.DateParameter(default=date(2019, 1, 1))
    end_date = luigi.DateParameter(default=date(2021, 6, 1))
    model = luigi.Parameter(default="spectrogram_v0.0.11_2021-07-13.h5")
    roi = luigi.Parameter(default="test_patch")
    patch_model = luigi.Parameter(default="v1.1_weak_labels_28x28x24.h5")
    run_id = luigi.Parameter()
    patch_product_id = luigi.Parameter(default=f'patch_product_{run_id}')
    product_id = luigi.Parameter(default=f'earthrise:product_{run_id}')

    def output(self):
        return S3Target(f'{GlobalConfig().s3_base_url}/{self.run_id}/run.id')

    def requires(self):
        # task id is a long string created from all the parameters
        # we're going to hash it to make it more usable and it should still be idempotent
        #self.run_id = sha256(self.task_id.encode('utf-8')).hexdigest()[:20]

        # patch_product_id = f'patch_product_{self.run_id}'
        # product_id = f'earthrise:product_{self.run_id}'

        # print(self.run_id)

        return DownloadHeatmaps(run_id=self.run_id,
                         patch_product_id=self.patch_product_id,
                         product_id=self.product_id,
                         start_date=self.start_date,
                         end_date=self.end_date,
                         model=self.model,
                         patch_model=self.patch_model,
                         roi=self.roi)

    def run(self):

        with self.output().open('w') as outfile:
            outfile.write(self.run_id)


class DetectCandidates(luigi.Task):

    # input params
    start_date = luigi.DateParameter(default=date(2019, 1, 1))
    end_date = luigi.DateParameter(default=date(2021, 6, 1))
    model = luigi.Parameter(default="spectrogram_v0.0.11_2021-07-13.h5")
    roi = luigi.Parameter(default="test_patch")
    patch_model = luigi.Parameter(default="v1.1_weak_labels_28x28x24.h5")
    run_id = luigi.Parameter()

    def requires(self):
        return { 'sr': SpectrogramRun(start_date=self.start_date,
                       end_date=self.end_date,
                       model=self.model,
                       roi=self.roi,
                       patch_model=self.patch_model)
                 }

    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/candidates/candidates.geojson")

    def run(self):
        # self.run_id = S3Client().get_as_string(self.input()['sr'].path)
        yield DetectBlobsTiled(run_id=self.run_id)


class ComparePatchClassifier(luigi.Task):

    """
    Compares candidate point sites with patch polygons and outputs a final geojson
    """

    #intenal params
    gpd.options.use_pygeos = True

    start_date = luigi.DateParameter(default=date(2019, 1, 1))
    end_date = luigi.DateParameter(default=date(2021, 6, 1))
    model = luigi.Parameter(default="spectrogram_v0.0.11_2021-07-13.h5")
    roi = luigi.Parameter(default="test_patch")
    patch_model = luigi.Parameter(default="v1.1_weak_labels_28x28x24.h5")
    run_id = luigi.Parameter()

    def requires(self):
        return {'sr': SpectrogramRun(start_date=self.start_date,
                                     end_date=self.end_date,
                                     model=self.model,
                                     roi=self.roi,
                                     patch_model=self.patch_model,
                                     run_id=self.run_id),
                'dc': DetectCandidates(start_date=self.start_date,
                                     end_date=self.end_date,
                                     model=self.model,
                                     roi=self.roi,
                                     patch_model=self.patch_model,
                                     run_id=self.run_id),
                'pgt': DownloadPatchGeojson(run_id=self.run_id,
                             # patch_product_id=self.patch_product_id,
                             # product_id=self.product_id,
                             start_date=self.start_date,
                             end_date=self.end_date,
                             model=self.model,
                             patch_model=self.patch_model,
                             roi=self.roi),
                'dbt': DetectBlobsTiled(run_id=self.run_id)
                }

    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/output/sites.geojson")

    def run(self):

        # These are both S3Targets
        patch = gpd.read_file(self.input()['pgt'].path)
        pixel = gpd.read_file(self.input()['dc'].path)

        threshold = 0.3
        patch_threshold = patch[patch['mean'] > threshold]
        patch_index = patch_threshold['geometry'].sindex

        overlap = []
        for candidate in pixel['geometry']:
            if len(patch_index.query(candidate)) > 0:
                overlap.append(True)
            else:
                overlap.append(False)
        union = pixel[overlap]
        print(f"{len(union)} candidate points intersect with patch classifier predictions greater than {threshold}")

        tmp = AtomicS3File(self.output().path, S3Client())

        union.to_file(tmp, driver='GeoJSON')
        # I don't think this should be required but was getting 0 length files in S3 without it
        tmp.flush()
        tmp.move_to_final_destination()


class DownloadPatches(luigi.Task):
    """
    Downloads all of the patches for a given site
    """

    #params
    polygon_string = luigi.Parameter()
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()

    location_name = luigi.Parameter()

    rect_width = luigi.FloatParameter(default=0.008)
    mosaic_period = luigi.IntParameter(default=1)
    spectrogram_interval = luigi.IntParameter(2)


    def output(self):
        # this is a directory containing patches for the given polygon
        return S3Target(f"{GlobalConfig().s3_base_url}/patches/{self.location_name}/")

    def requires(self):
        pass

    def run(self):
        batches, raster_infos = [], []
        delta = relativedelta(months=self.mosaic_period)
        start = self.start_date
        end = start + delta
        sub_tasks = []
        while end <= self.end_date:
            sub_tasks.append(DownloadPatch(polygon_string=self.polygon_string,
                                           start_date=start,
                                           end_date=end,
                                           location_name=self.location_name))
            start += delta
            end += delta
        yield sub_tasks


class DownloadPatch(luigi.Task):

    s2_id = luigi.Parameter(default='sentinel-2:L1C')
    s2cloud_id = luigi.Parameter(default='sentinel-2:L1C:dlcloud:v1')
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    location_name = luigi.Parameter()
    polygon_string = luigi.Parameter()

    def output(self):
        # this is a directory containing patches for the given polygon
        start_str = self.start_date.isoformat()
        end_str = self.end_date.isoformat()
        return S3Target(f"{GlobalConfig().s3_base_url}/patches/{self.location_name}/{start_str}-{end_str}.npy")

    def requires(self):
        pass

    def run(self):
        polygon = json.loads(self.polygon_string)

        cloud_scenes, _ = dl.scenes.search(
            polygon,
            products=[self.s2cloud_id],
            start_datetime=self.start_date.isoformat(),
            end_datetime=self.end_date.isoformat(),
            limit=None
        )

        scenes, geoctx = dl.scenes.search(
            polygon,
            products=[self.s2_id],
            start_datetime=self.start_date.isoformat(),
            end_datetime=self.end_date.isoformat(),
            limit=None
        )

        # select only scenes that have a cloud mask
        cloud_dates = [scene.properties.acquired for scene in cloud_scenes]
        dates = [scene.properties.acquired for scene in scenes]
        shared_dates = set(cloud_dates) & set(dates)
        scenes = scenes.filter(
            lambda x: x.properties.acquired in shared_dates)
        cloud_scenes = cloud_scenes.filter(
            lambda x: x.properties.acquired in shared_dates)

        # A cloud stack is an array with shape (num_img, data_band, height, width)
        # A value of 255 means that the pixel is cloud free, 0 means cloudy
        cloud_stack = cloud_scenes.stack(bands=['valid_cloudfree'], ctx=geoctx)

        img_stack, raster_info = scenes.stack(
            bands=SENTINEL_BANDS, ctx=geoctx, raster_info=True)
        cloud_masks = np.repeat(cloud_stack, repeats=12, axis=1)

        # Add cloud masked pixels to the image mask
        img_stack.mask[cloud_masks.data == 0] = True

        # Remove fully masked images and reorder to channels last
        # TODO: remove raster info for fully masked images too
        img_stack = [np.moveaxis(img, 0, -1) for img in img_stack
                     if np.sum(img) > 0]

        numpy_to_s3(img_stack, self.output().path)

        print(f"Saved array stack to {self.output().path}")


class GenerateContoursForSite(luigi.Task):

    site_name = luigi.Parameter()
    polygon_string = luigi.Parameter()
    run_id = luigi.Parameter()
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    location_name = luigi.Parameter()

    def requires(self):

        return DownloadPatches(polygon_string=self.polygon_string,
                               start_date=self.start_date,
                               end_date=self.end_date,
                               location_name=self.location_name)

    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/output/{self.site_name}.geojson")

    def run(self):
        raise Exception("not implemented yet")


class GenerateContourForSites(luigi.Task):
    # intenal params
    gpd.options.use_pygeos = True

    start_date = luigi.DateParameter(default=date(2019, 1, 1))
    end_date = luigi.DateParameter(default=date(2021, 6, 1))
    model = luigi.Parameter(default="spectrogram_v0.0.11_2021-07-13.h5")
    roi = luigi.Parameter(default="test_patch")
    patch_model = luigi.Parameter(default="v1.1_weak_labels_28x28x24.h5")
    # fine tuning params
    rect_width = luigi.FloatParameter(default=0.008)
    mosaic_period = luigi.IntParameter(default=3)
    spectrogram_interval = luigi.IntParameter(2)

    # for debugging
    run_id = "9e4b2a4463f49486e40b"


    def requires(self):
        return ComparePatchClassifier(start_date=self.start_date,
                                     end_date=self.end_date,
                                     model=self.model,
                                     roi=self.roi,
                                     patch_model=self.patch_model,
                                     run_id=self.run_id)
    def output(self):
        return S3Target(f"{GlobalConfig().s3_base_url}/{self.run_id}/output/contours.geojson")

    def run(self):
        # Load sites from previous step
        sites = gpd.read_file(self.input().path)
        coords = [[site.x, site.y] for site in sites['geometry']]
        names = sites['name']
        subtasks = []
        # "download" data to S3
        for coord, name in zip(coords, names):
            location_name = h3.geo_to_h3(coord[0], coord[1], 7)
            poly = dl_utils.rect_from_point(coord, self.rect_width)
            subtasks.append(GenerateContoursForSite(
                site_name=name,
                polygon_string=json.dumps(poly),
                run_id=self.run_id,
                location_name=location_name,
                start_date=self.start_date,
                end_date=self.end_date,
            ))
        yield subtasks


class GenerateContourFromImage(luigi.Task):
    def output(self):
        pass
    def requires(self):
        pass
    def run(self):
        pass