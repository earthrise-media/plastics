import glob
import os
import pickle
from os.path import exists
from hashlib import sha256
from datetime import date
import json, functools, time
import numpy as np
import luigi
from luigi.contrib.s3 import S3Target, S3Client
from skimage.feature import blob_doh
from skimage.feature.peak import peak_local_max
from sklearn.neighbors import KDTree
import descarteslabs as dl
from descarteslabs.catalog import Image, properties
import geopandas as gpd
import rasterio as rs
from rasterio.merge import merge

from tensorflow.keras.models import load_model
from tensorflow import keras

from scripts import deploy_nn_v1


class ModelFile(luigi.Task):
    model: str = luigi.Parameter()

    def output(self):
        s3file = f's3://flyte-plastic-artifacts/models/{self.model}'
        dest = f'/tmp/{self.model}'
        S3Client().get(s3_path=s3file, destination_local_path=dest)
        return luigi.LocalTarget(dest)


class ROIFile(luigi.Task):
    # test_patch
    roi: str = luigi.Parameter()

    def output(self):
        s3file = f's3://flyte-plastic-artifacts/boundaries/{self.roi}.geojson'
        dest = f'/tmp/{self.roi}.geojson'
        S3Client().get(s3_path=s3file, destination_local_path=dest)
        return luigi.LocalTarget(dest)


class PatchModelFile(luigi.Task):
    model: str = luigi.Parameter()

    def output(self):
        s3file = f's3://flyte-plastic-artifacts/models/{self.model}'
        dest = f'/tmp/{self.model}'
        S3Client().get(s3_path=s3file, destination_local_path=dest)
        return luigi.LocalTarget(dest)


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
        return luigi.LocalTarget(f'/tmp/{self.run_id}/{self.run_id}.json')

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
        anticipated_results = deploy_nn_v1.main(args)

        while True:
            groups = dl.tasks.list_groups(sort_order="desc", sort_field="created", limit=10)
            group_list = groups.get("groups")

            for group in group_list:
                if group.get("name") == self.run_id:
                    print(f'Found task id: {group.get("id")}')
                    # add run args to metadata file so we have it
                    group['run_args'] = args
                    group['anticipated_results'] = anticipated_results
                    with self.output().open('w') as outfile:
                        json.dump(group, outfile)
                    return
            print("Waiting 10 seconds for task to appear...")
            time.sleep(10)


class DownloadPatchGeojson(luigi.Task):
    run_id = luigi.Parameter()
    patch_product_id = luigi.Parameter()
    product_id = luigi.Parameter()
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    model = luigi.Parameter()
    roi = luigi.Parameter()
    patch_model = luigi.Parameter()
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
        return luigi.LocalTarget(f"/tmp/{self.run_id}/{self.run_id}_patch.geojson")

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
                if group['queue']['pending'] == -1:
                    if group['queue']['successes'] > 0:
                        # at least partial success (I think)
                        print('Task appears to be complete:')
                        print(
                            f"{group['queue']['successes']} successful jobs and +"
                            f"{group['queue']['failures']} failed jobs")
                        break
                    if group['queue']['successes'] == 0 and group['queue']['failures'] > 0:
                        print("All jobs appear to have failed")
                        raise Exception("All Descartes Labs jobs appear to have failed")
                elif group['status'] == 'running':
                    print(
                        f"Task still running with {group['queue']['pending']} pending jobs, +"
                        f"{group['queue']['successes']} successful jobs and {group['queue']['failures']} failed jobs")
                    time.sleep(100)
            except dl.client.exceptions.BadRequestError as e:
                print("API error....waiting to try again")
                time.sleep(100)
            except dl.client.exceptions.ServerError as e:
                print("API error....waiting to try again")
                time.sleep(100)
            ## old way of checking
            # results = dl.tasks.get_task_results(self.dl_run['id'])
            # # descartes breaks the job into tiles, each tile should be a result
            # if len(results["results"]) != self.dl_run['anticipated_results']:
            #     print("Results not yet ready....waiting to check again")
            #     time.sleep(100)
            #     continue
            #
            # for res in results["results"]:
            #     print(f"Result status: {res['status']}")
            #     if res["status"] == 'FAILURE':
            #         # task failed, error out
            #         raise Exception("At least 1 Descartes Task Failed")
            # break

        # get the first file that contains our patch_product_id -- naming really needs to be cleaned up
        fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if self.patch_product_id in fc.id][0]
        fc = dl.vectors.FeatureCollection(fc_id)
        # use our ROI as a geo filter -- this seems unnecessary
        region = gpd.read_file(self.input()['rf'].path)['geometry']
        filtered_features = []
        # add matching features to temp collection
        for f in fc.filter(region).features():
            filtered_features.append(f.geojson)

        results = gpd.GeoDataFrame.from_features(filtered_features)
        outfile = self.output().path
        results.to_file(outfile, driver='GeoJSON')


class DownloadHeatmaps(luigi.Task):
    run_id = luigi.Parameter()
    patch_product_id = luigi.Parameter()
    product_id = luigi.Parameter()
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    model = luigi.Parameter()
    roi = luigi.Parameter()
    patch_model = luigi.Parameter()
    dl_run = {}

    def requires(self):
        return {'dg': DownloadPatchGeojson(run_id=self.run_id,
                                           patch_product_id=self.patch_product_id,
                                           product_id=self.product_id,
                                           start_date=self.start_date,
                                           end_date=self.end_date,
                                           model=self.model,
                                           patch_model=self.patch_model,
                                           roi=self.roi), 'rf': ROIFile(roi=self.roi)}

    def output(self):
        return luigi.LocalTarget(f"/tmp/{self.run_id}/heatmaps/")

    def run(self):

        search = Image.search().filter(properties.product_id == self.product_id)

        band = 'median'

        basepath = f"/tmp/{self.run_id}/heatmaps/"

        print("Saving to", basepath)
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        image_list = [image.id for image in search]
        raster_client = dl.Raster()
        for image in image_list:
            try:
                print(f"Saving {image}")
                raster_client.raster(inputs=image,
                                     bands=[band],
                                     save=True,
                                     outfile_basename=os.path.join(basepath, image),
                                     srs='WGS84')
            except dl.client.exceptions.BadRequestError as e:
                print(f'Warning: {repr(e)}\nContinuing...')
            except dl.client.exceptions.ServerError as e:
                print(f'Warning: {repr(e)}\nContinuing...')


# class MergeSimilarSites(luigi.Task):
#
#     search_radius = luigi.FloatParameter(default=0.0025)
#
#     def output(self):
#         pass
#     def requires(self):
#         pass
#     def run(self):

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


def detect_blobs(source, name, pred_threshold=0.75, min_sigma=3.5, max_sigma=100, area_threshold=0.0025,
                 window_size=5000, save=True):
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
            print(
                f"Processing row {(x // window_size) + 1} of {int(source.shape[0] / window_size) + 1}, column {(y // window_size) + 1} of {int(source.shape[1] / window_size) + 1}")
            # Set min and max to analyze a subset of the image
            window = Window.from_slices((x, x + window_size), (y, y + window_size))
            window_median = (source.read(1, window=window) / max_val).astype('float')
            # mask predictions below a threshold
            mask = np.ma.masked_where(window_median < pred_threshold, window_median).mask
            window_median[mask] = 0

            blobs = blob_doh(window_median, min_sigma=min_sigma, max_sigma=max_sigma, threshold=area_threshold)
            print(len(blobs), "candidates detected in window")

            overlap_threshold = 0.01
            transform = source.window_transform(window)
            for candidate in blobs:
                lon, lat = (transform * [candidate[1], candidate[0]])
                # Size doesn't mean anything at the moment. Should look into this later
                # size = candidate[2]
                candidate_sites.append([lon, lat])

    print(len(candidate_sites), "candidate sites detected in total")

    candidate_gdf = merge_similar_sites(candidate_sites, search_radius=0.01)
    display(candidate_gdf)

    if save:
        file_path = f"../data/model_outputs/candidate_sites/{name}_blobs_thresh_{pred_threshold}_min-sigma_{min_sigma}_area-thresh_{area_threshold}"
        # candidate_gdf.loc[:, ['lon', 'lat', 'name']].to_csv(file_path + '.csv', index=False)
        candidate_gdf.to_file(file_path + '.geojson', driver='GeoJSON')

    return candidate_gdf



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

        return luigi.LocalTarget(self.file+".json")

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
    source_dir = luigi.Parameter()
    candidate_dir = luigi.Parameter()
    pred_threshold = luigi.FloatParameter(default=0.75)
    min_sigma = luigi.FloatParameter(default=3.5)
    max_sigma = luigi.IntParameter(default=100)
    area_threshold = luigi.FloatParameter(0.0025)
    merge_radius = luigi.FloatParameter(default=0.005)

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(self.candidate_dir+"/candidates.geojson")

    def run(self):
        # files = os.listdir(self.source_dir)
        # file_paths = [os.path.join(self.source_dir, file) for file in files]

        # this should produce absolute paths
        files = glob.glob(self.source_dir+"**.tif")


        # blob_detect_partial = functools.partial(blob_detect,
        #                                     pred_threshold=self.pred_threshold,
        #                                     min_sigma=self.min_sigma,
        #                                     max_sigma=self.max_sigma,
        #                                     area_threshold=self.area_threshold)
        #
        # site_list = process_map(blob_detect_partial, file_paths)

        # instead of the above we'll yield a list of tasks
        detection_tasks = [BlobDetect(file=file,
                                      pred_threshold=self.pred_threshold,
                                      min_sigma=self.min_sigma,
                                      max_sigma=self.max_sigma,
                                      area_threshold=self.area_threshold
                                      ) for file in files]
        # this will run all the detection tasks in luigi
        task_result = yield detection_tasks

        # result files should be in the same dir
        json_files = glob.glob(self.source_dir+"**.json")

        candidate_sites = []

        for jf in json_files:
            with open(jf, 'r') as json_file:
                more_sites = json.load(json_file)
                if isinstance(more_sites, list):
                    candidate_sites.extend(more_sites)
                else:
                    candidate_sites.append(more_sites)

        print(len(candidate_sites), "sites detected overall")

        # todo: make this into a task (maybe?)
        candidate_gdf = merge_similar_sites(candidate_sites, search_radius=self.merge_radius)

        print(len(candidate_sites) - len(candidate_gdf), "sites merged")

        candidate_gdf.to_file(self.candidate_dir+'/candidates.geojson', driver='GeoJSON')

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
            window = Window.from_slices((x, x + window_size), (y, y + window_size))
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
    candidate_gdf['name'] = [f"{name}_{i + 1}" for i in candidate_gdf.index]

    if save:
        file_path = f"../data/model_outputs/candidate_sites/{name}_peaks_thresh_{threshold_abs}_min_dist_{min_distance}"
        # candidate_gdf.loc[:, ['lon', 'lat', 'name']].to_csv(file_path + '.csv', index=False)
        candidate_gdf.to_file(file_path + '.geojson', driver='GeoJSON')

    return candidate_gdf


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
                    continue
            # upload file
            S3Client().put(filename, s3file)

class SpectrogramRun(luigi.WrapperTask):
    # input params
    start_date = luigi.DateParameter(default=date(2019, 1, 1))
    end_date = luigi.DateParameter(default=date(2021, 6, 1))
    model = luigi.Parameter(default="spectrogram_v0.0.11_2021-07-13.h5")
    roi = luigi.Parameter(default="test_patch")
    patch_model = luigi.Parameter(default="v1.1_weak_labels_28x28x24.h5")

    def output(self):
        pass

    def requires(self):
        # task id is a long string created from all the parameters
        # we're going to hash it to make it more usable and it will still be idempotent
        run_id = sha256(self.task_id.encode('utf-8')).hexdigest()[:20]
        # setup a dir to store all our files (even if they end up in s3)
        if not exists(f'/tmp/{run_id}/'):
            os.mkdir(f"/tmp/{run_id}/")

        patch_product_id = f'patch_product_{run_id}'
        product_id = f'earthrise:product_{run_id}'

        yield DownloadHeatmaps(run_id=run_id,
                               patch_product_id=patch_product_id,
                               product_id=product_id,
                               start_date=self.start_date,
                               end_date=self.end_date,
                               model=self.model,
                               patch_model=self.patch_model,
                               roi=self.roi)
        yield SyncToS3(source_dir=f"/tmp/{run_id}/", dest_dir=f"s3://flyte-plastic-artifacts/runs/{run_id}/")


class DetectCandidates(luigi.Task):

    # input params
    start_date = luigi.DateParameter(default=date(2019, 1, 1))
    end_date = luigi.DateParameter(default=date(2021, 6, 1))
    model = luigi.Parameter(default="spectrogram_v0.0.11_2021-07-13.h5")
    roi = luigi.Parameter(default="test_patch")
    patch_model = luigi.Parameter(default="v1.1_weak_labels_28x28x24.h5")

    def requires(self):
        pass
        # SpectrogramRun(start_date=self.start_date,
        #                end_date=self.end_date,
        #                model=self.model,
        #                roi=self.roi,
        #                patch_model=self.patch_model)

    def output(self):
        pass

    def run(self):
        sd = "/tmp/9e4b2a4463f49486e40b/heatmaps/"
        parent_dir = os.path.abspath(os.path.join(sd,os.pardir))
        candidate_dir= os.path.join(parent_dir,"candidates")
        if not os.path.exists(candidate_dir):
            os.mkdir(candidate_dir)
        yield DetectBlobsTiled(source_dir="/tmp/9e4b2a4463f49486e40b/heatmaps/", candidate_dir=candidate_dir)