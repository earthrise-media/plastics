import datetime
import os

import cv2
from dateutil.relativedelta import relativedelta
import descarteslabs as dl
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio as rs
import requests
import shapely
from shapely.validation import make_valid
from tensorflow import keras
from tqdm import tqdm

try:
    import dl_utils
    from dl_utils import predict_spectrogram, rect_from_point
except:
    from scripts import dl_utils
    from scripts.dl_utils import predict_spectrogram, rect_from_point

class SiteContours(object):
    def __init__(
        self,
        coord,
        name,
    ):
        self.coord = coord
        self.name = name

    def download_pairs(
        self,
        start_date="2016-06-01",
        end_date=datetime.datetime.isoformat(datetime.datetime.now())[:10],
        mosaic_period=6,
        spectrogram_interval=1,
        rect_width=0.01,
    ):
        polygon = rect_from_point(self.coord, rect_width)
        self.data = dl_utils.SentinelData(polygon, start_date, end_date, mosaic_period, spectrogram_interval, method='min')
        self.rect_width = rect_width
        self.data.search_scenes()
        self.data.download_scenes()
        self.data.create_composites()
        self.data.create_pairs()
        self.pairs = self.data.pairs
        self.dates = self.data.pair_starts
        self.bounds = self.data.metadata[0]["wgs84Extent"]["coordinates"][0][:-1]
        

    def generate_predictions(self, model):
        # Generate predictions
        if type(model) == list:
            preds = dl_utils.predict_ensemble(self.pairs, model)
        else:
            preds = [predict_spectrogram(pair, model) for pair in self.pairs]
        self.preds = preds

    def threshold_predictions(self, threshold=0.5):
        # Set predictions below a threshold to 0
        preds_t = np.ma.copy(self.preds)
        for i in range(len(self.preds)):
            preds_t[i].data[self.preds[i] < threshold] = 0
        self.preds_t = preds_t

    def mask_predictions(self, window_size=6, threshold=0.1):
        # Create a median prediction mask
        if len(self.preds_t) <= window_size:
            window_size = len(self.preds_t)
            if window_size == 1:
                window_size += 1
        padded_preds = pad_preds(self.preds_t, window_size)
        masks = np.array(
            [
                np.median(padded_preds[i: i + window_size], axis=0)
                for i in range(0, len(self.preds_t))
            ]
        )
        masks[masks < threshold] = 0
        masks[masks > threshold] = 1
        # mask predictions
        masked_preds = np.ma.multiply(self.preds_t, masks)
        self.masked_preds = masked_preds

    def generate_contours(self, threshold=0.5, scale=4, plot=False):
        """
        Generate a list of contours for a set of predictions
        Inputs
            - preds: A list of numpy prediction arrays
            - dates: A list of dates for each scene in prediction list
            - threshold: Value under which pixels are masked. Given that the heatmaps are first
                        blurred, it is recommended to set this value lower than in blob detection
            - plot: Visualize outputs if True
        Outputs
            - A list of contours. Each prediction frame has a separate list of contours.
            Contours are defined as (x,y) pairs
            - A list of dates for instance when contours were generated
        """

        img_size = self.masked_preds[0].shape
        self.scale = scale

        # Set a prediction threshold. Given that the heatmaps are blurred, it is recommended
        # to set this value lower than you would in blob detection
        contour_list = []
        date_list = []
        hierarchies = []
        for index, (pred, date) in enumerate(zip(self.masked_preds, self.dates)):
            # If a scene is masked beyond a threshold, don't generate contours
            masked_percentage = np.sum(pred.mask) / np.size(pred.mask)
            if masked_percentage < 0.1:
                pred = np.array(
                    Image.fromarray(pred).resize(
                        (img_size[0] * scale, img_size[1]
                         * scale), Image.BICUBIC
                    )
                )
                # OpenCV works best with ints in range (0,255)
                input_img = (pred * 255).astype(np.uint8)
                # Blur the image to minimize the influence of single-pixel or mid-value model outputs
                blurred = cv2.GaussianBlur(
                    input_img, (8 * scale + 1, 8 * scale +
                                1), cv2.BORDER_DEFAULT
                )
                # Set all values below a threshold to 0
                _, thresh = cv2.threshold(
                    blurred, int(threshold * 255), 255, cv2.THRESH_TOZERO
                )
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                contour_list.append(contours)
                date_list.append(date)
                hierarchies.append(hierarchy)

                if plot:
                    plt.figure(figsize=(20, 4), dpi=150)
                    plt.subplot(1, 5, 1)
                    plt.imshow(
                        np.array(
                            Image.fromarray(pred).resize(
                                (img_size[0], img_size[1]), Image.BICUBIC
                            )
                        ),
                        vmin=0,
                        vmax=1,
                        cmap="RdBu_r",
                    )
                    plt.title("Pred")
                    plt.axis("off")
                    plt.subplot(1, 5, 3)
                    plt.imshow(thresh, vmin=0, vmax=255, cmap="RdBu_r")
                    plt.title("Thresholded Blur")
                    plt.axis("off")
                    plt.subplot(1, 5, 2)
                    plt.imshow(blurred, vmin=0, vmax=255, cmap="RdBu_r")
                    plt.title("Blurred")
                    plt.axis("off")
                    plt.subplot(1, 5, 4)
                    three_channel_preds = np.stack(
                        (blurred, blurred, blurred), axis=-1)
                    preds_contour_img = cv2.drawContours(
                        three_channel_preds, contours, -
                        1, (255, 0, 0), self.scale
                    )
                    plt.imshow(preds_contour_img / 255)
                    plt.title(f"{len(contours)} separate contours")
                    plt.axis("off")
                    plt.subplot(1, 5, 5)
                    rgb = np.clip(np.mean(self.pairs[index], axis=0)[
                                  :, :, 3:0:-1] / 3000, 0, 1)
                    plt.imshow(rgb)
                    plt.title(f"mean image")
                    plt.axis("off")
                    plt.suptitle(date)
                    plt.show()

        self.contour_list = contour_list
        self.date_list = date_list
        self.contour_hierarchies = hierarchies

    def generate_polygons(self, plot=False):
        """
        Convert a list of coordinates into georeferenced shapely polygons
        Inputs
            - List of contours
            - List of patch coordinate boundaries
        Returns
            - A list of shapely MultiPolygons. One for each coordinate in the input list
        """
        contour_multipolygons = []
        for contours, hierarchy in zip(
            self.contour_list, self.contour_hierarchies
        ):
            # Define patch coordinate bounds to set pixel scale
            #bounds = shapely.geometry.Polygon(bounds).bounds
            polygon = rect_from_point(self.coord, self.rect_width)
            bounds = shapely.geometry.Polygon(polygon['coordinates'][0]).bounds
            transform = rs.transform.from_bounds(
                *bounds,
                self.masked_preds[0].shape[0] * self.scale,
                self.masked_preds[0].shape[1] * self.scale,
            )
            polygon_coords = []
            for contour in contours:
                # Convert from pixels to coords
                contour_coords = []
                for point in contour[:, 0]:
                    lon, lat = rs.transform.xy(transform, point[1], point[0])
                    contour_coords.append([[lon, lat]])
                polygon_coords.append(np.array(contour_coords))
            if len(polygon_coords) > 0:
                contour_polygons = generate_sp_polygons(
                    polygon_coords, hierarchy)
            else:
                contour_polygons = []

            multipolygon = shapely.geometry.MultiPolygon(contour_polygons)
            if not multipolygon.is_valid:
                multipolygon = multipolygon.buffer(0)
                multipolygon = make_valid(multipolygon)
            if plot:
                display(multipolygon)
            contour_multipolygons.append(multipolygon)
        self.polygons = contour_multipolygons

    def compile_contours(self):
        if len(self.polygons) > 0:
            gdf = gpd.GeoDataFrame(geometry=self.polygons)
            gdf.crs = "epsg:4326"
            gdf["date"] = [datetime.datetime.fromisoformat(
                date) for date in self.date_list]
            gdf["area (km^2)"] = (
                gdf["geometry"].to_crs("epsg:3395").map(
                    lambda p: p.area / 10**6)
            )
            gdf["h3_id"] = [self.name for _ in range(len(self.date_list))]
            self.contour_gdf = gdf


def load_ensemble(folder_path):
    model_files = [fn for fn in os.listdir(folder_path) if ".h5" in fn]
    model_list = []
    for fn in model_files:
        model_list.append(keras.models.load_model(
            os.path.join(folder_path, fn)))
    return model_list


def predict_ensemble(pairs, model_list):
    ensemble_preds = []
    for pair in pairs:
        pred_stack = []
        for ensemble_model in model_list:
            pred_stack.append(predict_spectrogram(
                pair, ensemble_model, unit_norm=True))
        ensemble_preds.append(np.median(pred_stack, axis=0))
    return ensemble_preds


def merge_ensemble(model_list):
    """
    Create a single keras model from an ensemble of models.
    Output predictions are averaged
    """
    model_input = keras.layers.Input(model_list[0].input_shape[1:])
    averaged = keras.layers.Average()(
        [model(model_input) for model in model_list])
    mean_model = keras.models.Model(inputs=model_input, outputs=averaged)
    return mean_model


def pad_preds(preds, window_size):
    pad_len = window_size - 1
    padded_preds = np.concatenate(
        ([np.mean(preds[:pad_len], axis=0) for _ in range(pad_len)], preds)
    )
    return padded_preds


def _DFS(polygons, contours, hierarchy, sibling_id, is_outer, siblings):
    while sibling_id != -1:
        contour = contours[sibling_id].squeeze(axis=1)
        if len(contour) >= 3:
            first_child_id = hierarchy[sibling_id][2]
            children = [] if is_outer else None
            _DFS(polygons, contours, hierarchy,
                 first_child_id, not is_outer, children)

            if is_outer:
                polygon = shapely.geometry.Polygon(contour, holes=children)
                # display(polygon)
                # print(hierarchy[sibling_id])
                polygons.append(polygon)
            else:
                siblings.append(contour)

        sibling_id = hierarchy[sibling_id][0]


def generate_sp_polygons(contours, hierarchy):
    """Generates a list of Shapely polygons from the contours hirarchy returned by cv2.find_contours().
       The list of polygons is generated by performing a depth-first search on the contours hierarchy tree.
       Code from https://gist.github.com/stefano-malacrino/7d429e5d12854b9e51b187170e812fa4
    Parameters
    ----------
    contours : list
      The contours returned by cv2.find_contours()
    hierarchy : list
      The hierarchy returned by cv2.find_contours()
    Returns
    -------
    list
      The list of generated Shapely polygons
    """

    hierarchy = hierarchy[0]
    polygons = []
    _DFS(polygons, contours, hierarchy, 0, True, [])
    return polygons


def resolve_overlapping_contours(confirmed_sites, contour_df):
    """
    This is not a function I am proud of writing. The intention is to
    look at a dataframe of contours, find sets of overlapping contours,
    and then assign a contour to the site which is nearest. It seems that
    this could be done with less code and fewer loops.
    Inputs:
      - confirmed_sites: the original dataframe of site names and coordinates
      - contour_gdf: a dataframe of contours for each site through time
    Returns:
      - a contour dataframe with no overlapping contours.
    """
    rect_height = 0.02
    rects = []
    coords = [[site.x, site.y] for site in confirmed_sites["geometry"]]
    for candidate in coords:
        rect = dl_utils.rect_from_point(candidate, rect_height)
        rects.append(shapely.geometry.Polygon(rect["coordinates"][0]))

    confirmed_sites["rect"] = rects
    names = confirmed_sites["name"]
    overlapping = []
    for name, rect in zip(names, rects):
        overlapping.append([rect.overlaps(other_rects)
                           for other_rects in rects])
    confirmed_sites["overlap"] = [
        list(confirmed_sites["name"][overlap]) for overlap in overlapping
    ]

    contour_df["updated_geometry"] = [None for _ in range(len(contour_df))]

    all_contours = []
    contour_indices = []
    for site in tqdm(names):
        site_contours = list(
            contour_df[contour_df["name"] == site]["geometry"])
        site_indices = list(
            contour_df[contour_df["name"] == site]["geometry"].index)
        site_dates = list(contour_df[contour_df["name"] == site]["date"])
        site_center = confirmed_sites[confirmed_sites["name"] == site][
            "geometry"
        ].item()
        overlapping_sites = list(
            confirmed_sites[confirmed_sites["name"] == site]["overlap"]
        )[0]

        # iterate through each site with a rect that overlaps the site of interest rect
        for external_site in overlapping_sites:
            external_contours = list(
                contour_df[contour_df["name"] == external_site]["geometry"]
            )
            external_dates = list(
                contour_df[contour_df["name"] == external_site]["date"]
            )
            external_center = confirmed_sites[confirmed_sites["name"] == external_site][
                "geometry"
            ].item()

            # iterate through each date where the site has a contour
            for index, date in enumerate(site_dates):

                # check if overlapping site has a contour at that date
                if date in external_dates:
                    site_index = external_dates.index(date)

                    # sometimes contours can be none. Check if contours exist for both
                    if site_contours[index] and external_contours[site_index]:

                        # check if the multipolygon contours overlap
                        try:
                            contours_overlap = site_contours[index].overlaps(
                                external_contours[site_index]
                            )
                            if contours_overlap == True:
                                # if the contours overlap, create a new list of polygons
                                site_polygon_list = []

                                # make sure sites are multipolygons rather than polygons
                                if (
                                    type(external_contours[site_index])
                                    != shapely.geometry.multipolygon.MultiPolygon
                                ):
                                    external_contours[
                                        site_index
                                    ] = shapely.geometry.MultiPolygon(
                                        [external_contours[site_index]]
                                    )

                                # for each polygon in the external site multipolygon
                                for external_polygon in external_contours[
                                    site_index
                                ].geoms:
                                    # check each polygon in the site multipolygon to see if they overlap
                                    if (
                                        type(site_contours[index])
                                        != shapely.geometry.multipolygon.MultiPolygon
                                    ):
                                        site_contours[
                                            index
                                        ] = shapely.geometry.MultiPolygon(
                                            [site_contours[index]]
                                        )
                                    for site_polygon in site_contours[index].geoms:

                                        # if the site polygon overlaps, check which rect center is nearest
                                        if site_polygon.overlaps(external_polygon):
                                            site_centroid = site_polygon.centroid
                                            site_distance = site_center.distance(
                                                site_centroid
                                            )
                                            external_centroid = (
                                                external_polygon.centroid
                                            )
                                            external_distance = (
                                                external_center.distance(
                                                    external_centroid
                                                )
                                            )
                                            if site_distance > external_distance:
                                                # print(f"Site {site} overlaps {external_site} on {date}, {site}'s polygon is closer")
                                                try:
                                                    site_contours[index] -= site_polygon
                                                except Exception as e:
                                                    print(e)
                                            # else:
                                            # print(f"Site {site} overlaps {external_site} on {date}, {external_site}'s polygon is closer")
                        except Exception as e:
                            print(e)

        all_contours += site_contours
        contour_indices += site_indices

    resolved_df = gpd.GeoDataFrame(
        {
            "name": contour_df["name"][contour_indices],
            "date": contour_df["date"][contour_indices],
        },
        geometry=all_contours,
    ).set_crs("EPSG:4326")

    resolved_df["area (km^2)"] = (
        resolved_df["geometry"]
        .to_crs("epsg:3395")
        .map(lambda p: p.area / 10**6 if p != None else None)
    )

    return resolved_df


def filter_scattered_contours(contours, polygon_threshold=3, area_threshold=0.003):
    # Bad contours are oftentimes small point sources of heat scattered throughout a frame
    # If there are more than `polygon_threshold` contours in a multipolygon, and the average
    # contour area is above `area_threshold`, then delete the polygons for that time point.
    # area threshold is in km^2

    filtered_contours = contours.copy()
    for i in range(len(contours)):
        site = contours.set_crs('EPSG:4326').to_crs('epsg:3857').iloc[i]
        if (site["geometry"] != None and type(site["geometry"]) == shapely.geometry.multipolygon.MultiPolygon):
            num_polygons = len(site["geometry"].geoms)
            area = site["area (km^2)"]
            if (num_polygons >= polygon_threshold and area / num_polygons < area_threshold):
                [print(poly.area) for poly in site["geometry"].geoms]
                print(i, len(site))
                filtered_polygons = [poly for poly in site['geometry'].geoms if poly.area > 100]
                filtered_contours['geometry'][i] = shapely.geometry.MultiPolygon(filtered_polygons)
                print(len(filtered_contours['geometry'][i]))
    print(
        sum(filtered_contours["geometry"] == None) -
        sum(contours["geometry"] == None),
        "contours removed",
    )
    return filtered_contours


class DescartesContourRun(object):
    """Class to manage bulk model prediction on the Descartes Labs platform.

    Attributes:
        product_id: DL id for output rasters
        product_name: String identifer for output rasters
        product: Instantiated dl.catalog.Product
        model_name: String identifier for learned Keras model
        model: Instantiated Keras model
        mosaic_period: Integer number of months worth of data to mosaic
        mosaic_method: Compositing method for the mosaic() function
        spectrogram_interval: Integer number of mosaic periods between mosaics
            input to spectrogram
        spectrogram_length: Total duration of a spectrogram in months
        input_bands: List of DL names identifying Sentinel bands

    External methods:
        init_product: Create or get DL catalog product with specified bands.
        upload_model: Upload model to DL storage.
        init_model: Instantiate model from DL storage.
        __call__: Run model on a geographic tile.
        predict: Predict on image-mosaic spectrograms.
        add_band: Create a band in the DL product.
        upload_raster: Upload a raster to DL storage.
    """

    def __init__(self,
                 product_id,
                 model_name,
                 rect_width,
                 product_name='',
                 model_file='',
                 mosaic_period=1,
                 mosaic_method='min',
                 spectrogram_interval=6,
                 nodata=-1,
                 scale=4,
                 input_bands=dl_utils.SENTINEL_BANDS,
                 start_date='',
                 end_date='',
                 endpoint='',
                 **kwargs):

        self.model_name = model_name
        self.rect_width = float(rect_width)
        self.start_date = start_date
        self.end_date = end_date
        if product_id.startswith('earthrise:'):
            self.product_id = product_id
        else:
            self.product_id = f'earthrise:{product_id}'
        self.product_name = product_name if product_name else self.product_id
        self.nodata = nodata
        self.scale = scale
        self.product = self.init_product()
        if model_file:
            self.upload_model(model_file)
        self.model = self.init_model()
        self.mosaic_period = mosaic_period
        self.mosaic_method = mosaic_method
        self.spectrogram_interval = spectrogram_interval
        self.spectrogram_length = self._get_gram_length()
        self.input_bands = input_bands
        self.endpoint = endpoint

    def init_product(self):
        """Create or get DL catalog product."""
        fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list()]
        product_id = None
        for fc in fc_ids:
            if self.product_id in fc:
                product_id = fc

        if not product_id:
            print("Creating product", self.product_id + '_contours')
            product = dl.vectors.FeatureCollection.create(product_id=self.product_id + '_contours',
                                                          title=self.product_name + '_contours',
                                                          description=self.model_name)
        else:
            print(f"Product {self.product_id}_patches already exists...")
            product = dl.vectors.FeatureCollection(product_id)
        return product

    def upload_model(self, model_file):
        """Upload model to DL storage."""
        if dl.Storage().exists(self.model_name):
            print(f'Model {self.model_name} found in DLStorage.')
        else:
            dl.Storage().set_file(self.model_name, model_file)
            print(f'Model {model_file} uploaded with key {self.model_name}.')

    def init_model(self):
        """Instantiate model from DL storage."""
        temp_file = 'tmp-' + self.model_name
        dl.Storage().get_file(self.model_name, temp_file)
        model = keras.models.load_model(temp_file, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                                   'ELU': keras.layers.ELU,
                                                                   'ReLU': keras.layers.ReLU
                                                                   })
        os.remove(temp_file)
        return model

    def _get_gram_length(self):
        """Compute the length of the spectrogram in months."""
        interval_months = self.mosaic_period * self.spectrogram_interval
        n_intervals = self.model.input_shape[2]
        last_interval_months = self.mosaic_period
        return interval_months * (n_intervals - 1) + last_interval_months

    def __call__(self, coord, name):
        """
        Generate contours for a given site

        Args:
            coord: coordinate pair for waste site
            name: h3_id of waste site

        Returns: None. Adds contours to API
        """
        site = SiteContours(coord, name)
        print("Generating contours for site", name)
        print("Downloading data at", datetime.datetime.isoformat(datetime.datetime.now())[:19])
        site.download_pairs(
            start_date=self.start_date,
            end_date=self.end_date,
            mosaic_period=self.mosaic_period,
            spectrogram_interval=self.spectrogram_interval,
            rect_width=self.rect_width
        )
        if len(site.pairs) > 0:
            print("Generating predictions at", datetime.datetime.isoformat(datetime.datetime.now())[:19])
            site.generate_predictions(self.model)
            print("Preds finished at", datetime.datetime.isoformat(datetime.datetime.now())[:19])
            site.threshold_predictions(threshold=0.5)
            site.mask_predictions(window_size=6, threshold=0.1)
            site.generate_contours(threshold=0.5, scale=self.scale, plot=False)
            site.generate_polygons(plot=False)
            site.compile_contours()
            #contour_gdf = contour_gdf.append(site.contour_gdf, ignore_index=True)
            site_endpoint = f'{self.endpoint}/sites/{site.name}/contours'
            auth = requests.auth.HTTPBasicAuth('admin', 'plastics')
            delete = requests.delete(site_endpoint, auth=auth)
            print("delete status", delete.status_code)
            site.contour_gdf['date'] = [datetime.datetime.isoformat(date) for date in site.contour_gdf['date']]
            r = requests.post(site_endpoint, site.contour_gdf.to_json())
            print("request status", r.status_code)
            print("Finished at", datetime.datetime.isoformat(datetime.datetime.now())[:19])
            """
            I'm adding sites directly to the API now, so I don't need to add to a DL product.
            The DL product would throw an error for some geometries.
            That error would need to be fixed if we did want to go back to DL storage.
            Keeping this in case we want it in the future
            feature_list = []
            for feature in site.contour_gdf.iterfeatures():
                if feature['geometry'] == None:
                    feature['geometry'] = shapely.geometry.Point(site.coord)
                feature['properties']['date'] = datetime.datetime.isoformat(
                    feature['properties']['date'])
                feature_list.append(
                    dl.vectors.Feature(
                        geometry=feature['geometry'],
                        properties=feature['properties']
                    )
                )
            if len(feature_list) > 0:
                self.product.add(feature_list)
            """
            return site.contour_gdf
