#!/usr/bin/env python
# coding: utf-8

# # Site Monitoring and Contour Generation

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import datetime
import os

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio as rs
import shapely
from tensorflow import keras
from tqdm.notebook import tqdm

from scripts import dl_utils
from scripts.dl_utils import predict_spectrogram, rect_from_point


# ## Download Data

# In[ ]:


def load_ensemble(folder_path):
    model_files = [file for file in os.listdir(folder_path) if '.h5' in file]
    model_list = []
    for file in model_files:
        model_list.append(keras.models.load_model(os.path.join(folder_path,file)))
    return model_list

def predict_ensemble(pairs, model_list):
    ensemble_preds = []
    for pair in pairs:
        pred_stack = []
        for ensemble_model in model_list:
            pred_stack.append(predict_spectrogram(pair, ensemble_model, unit_norm=True))
        ensemble_preds.append(np.median(pred_stack, axis=0))
    return ensemble_preds

def generate_contours(preds, dates, threshold=0.5, plot=False):
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
    
    img_size = preds[0].shape
    
    # Set a prediction threshold. Given that the heatmaps are blurred, it is recommended
    # to set this value lower than you would in blob detection
    contour_list = []
    date_list = []
    for pred, date in zip(preds, dates):
        # If a scene is masked beyond a threshold, don't generate contours
        masked_percentage = np.sum(pred.mask / np.size(pred.mask))
        if masked_percentage < 0.1:
            pred = np.array(Image.fromarray(pred).resize((img_size[0] * SCALE, img_size[1] * SCALE), Image.BICUBIC))
            # OpenCV works best with ints in range (0,255)
            input_img = (pred*255).astype(np.uint8)
            # Blur the image to minimize the influence of single-pixel or mid-value model outputs
            blurred = cv2.GaussianBlur(input_img, (8 * SCALE + 1,8 * SCALE + 1), cv2.BORDER_DEFAULT)
            # Set all values below a threshold to 0
            _, thresh = cv2.threshold(blurred, int(threshold * 255), 255, cv2.THRESH_TOZERO)
            # Note that cv2.RETR_CCOMP returns a hierarchy of parent and child contours
            # Needed for fixing the case with polygon holes 
            # https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = [contour for contour in contours if cv2.contourArea(contour) > 40 * SCALE]
            contour_list.append(contours)
            date_list.append(date)
            
            if plot:
                plt.figure(figsize=(16,4), dpi=150)
                plt.subplot(1,4,1)
                plt.imshow(np.array(Image.fromarray(pred).resize((img_size[0], img_size[1]), Image.BICUBIC)), vmin=0, vmax=1, cmap='RdBu_r')
                plt.title('Pred')
                plt.axis('off')
                plt.subplot(1,4,3)
                plt.imshow(thresh, vmin=0, vmax=255, cmap='RdBu_r')
                plt.title('Thresholded Blur')
                plt.axis('off')
                plt.subplot(1,4,2)
                plt.imshow(blurred, vmin=0, vmax=255, cmap='RdBu_r')
                plt.title('Blurred')
                plt.axis('off')
                plt.subplot(1,4,4)
                three_channel_preds = np.stack((blurred, blurred, blurred), axis=-1)
                preds_contour_img = cv2.drawContours(three_channel_preds, contours, -1, (255, 0, 0), SCALE)
                plt.imshow(preds_contour_img /255, vmin=0, vmax=255)
                plt.title(f"{len(contours)} separate contours")
                plt.axis('off')
                plt.suptitle(date)
                plt.show()
                
    return contour_list, date_list

def generate_polygons(contour_list, bounds_list, preds, plot=False):
    """
    Convert a list of coordinates into georeferenced shapely polygons
    Inputs
        - List of contours
        - List of patch coordinate boundaries
    Returns
        - A list of shapely MultiPolygons. One for each coordinate in the input list
    """
    contour_multipolygons = []
    for contours, bounds in zip(contour_list, bounds_list):
        # Define patch coordinate bounds to set pixel scale
        bounds = shapely.geometry.Polygon(bounds).bounds
        transform = rs.transform.from_bounds(*bounds, preds[0].shape[0] * SCALE, preds[0].shape[1] * SCALE)
        polygon_coords = []
        for contour in contours:
            # Convert from pixels to coords
            contour_coords = []
            for point in contour[:,0]:
                lon, lat = rs.transform.xy(transform, point[1], point[0])
                contour_coords.append([lon, lat])
            if len(contour_coords) > 1:
                # Close the loop
                contour_coords.append(contour_coords[0])
                # Add individual contour to list of contours for the month
                polygon_coords.append(contour_coords)

        contour_polygons = []
        for coord in polygon_coords:
            poly = shapely.geometry.Polygon(coord)
            # A single line of pixels will be recognized as a line rather than a polygon
            # Inflate the area by a small amount to create a polygon
            if poly.area == 0:
                poly = poly.buffer(0.00002)
            contour_polygons.append(poly)
        multipolygon = shapely.geometry.MultiPolygon(contour_polygons)
        # Currently, "holes" in a polygon are seen as separate contours.
        # This means that there will be overlapping polygons. Shapely can 
        # detect this case, but can't fix it automatically. To rectify, the
        # unary_union operator and .buffer(0) hack removes interior polygons.
        if not multipolygon.is_valid:
            multipolygon = multipolygon.buffer(0)
        if plot:
            display(multipolygon)
        contour_multipolygons.append(multipolygon)
    return contour_multipolygons

def pad_preds(preds, window_size):
    pad_len = window_size - 1
    padded_preds = np.concatenate(([np.mean(preds[:pad_len], axis=0) for _ in range(pad_len)], preds))
    return padded_preds


def mask_predictions(preds, window_size=6, threshold=0.1):
    # Create a median prediction mask
    if len(preds) <= window_size:
        window_size = len(preds)
    padded_preds = pad_preds(preds, window_size)
    masks = np.array([np.median(padded_preds[i:i+window_size], axis=0) for i in range(0, len(preds))])
    masks[masks < threshold] = 0
    masks[masks > threshold] = 1
    
    # mask predictions
    masked_preds = np.ma.multiply(preds, masks)
    return masked_preds


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
    coords = [[site.x, site.y] for site in confirmed_sites['geometry']]
    for candidate in coords:
        rect = dl_utils.rect_from_point(candidate, rect_height)
        rects.append(shapely.geometry.Polygon(rect['coordinates'][0]))

    confirmed_sites['rect'] = rects
    names = confirmed_sites['name']
    overlapping = []
    for name, rect in zip(names, rects):
        overlapping.append([rect.overlaps(other_rects) for other_rects in rects])
    confirmed_sites['overlap'] = [list(confirmed_sites['name'][overlap]) for overlap in overlapping]

    contour_df['updated_geometry'] = [None for _ in range(len(contour_df))]

    all_contours = []
    contour_indices = []
    for site in tqdm(names):
        site_contours = list(contour_df[contour_df['name'] == site]['geometry'])
        site_indices = list(contour_df[contour_df['name'] == site]['geometry'].index)
        site_dates = list(contour_df[contour_df['name'] == site]['date'])
        site_center = confirmed_sites[confirmed_sites['name'] == site]['geometry'].item()
        overlapping_sites = list(confirmed_sites[confirmed_sites['name'] == site]['overlap'])[0]
        
        # iterate through each site with a rect that overlaps the site of interest rect
        for external_site in overlapping_sites:
            external_contours = list(contour_df[contour_df['name'] == external_site]['geometry'])
            external_dates = list(contour_df[contour_df['name'] == external_site]['date'])
            external_center = confirmed_sites[confirmed_sites['name'] == external_site]['geometry'].item()
            
            # iterate through each date where the site has a contour
            for index, date in enumerate(site_dates):
                
                # check if overlapping site has a contour at that date
                if date in external_dates:
                    site_index = external_dates.index(date)
                    
                    # sometimes contours can be none. Check if contours exist for both
                    if site_contours[index] and external_contours[site_index]:
                        
                        # check if the multipolygon contours overlap
                        try:
                            contours_overlap = site_contours[index].overlaps(external_contours[site_index])
                            if contours_overlap == True:
                                # if the contours overlap, create a new list of polygons
                                site_polygon_list = []

                                # make sure sites are multipolygons rather than polygons
                                if type(external_contours[site_index]) != shapely.geometry.multipolygon.MultiPolygon:
                                    external_contours[site_index] = shapely.geometry.MultiPolygon([external_contours[site_index]])

                                # for each polygon in the external site multipolygon
                                for external_polygon in external_contours[site_index].geoms:
                                    # check each polygon in the site multipolygon to see if they overlap
                                    if type(site_contours[index]) != shapely.geometry.multipolygon.MultiPolygon:
                                        site_contours[index] = shapely.geometry.MultiPolygon([site_contours[index]])
                                    for site_polygon in site_contours[index].geoms:

                                        # if the site polygon overlaps, check which rect center is nearest
                                        if site_polygon.overlaps(external_polygon):
                                            site_centroid = site_polygon.centroid
                                            site_distance = site_center.distance(site_centroid)
                                            external_centroid = external_polygon.centroid
                                            external_distance = external_center.distance(external_centroid)
                                            if site_distance > external_distance:
                                                #print(f"Site {site} overlaps {external_site} on {date}, {site}'s polygon is closer")
                                                try:
                                                    site_contours[index] -= site_polygon
                                                except Exception as e:
                                                    print(e)
                                            # else:
                                                #print(f"Site {site} overlaps {external_site} on {date}, {external_site}'s polygon is closer")
                        except Exception as e:
                            print(e)

        all_contours += site_contours
        contour_indices += site_indices

    resolved_df = gpd.GeoDataFrame({
            'name': contour_df['name'][contour_indices], 
            'date': contour_df['date'][contour_indices]}, 
            geometry=all_contours).set_crs('EPSG:4326')

    resolved_df['area (km^2)'] = resolved_df['geometry'].to_crs('epsg:3395').map(lambda p: p.area / 10**6 if p != None else None)
    
    return resolved_df

def filter_scattered_contours(contours, polygon_threshold=3, area_threshold=0.003):
    # Bad contours are oftentimes small point sources of heat scattered throughout a frame
    # If there are more than `polygon_threshold` contours in a multipolygon, and the average
    # contour area is above `area_threshold`, then delete the polygons for that time point.
    # area threshold is in km^2

    filtered_contours = contours.copy()
    for i in range(len(contours)):
        site = contours.iloc[i]
        if site['geometry'] != None and type(site['geometry']) == shapely.geometry.multipolygon.MultiPolygon:
            num_polygons = len(site['geometry'].geoms)
            area = site['area (km^2)']
            if num_polygons >= polygon_threshold and area / num_polygons < area_threshold:
                filtered_contours.iloc[i] = None
    print(sum(filtered_contours['geometry'] == None) - sum(contours['geometry'] == None), "contours removed")
    return filtered_contours


# In[ ]:


version_num = str(2.1)
contour_output_path = f'../data/model_outputs/site_contours/{version_num}'
if not os.path.exists(contour_output_path):
    os.mkdir(contour_output_path)
# Image upsampling factor. Makes for smoother contours
SCALE = 4
# Rect needs to be large enough to cover big sites. 
# But large rects take longer to process and increase false positive likelihood
RECT_WIDTH = 0.008
START_DATE = '2016-06-01'
END_DATE = '2021-09-01'
MOSAIC_PERIOD = 3
SPECTROGRAM_INTERVAL = 2


# In[ ]:


#model_name = 'spectrogram_v0.0.11_2021-07-13'
#model = keras.models.load_model(f'../models/{model_name}.h5')

ensemble_name = 'v0.0.11_ensemble-8-25-21'
model_list = load_ensemble(f'../models/{ensemble_name}')


# In[ ]:


confirmed_sites_file = 'philippines_confirmed_v0.0.7_3.5_0.6_blob_detect_v0.1_classifier'
confirmed_sites = gpd.read_file(f"../data/sampling_locations/{confirmed_sites_file}.geojson")
coords = [[site.x, site.y] for site in confirmed_sites['geometry']]
names = confirmed_sites['name']
print(len(confirmed_sites), 'sites loaded')
display(confirmed_sites.head())


# In[ ]:


# Initialize a contour GeoDataFrame. I'm not certain that EPSG:4326 is the correct coordinate system.
# Coordinate system must be set in order to compute polygon areas
contour_gdf = gpd.GeoDataFrame(columns=['geometry', 'area (km^2)', 'date', 'name']).set_crs('EPSG:4326')
failures = []
for coord, name in zip(tqdm(coords), names):
    try:
        # Download data
        mosaics, metadata = dl_utils.download_mosaics(
            rect_from_point(coord, RECT_WIDTH), START_DATE, END_DATE, MOSAIC_PERIOD, method='min')
        pairs = dl_utils.pair(mosaics, SPECTROGRAM_INTERVAL)

        # Generate predictions
        #preds = [predict_spectrogram(pair, model) for pair in pairs] # single model
        preds = predict_ensemble(pairs, model_list) # ensemble model
        for i in range(len(preds)):
            preds[i][preds[i] < 0.5] = 0
        window_size = 8
        preds_m = mask_predictions(preds, window_size=window_size, threshold=0.1)

        # Generate contours and polygons
        dates = dl_utils.get_starts(START_DATE, END_DATE, 3, 2)[SPECTROGRAM_INTERVAL:]
        bounds = [sample['wgs84Extent']['coordinates'][0][:-1] for sample in metadata[SPECTROGRAM_INTERVAL:]]
        contours, contour_dates = generate_contours(preds_m, dates, threshold=0.2, plot=False)
        polygons = generate_polygons(contours, bounds, preds_m, plot=False)

        # Write contours to a GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons).set_crs('EPSG:4326')
        gdf['date'] = [datetime.datetime.fromisoformat(date) for date in contour_dates]

        # Calculate contour area. I'm not certain this is a valid technique for calculating area
        gdf['area (km^2)'] = gdf['geometry'].to_crs('epsg:3395').map(lambda p: p.area / 10**6)
        gdf['name'] = [name for _ in range(len(contour_dates))]
        contour_gdf = contour_gdf.append(gdf, ignore_index=True)
    except Exception as e:
        print(name, 'failed')
        print(e)
        failures.append(name)

print(f'{1 - len(failures) / len(confirmed_sites):.1%} success rate')
print(f'Failed sites: {failures}')


# In[ ]:


polygon_threshold = 3
min_area_threshold = 0.003
filtered_df = filter_scattered_contours(contour_gdf, polygon_threshold=polygon_threshold, area_threshold=min_area_threshold)


# In[ ]:


resolved_df = resolve_overlapping_contours(confirmed_sites, filtered_df)
resolved_df.to_file(f'{contour_output_path}/{confirmed_sites_file}_mask_window_{window_size}_upsampled_{SCALE}_contours_model_{ensemble_name}.geojson', driver='GeoJSON')


# In[ ]:




