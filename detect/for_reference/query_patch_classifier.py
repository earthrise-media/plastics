#!/usr/bin/env python
# coding: utf-8

# # Patch Classifier Query
# We are now running both a pixel and patch classifier on Descartes. This notebook takes in a set of patch classifier outputs along with a set of pixel classifier candidates. This notebook finds the intersection between candidate points and patch classifier predictions above a given threshold.

# In[ ]:


import geopandas as gpd
# pygeos is required to use geopandas spatial indexing
gpd.options.use_pygeos = True


# In[ ]:


patch_model_version = '0.4'
patch_classifier_fname= 'malaysia_patch_weak_labels_0.4_2019-01-01_2021-06-01_stride_8'
patch = gpd.read_file(f'../data/model_outputs/patch_classifier/{patch_classifier_fname}.geojson')


# In[ ]:


pixel_model_version = '0.0.11'
pixel_classifier_fname = 'malaysia_v0.0.11_unit_norm_2019-01-01_2021-06-01mosaic-median_blobs_thresh_0.6_min-sigma_3.5_area-thresh_0.00025'
pixel = gpd.read_file(f'../data/model_outputs/candidate_sites/{pixel_model_version}/{pixel_classifier_fname}.geojson')


# In[ ]:


threshold = 0.3
patch_threshold = patch[patch['mean'] > threshold]
patch_index = patch_threshold['geometry'].sindex


# In[ ]:


overlap = []
for candidate in pixel['geometry']:
    if len(patch_index.query(candidate)) > 0:
        overlap.append(True)
    else:
        overlap.append(False)
union = pixel[overlap]
print(f"{len(union)} candidate points intersect with patch classifier predictions greater than {threshold}")
union


# In[ ]:


filename = 'malaysia_intersection'
union.to_file(f'../data/model_outputs/candidate_sites/{pixel_model_version}/{filename}_patch_v{patch_model_version}_threshold_{threshold}.geojson', driver='GeoJSON')


# In[ ]:




