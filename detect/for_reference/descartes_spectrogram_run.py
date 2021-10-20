import os
import shutil

import descarteslabs as dl
from descarteslabs.catalog import Image, properties
import geopandas as gpd
import rasterio as rs
from rasterio.merge import merge
from tensorflow.keras.models import load_model
from tensorflow import keras
from tqdm.notebook import tqdm

from scripts import deploy_nn_v1


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

run_local = False # If False, the model prediction tasks are async queued and sent to DL for processing.

# If running locally, get results faster by setting small tile size (100?)
# If running on Descartes, use tile size 900

if run_local:
    tilesize = 100
else:
    tilesize = 900

# Generally, leave padding at 0
padding = patch_input_shape - patch_stride

args = [
    '--roi_file',
    roi_file,
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
    str(padding),
    '--tilesize',
    str((tilesize // patch_input_shape) * patch_input_shape - padding)
]
if run_local:
    args.append('--run_local')


# Launch Descartes job. Monitor at https://monitor.descarteslabs.com/

# In[ ]:
#
#
# # Because of the way DL uploads modules when queuing async tasks, we need to launch from the scripts/ folder
# get_ipython().run_line_magic('cd', '../scripts')
# get_ipython().run_line_magic('pwd', '')


# In[ ]:


# Check if patch feature collection exists. If it does, delete the FC
fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list() if patch_product_id in fc.id]
if len(fc_ids) > 0:
    fc_id = fc_ids[0]
    print("Existing product found.\nDeleting", fc_id)
    dl.vectors.FeatureCollection(fc_id).delete()


# In[ ]:


deploy_nn_v1.main(args)


# # Download Data

# In[ ]:


# Alternatively, input a known product_id to download for an earlier model run: 
#product_id = 'earthrise:Bali_spectrogramV0.0.7_2020-06-01_2021-04-01'
#
#roi = 'bali'
#product_id = f'earthrise:{roi}_v{model_version}_{start_date}_{end_date}'
#patch_product_id = f'earthrise:{roi}_patch_v{patch_model_version}_{start_date}_{end_date}' 


# ### Download Patch Classifier Feature Collection

# In[ ]:


print("Downloading", patch_product_id)
fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if patch_product_id in fc.id][0]
fc = dl.vectors.FeatureCollection(fc_id)
region = gpd.read_file(roi_file)['geometry']
    
features = []
for elem in tqdm(fc.filter(region).features()):
    features.append(elem.geojson)
results = gpd.GeoDataFrame.from_features(features)
results.to_file(f"../data/model_outputs/patch_classifier/{patch_product_id.split(':')[-1]}.geojson", driver='GeoJSON')
print(len(features), 'features found')


# ### Download pixel classifier raster

# After the job is complete (only signaled by looking at the console), use this script to download the files. This process downloads each file individually because Descartes may throw a 502 error when trying to download too many tiles.

# In[ ]:


search = Image.search().filter(properties.product_id == product_id)
search.summary()


# In[ ]:


# Select one of these available bands
product = dl.catalog.Product.get(product_id)
for b in product.bands():
    print(b.id)


# In[ ]:


band = 'median'


# In[ ]:


basepath = os.path.join('../data/model_outputs/heatmaps', model_version, product_id.split(':')[-1] + f'mosaic-{band}')
print("Saving to", basepath)
if not os.path.exists(basepath):
    os.makedirs(basepath)


# In[ ]:


image_list = [image.id for image in search]
raster_client = dl.Raster()
for image in tqdm(image_list):
    try:
        raster_client.raster(inputs = image,
                             bands = [band],
                             save=True,
                             outfile_basename = os.path.join(basepath, image),
                             srs='WGS84')
    except dl.client.exceptions.BadRequestError as e:
        print(f'Warning: {repr(e)}\nContinuing...')
    except dl.client.exceptions.ServerError as e:
        print(f'Warning: {repr(e)}\nContinuing...')


# ## Combine tiles into single raster
# Not recommended. Only run these cells if necessary. Mosaicing dramatically increases file size

# In[ ]:


files_to_mosaic = []
for file in os.listdir(basepath):
    src = rs.open(os.path.join(basepath, file))
    files_to_mosaic.append(src)
mosaic, out_trans = merge(files_to_mosaic)


# In[ ]:


output_metadata = src.meta.copy()

output_metadata.update({"height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans
                 }
                )
output_metadata
with rs.open(basepath + '.tif', 'w', **output_metadata) as f:
    f.write(mosaic)


# In[ ]:


# Delete individual files and folder
shutil.rmtree(basepath)


# # Batched Run
# Deploy model on to run using a folder of boundary files

# In[ ]:


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

run_local = False # If False, the model prediction tasks are async queued and sent to DL for processing.


# In[ ]:



# ## Bulk Deploy

# In[ ]:


for roi in region_list:
    roi_file = os.path.join(boundary_folder, roi + '.geojson')
    product_id = f'earthrise:{roi}_v{model_version}_{start_date}_{end_date}'
    patch_product_id = f'earthrise:{roi}_patch_v{patch_model_version}_{start_date}_{end_date}'
    product_name = product_id.split(':')[-1]  # Arbitrary string - optionally set this to something more human readable.
    tilesize = 900

    args = [
        '--roi_file',
        roi_file,
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
        str(padding),
        '--tilesize',
        str((tilesize // patch_input_shape) * patch_input_shape - padding)
    ]
    if run_local:
        args.append('--run_local')

    # Because of the way DL uploads modules when queuing async tasks, we need to launch from the scripts/ folder
    get_ipython().run_line_magic('cd', '../scripts')
    get_ipython().run_line_magic('pwd', '')

    # Check if patch feature collection exists. If it does, delete the FC
    fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list() if patch_product_id in fc.id]
    if len(fc_ids) > 0:
        fc_id = fc_ids[0]
        print("Existing product found.\nDeleting", fc_id)
        dl.vectors.FeatureCollection(fc_id).delete()
    print("Deploying", roi)
    deploy_nn_v1.main(args)


# ## Bulk Download

# In[ ]:


# Patch classifier product download
for roi in region_list:
    roi_file = f'../data/boundaries/{roi}.geojson'
    patch_product_id = f'earthrise:{roi}_patch_v{patch_model_version}_{start_date}_{end_date}' 
    print("Downloading", patch_product_id)
    fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if patch_product_id in fc.id][0]
    fc = dl.vectors.FeatureCollection(fc_id)
    region = gpd.read_file(roi_file)['geometry']

    features = []
    for elem in tqdm(fc.filter(region).features()):
        features.append(elem.geojson)
    results = gpd.GeoDataFrame.from_features(features)
    results.to_file(f"../data/model_outputs/patch_classifier/{patch_product_id.split(':')[-1]}.geojson", driver='GeoJSON')
    print(len(features), 'features found')


# In[ ]:


# Pixel classifier product download
for roi in region_list:
    product_id = f'earthrise:{roi}_v{model_version}_{start_date}_{end_date}'
    patch_product_id = f'earthrise:{roi}_patch_v{patch_model_version}_{start_date}_{end_date}' 
    
    search = Image.search().filter(properties.product_id == product_id)
    print(search.summary())
    
    # Select one of these available bands
    product = dl.catalog.Product.get(product_id)
    for b in product.bands():
        print(b.id)
    band = 'median'
    
    basepath = os.path.join('../data/model_outputs/heatmaps', model_version, product_id.split(':')[-1] + f'mosaic-{band}')
    if not os.path.exists(basepath):
        os.makedirs(basepath)
        
    image_list = [image.id for image in search]
    raster_client = dl.Raster()
    for image in tqdm(image_list):
        try:
            raster_client.raster(inputs = image,
                                 bands = [band],
                                 save=True,
                                 outfile_basename = os.path.join(basepath, image),
                                 srs='WGS84')
        except dl.client.exceptions.BadRequestError as e:
            print(f'Warning: {repr(e)}\nContinuing...')
        except dl.client.exceptions.ServerError as e:
            print(f'Warning: {repr(e)}\nContinuing...')


# In[ ]:




