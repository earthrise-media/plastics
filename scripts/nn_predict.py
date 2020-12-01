import argparse
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm.notebook import tqdm
from tensorflow import keras
from tensorflow.keras import layers

from get_s2_data_ee import get_history

# Sentinel 2 band descriptions
band_descriptions = {
    'B1': 'Aerosols, 442nm',
    'B2': 'Blue, 492nm',
    'B3': 'Green, 559nm',
    'B4': 'Red, 665nm',
    'B5': 'Red Edge 1, 704nm',
    'B6': 'Red Edge 2, 739nm',
    'B7': 'Red Edge 3, 779nm',
    'B8': 'NIR, 833nm',
    'B8A': 'Red Edge 4, 864nm',
    'B9': 'Water Vapor, 943nm',
    'B11': 'SWIR 1, 1610nm',
    'B12': 'SWIR 2, 2186nm'
}

band_wavelengths = [442, 492, 559, 665, 704, 739, 779, 833, 864, 943, 1610, 2186]


def get_pixel_vectors(data_source, month):
    pixel_vectors = []
    width, height = 0, 0
    for site in data_source[list(data_source.keys())[0]]:
        #for month in data_source.keys():
        if -999 not in data_source[month][site]['B2']:
            if len(np.shape(data_source[month][site]['B2'])) > 1:
                width, height = np.shape(data_source[month][site]['B2'])
                for i in range(width):
                    for j in range(height):
                        pixel_vector = []
                        for band in band_descriptions:
                            pixel_vector.append(data_source[month][site][band][i][j])
                        pixel_vectors.append(pixel_vector)
    return pixel_vectors, width, height

def make_predictions(model_path, data, site_name, threshold):
    test_image = data
    model = keras.models.load_model(model_path)

    rgb_stack = []
    preds_stack = []
    threshold_stack = []

    for month in tqdm(list(test_image.keys())):
        test_pixel_vectors, width, height = get_pixel_vectors(test_image, month)
        if width > 0:
            test_pixel_vectors = normalize(test_pixel_vectors)

            r = np.reshape(np.array(test_pixel_vectors)[:,3], (width, height))
            g = np.reshape(np.array(test_pixel_vectors)[:,2], (width, height))
            b = np.reshape(np.array(test_pixel_vectors)[:,1], (width, height))
            rgb = np.moveaxis(np.stack((r,g,b)), 0, -1)
            rgb_stack.append(rgb)

            preds = model.predict(np.expand_dims(test_pixel_vectors, axis=-1))
            preds_img = np.reshape(preds, (width, height, 2))[:,:,1]
            preds_stack.append(preds_img)

            thresh_img = preds_img > threshold
            threshold_stack.append(thresh_img)

    output_dir = '../notebooks/figures/neural_network'
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)


    rgb_median = np.median(rgb_stack, axis=0)
    preds_median = np.median(preds_stack, axis=0)
    threshold_median = np.median(threshold_stack, axis=0)

    plt.figure(dpi=150, facecolor=(1,1,1), figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(rgb_median / np.max(rgb_median))
    plt.title(f'{site_name} Median', size=8)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(preds_median, vmin=0, vmax=1, cmap='seismic')
    plt.title('Classification Median', size=8)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(threshold_median, vmin=0, vmax=1, cmap='gray')
    plt.title(f"Positive Pixels Median: Threshold {threshold}", size=8)
    plt.axis('off')

    title = f"{site_name} - Median Values - Neural Network Classification - Threshold {threshold}"
    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, title + '.png'), bbox_inches='tight')
    plt.show()


    fig, ax = plt.subplots(dpi=200, facecolor=(1,1,1), figsize=(4,4))
    ax.set_axis_off()
    clipped_img = np.moveaxis([channel * (preds_median > 0) for channel in np.moveaxis(rgb_median, -1, 0)], 0, -1)
    img = plt.imshow(clipped_img / (clipped_img.max()))
    ax.set_title('Threshold 0')
    plt.tight_layout()
    plt.show()

    def animate(i):
        i /= 100
        clipped_img = np.moveaxis([channel * (preds_median > i) for channel in np.moveaxis(rgb_median, -1, 0)], 0, -1)
        img.set_data(clipped_img / (clipped_img.max()))
        #img.set_data((preds_stack > i) * 1)
        ax.set_title(site_name + ' Threshold ' + str(i))
        return img,

    ani = animation.FuncAnimation(fig, animate, frames=100, interval=60, blit=True, repeat_delay=500)
    ani.save(os.path.join(output_dir, site_name + '_threshold_visualization' + '.mp4'))
    plt.close()

    return rgb_median, preds_median, threshold_median


def main():
    parser = argparse.ArgumentParser(description='Configure patch prediction')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of patch center')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of patch center')
    parser.add_argument('--width', type=float, required=False, default=0.002, help='Width of patch in degrees')
    parser.add_argument('--network', type=str, required=True, help='Path to neural network')
    args = parser.parse_args()

    lat = args.lat
    lon = args.lon
    width = args.width
    model_path = args.network

    name = f"Predictions {lat:.2f}, {lon:.2f}, {width} patch"

    patch_history = get_history(lon, lat, width, name)
    rgb_median, preds_median, threshold_median = make_predictions(model_path, patch_history, name, 0.95)

if __name__ == '__main__':
    main()
