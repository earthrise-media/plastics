import argparse
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

from scripts import dl_utils
from scripts.viz_tools import normalize

def make_predictions(patches, model):
    pred_stack = []
    for patch in patches:
        h, w, c = patch.shape
        preds = model.predict(normalize(patch.reshape(h * w, c, 1)))[:,1]
        preds = preds.reshape(h,w)
        preds = np.ma.masked_where(patch.mask[:,:,0], preds)
        pred_stack.append(preds)
    return pred_stack

def visualize_predictions(patches, pred_stack, threshold=0.8, name=None, save=False):

    rgb = normalize(np.ma.mean(patches, axis=0))[:,:,3:0:-1]
    pred = np.ma.mean(pred_stack, axis=0)
    overlay = np.copy(rgb)
    overlay[pred > threshold, 0] = .9
    overlay[pred > threshold, 1] = 0
    overlay[pred > threshold, 2] = .1

    plt.figure(figsize=(15, 5), dpi=150, facecolor=(1,1,1))
    if name:
        title = f"{name} - Mean Values - Threshold {threshold}"
        plt.suptitle(title, y=1.01)
    plt.subplot(1,3,1)
    plt.title('RGB Mean')
    plt.imshow(np.clip(rgb, 0, 1))
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title('Predictions Mean')
    plt.imshow(pred, vmin=0, vmax=1, cmap='RdBu_r')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title(f"Pred Overlay, Threshold {threshold}")
    plt.imshow(np.clip(overlay, 0, 1))
    plt.axis('off')

    plt.tight_layout()
    if save:
        output_dir = '../figures/neural_network'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.savefig(os.path.join(output_dir, title + '.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Configure patch prediction')
    parser.add_argument('--coords', nargs='+', required=True, type=float, help='Lat Lon of patch center')
    parser.add_argument('--width', type=float, required=False, default=0.02, help='Width of patch in degrees')
    parser.add_argument('--network', type=str, required=True, help='Path to neural network')
    parser.add_argument('--threshold', type=float, required=False, default=0.95, help='Classifier masking threshold')
    parser.add_argument('--start_date', type=str, required=False, default='2019-03-01', help='Start date for predictions')
    parser.add_argument('--end_date', type=str, required=False, default='2019-06-01', help='End date for predictions')
    args = parser.parse_args()

    coords = args.coords
    lat = coords[0]
    lon = coords[1]
    width = args.width
    model_path = args.network
    threshold = args.threshold
    start_date = args.start_date
    end_date = args.end_date

    name = f"{lat:.2f}, {lon:.2f}, {width} patch, {start_date}:{end_date}"
    model = keras.models.load_model(model_path)
    print("Downloading data")
    patches = dl_utils.download_patch(
        dl_utils.rect_from_point([lon, lat], width), start_date, end_date)
    print("Making Predictions")
    pred_stack = make_predictions(patches, model)
    visualize_predictions(patches, pred_stack, threshold=threshold, name=name, save=True)

if __name__ == '__main__':
    main()
