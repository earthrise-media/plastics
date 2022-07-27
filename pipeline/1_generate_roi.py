import argparse
import json

from scripts import get_roi

def main(config_path):
    config = json.load(config_path)
    roi_name = config['roi']['name']
    region = get_roi.GetROI(roi_name, region=config['roi']['region'], pop_threshold=0.25, plot=True, save_polygons=True)
    region.run()
    config['roi']['region file'] = region.region_file_name
    config['roi']['dlkey file'] = region.dl_key_name
    config['roi']['populated dltiles file'] = region.populated_dltiles_name
    with open(config_path, 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    # parse default arguments
    parser = argparse.ArgumentParser(description='Select config file')
    parser.add_argument('--config_path', type=str, default='../pipeline/configs/config.json', help="Path to run's config file")
    args = parser.parse_args()

    main(
        config_path=args.config_path,
    )