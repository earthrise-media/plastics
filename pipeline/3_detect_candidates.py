import argparse
import json

from scripts import deploy_candidate_detect

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    product_name = config['pixel']['product_name']
    pred_threshold = config['candidate_detect']['pred_threshold']
    min_sigma = config['candidate_detect']['min_sigma']
    band = 'median'
    product_id = f"{product_name}_blobs_thresh_{pred_threshold}_min-sigma_{min_sigma}_area-thresh_0.0025_band-{band}"
    config['candidate_detect']['product_id'] = product_id

    run_local = bool(config['run_local'])

    with open(config_path, 'w') as f:
        json.dump(config, f)

    args = [
        '--product_name',
        product_name,
        '--pred_threshold',
        str(pred_threshold),
        '--min_sigma',
        str(min_sigma),
        '--band',
        band,
    ]
    if run_local:
        args.append('--run_local')

    deploy_candidate_detect.main(args)

if __name__ == "__main__":
    # parse default arguments
    parser = argparse.ArgumentParser(description='Select config file')
    parser.add_argument('--config_path', type=str, default='../pipeline/configs/config.json', help="Path to run's config file")
    args = parser.parse_args()

    main(
        config_path=args.config_path,
    )