import argparse
import json

from scripts import deploy_query_patch

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    roi_name = config['roi']['name']
    pixel_product_name = config['pixel']['product_name']
    pixel_version = config['pixel']['version']
    patch_product_name = config['patch']['product_name']
    patch_version = config['patch']['version']
    pred_threshold = config['candidate_detect']['pred_threshold']
    pred_threshold = config['candidate_detect']['pred_threshold']
    run_local = bool(config['run_local'])

    output_id = f"earthrise:intersect_{pixel_product_name.split('earthrise:')[-1]}-{patch_product_name.split('earthrise:')[-1]}_threshold_{pred_threshold}.geojson"
    config['intersect']['product_id'] = output_id

    with open(config_path, 'w') as f:
        json.dump(config, f)

    args = [
            '--roi_name', roi_name,
            '--pixel_product_name', pixel_product_name,
            '--pixel_version', pixel_version,
            '--patch_product_name', patch_product_name,
            '--patch_version', patch_version,
            '--pred_threshold', pred_threshold
        ]
    if run_local:
        args.append('--run_local')

    deploy_query_patch.main(args)

if __name__ == "__main__":
    # parse default arguments
    parser = argparse.ArgumentParser(description='Select config file')
    parser.add_argument('--config_path', type=str, default='../pipeline/configs/config.json', help="Path to run's config file")
    args = parser.parse_args()

    main(
        config_path=args.config_path,
    )