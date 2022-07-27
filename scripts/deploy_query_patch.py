import argparse
import descarteslabs as dl
import geopandas as gpd
from scripts import query_patch


DL_SYSTEM_PARAMS = {
    'image': ('us.gcr.io/dl-ci-cd/images/tasks/public/' +
              'py3.8:v2020.09.22-5-ga6b4e5fa'),
    'cpus': 1,
    'maximum_concurrency': 1,
    'memory': '8Gi',
    'retry_count': 4,
    'task_timeout': 20000,
    'include_modules': ['scripts.dl_utils', 'scripts.query_patch'],
    'requirements': ['tqdm', 'h3', 'geopandas==0.10.2', 'pygeos']
}

def run_query(roi, **kwargs):
    import query_patch
    runner = query_patch.DescartesQueryRun(roi, **kwargs)
    runner()

def main(*args):
    """
    Find pixel candidate points that overlap with patch classifier predictions 
    greater than a threshold.

    Args:
        args:list: Can include any pair of form (flag, argument) passed to
            the argument parser, e.g. ['--roi_file', '../data/bali.json'].
            Cannot be None if calling from an interpreter. Give [] instead.
    """
    parser = argparse.ArgumentParser('Configure patch classifier check')
    parser.add_argument('--roi_name', type=str)
    parser.add_argument('--pixel_product_name',
                        help='Name of the pixel candidates product',
                        type=str)
    parser.add_argument('--pixel_version', type=str)
    parser.add_argument('--patch_product_name', 
                        help='Name of the patch product',
                        type=str)
    parser.add_argument('--patch_version', type=str)
    parser.add_argument('--pred_threshold',
                        type=float,
                        default=0.3)
    parser.add_argument('--run_local',
                        action='store_true',
                        help='Run model locally rather than async on DL.')

    args = parser.parse_args(*args)
    roi = gpd.read_file(f'../data/boundaries/{args.roi_name}.geojson')['geometry']
    roi = roi.simplify(0.05).buffer(1).to_json()
    runner = query_patch.DescartesQueryRun(roi, **vars(args))
    if args.run_local:
        runner()
    else:
        async_func = dl.Tasks().create_function(
            run_query, 
            name=f"query_patch_{args.pixel_product_name.split('earthrise:')[-1]}_thresh_{args.pred_threshold}", 
            **DL_SYSTEM_PARAMS
        )
        async_func(roi, **vars(args))
            


if __name__ == "__main__":
    main()