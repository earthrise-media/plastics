import argparse

from tqdm import tqdm
import descarteslabs as dl
from descarteslabs.catalog import Image, properties

from scripts import candidate_detect


DL_SYSTEM_PARAMS = {
    'image': ('us.gcr.io/dl-ci-cd/images/tasks/public/' +
              'py3.8:v2020.09.22-5-ga6b4e5fa'),
    'cpus': 1,
    'maximum_concurrency': 150,
    'memory': '2Gi',
    'retry_count': 4,
    'task_timeout': 20000,
    'include_modules': ['scripts.dl_utils', 'scripts.candidate_detect'],
    'requirements': ['scikit-image', 'tqdm', 'h3']
}

def run_candidates(image_name, **kwargs):
    import candidate_detect
    runner = candidate_detect.DescartesDetectRun(**kwargs)
    runner(image_name)

def main(*args):
    """Deploy a model on the Descartes Labs platform.

    Args:
        args:list: Can include any pair of form (flag, argument) passed to
            the argument parser, e.g. ['--roi_file', '../data/bali.json'].
            Cannot be None if calling from an interpreter. Give [] instead.
    """
    parser = argparse.ArgumentParser('Configure candidate detection')
    parser.add_argument('--product_name',
                        type=str)
    parser.add_argument('--pred_threshold',
                        type=float,
                        default=0.6)
    parser.add_argument('--min_sigma',
                        type=float,
                        default=3.5)
    parser.add_argument('--run_local',
                        action='store_true',
                        help='Run model locally rather than async on DL.')
    parser.add_argument('--band',
                        type=str,
                        default='median',
                        help='Band to analyze')
    args = parser.parse_args(*args)

    product_id = 'earthrise:' + args.product_name

    search = Image.search().filter(properties.product_id == product_id)
    image_list = [image.id for image in search]

    runner = candidate_detect.DescartesDetectRun(**vars(args))

    if args.run_local:
        for image_name in tqdm(image_list):
            runner(image_name)
    else:
        async_func = dl.Tasks().create_function(
            run_candidates, 
            name=args.product_name + f'_candidate_detect_thresh_{args.pred_threshold}_min_sigma_{args.min_sigma}_band-{args.band}', 
            **DL_SYSTEM_PARAMS
        )
        for image_name in tqdm(image_list):
            async_func(image_name, **vars(args))


if __name__ == "__main__":
    main()