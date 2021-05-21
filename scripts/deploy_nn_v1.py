import argparse

from dateutil.relativedelta import relativedelta
import descarteslabs as dl
from tqdm import tqdm

import dl_utils

SYSTEM_PARAMS = {
    'image': ('us.gcr.io/dl-ci-cd/images/tasks/public/' +
              'py3.8:v2020.09.22-5-ga6b4e5fa'),
    'cpus': 1,
    'memory': '12Gi',
    'maximum_concurrency': 60,
    'retry_count': 0,
    'task_timeout': 20000,
    'include_modules': ['dl_utils']
}

def run_model(dlkey, product_id, model_name, start_date, end_date):
    import dl_utils
    runner = dl_utils.DescartesRun(product_id, model_name)
    runner(dlkey, start_date, end_date)

def main(*args):
    """Deploy a model on the Descartes Labs platform.

    Args:
        args:list: Can include any pair of form (flag, argument) passed to  
            the argument parser, e.g. ['--roi_file', '../data/bali.json'].
            Cannot be None if calling from an interpreter. Give [] instead. 
    """
    parser = argparse.ArgumentParser('Configure TPA detector deployment')
    parser.add_argument('--roi_file',
                        type=str,
                        help='GeoJSON file with ROI to deploy over',
                        default='../data/bali.json')
    parser.add_argument('--product_id',
                        type=str,
                        help='ID of catalog product (prefix with earthrise)',
                        default='earthrise:tpa_nn_toa')
    parser.add_argument('--product_name',
                        type=str,
                        help='Name of catalog product',
                        default='TPA NN TOA')
    parser.add_argument('--tilesize',
                        type=int,
                        help='Tilesize in pixels',
                        default=840)
    parser.add_argument('--pad',
                        type=int,
                        help='Padding in pixels',
                        default=16)
    parser.add_argument('--model_file',
                        type=str,
                        help='Model file for prediction',
                        default='../models/model_filtered_toa-12-09-2020.h5')
    parser.add_argument('--model_name',
                        type=str,
                        help='Model name in DLStorage',
                        default='model_filtered_toa-12-09-2020.h5')
    parser.add_argument('--mosaic_period',
                        type=int,
                        help='Months over which to mosaic image data',
                        default=1)
    parser.add_argument('--spectrogram_interval',
                        type=int,
                        help=('Spectrogram time interval, in mosaic periods'),
                        default=6)
""" To delete:
    parser.add_argument('--spectrogram_steps',
                        type=int,
                        help=('Number of time steps in the spectrogram:' +
                              'Use 1 for no temporal dimension.'),
                        default=2)
"""
    # Note on dates: A number of mosaics will be created within these bounds,
    # depending on the mosaic_period. Additional data to fill out a spectrogram
    # for each mosaic will be sought from earlier times, as dictated
    # by spectrogram_interval and the number of time steps in the spectrogram. 
    parser.add_argument('--start_date',
                        type=str,
                        help='Isoformat start date for predictions',
                        default='2020-09-01')
    parser.add_argument('--end_date',
                        type=str,
                        help='Isoformat end date for predictions',
                        default='2021-01-01')
    args = parser.parse_args(*args)

    tiles = dl_utils.get_tiles_from_roi(args.roi_file, args.tilesize, args.pad)
    
    mosaic_starts = dl_utils.get_starts(
        args.start_date, args.end_date, args.mosaic_period)
    band_names = mosaic_starts + ['median']

    # This first init handles product creation and model upload.
    dl_utils.DescartesRun(output_band_names=band_names, **vars(args))

    async_func = dl.Tasks().create_function(
        run_model, name=args.product_name, **SYSTEM_PARAMS)

    for dlkey in tqdm(tiles):
        task = async_func(dlkey, args.product_id, args.model_name,
                              args.start_date, args.end_date)
        print(task.result)
        print(task.log)
        
if __name__ == "__main__":
    main()
    

