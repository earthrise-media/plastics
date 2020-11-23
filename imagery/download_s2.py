'''
download_s2.py

Download Sentinel-2 imagery over dump sites
'''


from argparse import ArgumentParser
import geemap
import geopandas as gpd
import ee
import os
from shapely.geometry import mapping


# initialize GEE
ee.Initialize()


# set some defaults
S2_ID = 'COPERNICUS/S2'
S2CLOUDLESS_ID = 'COPERNICUS/S2_CLOUD_PROBABILITY'
S2_SCALE = 10

# red, green, blue, nir, swir1, swir2, cloud probability
S2_BANDS = ['B4', 'B3', 'B2', 'B8', 'B11', 'B12', 'probability']


def build_ic():
    '''
    build a base Sentinel-2 image collection
    add s2cloudless band as cloud_mask
    '''

    s2 = ee.ImageCollection(S2_ID)
    s2cloud = ee.ImageCollection(S2CLOUDLESS_ID)

    s2joined = ee.Join.saveFirst('cloud_mask').apply(
        primary=s2,
        secondary=s2cloud,
        condition=ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )
    )

    s2combined = ee.ImageCollection(s2joined).map(
        lambda img: img.addBands(img.get('cloud_mask'))
    )

    return s2combined


def download_feature(feature, output_dir):
    '''
    download S2 image collection over a feature
    '''

    # grab base image collection
    ic = build_ic()

    # build ROI geometry in WGS
    geometry = mapping(feature['bbox'])['coordinates']
    roi = ee.Geometry.Polygon(geometry)

    ic = ic.filterBounds(roi).select(S2_BANDS)

    # download image collection
    feature_name = feature['Name'].replace(' ', '_')
    outdir = os.path.join(output_dir, feature_name)
    geemap.ee_export_image_collection(ic, outdir, region=roi, scale=S2_SCALE)

    print('Downloaded feature {}'.format(feature_name))
    return True


def read_input_file(input_file, tile_width):
    '''
    read in an input file 
    output a geodataframe with bounding boxes at a specified width (deg)
    '''

    df = gpd.read_file(args.input_file)
    print('Read in {} locations'.format(len(df)))

    # add a new field which contains bounding boxes corresponding to each point
    df['bbox'] = df.set_crs(epsg=4326).buffer(tile_width).envelope

    return df


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('Created directory {}'.format(args.output_dir))

    df = read_input_file(args.input_file, args.tile_width)

    for ridx, row in df.iterrows():
        download_feature(row, args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, help='File with TPA locations',
                        default='../data/tpa_points.json')
    parser.add_argument('--output_dir', type=str, help='Output directory to write images',
                        default='../data/s2')
    parser.add_argument('--tile_width', type=float, help='Tile width in WGS degrees',
                        default=0.01) # about 225 pixels wide, slightly varies

    args = parser.parse_args()

    main(args)
