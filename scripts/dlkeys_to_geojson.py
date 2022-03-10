"""
Write out footprints specified by DLKeys as GeoJSON
Useful for visualizing which tiles remain after population filtering
"""

from argparse import ArgumentParser
import descarteslabs as dl
import json
from shapely.geometry import mapping
from tqdm.contrib.concurrent import process_map


def read_dlkeys_file(dlkeys_file):
    with open(dlkeys_file, 'r') as f:
        dlkeys = [l.rstrip() for l in f]

    return dlkeys


def dlkey_to_geojson(dlkey):
    tile = dl.scenes.DLTile.from_key(dlkey)
    geom = mapping(tile.geometry)
    return geom


def build_feature(dlkey):
    geom = dlkey_to_geojson(dlkey)
    feature = {'type': 'Feature', 'geometry': geom, 'properties': {'dlkey': dlkey}}
    return feature


def features_to_geojson(features, geojson_file):
    fc = {'type': 'FeatureCollection', 'features': features}
    with open(geojson_file, 'w') as f:
        json.dump(fc, f)


def main(args):
    dlkeys = read_dlkeys_file(args.dlkeys_in)
    print(f"Found {len(dlkeys)} tiles in {args.dlkeys_in}")
    features = process_map(build_feature, dlkeys, chunksize=500)
    features_to_geojson(features, args.geojson_out)
    print(f"Wrote output file {args.geojson_out}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('dlkeys_in', type=str, help='Input dlkeys text file')
    parser.add_argument('geojson_out', type=str, help='Output GeoJSON file')
    
    args = parser.parse_args()

    main(args)
