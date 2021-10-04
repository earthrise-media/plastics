import requests
import pickle
import itertools
import json
import argparse, glob, csv
from geopy.geocoders import Nominatim
from requests.api import get
from shapely.geometry import mapping, shape
from shapely.geometry.base import CAP_STYLE, geom_from_wkt


# --------------
# CONSTANTS
# --------------

HEADERS = {"Content-type": "application/json", "Accept": "application/json"}
AREA_KEY = "area (km^2)"
locator = Nominatim(user_agent="Earthrise GPW")
cache_map = {}
nominatim_calls = 0 
cached_addresses = 0

# --------------
# HELPER METHODS
# --------------


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def keyfunc(feature):
    if "name" not in feature["properties"]:
        return None
    return feature["properties"]["name"]


def idfunc(feature):
    return int(feature["id"])


def require_positive_area(feature):
    area_km = feature["properties"][AREA_KEY]
    return type(area_km) == float and area_km > 0


# Use nominatim to get the address for a point,
# but check the cache first to see if it's already cached.
def geocode_centroid(centroid):
    global cached_addresses
    global nominatim_calls
    pair = tuple(reversed(centroid["coordinates"]))
    if pair not in cache_map:
        cache_map[pair] = locator.reverse(pair)
        nominatim_calls = nominatim_calls + 1
        pickle.dump(cache_map, open("./nominatim_cache", "wb"))
    else:
        cached_addresses = cached_addresses + 1
    return cache_map[pair]


def find_old_site(centroid_shape, existing_shapes):
    for item in existing_shapes:
        if item["shape"].distance(centroid_shape) < 0.0005:
            existing_shapes.remove(item)
            return item


def transfer_streetview(old, new):
    if "heading" in old:
        for key in ["heading", "pitch", "zoom", "lat", "lng"]:
            new[key] = old[key]
    return new


def load_site_file(index, AUTH, API):
    skips = []
    zero_sites = []
    site_map = {}
    nominatim_calls = 0
    cached_addresses = 0

    existing_sites = requests.get(f"{API}/sites?limit=1000").json()

    existing_shapes = list(
        map(
            lambda feature: {"shape": shape(
                feature["geometry"]), "feature": feature},
            existing_sites["features"],
        )
    )

    old_ids_to_remove = map(lambda site: site['id'], existing_sites)

    # requests.delete(f"{API}/sites", auth=AUTH)

    for name, contours in itertools.groupby(
        sorted(index["features"], key=keyfunc), keyfunc
    ):
        contours = list(filter(require_positive_area, list(contours)))
        if len(contours) == 0:
            continue
        first_contour = contours[0]
        centroid_shape = shape(first_contour["geometry"]).centroid
        centroid_geojson = mapping(centroid_shape)
        geocoded = geocode_centroid(centroid_geojson)
        old_site = find_old_site(centroid_shape, existing_shapes)
        properties = {
            "name": name,
            "place_name": str(geocoded.address),
            "area": str(first_contour["properties"][AREA_KEY] * 1000000),
        }
        if old_site:
            properties = transfer_streetview(
            old_site,
            properties,
            )
        site_map[name] = {
            "old_id": old_site["feature"]["id"] if old_site else None,
            "centroid": {
                "type": "Feature",
                "properties": properties,
                "geometry": centroid_geojson,
            },
            "contours": contours,
        }

    print(
        f"Transformed sites. Addresses: cached={cached_addresses} called={nominatim_calls}"
    )    

    count = 0
    for name, record in site_map.items():
        count = count + 1

        old_id = record["old_id"]

        print("POST (new feature)")
        new_feature = requests.post(
            f"{API}/sites",
            json=feature_collection([record["centroid"]]),
            headers=HEADERS,
        )
        new_id = new_feature.json()["features"][0]["id"]
        print(f"POST (new contours for {new_id})")
        contour_count = len(record["contours"])
        print(f"{count} / Creating {contour_count} contours for {name}")
        resp = requests.post(
            f"{API}/sites/{new_id}/contours",
            json=feature_collection(record["contours"]),
            headers=HEADERS,
        ).raise_for_status()

    print(f"Removed names because they had no contours: {zero_sites}")

def add_metadata(csv_file, API):
    found = 0
    total = 0
    # find site based on csv 
    bboxes = open("bbox.csv","w+")
    bbox_writer =csv.writer(bboxes)
    with open(csv_file) as csvfile:
        site_reader = csv.DictReader(csvfile)  
        ignored_columns = ('', 'lat', 'lon', 'geometry')
        for site in site_reader:            
            # skip header
            if site_reader.line_num == 0:
                 continue
            total += 1
            site_point = geom_from_wkt(site["geometry"])
            buff = site_point.buffer(0.01,cap_style=CAP_STYLE.square,resolution=1) # this gets almost all of them, ~ 1km            
            bbox = buff.bounds
            coords =  str(bbox[0])+","+str(bbox[1])+","+str(bbox[2])+","+str(bbox[3])  # prob do something more pythonic 

            url = API+"/sites?bbox="+coords
            resp = requests.get(url).json()

            if len(resp["features"]) == 0:               
                continue
            
            found += 1
            feature = resp["features"][0] 
            
            for key in list(site.keys()):
                if key in ignored_columns:
                    continue                
                feature['properties'][key] = site[key]
            # update feature
            requests.put(f"{API}/sites/{feature['id']}",
                json=feature,
                headers=HEADERS).raise_for_status()
        
        print(f'found and updated {found} matching sites for {total} metadata rows')
     


if __name__ == "__main__":
    
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Site Data Loader')
    parser.add_argument("--user",required=False,nargs=1,default=["admin"], type=str,help="Username for transactional API operations", dest="user")
    parser.add_argument("--pw",required=False,nargs=1,default=["plastics"], type=str,help="Password for transactional API operations", dest="password")
    parser.add_argument("--api",required=False,nargs=1,default=["https://api.dev.plastic.watch.earthrise.media"], type=str,help="API Endpoint", dest="api")
    parser.add_argument("--dir",required=True,nargs=1, type=str, help="Directory to find geojson files", dest="dir")

    args = vars(parser.parse_args())

    auth = (args["user"][0], args["password"][0])   
    api = args["api"][0]
    
    # set up cache
    try:
        cache_map = pickle.load(open("./nominatim_cache", "rb"))
    except:
        cache_map = {}

    print(f"DELETE (old features)")
    requests.delete(
        f"{api}/sites", headers=HEADERS, auth=auth
        ).raise_for_status()

    # get all geojson files in provided directory
    print(auth)
    files = glob.glob(args["dir"][0]+"/*.geojson")
    print("Found "+str(len(files))+" geojson files to import")
    for file in files:                    
        index = json.load(open(file))
        load_site_file(index, AUTH=auth, API=api)
    add_metadata("../data/site_metadata/SE_ASIA_METADATA.csv",api)
