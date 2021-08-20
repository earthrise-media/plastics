import requests
import pickle
import itertools
import json
from geopy.geocoders import Nominatim
from shapely.geometry import mapping, shape

auth = ("admin", "plastics")
headers = {"Content-type": "application/json", "Accept": "application/json"}
API = "https://plastic-api-dorvc455lq-uc.a.run.app"
source_data = "../data/model_outputs/site_contours/indonesia_v0_contours_model_spectrogram_v0.0.11_2021-07-13_de-duped.geojson"
area_key = "area (km^2)"

skips = []
zero_sites = []


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


locator = Nominatim(user_agent="Earthrise GPW")

# print("Deleting current data")

requests.delete(f"{API}/sites", auth=auth)


def keyfunc(feature):
    if "name" not in feature["properties"]:
        return None
    return feature["properties"]["name"]


def idfunc(feature):
    print(feature["id"])
    return int(feature["id"])


def require_positive_area(feature):
    area_km = feature["properties"][area_key]
    return type(area_km) == float and area_km > 0


site_map = {}

index = json.load(open(source_data))

cache_map = pickle.load(open("./nominatim_cache", "rb"))

# set to infinity after dev. just here to make testing faster.
features = sorted(index["features"], key=keyfunc)
for name, contours in itertools.groupby(features, keyfunc):
    contours = list(filter(require_positive_area, list(contours)))
    if len(contours) == 0:
        continue
    centroid = mapping(shape(contours[0]["geometry"]).centroid)
    pair = tuple(reversed(centroid["coordinates"]))
    results = None
    if pair in cache_map:
        results = cache_map[pair]
    else:
        results = locator.reverse(pair)
        cache_map[pair] = results
    pickle.dump(cache_map, open("./nominatim_cache", "wb"))
    site_map[name] = {
        "centroid": {
            "type": "Feature",
            "properties": {
                # TODO: calculate area
                "name": name,
                "place_name": str(results.address),
                "area": str(contours[0]["properties"][area_key] * 1000000),
            },
            "geometry": centroid,
        },
        "contours": contours,
    }

count = 0
for name, record in site_map.items():
    print(f"{count} {name}")
    count = count + 1
    print(f"Creating site for {name}")
    resp = requests.post(
        f"{API}/sites?limit=1000",
        json=feature_collection([record["centroid"]]),
        headers=headers,
    ).raise_for_status()
    # TODO: this API should return the created ID, but it does not
    # https://github.com/earthrise-media/plastics/issues/31
    set_features = requests.get(f"{API}/sites?limit=1000").json()["features"]
    new_id = None
    for feature in set_features:
        if feature["properties"]["name"] == name:
            new_id = feature["id"]
            break
    contour_count = len(record["contours"])
    print(f"Creating {contour_count} contours for {name}, id {new_id}")
    resp = requests.post(
        f"{API}/sites/{new_id}/contours",
        json=feature_collection(record["contours"]),
        headers=headers,
    ).raise_for_status()

# print(f"In site map list: {len(site_map)}")
# print(f"Skipped names because they were not in positives: {skips}")
print(f"Removed names because they had no contours: {zero_sites}")
