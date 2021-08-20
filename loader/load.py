import requests
import pickle
import itertools
import json
from geopy.geocoders import Nominatim
from shapely.geometry import mapping, shape

# CONSTANTS
# ---------

AUTH = ("admin", "plastics")
HEADERS = {"Content-type": "application/json", "Accept": "application/json"}
API = "https://plastic-api-dorvc455lq-uc.a.run.app"
SOURCE_DATA = "../data/model_outputs/site_contours/indonesia_v0_contours_model_spectrogram_v0.0.11_2021-07-13_de-duped.geojson"
AREA_KEY = "area (km^2)"

skips = []
zero_sites = []

# CACHE MAP
# ---------
cache_map = pickle.load(open("./nominatim_cache", "rb"))


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
    pair = tuple(reversed(centroid["coordinates"]))
    if pair not in cache_map:
        cache_map[pair] = locator.reverse(pair)
        pickle.dump(cache_map, open("./nominatim_cache", "wb"))
    return cache_map[pair]


locator = Nominatim(user_agent="Earthrise GPW")


requests.delete(f"{API}/sites", auth=AUTH)

site_map = {}

index = json.load(open(SOURCE_DATA))


features = sorted(index["features"], key=keyfunc)
for name, contours in itertools.groupby(features, keyfunc):
    contours = list(filter(require_positive_area, list(contours)))
    if len(contours) == 0:
        continue
    first_contour = contours[0]
    centroid = mapping(shape(first_contour["geometry"]).centroid)
    geocoded = geocode_centroid(centroid)
    site_map[name] = {
        "centroid": {
            "type": "Feature",
            "properties": {
                "name": name,
                "place_name": str(geocoded.address),
                "area": str(first_contour["properties"][AREA_KEY] * 1000000),
            },
            "geometry": centroid,
        },
        "contours": contours,
    }

count = 0
for name, record in site_map.items():
    count = count + 1
    print(f"{count} / Creating site for {name}")
    resp = requests.post(
        f"{API}/sites?limit=1000",
        json=feature_collection([record["centroid"]]),
        headers=HEADERS,
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
    print(f"{count} / Creating {contour_count} contours for {name}")
    resp = requests.post(
        f"{API}/sites/{new_id}/contours",
        json=feature_collection(record["contours"]),
        headers=HEADERS,
    ).raise_for_status()

print(f"Removed names because they had no contours: {zero_sites}")
