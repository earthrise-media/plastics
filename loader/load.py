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
site_map = {}
index = json.load(open(SOURCE_DATA))

nominatim_calls = 0
cached_addresses = 0

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
        if item["shape"].distance(centroid_shape) < 0.0001:
            return item


def transfer_streetview(old, new):
    if "heading" in old:
        for key in ["heading", "pitch", "zoom", "lat", "lng"]:
            new[key] = old[key]
    return new


locator = Nominatim(user_agent="Earthrise GPW")

existing_sites = requests.get(f"{API}/sites?limit=1000").json()


existing_shapes = list(
    map(
        lambda feature: {"shape": shape(
            feature["geometry"]), "feature": feature},
        existing_sites["features"],
    )
)


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
        "old_id": old_site['feature']["id"] if old_site else None,
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

    old_id = record['old_id']

    print("POST (new feature)")
    new_feature = requests.post(
        f"{API}/sites",
        json=feature_collection([record["centroid"]]),
        headers=HEADERS,
    )
    new_id = new_feature.json()['features'][0]['id']
    print(f"POST (new contours for {new_id})")
    contour_count = len(record["contours"])
    print(f"{count} / Creating {contour_count} contours for {name}")
    resp = requests.post(
        f"{API}/sites/{new_id}/contours",
        json=feature_collection(record["contours"]),
        headers=HEADERS,
    ).raise_for_status()

    if old_id:  
        print(f"DELETE (old feature {old_id})")
        requests.delete(
            f"{API}/sites/{old_id}?site_id={old_id}",
            headers=HEADERS,
            auth=AUTH
        ).raise_for_status()


print(f"Removed names because they had no contours: {zero_sites}")
