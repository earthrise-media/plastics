import requests
import itertools
import json
import glob

auth = ("admin", "plastics")
API = "https://plastic-api-dorvc455lq-uc.a.run.app"

print("Deleting current data")
requests.delete(f"{API}/sites", auth=auth)


def keyfunc(feature):
    if not "name" in feature["properties"]:
        print(feature)
    return feature["properties"]["name"]


def require_positive_area(feature):
    return feature["properties"]["area (km^2)"] > 0


site_map = {}

index = json.load(open("./v12_java_bali_validated_positives.geojson"))

# Some sites are anonymous, like this. They are not usable
# because there needs to be something to link a site to contours.
# Ignore them.
# {
#   "type": "Feature",
#   "geometry": {
#      "type": "Point",
#      "coordinates":  [ 107.03102180094808,-6.161674096496641 ]
#   },
#   "properties": {}
# },
for feature in index["features"]:
    if "name" in feature["properties"]:
        site_map[feature["properties"]["name"]] = {"point": feature}

skips = []

headers = {"Content-type": "application/json", "Accept": "application/json"}

for file in [
    "./v12_bali_validated_positives_contours_model_spectrogram_v0.0.8_2021-06-03.geojson",
    "./v12_java_validated_positives_contours_model_spectrogram_v0.0.8_2021-06-03.geojson",
]:
    gj = json.load(open(file))
    features = sorted(gj["features"], key=keyfunc)
    for name, contours in itertools.groupby(features, keyfunc):
        if name in site_map:
            site_map[name] = {
                "point": site_map[name]["point"],
                "contours": list(filter(require_positive_area, list(contours))),
            }
        else:
            skips.append(name)


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


zero_sites = []
for name, record in site_map.items():
    contour_count = len(record["contours"])
    if contour_count == 0:
        zero_sites.append(name)

for name in zero_sites:
    site_map.pop(name)


for name, record in site_map.items():
    print(f"Creating site for {name}")
    resp = requests.post(
        f"{API}/sites", json=feature_collection([record["point"]]), headers=headers
    ).raise_for_status()
    # TODO: this API should return the created ID, but it does not
    # https://github.com/earthrise-media/plastics/issues/31
    new_id = requests.get(f"{API}/sites").json()["features"][-1]["id"]
    resp = requests.post(
        f"{API}/sites/{new_id}/contours",
        json=feature_collection(record["contours"]),
        headers=headers,
    ).raise_for_status()


print(f"Skipped names because they were not in positives: {skips}")
print(f"Removed names because they had no contours: {zero_sites}")
