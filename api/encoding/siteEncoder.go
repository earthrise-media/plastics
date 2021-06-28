package encoding

import (
	"errors"
	"github.com/earthrise-media/plastics/api/database"
	"github.com/paulmach/orb/geojson"
)

func SiteToGeoJsonFeature(site *database.Site) (*geojson.Feature, error) {
	feat := geojson.Feature{
		ID:       site.Id,
		Type:     "Point",
		Geometry: site.Location,
		Properties: map[string]interface{}{
			"name":       site.Name,
			"first_seen": site.FirstSeen,
			"last_seen":  site.LastSeen,
			"id":         site.Id,
		},
	}
	return &feat, nil

}

func SitesToFeatureCollection(sites []*database.Site) (*geojson.FeatureCollection, error) {

	fc := geojson.NewFeatureCollection()

	for _, site := range sites {

		feat := geojson.Feature{
			ID:       site.Id,
			Type:     "Point",
			Geometry: site.Location,
			Properties: map[string]interface{}{
				"name":       site.Name,
				"first_seen": site.FirstSeen,
				"last_seen":  site.LastSeen,
				"id":         site.Id,
			},
		}
		fc.Append(&feat)
	}

	return fc, nil

}

func FeatureCollectionToSites(fc *geojson.FeatureCollection) ([]*database.Site, error) {

	var sites []*database.Site

	for _, feat := range fc.Features {

		if feat.Geometry.GeoJSONType() != geojson.TypePoint {
			return nil, errors.New("Sites must have Point geometries")
		}

		site := database.Site{
			Location: feat.Point(),
		}
		sites = append(sites, &site)
	}
	return sites, nil

}
