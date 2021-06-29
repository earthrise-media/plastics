package encoding

import (
	"errors"
	"github.com/earthrise-media/plastics/api/model"
	"github.com/paulmach/orb/geojson"
)

func SiteToGeoJsonFeature(site *model.Site) (*geojson.Feature, error) {
	feat := geojson.Feature{
		ID:       site.Id,
		Type:     "Point",
		Geometry: site.Location,
		Properties: map[string]interface{}{
			"id":         site.Id,
		},
	}
	for k,v := range site.Properties{
		feat.Properties[k] = v
	}
	return &feat, nil

}

func SitesToFeatureCollection(sites []*model.Site) (*geojson.FeatureCollection, error) {

	fc := geojson.NewFeatureCollection()

	for _, site := range sites {

		feat := geojson.Feature{
			ID:       site.Id,
			Type:     "Point",
			Geometry: site.Location,
			Properties: map[string]interface{}{
				"id":         site.Id,
			},
		}
		for k,v := range site.Properties{
			feat.Properties[k] = v
		}
		fc.Append(&feat)
	}

	return fc, nil

}

func FeatureCollectionToSites(fc *geojson.FeatureCollection) ([]*model.Site, error) {

	var sites []*model.Site

	for _, feat := range fc.Features {

		if feat.Geometry.GeoJSONType() != geojson.TypePoint {
			return nil, errors.New("Sites must have Point geometries")
		}

		site := model.Site{
			Location: feat.Point(),
			Properties: make(map[string]string),
		}
		if len(feat.Properties) == 0 {
			site.Properties[model.SiteName] = ""
		} else {
			for k, _ := range feat.Properties{
				site.Properties[k] = feat.Properties.MustString(k,"")
			}

		}

		sites = append(sites, &site)
	}
	return sites, nil

}