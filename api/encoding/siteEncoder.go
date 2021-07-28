package encoding

import (
	"errors"
	"fmt"
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

func FeatureToSite(feature *geojson.Feature) (*model.Site, error){

	if feature.Geometry.GeoJSONType() != geojson.TypePoint {
		return nil, errors.New("sites must have point geometries")
	}

	site := model.Site{
		Location: feature.Point(),
		Properties: make(map[string]string,0),
	}
	if feature.ID != nil{
		site.Id = int64(feature.ID.(float64))
	}

	if len(feature.Properties) == 0 {
		site.Properties[model.SiteName] = ""
	} else {
		for k, _ := range feature.Properties{
			site.Properties[k] = fmt.Sprintf("%v",feature.Properties[k])
		}
	}
	return &site, nil
}

func FeatureCollectionToSites(fc *geojson.FeatureCollection) ([]*model.Site, error) {

	var sites []*model.Site

	for _, feat := range fc.Features {

		site, err := FeatureToSite(feat)
		if err != nil{
			return nil, err
		}
		sites = append(sites, site)
	}
	return sites, nil
}