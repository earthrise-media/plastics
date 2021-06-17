package encoding

import (
	"github.com/earthrise-media/plastics/api/database"
	"github.com/paulmach/orb/geojson"
)

func ToGeoJson(sites []*database.Site)(*geojson.FeatureCollection, error){

	fc := geojson.NewFeatureCollection()


	for _, site :=  range sites {

		feat := geojson.Feature{
			ID:         site.Id,
			Type:       "Point",
			Geometry:   site.Location,
			Properties: map[string]interface{}{
				"name":site.Name,
				"first_seen":site.FirstSeen,
				"last_seen":site.LastSeen,
				"id":site.Id,
			},
		}
		fc.Append(&feat)
	}

	return fc, nil

}
