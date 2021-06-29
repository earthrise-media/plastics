package encoding

import (
	"github.com/earthrise-media/plastics/api/model"
	"github.com/paulmach/orb/geojson"
)

func ContourFeatureCollection(contours []*model.Contour) (*geojson.FeatureCollection, error) {

	fc := geojson.NewFeatureCollection()

	for _, con := range contours {
		feat := geojson.Feature{
			ID:         con.Id,
			Type:       con.Geometry.GeoJSONType(),
			BBox:       nil,
			Geometry:   con.Geometry,
			Properties: nil,
		}
		fc.Append(&feat)
	}
	return fc, nil
}
