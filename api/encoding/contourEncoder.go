package encoding

import (
	"fmt"
	"github.com/earthrise-media/plastics/api/model"
	"github.com/paulmach/orb"
	"github.com/paulmach/orb/geojson"
	"go.uber.org/zap"
)

func ContourFeatureCollection(contours []*model.Contour) (*geojson.FeatureCollection, error) {

	fc := geojson.NewFeatureCollection()

	for _, con := range contours {
		feat := geojson.Feature{
			ID:         con.Id,
			Type:       con.Geometry.GeoJSONType(),
			BBox:       nil,
			Geometry:   con.Geometry,
			Properties: make(map[string]interface{},len(con.Properties)),
		}
		for k,v := range con.Properties {
			feat.Properties[k] = v
		}

		fc.Append(&feat)
	}
	return fc, nil
}
func FeatureCollectionToContours(collection *geojson.FeatureCollection)([]*model.Contour, error){

	 	contours := make([]*model.Contour,0)
		for _,feat := range collection.Features{

			if feat.Geometry == nil || feat.Geometry.GeoJSONType() != geojson.TypeMultiPolygon {
				zap.L().Error("ignoring contour with wrong geometry type")
				continue
			}

			c := model.Contour{
				Id:         int64(feat.Properties.MustInt(model.Id,0)),
				SiteId:     int64(feat.Properties.MustInt(model.SiteId, 0)),
				Geometry:   feat.Geometry.(orb.MultiPolygon),
			}
			if len(feat.Properties) == 0 {
				c.Properties = make(map[string]string,0)
			} else {
				c.Properties = make(map[string]string,len(feat.Properties))

				for k, _ := range feat.Properties{
					c.Properties[k] = fmt.Sprintf("%v",feat.Properties[k])
				}
			}
			contours = append(contours, &c)
		}
	
		return contours, nil
}