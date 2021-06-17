package database

import (
	"context"
	"github.com/jackc/pgx/v4"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/paulmach/orb"
	"github.com/paulmach/orb/encoding/wkb"
	"github.com/paulmach/orb/encoding/wkt"
	"go.uber.org/zap"
	"time"
)

type SiteController struct {
	db *pgxpool.Pool
}

func NewSiteController(db *pgxpool.Pool) *SiteController {

	return &SiteController{db: db}
}


func (sc *SiteController) FindSites(start int, limit int, bound orb.Bound) ([]*Site, error) {

	sql := "SELECT id, name, first_seen, last_seen, st_asbinary(geom) FROM sites WHERE ST_WITHIN(geom,ST_GeometryFromText($1,4326)) LIMIT $2 OFFSET $3"
	wkt := wkt.MarshalString(bound)
	rows, err := sc.db.Query(context.Background(), sql, wkt, limit, start)
	defer rows.Close()

	if err != nil {
		zap.L().Error(err.Error())
		return nil, err
	}
	return scanToSite(rows)


}

//scanToSite does all the nasty geometry stuff
func scanToSite(rows pgx.Rows)([]*Site, error){

	var sites []*Site
	var p orb.Point
	scanner := wkb.Scanner(&p)
	for rows.Next() {
		var id int64
		var name string
		var first time.Time
		var last time.Time
		var geom []byte
		err := rows.Scan(&id, &name, &first, &last, &geom)
		err = scanner.Scan(geom)
		if err != nil {
			zap.L().Warn("error scanning row:" + err.Error())
		}
		s := Site{
			Id:        id,
			Name:      name,
			FirstSeen: first,
			LastSeen:  last,
			Location:  scanner.Geometry.(orb.Point),
		}
		sites = append(sites, &s)

	}
	zap.L().Info("returned ", zap.Int("sites", len(sites)))
	return sites,nil
}
