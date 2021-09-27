package database

import (
	"context"
	"github.com/earthrise-media/plastics/api/model"
	"github.com/jackc/pgtype"
	"github.com/jackc/pgx/v4"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/paulmach/orb"
	"github.com/paulmach/orb/encoding/wkb"
	"github.com/paulmach/orb/encoding/wkt"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"strconv"
)

type SiteController struct {
	db *pgxpool.Pool
}

type ContourController struct {
	db *pgxpool.Pool
}

func NewContourController(db *pgxpool.Pool) *ContourController {
	return &ContourController{db: db}
}

func NewSiteController(db *pgxpool.Pool) *SiteController {

	return &SiteController{db: db}
}

//FindSiteById returns a single site based on the site id
func (sc *SiteController) FindSiteById(id string) (*model.Site, error) {
	site_id, err := strconv.Atoi(id)
	if err != nil {
		zap.L().Error("badly formatted site id")
		return nil,err
	}
	sql := "SELECT id, props, st_asbinary(geom) from sites WHERE id = $1"
	rows, err:= sc.db.Query(context.Background(), sql, site_id)
	defer rows.Close()

	if err != nil {
		return nil, err
	}
	sites, err := scanToSites(rows)
	if sites != nil {
		return sites[0], nil
	}
	return nil, err

}

//FindSites returns a site slice based on provided parameters
func (sc *SiteController) FindSites(start int, limit int, bound *orb.Bound) ([]*model.Site, error) {

	sql := "SELECT id, props, st_asbinary(geom) FROM sites WHERE ST_WITHIN(geom,ST_GeometryFromText($1,4326)) LIMIT $2 OFFSET $3"
	wkt := wkt.MarshalString(bound.Bound())
	rows, err := sc.db.Query(context.Background(), sql, wkt, limit, start)

	defer rows.Close()

	if err != nil {
		zap.L().Error(err.Error())
		return nil, err
	}

	return scanToSites(rows)

}

//AddSite creates a site
func (sc *SiteController) AddSites(sites []*model.Site) error {

	sql := "INSERT INTO sites(props, geom) values($1, ST_GeometryFromText($2,4326)) RETURNING id"

	tx, err := sc.db.Begin(context.Background())
	if err != nil {
		zap.L().Error("error starting transaction")
		return err
	}
	for _, site := range sites{
		wkt := wkt.MarshalString(site.Location)
		store :=  pgtype.Hstore{}
		store.Set(site.Properties)

		row := sc.db.QueryRow(context.Background(), sql, store, wkt)
		var id int64
		err := row.Scan(&id)
		if err != nil {
			zap.S().Errorf("error adding site: %s", err.Error())
			tx.Rollback(context.Background())
			return err
		}
		site.Id = id
	}

	err = tx.Commit(context.Background())
	if err != nil {
		zap.S().Error("error commiting: %s", err.Error())
		return err
	}
	return nil
}


//DeleteAllSites removes all sites from the database -- also removes all contours
func (sc *SiteController) DeleteAllSites() error {

	sql := "TRUNCATE TABLE sites CASCADE"
	tag, err := sc.db.Exec(context.Background(), sql)
	if err != nil {
		return err
	}
	zap.S().Info("truncated table sites - deleted %n rows", tag.RowsAffected())
	return nil

}

//DeleteSites deletes the provided sites from the database as well as their contours
func (sc *SiteController) DeleteSites(sites []*model.Site) error {

	idmap := make(map[int64]bool)
	for _, site := range sites {
		idmap[site.Id] = true
	}
	ids := make([]int64, 0, len(idmap))
	for k := range idmap {
		ids = append(ids, k)
	}
	sql := "DELETE FROM sites WHERE Id in $1"
	tag, err := sc.db.Exec(context.Background(), sql, ids)
	if err != nil {
		return err
	}
	zap.S().Infof("deleted %n sites", tag.RowsAffected())
	return nil

}

//DeleteSite deletes a single site based on id
func (sc *SiteController) DeleteSiteById(site *model.Site) error {

	//TODO implement this (if needed)
	return nil
}

//FindSiteByRadius finds the nearest site to a point limited to within a given radius
func (sc *SiteController) FindSiteByRadius(point *orb.Point) (*model.Site, error) {

	// long, lat , distance
	sql := "SELECT id,st_asbinary(geom) FROM sites WHERE ST_DWithin(geom, ST_MakePoint($1,$2)::geography, $3) ORDER BY geom <-> ST_MakePoint($4,$5)::geographyLIMIT 1"
	row := sc.db.QueryRow(context.Background(), sql, point.X(), point.Y(), viper.GetInt("SITE_MATCH_DISTANCE_METERS"), point.X(), point.Y())
	return scanToSite(row)

}

func (sc *SiteController) UpdateSite(site *model.Site) error {

	sql := "UPDATE sites SET geom = ST_GeometryFromText($1,4326), props = $2 WHERE sites.id = $3"
	wkt := wkt.MarshalString(site.Location)
	store :=  pgtype.Hstore{}
	store.Set(site.Properties)
	_, err := sc.db.Exec(context.Background(), sql, wkt,store, site.Id)
	return err
}

func (sc *SiteController) AddContoursToSite(site *model.Site, contours []*model.Contour) error {

	sql := "INSERT INTO contours(site_id, props, geom) VALUES($1, $2, ST_GeometryFromText($3,4326)) RETURNING id"

	tx, err := sc.db.Begin(context.Background())
	if err != nil {
		zap.L().Error("error starting transaction")
		return err
	}
	for _, contour := range contours{

		wkt := wkt.MarshalString(contour.Geometry)
		store :=  pgtype.Hstore{}
		store.Set(contour.Properties)
		row := tx.QueryRow(context.Background(),sql,site.Id,store, wkt)
		var contourId int64
		err = row.Scan(&contourId)
		if err != nil {
			zap.S().Error("error adding contour: %s", err.Error())
			tx.Rollback(context.Background())
			return err
		}
		contour.Id = contourId
		contour.SiteId = site.Id
	}
	err = tx.Commit(context.Background())
	if err != nil {
		zap.S().Error("error commiting: %s", err.Error())
		return err
	}
	return nil



}

func (sc *SiteController) GetContoursBySite(site *model.Site) ([]*model.Contour, error) {

	sql := "SELECT id, site_id, props, st_asbinary(geom) FROM contours WHERE site_id = $1"
	rows, err := sc.db.Query(context.Background(),sql, site.Id)
	defer rows.Close()
	if err != nil {
		zap.S().Errorf("error querying contours for site %v: %s", err.Error())
		return nil, err
	}
	return scanToContours(rows)

}

//func scanToContour(row pgx.Row)(*model.Contour, error){}

func scanToContours(rows pgx.Rows)([]*model.Contour, error){

	var contours []*model.Contour
	var mp orb.MultiPolygon
	scanner := wkb.Scanner(&mp)

	for rows.Next(){

		var id int64
		var site_id int64
		var geom []byte
		var props pgtype.Hstore
		err := rows.Scan(&id,&site_id, &props, &geom)
		if err != nil {
			zap.S().Warnf("error scanning row: %s", err.Error())
			continue
		}
		err = scanner.Scan(geom)
		if err != nil {
			zap.S().Warnf("error scanning geometry from row: %s", err.Error() )
			continue
		}
		c := model.Contour{
			Id:       id,
			SiteId: site_id,
			Geometry: scanner.Geometry.(orb.MultiPolygon),
			Properties: make(map[string]string,0),
		}
		for k, v := range props.Map {
			c.Properties[k] = v.String
		}
		contours = append(contours, &c)
	}
	return contours, nil
}

//scanToSite scans a single row into a Site object
func scanToSite(row pgx.Row) (*model.Site, error) {

	var id int64
	var props pgtype.Hstore
	var geom []byte
	var p orb.Point
	scanner := wkb.Scanner(&p)
	err := row.Scan(&id, &props, &geom)

	if err != nil {
		zap.S().Warnf("error scanning row: %s",  err.Error())
		return nil, err
	}
	err = scanner.Scan(geom)
	if err != nil {
		zap.S().Warnf("error scanning geometry: %s",  err.Error())
		return nil, err
	}
	s := model.Site{
		Id:       id,
		Location: scanner.Geometry.(orb.Point),
	}
	for k, v := range props.Map {
		s.Properties[k] = v.String
	}
	return &s, nil
}

//scanToSites does all the nasty geometry stuff
func scanToSites(rows pgx.Rows) ([]*model.Site, error) {

	var sites []*model.Site
	var p orb.Point
	scanner := wkb.Scanner(&p)
	for rows.Next() {

		var id int64
		var geom []byte
		var props pgtype.Hstore
		err := rows.Scan(&id, &props, &geom)

		if err != nil {
			zap.S().Warnf("error scanning row: %s", err.Error())
			continue
		}
		err = scanner.Scan(geom)
		if err != nil {
			zap.S().Warnf("error scanning geometry from row: %s", err.Error() )
			continue
		}
		s := model.Site{
			Id:       id,
			Location: scanner.Geometry.(orb.Point),
			Properties: make(map[string]string,0),
		}
		for k, v := range props.Map {
			s.Properties[k] = v.String
		}
		sites = append(sites, &s)

	}
	zap.L().Info("returned ", zap.Int("sites", len(sites)))
	return sites, nil
}


