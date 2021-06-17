package database

import (
	"context"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

//DeleteSchema cleans up the tables and data - useful for testing but not exposed to web
func DeleteSchema(db *pgxpool.Pool) error {
	dropTables := `drop table sites cascade;
					drop table contours cascade;`
	_, err := db.Exec(context.Background(), dropTables)
	return err
}

//SetupSchema creates the required tables if they don't exist
func SetupSchema(db *pgxpool.Pool) error {

	ctx := context.Background()

	//is postgis installed?
	row := db.QueryRow(ctx, "SELECT postgis_version()")
	var version string
	err := row.Scan(&version)

	if err != nil {
		zap.L().Warn("PostGIS not found...attempting to install")
		//probabaly need to install it
		_, err := db.Exec(ctx, "CREATE EXTENSION POSTGIS")
		if err != nil {
			return errors.Wrap(err, "unable to install postgis")
		} else {
			zap.L().Info("Installed PostGIS")
		}
	} else {
		zap.L().Info("Found PostGIS: " + version)
	}

	checkSql := "SELECT name FROM sites LIMIT 1"
	row = db.QueryRow(ctx,checkSql)
	err = row.Scan(&version)
	if err == nil{
		//table likely exists
		return nil
	}
	zap.L().Info("attempting to create tables")

	createSql := `CREATE TABLE IF NOT EXISTS sites(id serial primary key, name varchar(40), first_seen TIMESTAMP NOT NULL DEFAULT NOW(), last_seen TIMESTAMP);
	 CREATE TABLE IF NOT EXISTS contours(countour_id serial primary key, site_id int, model_run TIMESTAMP NOT NULL DEFAULT NOW());
	 SELECT AddGeometryColumn('sites', 'geom', 4326, 'POINT', 2, false);
	SELECT AddGeometryColumn('contours', 'geom', 4326, 'MULTIPOLYGON', 2, false);
	CREATE INDEX site_geom_index on sites USING GIST(geom);
	CREATE INDEX contour_geom_index on contours USING GIST(geom);
`
	_, err = db.Exec(ctx, createSql)
	return err
}
