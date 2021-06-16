package database

import (
	"context"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

func SetupSchema(db *pgxpool.Pool) error {

	ctx := context.Background()

	//is postgis installed?
	row := db.QueryRow(ctx,"SELECT postgis_version()")
	var version string
	err := row.Scan(&version)

	if err != nil {
		zap.L().Warn("PostGIS not found...attempting to install")
		//probabaly need to install it
		_, err := db.Exec(ctx, "CREATE EXTENSION POSTGIS")
		if err != nil{
			return errors.Wrap(err, "unable to install postgis")
		}
	} else {
		zap.L().Info("Found PostGIS: "+version)
	}
	createSql := `CREATE TABLE IF NOT EXISTS sites(id serial primary key, name varchar(40), first_seen TIMESTAMP NOT NULL DEFAULT NOW(), last_seen TIMESTAMP);
	 CREATE TABLE IF NOT EXISTS contours(countour_id serial primary key, site_id int, model_run TIMESTAMP NOT NULL DEFAULT NOW());
	 SELECT AddGeometryColumn('sites', 'geom', 4326, 'POINT', 2, false);
	SELECT AddGeometryColumn('contours', 'geom', 4326, 'MULTIPOLYGON', 2, false);`
	db.Exec(ctx,createSql)
	return nil
}
