package database

import (
	"github.com/paulmach/orb"
	"time"
)

type Site struct {
	Id        int64     `db:id`
	Name      string    `db:name`
	FirstSeen time.Time `db:first_seen`
	LastSeen  time.Time `db:last_seen`
	Location  orb.Point `db:geom`
}

type Contour struct {
	Id       int64
	SiteId   int64
	ModelRun time.Time
	Geometry orb.MultiPolygon
}
