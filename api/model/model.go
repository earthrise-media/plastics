package model

import (
	"github.com/paulmach/orb"
)


const (
	SiteName  = "name"
	SiteAdded  = "added"
	SiteMean  = "mean"
	SiteStd  = "std"
	SiteMin  = "min"
	SiteMax  = "max"
	SiteCount  = "count"
)

type ContourProperty string

type Site struct {
	Id        int64
	Location  orb.Point
	Properties map[string]string
}

type Contour struct {
	Id       int64
	SiteId   int64
	Properties map[string]string
	Geometry orb.MultiPolygon
}
