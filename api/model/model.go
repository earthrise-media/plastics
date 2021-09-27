package model

import (
	"github.com/paulmach/orb"
	"time"
)


const (
	SiteName  = "name"
	SiteAdded  = "added"
	SiteMean  = "mean"
	SiteStd  = "std"
	SiteMin  = "min"
	SiteMax  = "max"
	SiteCount  = "count"
	SiteDate = "date"
	SiteId = "site_id"
	Id = "id"
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

// implement sorting for Contours
type ByDate []*Contour

const byDateTimeFormat = "2006-01-02T15:04:05"

func (bd ByDate) Len() int {
	return len(bd)
}

func (bd ByDate)  Less(i,j int) bool{
	dateStringI, existI := bd[i].Properties[SiteDate]
	dateStringJ, existJ := bd[j].Properties[SiteDate]

	// if neither has a date they are equal and hence we return false
	if !existI && !existJ { return  false}
	// if one has a date but the other doesn't - we'll consider the one without a date to be "less"
	// i doesn't have a date but j does, so we return
	if !existI {return false}
	if !existJ {return true}


	// expect dates in this format 2019-06-01T00:00:00
	dateI , errI := time.Parse(byDateTimeFormat, dateStringI)
	dateJ , errJ := time.Parse(byDateTimeFormat, dateStringJ)
	// if one date is invalid we consider it as no date
	if errI != nil {return false}
	if errJ != nil {return true}
	// actually compare dates
	return dateI.Before(dateJ)
}

func (bd ByDate)  Swap(i,j int) {
	bd[i], bd[j] = bd[j], bd[i]
}

