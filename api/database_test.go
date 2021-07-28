package main

import (
	"github.com/earthrise-media/plastics/api/database"
	"github.com/earthrise-media/plastics/api/model"
	"github.com/paulmach/orb"
	"testing"
)

func TestSiteController_AddSite(t *testing.T) {

	s1 := model.Site{

		Location: orb.Point{-76.332395, 39.544287},
	}
	s2 := model.Site{

		Location: orb.Point{-76.333457, 39.544990},
	}
	s3 := model.Site{
		Location: orb.Point{-76.334648, 39.544758},
	}

	sites := []*model.Site{&s1, &s2, &s3}

	sc := database.NewSiteController(db)

	err := sc.AddSites(sites)
	if err != nil {
		t.Fail()
	}

}

func TestSiteController_DeleteAllSites(t *testing.T) {

	sc := database.NewSiteController(db)
	if err := sc.DeleteAllSites(); err != nil {
		t.Fail()
	}

}

func TestSiteController_DeleteSiteById(t *testing.T) {

	sc := database.NewSiteController(db)
	sites, err := sc.FindSites(0, 1, &orb.Bound{Max: orb.Point{-90, -180}, Min: orb.Point{90, 180}})
	if err != nil {
		t.FailNow()
	}
	if len(sites) == 0 {
		return
	}
	if err = sc.DeleteSiteById(sites[0]); err != nil {
		t.FailNow()
	}

}

func TestSiteController_FindSiteById(t *testing.T) {

}

func TestSiteController_FindSiteByRadius(t *testing.T) {
}

func TestSiteController_FindSites(t *testing.T) {

}
