package main

import (
	"github.com/earthrise-media/plastics/api/database"
	"github.com/earthrise-media/plastics/api/model"
	"github.com/paulmach/orb"
	"testing"
)

func TestSiteController_AddSite(t *testing.T) {

	sites := []model.Site{model.Site{

		Location:  orb.Point{-76.333457, 39.544990},
	},
		model.Site{

			Location:  orb.Point{-76.332395, 39.544287},
		},
		model.Site{
			Location:  orb.Point{-76.334648, 39.544758},
		},
	}

	sc := database.NewSiteController(db)

	for _,site := range sites{

		err := sc.AddSite(&site)
		if err != nil {
			t.Fail()
		}
		if site.Id == 0 {
			t.Fail()
		}
	}


}

func TestSiteController_DeleteAllSites(t *testing.T) {

}

func TestSiteController_DeleteSiteById(t *testing.T) {
}

func TestSiteController_DeleteSites(t *testing.T) {

}

func TestSiteController_FindSiteById(t *testing.T) {


}

func TestSiteController_FindSiteByRadius(t *testing.T) {
}

func TestSiteController_FindSites(t *testing.T) {

}

