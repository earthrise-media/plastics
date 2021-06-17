package handler

import (
	"github.com/earthrise-media/plastics/api/database"
	"github.com/earthrise-media/plastics/api/encoding"
	"github.com/kataras/iris/v12"
	"github.com/paulmach/orb"
)

type SiteHandler struct {

	SiteController *database.SiteController
}

func(sh *SiteHandler) GetSites(ctx iris.Context) {

	var offset = 0
	var limit = 100
	var bnds = orb.Bound{Max: orb.Point{-90,-180}, Min:orb.Point{90,180}}

	//TODO parse input

	sites, err  := sh.SiteController.FindSites(offset,limit, bnds)
	if err != nil{
		ctx.Problem(iris.NewProblem().Type("/sites").Detail("database issue").Status(500))
		return
	}
	fc, err := encoding.ToGeoJson(sites)
	if err != nil {
		ctx.Problem(iris.NewProblem().Type("/sites").Detail("encoding issue").Status(500))
		return
	}

	ctx.JSON(fc)
}
