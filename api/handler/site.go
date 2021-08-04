package handler

import (
	"github.com/earthrise-media/plastics/api/database"
	"github.com/earthrise-media/plastics/api/encoding"
	"github.com/earthrise-media/plastics/api/model"
	"github.com/kataras/iris/v12"
	"github.com/paulmach/orb"
	"github.com/paulmach/orb/geojson"
	"go.uber.org/zap"
	"sort"
)

type SiteHandler struct {
	SiteController *database.SiteController
}

//GetSites returns sites based on query params
func (sh *SiteHandler) GetSites(ctx iris.Context) {

	offset := ctx.URLParamIntDefault("offset", 0)
	limit := ctx.URLParamIntDefault("limit", 100)

	var bbox = &orb.Bound{Max: orb.Point{-180, -90}, Min: orb.Point{180, 90}}
	if ctx.URLParamExists("bbox") {
		bnds, err := encoding.ParseBbox(ctx.URLParam("bbox"))
		if err != nil {
			ctx.Problem(iris.NewProblem().Type("/sites").Detail(err.Error()).Status(400))
			return
		}
		bbox = bnds
	}

	sites, err := sh.SiteController.FindSites(offset, limit, bbox)
	if err != nil {
		ctx.Problem(iris.NewProblem().Type("/sites").Detail("database issue").Status(500))
		return
	}
	fc, err := encoding.SitesToFeatureCollection(sites)
	if err != nil {
		ctx.Problem(iris.NewProblem().Type("/sites").Detail("encoding issue").Status(500))
		return
	}

	ctx.JSON(fc)
}

//GetSiteById gets a site from a id
func (sh *SiteHandler) GetSiteById(ctx iris.Context) {

	id := ctx.Params().GetString("site_id")
	site, err := sh.SiteController.FindSiteById(id)
	if err != nil {
		ctx.Problem(iris.NewProblem().Type("/sites/" + id).Detail(err.Error()).Status(500))
		return
	}
	if site == nil {
		ctx.Problem(iris.NewProblem().Detail("site not found").Status(404))
		return
	}
	feature, err := encoding.SiteToGeoJsonFeature(site)
	if err != nil {
		ctx.Problem(iris.NewProblem().Type("/sites/" + id).Detail(err.Error()).Status(500))
		return
	}
	ctx.JSON(feature)

}

//CreateSites will add sites to the database
//It expects a feature collection with points in the POST Body
func (sh *SiteHandler) CreateSites(ctx iris.Context) {

	payload, err := ctx.GetBody()
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail("unable to ready body of POST").Status(500))
		return
	}
	var fc geojson.FeatureCollection
	err = fc.UnmarshalJSON(payload)
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail("unable to read GeoJson Feature Collection, check for errors").Status(400))
		return
	}
	sites, err := encoding.FeatureCollectionToSites(&fc)

	if err != nil {
		ctx.Problem(iris.NewProblem().Detail(err.Error()).Status(400))
		return
	}

	err = sh.SiteController.AddSites(sites)
	if err != nil {
		zap.L().Warn(err.Error())
	}

	jsonSites, err := encoding.SitesToFeatureCollection(sites)
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail(err.Error()).Status(500))
	}

	ctx.JSON(jsonSites)

}

//DeleteSites truncates all the sites
func (sh *SiteHandler) DeleteAllSites(ctx iris.Context) {

	err := sh.SiteController.DeleteAllSites()
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail(err.Error()).Status(500))
	}
	ctx.StatusCode(204)

}

//Deletes a single site
func (sh *SiteHandler) DeleteSiteById(ctx iris.Context) {

	s := model.Site{}
	id, err := ctx.URLParamInt64("site_id")
	if err != nil {
		ctx.Problem(iris.NewProblem().Status(400).Detail("Invalid Site ID"))
		return
	}
	s.Id = id
	if err = sh.SiteController.DeleteSiteById(&s); err != nil {
		ctx.Problem(iris.NewProblem().Status(500).Detail("Error deleting site").Cause(iris.NewProblem().Detail(err.Error())))
		return
	}
	ctx.StatusCode(204)
}

func (sh *SiteHandler) UpdateSite(ctx iris.Context) {

	payload, err := ctx.GetBody()
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail("unable to ready body of POST").Status(500))
		return
	}
	var f geojson.Feature
	err = f.UnmarshalJSON(payload)
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail("unable to read GeoJson Feature Collection, check for errors").Status(400))
		return
	}
	site, err := encoding.FeatureToSite(&f)
	if err = sh.SiteController.UpdateSite(site); err != nil {
		ctx.Problem(iris.NewProblem().Status(500).Detail("error updating site").Cause(iris.NewProblem().Detail(err.Error())))
		return
	}
	ctx.StatusCode(204)
}

func (sh *SiteHandler) GetContours(ctx iris.Context) {

	site, err := sh.SiteController.FindSiteById(ctx.Params().GetString("site_id"))
	if err != nil || site == nil {
		ctx.Problem(iris.NewProblem().Detail("invalid site id").Status(400))
		return
	}
	contours, err := sh.SiteController.GetContoursBySite(site)
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail(err.Error()).Status(500))

	}
	//sort from newest to olders
	sort.Sort(sort.Reverse(model.ByDate(contours)))

	fc, err := encoding.ContourFeatureCollection(contours)
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail(err.Error()).Status(500))
	}
	ctx.JSON(fc)

}

func (sh *SiteHandler) AddContours(ctx iris.Context) {
	payload, err := ctx.GetBody()
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail("unable to ready body of POST").Status(500))
		return
	}

	site, err := sh.SiteController.FindSiteById(ctx.Params().GetString("site_id"))
	if err != nil || site == nil {
		ctx.Problem(iris.NewProblem().Detail("invalid site id").Status(400))
		return
	}
	var fc geojson.FeatureCollection
	err = fc.UnmarshalJSON(payload)
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail("unable to read GeoJson Feature Collection, check for errors").Status(400))
		return
	}
	contours, err := encoding.FeatureCollectionToContours(&fc)
	zap.S().Infof("%d", len(contours))
	if err != nil {
		ctx.Problem(iris.NewProblem().Detail(err.Error()).Status(400))
		return
	}
	err = sh.SiteController.AddContoursToSite(site, contours)
	if err != nil {
		zap.L().Error(err.Error())
		ctx.Problem(iris.NewProblem().Detail(err.Error()).Status(500))
		return
	} else {
		ctx.StatusCode(201)
	}
}

func (sh *SiteHandler) DeleteAllContours(ctx iris.Context) {
	//TODO: implement
}

func (sh *SiteHandler) UpdateContour(ctx iris.Context) {
	//TODO: implement
}

func (sh *SiteHandler) DeleteContour(ctx iris.Context) {
	//TODO: implement
}
