package main

import (
	"github.com/earthrise-media/plastics/api/config"
	"github.com/earthrise-media/plastics/api/database"
	"github.com/earthrise-media/plastics/api/handler"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/kataras/iris/v12"
	"github.com/spf13/viper"
)

var db *pgxpool.Pool

func main() {

	config.Preflight()
	db = config.Db
	defer db.Close()
	app := plasticApi()
	app.Listen(":" + viper.GetString("port"))
}

func plasticApi() *iris.Application {


	app := iris.New()

	//healthcheck endpoint
	app.Get("/healthz", handler.Ok)
	app.Get("/health", handler.Ok)

	//site endpoint
	sc := database.NewSiteController(db)
	sh := handler.SiteHandler{SiteController: sc}
	allSiteEndpoint := app.Party("/sites")
	{
		allSiteEndpoint.Get("/", sh.GetSites)
		allSiteEndpoint.Get("/{site_id}", sh.GetSiteById)
		allSiteEndpoint.Post("/", sh.CreateSites)
		//allSiteEndpoint.Post("/", sh.GetSites)
		//allSiteEndpoint.Delete("/", sh.GetSites)

	}
	return app

}
