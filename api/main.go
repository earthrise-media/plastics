package main

import (
	"context"
	"github.com/earthrise-media/plastics/api/database"
	"github.com/earthrise-media/plastics/api/handler"
	"github.com/jackc/pgx/v4/log/zapadapter"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/kataras/iris/v12"
	"github.com/kataras/iris/v12/middleware/basicauth"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var db *pgxpool.Pool

func main() {

	preflight()
	defer db.Close()
	app := plasticApi()
	app.Listen(":" + viper.GetString("port"))
}

func plasticApi() *iris.Application {

	app := iris.New()

	//TODO add CORS

	//TODO do we need real auth?
	//auth
	auth := basicauth.Default(map[string]string{
		viper.GetString("ADMIN_USER"): viper.GetString("ADMIN_PASSWORD"),
	})

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
		allSiteEndpoint.Delete("/", auth, sh.DeleteAllSites)

	}
	return app

}


//Preflight sets up all the config and sanity checks
func preflight() {

	//setup configuration and defaults
	viper.New()
	viper.SetDefault("PORT", "8080")           //web service port
	viper.SetDefault("PGHOST", "localhost")    //database hostname or ip
	viper.SetDefault("PGPORT", "5432")         //  database port
	viper.SetDefault("PGDATABASE", "plastic")  // name of database
	viper.SetDefault("PGUSER", "postgis")      //database username
	viper.SetDefault("PGPASSWORD", "password") // database password
	viper.SetDefault("DB_INIT", true)          //flag to initialize database, ideally this is safe even if db is already initialized
	viper.SetDefault("LOG_LEVEL", "DEBUG")     //log levels as defined by Zap library -- pretty standard
	//Point matching thresholds
	viper.SetDefault("SITE_MATCH_DISTANCE_METERS", 1000) //if a point is within Xm of another site an update will treat them as the same site
	//Security (lol) here
	viper.SetDefault("ADMIN_USER", "admin")        //an admin user who can perform destructive actions
	viper.SetDefault("ADMIN_PASSWORD", "plastics") //admin user password

	viper.AutomaticEnv()

	//setup logging
	loggerConfig := zap.NewProductionConfig()
	loggerConfig.Sampling = nil
	loggerConfig.Level.UnmarshalText([]byte(viper.GetString("LOG_LEVEL")))
	loggerConfig.EncoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder
	loggerConfig.EncoderConfig.TimeKey = "ts"
	loggerConfig.EncoderConfig.LevelKey = "l"
	loggerConfig.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	loggerConfig.OutputPaths = []string{"stdout"}
	loggerConfig.ErrorOutputPaths = []string{"stderr"}
	logger, _ := loggerConfig.Build()
	zap.ReplaceGlobals(logger)

	//connect to database
	//postgres://username:password@localhost:5432/database_name
	connstring := "postgres://" + viper.GetString("PGUSER") + ":" + viper.GetString("PGPASSWORD") + "@" + viper.GetString("PGHOST") + ":" + viper.GetString("PGPORT") + "/" + viper.GetString("PGDATABASE")

	dbLogger := zapadapter.NewLogger(logger)

	poolConfig, err := pgxpool.ParseConfig(connstring)
	if err != nil {
		zap.L().Fatal("Unable to parse connection string")
	}

	poolConfig.ConnConfig.Logger = dbLogger

	db, err = pgxpool.ConnectConfig(context.Background(), poolConfig)

	if err != nil {
		zap.L().Fatal("failed to connect to database")
	}
	if viper.GetBool("DB_INIT") {
		err := database.SetupSchema(db)
		if err != nil {
			zap.L().Fatal("unable to setup database")
		}

	}

	zap.L().Info("Preflight complete!")
}

