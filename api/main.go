package main

import (
	"context"
	"github.com/earthrise-media/plastics/api/database"
	"github.com/jackc/pgx/v4/log/zapadapter"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/kataras/iris/v12"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var db *pgxpool.Pool


func main() {

	preflight()
	defer db.Close()
	app := iris.New()


	//healthcheck endpoint
	healthEndpoint := app.Party("/healthz")
	{
		healthEndpoint.Get("/", ok)
	}



	app.Listen(":"+viper.GetString("port"))
}


//preflight sets up all the config and sanity checks
func preflight(){

	//setup configuration and defaults
	viper.New()
	viper.SetDefault("PORT","8080") //web service port
	viper.SetDefault("PGHOST", "localhost") //database hostname or ip
	viper.SetDefault("PGPORT", "5432") //  database port
	viper.SetDefault("PGDATABASE", "plastic") // name of database
	viper.SetDefault("PGUSER", "postgis") //database username
	viper.SetDefault("PGPASSWORD", "password") // database password
	viper.SetDefault("DB_INIT", true) //flag to initialize database, ideally this is safe even if db is already initialized
	viper.SetDefault("LOG_LEVEL", "DEBUG") //log levels as defined by Zap library -- pretty standard


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
	connstring := "postgres://"+viper.GetString("PGUSER")+":"+viper.GetString("PGPASSWORD")+"@"+viper.GetString("PGHOST")+":"+viper.GetString("PGPORT")+"/"+viper.GetString("PGDATABASE")

	dbLogger := zapadapter.NewLogger(logger)

	poolConfig, err := pgxpool.ParseConfig(connstring)
	if err != nil {
		zap.L().Fatal("Unable to parse connection string")
	}

	poolConfig.ConnConfig.Logger = dbLogger

	db, err = pgxpool.ConnectConfig(context.Background(), poolConfig)

	if err != nil{
		zap.L().Fatal("failed to connect to database")
	}
	if viper.GetBool("DB_INIT"){
		err := database.SetupSchema(db)
		if err != nil{
			zap.L().Fatal("nable to setup database")
		}

	}
	zap.L().Info("Preflight complete!")
}

//ok is a simple health check endpoint for the service
func ok(ctx iris.Context){

	ctx.JSON(map[string]string{"status":"ok"})
}
