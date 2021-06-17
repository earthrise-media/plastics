package config

import (
	"github.com/earthrise-media/plastics/api/database"
	"github.com/jackc/pgx/v4/log/zapadapter"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"golang.org/x/net/context"
)

var Db *pgxpool.Pool

//Preflight sets up all the config and sanity checks
func Preflight() {

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

	Db, err = pgxpool.ConnectConfig(context.Background(), poolConfig)

	if err != nil {
		zap.L().Fatal("failed to connect to database")
	}
	if viper.GetBool("DB_INIT") {
		err := database.SetupSchema(Db)
		if err != nil {
			zap.L().Fatal("unable to setup database")
		}

	}

	zap.L().Info("Preflight complete!")
}