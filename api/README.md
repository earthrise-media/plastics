## Plastics API 

### Building 
Building requires Go v 1.15+ and a simple:

`go build .`

### Testing

Testing needs a PostGIS database. The easiest way is:

`docker run -e POSTGRES_DB=plastic -e POSTGRES_USER=postgis -p 5432:5432 -e POSTGRES_PASSWORD=password -d mdillon/postgis
`

Then: `go test`

### Running 
Configuration can be set in the environment, the cli or a config file. See [Viper's Automatic Env docs](https://github.com/spf13/viper) for more info.
This service exposes the following configuration options:

| Config | Default | Description |
|--------|---------|-------------|
| PORT | 8080 | listening port |
| PGHOST | localhost | database hostname or IP |
| PGPORT | 5432 | port database is listening on |
|PGDATABASE | plastic | name of database |
|PGUSER | postgis | database username |
|PGPASSWORD | password | database password |
|LOG_LEVEL| DEBUG |one of TRACE, DEBUG, INFO, WARN, ERROR | 
|SITE_MATCH_DISTANCE_METERS | 1000 | the distance within which sites will be considered the same site | 
|ADMIN_USER| admin | an admin user who can perform destructive actions |
|ADMIN_PASSWORD | plastics | admin user password |	

### Endpoints

See OpenAPI docs [here](https://app.swaggerhub.com/apis/tingold/Plastics/0.0.1)