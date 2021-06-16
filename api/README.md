## Plastics API 

### Building 
Building requires Go v 1.15+ and a simple:

`go build .`

### Testing

Testing needs a PostGIS database. Easiest way is:

`docker run --name test-postgis -e POSTGRES_DB=plastic -e POSTGRES_USER=postgis -e POSTGRES_PASSWORD=password -d mdillon/postgis`

Then: `go test`
