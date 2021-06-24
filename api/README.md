## Plastics API 

### Building 
Building requires Go v 1.15+ and a simple:

`go build .`

### Testing

Testing needs a PostGIS database. The easiest way is:

`docker run -e POSTGRES_DB=plastic -e POSTGRES_USER=postgis -p 5432:5432 -e POSTGRES_PASSWORD=password -d mdillon/postgis
`

Then: `go test`
