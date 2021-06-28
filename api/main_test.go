package main

import (
	"github.com/earthrise-media/plastics/api/config"
	"github.com/kataras/iris/v12/httptest"
	"os"
	"runtime"
	"testing"
)

func TestMain(m *testing.M) {

	setup()
	defer shutdown()
	code := m.Run()

	defer os.Exit(code)
	runtime.Goexit()
}

func TestGetSites(t *testing.T){

	config.Preflight()
	db = config.Db
	app := plasticApi()

	test := httptest.New(t, app)

	test.GET("/").Expect().Status(404)
	test.GET("/sites").Expect().Status(200).JSON()

}

func setup() {

}

func shutdown() {
	config.Db.Close()
}


func Test_preflight(t *testing.T) {

	config.Preflight()
}
