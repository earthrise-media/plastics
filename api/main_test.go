package main

import (
	"github.com/kataras/iris/v12"
	"github.com/kataras/iris/v12/httptest"
	"github.com/paulmach/orb/geojson"
	"github.com/spf13/viper"
	"io/ioutil"
	"os"
	"runtime"
	"testing"
)

var api *iris.Application

func TestMain(m *testing.M) {

	setup()
	defer shutdown()
	code := m.Run()

	defer os.Exit(code)
	runtime.Goexit()
}

func TestGetSites(t *testing.T) {

	test := httptest.New(t, api)
	test.GET("/").Expect().Status(404)
	test.GET("/sites").Expect().Status(200).JSON()

}

func TestHealth(t *testing.T) {

	test := httptest.New(t, api)
	test.GET("/health").Expect().Status(200).JSON()
	test.GET("/healthz").Expect().Status(200).JSON()

}

func TestInsertSites(t *testing.T) {
	test := httptest.New(t, api)
	file, _ := os.Open("sample_data/v12_java_bali_validated_positives.geojson")
	data, _ := ioutil.ReadAll(file)
	fc := geojson.FeatureCollection{}
	fc.UnmarshalJSON(data)
	startLength := len(fc.Features)

	//this assertion means that we successfully inserted all of the sites that were sent
	//ie the number of features sent = the number of features received
	respData := test.POST("/sites").WithJSON(fc).Expect().Status(200).Body().Raw()
	fc.UnmarshalJSON([]byte(respData))
	endLength := len(fc.Features)
	if startLength != endLength {
		t.Log("Not all features appear to have been properly inserted")
		t.Fail()
	}

}

func TestDeleteSites(t *testing.T) {

	test := httptest.New(t, api)
	test.DELETE("/sites").WithBasicAuth(viper.GetString("ADMIN_USER"), viper.GetString("ADMIN_PASSWORD")).Expect().Status(204)

}


func TestGetSiteByID(t *testing.T) {

	test := httptest.New(t, api)
	test.GET("/sites/932180398").Expect().Status(404)
	TestInsertSites(t)
	test.GET("/sites/1").Expect().Status(200).JSON()

}

func setup() {
	preflight()
	api = plasticApi()

}

func shutdown() {
	db.Close()
}

func Test_preflight(t *testing.T) {

	preflight()
}
