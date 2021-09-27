package main

import (
	"github.com/kataras/iris/v12"
	"github.com/kataras/iris/v12/httptest"
	"github.com/paulmach/orb/geojson"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"math/rand"
	"os"
	"runtime"
	"strconv"
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
	//startLength := len(fc.Features)

	//changed service signature to only return a 201 on successful creation
	test.POST("/sites").WithJSON(fc).Expect().Status(200).JSON()
	//fc.UnmarshalJSON([]byte(respData))
	//endLength := len(fc.Features)
	//if startLength != endLength {
	//	t.Log("Not all features appear to have been properly inserted")
	//	t.Fail()
	//}

}

func TestLimitSites(t *testing.T) {

	test := httptest.New(t, api)

	//how many sites do we have?
	feats := test.GET("/sites").WithQuery("limit", 500).Expect().Status(200).JSON().Path("$.features").Array()
	totalSites := len(feats.Raw())

	//lets do this randomly 10 times
	for i := 0; i < 10; i++ {
		//this should be a number between 0 and total number of sites, so the return count should always be the right size
		limit := rand.Intn(totalSites)
		feats = test.GET("/sites").WithQuery("limit", limit).Expect().Status(200).JSON().Path("$.features").Array()
		assert.Equal(t, len(feats.Raw()), limit)
	}
}

func TestDeleteSites(t *testing.T) {

	test := httptest.New(t, api)
	test.DELETE("/sites").WithBasicAuth(viper.GetString("ADMIN_USER"), viper.GetString("ADMIN_PASSWORD")).Expect().Status(204)

}

func TestGetSiteByID(t *testing.T) {

	test := httptest.New(t, api)
	test.GET("/sites/bob").Expect().Status(500)
	//add some sites
	TestInsertSites(t)
	//get 100 sites and use the first id in the test
	id := test.GET("/sites").Expect().JSON().Path("$.features").Array().First().Object().Value("id").Number().Raw()
	path := "/sites/" + strconv.Itoa(int(id))
	t.Logf("Path: %s", path)
	//make sure the id we ask for is the id of the object we get
	test.GET(path).Expect().Status(200).JSON().Object().Value("id").Number().Equal(int(id))
}

func TestInsertContours(t *testing.T) {

	test := httptest.New(t, api)
	//load the contours
	file, _ := os.Open("sample_data/v12_java_validated_positives_contours_model_spectrogram_v0.0.8_2021-06-03.geojson")
	data, _ := ioutil.ReadAll(file)
	fc := geojson.FeatureCollection{}
	err := fc.UnmarshalJSON(data)
	if err != nil {
		t.Logf("%s", err.Error())
		t.Fail()
	}
	//startLength := len(fc.Features)
	//pick a random site to associate them to
	id := test.GET("/sites").Expect().JSON().Path("$.features").Array().First().Object().Value("id").Number().Raw()
	path := "/sites/" + strconv.Itoa(int(id)) + "/contours"
	t.Logf("Path %s", path)
	test.POST(path).WithJSON(&fc).Expect().Status(201)

	//now test getting them back
	contourBody := test.GET(path).Expect().Status(200).Body()
	_ = fc.UnmarshalJSON([]byte(contourBody.Raw()))
	print("is sorted?")

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
