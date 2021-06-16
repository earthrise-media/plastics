package main

import (
	"github.com/kataras/iris/v12"
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

func setup() {



}
func shutdown() {


}

func Test_ok(t *testing.T) {
	type args struct {
		ctx iris.Context
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
		})
	}
}

func Test_preflight(t *testing.T) {

	preflight()
}
