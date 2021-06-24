package main

import (
	"github.com/earthrise-media/plastics/api/config"
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
	config.Db.Close()
}


func Test_preflight(t *testing.T) {

	config.Preflight()
}
