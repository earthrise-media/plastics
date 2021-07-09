package encoding

import (
	"github.com/paulmach/orb"
	"github.com/pkg/errors"
	"strconv"
	"strings"
)

func ParseBbox(bbox string) (*orb.Bound, error) {

	coords := strings.Split(bbox, ",")
	if len(coords) != 4 {
		return nil, errors.New("bbox does not have 4 elements")
	}

	// minlon,maxlon,minlat,maxlat
	minlon, err := strconv.ParseFloat(coords[0], 32)
	maxlon, err := strconv.ParseFloat(coords[1], 32)
	minlat, err := strconv.ParseFloat(coords[2], 32)
	maxlat, err := strconv.ParseFloat(coords[3], 32)

	if err != nil {

		return nil, errors.New("unable to parse coordinates from bbox")
	}

	return &orb.Bound{Max: orb.Point{minlat, minlon}, Min: orb.Point{maxlat, maxlon}}, nil

}
