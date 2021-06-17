package handler

import "github.com/kataras/iris/v12"

//ok is a simple health check endpoint for the service
func Ok(ctx iris.Context) {

	ctx.JSON(map[string]string{"status": "ok"})
}
