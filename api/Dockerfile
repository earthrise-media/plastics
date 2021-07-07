FROM golang AS build-env

WORKDIR /go/src/app
ADD . /go/src/app

RUN go get -d -v ./...

RUN go build -o /go/bin/plastic-api

FROM gcr.io/distroless/base
COPY --from=build-env /go/bin/plastic-api /
CMD ["/plastic-api"]