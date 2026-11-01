GO       ?= go
GOFLAGS  ?=
MODULE   := github.com/xiaoxin-agent/llama.go

.PHONY: all
all: fmt vet test

.PHONY: fmt
fmt:
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "unformatted files:"; echo "$$out"; exit 1; fi

.PHONY: vet
vet:
	$(GO) vet ./...

.PHONY: test
test:
	$(GO) test ./... -count=1

.PHONY: race
race:
	$(GO) test ./... -race -count=1

.PHONY: cover
cover:
	$(GO) test ./... -count=1 -cover

.PHONY: bench
bench:
	$(GO) test ./... -run '^$$' -bench . -benchmem

.PHONY: build
build:
	$(GO) build ./...

# verify is the single gate M7 requires to pass from a clean checkout.
.PHONY: verify
verify: fmt vet race cover

.PHONY: clean
clean:
	$(GO) clean -testcache
