PYTHON             := python
BUILD_VARIANT      := cpu
SERVICE_NAME  		 := yaclip
GITHUB_TOKEN       :=
TAG                :=

# Use `uv run python` when uv is on PATH (works without an activated venv);
# otherwise fall back to plain $(PYTHON), which then requires an already-active venv.
UV                 := $(shell command -v uv 2>/dev/null)
ifdef UV
RUN                := uv run -- python
else
RUN                := $(PYTHON)
endif

.PHONY:

.SILENT:

init:
	make clean
ifdef UV
	uv sync --locked --extra dev
else
	$(PYTHON) -m pip install --no-cache-dir --upgrade pip
	$(PYTHON) -m pip install --no-cache-dir -e ".[dev]"
endif

init-dist:
	mkdir -p dist

build:
	make init-dist
	$(RUN) build.py build --variant $(BUILD_VARIANT)
	echo "Build '$(SERVICE_NAME)' ($(BUILD_VARIANT)) complete."

build-cuda:
	make build BUILD_VARIANT=cuda

archive:
	$(RUN) build.py archive --variant $(BUILD_VARIANT)

checksum:
	$(RUN) build.py checksum

release:
	make clean-dist
	$(RUN) build.py build archive checksum --variant $(BUILD_VARIANT)
	echo "Release '$(SERVICE_NAME)' complete, please check dist directory."

publish:
	if [ -z "$(TAG)" ]; then echo "usage: make publish TAG=v0.1.0"; exit 1; fi
	GH_TOKEN=$(GITHUB_TOKEN) $(RUN) build.py publish --tag $(TAG)
	echo "Publish '$(SERVICE_NAME)' complete, please check your repository releases."

run:
	$(RUN) app.py

clean-dist:
	rm -rf dist

clean:
	make clean-dist
