# ── Stage 1: deps ─────────────────────────────────────────────────────────────
# Installs all Python dependencies (runtime + dev) without the package source.
# This layer is cached monthly by the dockerDependencies shared workflow so that
# CI builds are fast – only source-code changes ever need a rebuild.
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE} AS deps

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install R 4.4 via rig (R Installation Manager) — avoids apt dependency hell
# on the minimal pytorch base image. Rig bundles pre-compiled binaries.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://github.com/r-lib/rig/releases/download/v0.8.0/rig-linux-0.8.0.tar.gz \
    | tar xz -C /usr/local \
    && rig add 4.4 \
    && rig default $(rig list | awk '/^[0-9]/{print $1; exit}')

# System deps for R packages (libcurl, libssl, libxml2, libicu needed by Seurat/anndata)
# build-essential and gfortran needed to compile CRAN packages from source
# libuv-dev for httpuv (shiny), pandoc for rmarkdown/htmlwidgets
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libicu-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libuv-dev \
    cmake \
    pandoc \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency spec files so that source changes don't bust the cache.
# LICENSE and README.md are required because pyproject.toml declares
# `license = { file = "LICENSE" }` and `readme = "README.md"` — hatchling reads
# both when building the package metadata during `uv pip install`.
COPY pyproject.toml uv.lock LICENSE README.md ./

# Create a minimal package stub so uv can resolve the package metadata without
# needing the real source tree, then install all dependencies (runtime + dev),
# then remove the stub.  The real source is injected in the runtime stage.
RUN mkdir -p mil_ton \
    && touch mil_ton/__init__.py \
    && uv pip install --system --no-cache ".[dev]" \
    && rm -rf mil_ton

# Install required R packages (used by GEX_MERGE_COUNTS template)
# Must be after Python deps (anndata R package depends on Python anndata via reticulate)
RUN R -e 'install.packages(c("Matrix", "data.table"), repos = "https://cloud.r-project.org")'
RUN R -e 'Sys.setenv(CXX17="g++"); install.packages("fs", repos = "https://cloud.r-project.org")'
RUN R -e 'install.packages("Seurat", repos = "https://cloud.r-project.org"); stopifnot(require("Seurat", character.only = TRUE))'
RUN R -e 'install.packages("anndata", repos = "https://cloud.r-project.org"); stopifnot(require("anndata", character.only = TRUE))'

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Adds the actual package source on top of the dep layer.  This is the image
# pushed to GHCR as :latest and used for production inference.
FROM deps AS runtime

COPY mil_ton/ mil_ton/
COPY tests/ tests/

# Install the package itself without reinstalling its dependencies (already present).
RUN uv pip install --system --no-cache --no-deps .
