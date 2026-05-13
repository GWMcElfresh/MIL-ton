# ── Stage 1: deps ─────────────────────────────────────────────────────────────
# Installs all Python dependencies (runtime + dev) without the package source.
# This layer is cached monthly by the dockerDependencies shared workflow so that
# CI builds are fast – only source-code changes ever need a rebuild.
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE} AS deps

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install R from CRAN repository (system r-base on this base image is R 3.6.3,
# which is too old for Seurat/anndata — we need R >= 4.1).
# pytorch:2.1.0-cuda11.8 is based on Ubuntu 22.04 (jammy).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
      | gpg --dearmor -o /usr/share/keyrings/cran-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cran-archive-keyring.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
      > /etc/apt/sources.list.d/cran.list \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    cmake \
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
