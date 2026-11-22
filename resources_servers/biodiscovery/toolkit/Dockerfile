# Analysis toolkit for biodiscovery episodes: the tools the task briefs promise, no world data (worlds are mounted
# read-only at /worlds) and no network at run time (the sandbox provider runs it with network none).
#   docker build -t bdrl-toolkit:latest resources_servers/biodiscovery/toolkit                          # host arch
#   docker build --platform=linux/amd64 -t bdrl-toolkit:latest resources_servers/biodiscovery/toolkit   # x86 hosts
# plink2 and regenie publish Linux binaries for x86_64 only, so arm64 builds (for example a DGX Spark) compile them
# from pinned releases.

# regenie: the x86_64 release binary, or on arm64 a build from source against BGEN (whose current v1.1.7 tarball
# bundles Boost 1.86 where regenie 3.4.1 expects 1.55, so its paths and one removed Boost.Filesystem call are patched) with Eigen's own linear algebra.
FROM python:3.12-slim AS regenie
ARG TARGETARCH
ARG REGENIE_TAG=v3.4.1
ARG REGENIE_X86_ZIP=https://github.com/rgcgithub/regenie/releases/download/v3.4.1/regenie_v3.4.1.gz_x86_64_Linux_mkl.zip
ARG BGEN_TGZ=https://enkre.net/cgi-bin/code/bgen/tarball/release/v1.1.7/bgen.tgz
RUN set -eux; apt-get update; \
    if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
        apt-get install -y --no-install-recommends curl unzip ca-certificates libgomp1; \
        curl -fsSL -o /tmp/regenie.zip "$REGENIE_X86_ZIP"; unzip -q /tmp/regenie.zip -d /tmp/regenie; \
        install -m 0755 "$(find /tmp/regenie -type f | head -1)" /usr/local/bin/regenie; \
    else \
        apt-get install -y --no-install-recommends build-essential gfortran cmake git curl ca-certificates \
            zlib1g-dev libzstd-dev libbz2-dev liblzma-dev libssl-dev libcurl4-openssl-dev libboost-iostreams-dev; \
        ln -s "$(command -v gfortran)" /usr/local/bin/f77; \
        curl -fsSL -o /tmp/bgen.tgz "$BGEN_TGZ"; mkdir -p /opt/bgen; \
        tar -xzf /tmp/bgen.tgz -C /opt/bgen --strip-components=3; (cd /opt/bgen && python3 waf configure && python3 waf); \
        git clone --depth 1 --branch "$REGENIE_TAG" https://github.com/rgcgithub/regenie /tmp/regenie; \
        sed -i 's/boost_1_55_0/boost_1_86_0/g' /tmp/regenie/CMakeLists.txt; \
        sed -i 's/fs::extension(\([a-z_]*\))/fs::path(\1).extension().string()/g' /tmp/regenie/src/*.cpp; \
        mkdir /tmp/regenie/build; (cd /tmp/regenie/build && BGEN_PATH=/opt/bgen cmake .. && make -j"$(nproc)"); \
        install -m 0755 /tmp/regenie/build/regenie /usr/local/bin/regenie; \
    fi; \
    regenie --version

FROM python:3.12-slim
ARG TARGETARCH
ARG PLINK2_X86_ZIP=https://s3.amazonaws.com/plink2-assets/beta1/plink2_linux_x86_64_20261107.zip
ARG PLINK_NG_TAG=v2.0.0-b.1.4

# libgomp1, libgfortran5, libcurl4 and libboost-iostreams are regenie's run-time libraries on arm64.
RUN apt-get update && apt-get install -y --no-install-recommends bcftools samtools tabix curl unzip ca-certificates \
        procps libgomp1 libgfortran5 libcurl4 libboost-iostreams1.83.0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache pandas numpy scipy statsmodels lifelines scikit-learn pyreadstat pyarrow polars \
        duckdb cyvcf2 pysam matplotlib neurokit2
# On arm64, OpenBLAS, LAPACK(E) and zstd stay installed after the build: plink2 links against them at run time.
RUN set -eux; \
    if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
        curl -fsSL -o /tmp/plink2.zip "$PLINK2_X86_ZIP"; \
        unzip -q /tmp/plink2.zip -d /usr/local/bin; rm /tmp/plink2.zip; \
    else \
        apt-get update; apt-get install -y --no-install-recommends build-essential git zlib1g-dev libzstd-dev \
            libopenblas-dev liblapack-dev liblapacke-dev; \
        git clone --depth 1 --branch "$PLINK_NG_TAG" https://github.com/chrchang/plink-ng.git /tmp/plink-ng; \
        make -C /tmp/plink-ng/2.0/build_dynamic -j"$(nproc)"; \
        cp /tmp/plink-ng/2.0/build_dynamic/plink2 /usr/local/bin/plink2; \
        rm -rf /tmp/plink-ng; apt-mark manual libopenblas-dev liblapack-dev liblapacke-dev libzstd-dev; apt-get purge -y build-essential git zlib1g-dev; \
        apt-get autoremove -y; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    chmod 0755 /usr/local/bin/plink2; plink2 --version
COPY --from=regenie /usr/local/bin/regenie /usr/local/bin/regenie
RUN regenie --version && samtools --version | head -1 && python -c "import neurokit2"

RUN useradd -m agent && mkdir -p /trial/workspace && chown -R agent:agent /trial
USER agent
WORKDIR /trial/workspace
