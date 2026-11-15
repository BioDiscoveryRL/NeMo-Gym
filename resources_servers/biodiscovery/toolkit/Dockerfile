# Analysis toolkit for biodiscovery episodes: the tools the task briefs promise, no world data (worlds are mounted
# read-only at /worlds) and no network at run time (the sandbox provider runs it with network none).
#   docker build --platform=linux/amd64 -t bdrl-toolkit:latest resources_servers/biodiscovery/toolkit
FROM --platform=linux/amd64 python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends bcftools tabix curl unzip ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas numpy scipy statsmodels lifelines scikit-learn pyreadstat pyarrow polars duckdb \
        cyvcf2 pysam matplotlib
RUN curl -fsSL -o /tmp/plink2.zip https://s3.amazonaws.com/plink2-assets/beta1/plink2_linux_x86_64_20261107.zip \
    && unzip -q /tmp/plink2.zip -d /usr/local/bin && chmod 0755 /usr/local/bin/plink2 && rm /tmp/plink2.zip

RUN useradd -m agent && mkdir -p /trial/workspace && chown -R agent:agent /trial
USER agent
WORKDIR /trial/workspace
