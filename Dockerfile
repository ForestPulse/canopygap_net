FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4 AS builder
ARG VARIANT=cpu
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements-${VARIANT}.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN find . -name "*.py" -exec chmod +x {} \;

FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4
WORKDIR /app
COPY --from=builder /venv /venv
COPY --from=builder /app /app
ENV PATH="/venv/bin:/app:$PATH"

ENTRYPOINT ["python3", "predict_nf.py"]