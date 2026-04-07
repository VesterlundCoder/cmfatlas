FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Move DB seed files to /app/data_seed — this directory is NOT under the Railway
# volume mount point (/app/data), so these files survive across deployments and
# are used by the entrypoint to seed the volume on first startup.
RUN mkdir -p /app/data_seed && \
    if [ -f /app/data/atlas_2d.db.gz ]; then mv /app/data/atlas_2d.db.gz /app/data_seed/; fi && \
    if [ -f /app/data/atlas.db ];       then cp  /app/data/atlas.db       /app/data_seed/; fi

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
