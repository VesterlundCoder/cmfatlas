#!/bin/bash
set -e

DATA_DIR="/app/data"
SEED_DIR="/app/data_seed"

mkdir -p "$DATA_DIR"

# Seed atlas_2d.db if it's missing (happens when volume is mounted fresh)
if [ ! -f "$DATA_DIR/atlas_2d.db" ] && [ ! -f "$DATA_DIR/atlas.db" ]; then

    if [ -f "$SEED_DIR/atlas_2d.db.gz" ]; then
        # Best case: gz was baked into the image (local railway up deploy)
        echo "[startup] Seeding atlas_2d.db from compressed archive (~30s)..."
        gzip -dc "$SEED_DIR/atlas_2d.db.gz" > "$DATA_DIR/atlas_2d.db"
        echo "[startup] Done: $(du -sh $DATA_DIR/atlas_2d.db)"

    elif [ -n "$DB_DOWNLOAD_URL" ]; then
        # Fallback: download from a URL set as Railway env var
        echo "[startup] Downloading atlas_2d.db.gz from \$DB_DOWNLOAD_URL..."
        curl -fsSL "$DB_DOWNLOAD_URL" -o /tmp/atlas_2d.db.gz
        echo "[startup] Decompressing..."
        gzip -dc /tmp/atlas_2d.db.gz > "$DATA_DIR/atlas_2d.db"
        rm -f /tmp/atlas_2d.db.gz
        echo "[startup] Done: $(du -sh $DATA_DIR/atlas_2d.db)"

    elif [ -f "$SEED_DIR/atlas.db" ]; then
        echo "[startup] Seeding atlas.db (small fallback)..."
        cp "$SEED_DIR/atlas.db" "$DATA_DIR/atlas.db"

    else
        echo "[startup] ERROR: No database found. Set DB_DOWNLOAD_URL or bake atlas_2d.db.gz into the image." >&2
        echo "[startup] data_seed contents:" >&2
        ls -la "$SEED_DIR" 2>/dev/null || echo "  (missing)" >&2
        exit 1
    fi
fi

exec "$@"
