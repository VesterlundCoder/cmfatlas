#!/bin/bash
set -e

DATA_DIR="/app/data"
SEED_DIR="/app/data_seed"

mkdir -p "$DATA_DIR"

# Seed atlas_2d.db if it's missing (happens when volume is mounted fresh)
if [ ! -f "$DATA_DIR/atlas_2d.db" ] && [ ! -f "$DATA_DIR/atlas.db" ]; then
    if [ -f "$SEED_DIR/atlas_2d.db.gz" ]; then
        echo "[startup] Seeding atlas_2d.db from compressed archive (~30s)..."
        gzip -dc "$SEED_DIR/atlas_2d.db.gz" > "$DATA_DIR/atlas_2d.db"
        echo "[startup] Done: $(du -sh $DATA_DIR/atlas_2d.db)"
    elif [ -f "$SEED_DIR/atlas.db" ]; then
        echo "[startup] Seeding atlas.db (small fallback)..."
        cp "$SEED_DIR/atlas.db" "$DATA_DIR/atlas.db"
    else
        echo "[startup] ERROR: No database seed found in $SEED_DIR" >&2
        ls -la "$SEED_DIR" 2>/dev/null || echo "  (seed dir missing)"
        exit 1
    fi
fi

exec "$@"
