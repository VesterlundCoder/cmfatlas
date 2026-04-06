"""Quick query: list all 3D CMFs in the Atlas."""
import sqlite3, json, os

DB = os.path.join(os.path.dirname(__file__), "data", "atlas_2d.db")
con = sqlite3.connect(DB)

rows = con.execute("""
    SELECT id, cmf_payload FROM cmf
    WHERE dimension = 3
    AND (json_extract(cmf_payload,'$.hidden') IS NULL
         OR json_extract(cmf_payload,'$.hidden') = 0)
    ORDER BY id
""").fetchall()

print(f"Found {len(rows)} non-hidden 3D CMFs\n")
for row_id, payload_str in rows:
    p = json.loads(payload_str) if payload_str else {}
    print(f"  #{row_id:5d}  cert={p.get('certification_level','?'):<20s}  "
          f"source={p.get('source_category','?'):<15s}  "
          f"const={p.get('primary_constant','?')}")

con.close()
