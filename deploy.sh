#!/bin/bash
# Run this once from the cmf_atlas directory to push all pending fixes
set -e
cd "$(dirname "$0")"
git add api.py frontend/entry.html frontend/explorer.html
git commit -m "fix: 3D walk k_start, _compute_matrices b_kn, A_certified badge, 3D cert panel"
git push origin main
zip -r frontend_cpanel.zip frontend/ -x "*.DS_Store"
echo ""
echo "✓ Pushed to GitHub. Now:"
echo "  1. Trigger Railway redeploy"
echo "  2. Upload frontend_cpanel.zip to cPanel"
