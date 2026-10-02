#!/usr/bin/env bash
# Assemble the static demo site (GitHub Pages) into <out>: the page and its module from web/, the fallback recording
# from docs/, the model and starting histories from <assets> (the files of the web-demo release, checked against
# configs/web_demo_assets.sha256), and ONNX Runtime Web vendored from npm (the version in configs/web_demo.yaml; npm
# verifies the package against the registry's integrity hash), so the page loads no third-party script.
#   bash tools/build_site.sh <out_dir> <assets_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out dir}; ASSETS=${2:?assets dir}
VER=$(grep '^ort_version:' configs/web_demo.yaml | awk '{print $2}')
rm -rf "$OUT" && mkdir -p "$OUT/ort"
( cd "$ASSETS" && shasum -a 256 -c "$OLDPWD/configs/web_demo_assets.sha256" )
cp "$ASSETS"/model_fp16.onnx "$ASSETS"/starts.bin "$ASSETS"/starts.json "$OUT"/
cp web/index.html web/pacworld.js "$OUT"/
cp docs/demo.gif "$OUT"/
TMP=$(mktemp -d)
( cd "$TMP" && npm pack --silent "onnxruntime-web@$VER" >/dev/null && tar -xzf onnxruntime-web-*.tgz )
cp "$TMP"/package/dist/ort.webgpu.min.mjs "$TMP"/package/dist/ort-wasm-simd-threaded.asyncify.* "$OUT"/ort/
cp "$TMP"/package/LICENSE "$OUT"/ort/LICENSE 2>/dev/null || true
rm -rf "$TMP"
# the page's module imports the vendored runtime instead of the CDN
sed -i.bak "s#https://cdn.jsdelivr.net/npm/onnxruntime-web@[0-9.]*/dist/ort.webgpu.min.mjs#./ort/ort.webgpu.min.mjs#" "$OUT"/pacworld.js && rm "$OUT"/pacworld.js.bak
grep -q './ort/ort.webgpu.min.mjs' "$OUT"/pacworld.js || { echo "runtime import was not rewritten"; exit 1; }
touch "$OUT"/.nojekyll
echo "site in $OUT: $(du -sh "$OUT" | cut -f1); files: $(ls "$OUT" "$OUT"/ort | tr '\n' ' ')"
