#!/usr/bin/env bash
# Build the arXiv source bundle of the paper: main.tex, main.bbl (arXiv does not run BibTeX on refs.bib reliably),
# refs.bib and the figures, in one flat folder with no ../ paths. Output: paper/arxiv/ and paper/arxiv.tar.gz
# (both untracked). The bundle is compiled once more from scratch as a check. Submission is the owner's step.
#   bash tools/make_arxiv_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/../paper"
T=../.venv/bin/tectonic
rm -rf arxiv arxiv.tar.gz && mkdir arxiv
$T --keep-intermediates --outdir arxiv main.tex >/dev/null 2>&1
cp main.tex refs.bib ghost_ratio.png arxiv/
( cd arxiv && find . -type f ! -name main.tex ! -name main.bbl ! -name refs.bib ! -name ghost_ratio.png -delete )
if grep -n '\.\./' arxiv/main.tex; then echo "main.tex still has ../ paths"; exit 1; fi
if grep -n '\\todo{' arxiv/main.tex; then echo "main.tex still has TODOs"; exit 1; fi
( cd arxiv && ../$T main.tex >/dev/null 2>&1 && rm -f main.pdf ) || { echo "the bundle does not compile on its own"; exit 1; }
tar -czf arxiv.tar.gz -C arxiv .
echo "arXiv bundle: paper/arxiv.tar.gz ($(du -h arxiv.tar.gz | cut -f1)); files: $(ls arxiv | tr '\n' ' ')"
