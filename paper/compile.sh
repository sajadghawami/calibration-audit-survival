#!/bin/bash
# Compile both paper versions in isolation using temp directories.
# Usage: ./compile.sh

set -e
export PATH="/usr/local/texlive/2026basic/bin/universal-darwin:$PATH"
DIR="$(cd "$(dirname "$0")" && pwd)"

compile_one() {
    local TEX="$1"
    local BASE="${TEX%.tex}"
    local TMPDIR=$(mktemp -d)

    # Copy source files
    cp "$DIR/$TEX" "$DIR/references.bib" "$DIR/jmlr.cls" "$DIR/jmlrutils.sty" "$TMPDIR/"
    ln -s "$DIR/../figures" "$TMPDIR/figures"

    # Compile
    cd "$TMPDIR"
    pdflatex -interaction=nonstopmode "$TEX" > /dev/null 2>&1
    bibtex "$BASE" > /dev/null 2>&1
    pdflatex -interaction=nonstopmode "$TEX" > /dev/null 2>&1
    pdflatex -interaction=nonstopmode "$TEX" > /dev/null 2>&1

    # Copy PDF back
    cp "$TMPDIR/$BASE.pdf" "$DIR/"

    # For arxiv: also copy bbl
    if [ "$BASE" = "paper_arxiv" ]; then
        cp "$TMPDIR/$BASE.bbl" "$DIR/arxiv_upload/"
        cp "$DIR/$TEX" "$DIR/arxiv_upload/"
        cp "$DIR/references.bib" "$DIR/arxiv_upload/"
    fi

    # Cleanup
    rm -rf "$TMPDIR"
    echo "  $BASE.pdf compiled OK"
}

echo "Compiling papers..."
compile_one "paper_mlhc2026.tex"
compile_one "paper_arxiv.tex"

# Verify
echo ""
echo "Verification:"
grep -o "Anonymous\|Sajad" "$DIR/paper_mlhc2026.tex" | head -1 | xargs -I{} echo "  MLHC author: {}"
grep -o "Anonymous\|Sajad" "$DIR/paper_arxiv.tex" | head -1 | xargs -I{} echo "  arXiv author: {}"
echo "Done."
