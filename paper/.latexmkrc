# latexmk configuration — pdflatex + bibtex, build artifacts in build/
$pdf_mode = 1;                       # produce a PDF via pdflatex
$out_dir  = 'build';                 # keep .aux/.log/.pdf out of the source tree
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;                     # run bibtex when a .bib is present
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml bbl';
