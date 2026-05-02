# Always compile from the directory that contains the main .tex file.
# Without this, `latexmk submission/final/foo.tex` run from the repo root
# skips ./submission/final/.latexmkrc and drops .aux/.log/.synctex.gz next to the source.
$do_cd = 1;
