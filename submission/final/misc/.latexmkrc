# PDF + SyncTeX stay next to the .tex ($out_dir = '.'). All other engine outputs
# (.aux, .log, .fls, .xdv, .fdb_latexmk, …) go under ./misc/ (TeX Live: $emulate_aux).
# SyncTeX must stay beside the PDF for VS Code’s LaTeX Workshop; do not set $out2_dir.
$aux_dir  = './misc';
$out_dir  = '.';
$pdf_mode = 5;       # xelatex → xdv → pdf (avoids accidental `latex` → .dvi)
$emulate_aux = 1;
