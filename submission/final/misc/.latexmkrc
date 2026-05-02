# Keep *all* LaTeX build artifacts out of submission/final/.
# We build into a hidden folder, and LaTeX Workshop copies only the PDF back.
$aux_dir  = './.latex-out';
$out_dir  = './.latex-out';
$pdf_mode = 5;       # xelatex → xdv → pdf (avoids accidental `latex` → .dvi)
$emulate_aux = 1;
