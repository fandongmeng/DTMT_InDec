./delbpe.sh valid_src.out
./multi-bleu.perl -lc valid_trg < valid_src.out.delbpe > $1
