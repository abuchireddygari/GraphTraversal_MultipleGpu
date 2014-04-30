#!/bin/bash

BIN_DIR=..
DATA_DIR=/data/xdata/GraphDatasets
#TYPES="BFS SSSP cc PR SSSP-nogather SSSP-RC BFS-RC"
TYPES="SSSP"
SYM_DATA="ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 kron_g500-logn20 bitcoin"
GEN_DATA="webbase-1M wikipedia-20070206 twitter_d1";# traceroute"
DATA="$SYM_DATA $GEN_DATA"

for data in $DATA
do
   for type in $TYPES
   do
     EXEC="$BIN_DIR/simple$type"
     $
     if [[ -a "out.$type.$data.kepler.log" ]]; then
	 echo out.$type.$data.kepler.log
          #grep "Kernel" out.$type.$data.kepler.log > out.$type.$data.kernel.timing
	 awk '/M-Edges/{print $4}' out.$type.$data.kepler.log > out.$type.$data.00_M_edges.timing
	 awk '/Total/{print $3}' out.$type.$data.kepler.log > out.$type.$data.01_iter.timing
	 awk '/Kernel/{print $4}' out.$type.$data.kepler.log > out.$type.$data.02_kernel.timing
	 awk '/retval/{print $2}' out.$type.$data.kepler.log > out.$type.$data.03_retval.timing
	 awk '/Wall/{print $4}' out.$type.$data.kepler.log > out.$type.$data.04_wall.timing
          #awk '/Number/{print $5}' out.$type.$data.kepler.log > out.$type.$data.05_comp.timing
	 paste `ls out.$type.$data.*.timing` > out.$type.$data.kepler.txt
     fi
   done
done
# clean up tmp files
rm -f out.*.timing
