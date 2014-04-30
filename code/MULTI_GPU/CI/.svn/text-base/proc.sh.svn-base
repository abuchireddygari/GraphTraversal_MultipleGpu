#!/bin/bash
# Directory containing the mapgraph executables.
BIN_DIR=../Algorithms
# Directory containing the graph datasets to be processed.
DATA_DIR=/data/xdata/GraphDatasets
# Directory containing the matching files of random vertices for the initial frontier.
RAND_DIR=/data/xdata/starting_vertices

# The algorithm type (accepts sampled starting vertices).
TYPES="SSSP"; # BFS SSSP PR CC BC

##
# NOTE: The .mtx data files MUST have the correct #of rows, cols, and edges (no -1, -1 in the header row of the file).
##

# Symmetric matrices.
#SYM_DATA="ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 kron_g500-logn20 bitcoin"
SYM_DATA="ak2010"

# Generalized (aka non-symmetric) matrices.
#GEN_DATA="webbase-1M wikipedia-20070206 twitter_d1";# traceroute"

# Set of all graphs to be run.
DATA="$SYM_DATA $GEN_DATA"

# Suffix for the matrix files.
suffix=".mtx"

# The #of random vertices to use (first N, up to how every many are in the file).
#N=100
N=1

#### end changable section

nerr=0
for TYPE in $TYPES
  do
  for graph in $DATA
    do 
          # The name of the executable.
    exec="$BIN_DIR/$TYPE/$TYPE"
          # remove old log file for this graph.
    rm -f out.$TYPE.$graph.log
	    # 1 iff this is an undirected graph (aka symmetric). 0 otherwise.
    export directedGraph=`echo $SYM_DATA | grep -c "$graph"`
    export vertices=`head -n $N "$RAND_DIR/$graph.rand_indices"`
	    #echo "graph=[$graph], directedGraph=[$directedGraph]"
    for src in ${vertices}
      do
      cmd="$exec -g $DATA_DIR/$graph$suffix foo -p src=$src directed=$directedGraph"
      echo "Running: $cmd"
      $cmd >> out.$TYPE.$graph.log 2>&1
      if [[ $rc != 0 ]] ; then
	  echo "Error: $cmd"
	  nerr=`expr $nerr + 1`
      fi
    done
  done
done
if [[ $nerr != 0 ]] ; then
    echo "There were $nerr errors."
    exit 1
fi
echo "Success."
exit 0
