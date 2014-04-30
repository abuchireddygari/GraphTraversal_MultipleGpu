#!/bin/bash
#ZHI_BIN=/home/zhisong/Dropbox/systap/XDATA/mpgraph-code/branches/SSSP-scattercheckandonlyscatter
ZHI_BIN=/home/zhisong/Dropbox/systap/XDATA/mpgraph-code/branches/BFS-kepler
DATA_DIR=/home/zhisong/workspace/GraphDatasets
RAND_DIR=/home/zhisong/Dropbox/XDATA-PROJECT/Software/mpgraph/first-year-final-report/starting_vertices

# The algorithm type (accepts sampled starting vertices).
type="BFS"

# The name of the executable.
exec="$ZHI_BIN/simpleSSSP"

##
# NOTE: The .mtx data files MUST have the correct #of rows, cols, and edges (no -1, -1 in the file).
##

# Symmetric matrices.
#
# Unused: kron_g500_logn21
#SYM_DATA="ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 kron_g500-logn20"
SYM_DATA="delaunay_n21 ak2010 belgium_osm coAuthorsDBLP delaunay_n13 kron_g500-logn20"


# Generalized (aka non-symmetric) matrices.
#
# Unused: soc_LiveJournal1 
#GEN_DATA="webbase-1M wikipedia-20070206 twitter_d1 traceroute"
#GEN_DATA="twitter_d1 traceroute bitcoin"
GEN_DATA="bitcoin twitter_d1 webbase-1M wikipedia-20070206 traceroute"
# Set of all graphs to be run.
export GRAPHS=`echo $GEN_DATA $SYM_DATA`

# Suffix for the matrix files.
suffix=".mtx"

# The #of random vertices to use (first N, up to how every many are in the file).
N=100

#### end changable section

#for data in $SYM_DATA
#do
    for graph in $GRAPHS
       do
          # remove old log file for this graph.
          #rm -f zhi.$type.$graph.log
	  # 1 iff this is an undirected graph (aka symmetric). 0 otherwise.
	  export undirectedGraph=`echo $SYM_DATA | grep -c "$graph"`
          export vertices=`head -n $N "$RAND_DIR/$graph.rand_indices"`
	  #echo "graph=[$graph], directedGraph=[$directedGraph]"
          for src in ${vertices}
          do
	   cmd="$exec $DATA_DIR/$graph$suffix foo $src $undirectedGraph 2 2.0"
           echo "Running: $cmd"
           $cmd >> zhi.$type.$graph.fermi.log 2>&1
          done
       done
#done
