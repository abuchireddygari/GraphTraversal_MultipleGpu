GRAPH_DIR = ../largePerformanceGraphs

#Can restrict this to the graphs you have downloaded. Largest is kron_g500-logn21.
GRAPHS = ak2010 belgium_osm delaunay_n13 coAuthorsDBLP delaunay_n21 kron_g500-logn21 soc-LiveJournal1 webbase-1M
#GRAPHS = ak2010 belgium_osm delaunay_n13 coAuthorsDBLP delaunay_n21 soc-LiveJournal1 webbase-1M

#which algorithms to check
ALGORITHMS = pagerank sssp bfs cc
ALGORITHMS = cc

#Match this to number of cores on your system
POWERGRAPH_OPTS = --ncpus 4

GOLD_BINARIES = $(foreach x,$(ALGORITHMS),../PowerGraphReferenceImplementations/$x.x)
GOLD_FILES = $(foreach P,$(ALGORITHMS),$(foreach G,$(GRAPHS),$G.$P.gold))
TEST_FILES = $(foreach P,$(ALGORITHMS),$(foreach G,$(GRAPHS),$G.$P.test))

REGRESSIONS = $(foreach P,$(ALGORITHMS),$(foreach G,$(GRAPHS),$G.$P.pass))

all: regress

gold: $(GOLD_BINARIES) $(GOLD_FILES)

test: $(TEST_FILES)

regress: $(REGRESSIONS)

define MAKEGOLD
  rm -f __$(1).$(2).out* ;
  ../PowerGraphReferenceImplementations/$(2).x --graph $(GRAPH_DIR)/$(1)/$(1).mtx --graph_opts ingress=batch --save __$(1).$(2).out | awk '/Finished Running/{print $$5}' > $(1).$(2).timing ; 
  cat __$(1).$(2).out* | sort -n > $(1).$(2).gold ;
  rm -f __$(1).$(2).out* ;
endef

# TODO Add reference implementation and comparison for CC. See the pagerank example below.
%.cc.test: ../simpleCC
	../simpleCC $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.cc.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.pagerank.gold: ../PowerGraphReferenceImplementations/pagerank.x
	$(call MAKEGOLD,$*,pagerank)

%.pagerank.test: ../simplePageRank
	../simplePageRank $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.pagerank.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.pagerank.pass: %.pagerank.test %.pagerank.gold
	./checkPageRank.py $*.pagerank.test $*.pagerank.gold
	touch $*.pagerank.pass

%.sssp.gold: ../PowerGraphReferenceImplementations/sssp.x
	$(call MAKEGOLD,$*,sssp)

%.sssp.test: ../simpleSSSP
	../simpleSSSP $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.sssp.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.sssp.pass: %.sssp.test %.sssp.gold
	diff -q $*.sssp.test $*.sssp.gold
	touch $*.sssp.pass

%.bfs.gold: ../PowerGraphReferenceImplementations/bfs.x
	$(call MAKEGOLD,$*,bfs)

%.bfs.test: ../simpleBFS
	../simpleBFS $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.bfs.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.bfs.pass: %.bfs.test %.bfs.gold
	diff -q $*.bfs.test $*.bfs.gold
	touch $*.bfs.pass

clean:
	rm -f *.test *.timing_gpu *.pass

clean-gold:
	rm -f *.gold *.timing

clean-all: clean clean-gold
