# The software release version.
ver=0.3.2
version=mapgraph.${ver}.tgz
release.dir=releases

# The list of directories to operate on.  Could also be defined using
# wildcards.

#SUBDIRS = Algorithms/InterestingSubgraph Algorithms/BFS Algorithms/CC Algorithms/SSSP Algorithms/PageRank
SUBDIRS = moderngpu2/src Algorithms/BFS Algorithms/CC Algorithms/SSSP Algorithms/PageRank

# Setup mock targets. There will be one per subdirectory. Note the
# ".all" or ".clean" extension. This will trigger the parameterized
# rules below.
#

ALL = $(foreach DIR,$(SUBDIRS),$(DIR).all)
CLEAN = $(foreach DIR,$(SUBDIRS),$(DIR).clean)

# Define top-level targets.
#

all: $(ALL) doc.all

clean: $(CLEAN) doc.clean

realclean: clean realclean.create

realclean.create:
	rm -rf ${release.dir}

# Parameterized implementation of the mock targets, invoked by
# top-level targets for each subdirectory.
#

%.all:
	$(MAKE) -C $*

%.clean:
	$(MAKE) -C $* clean

# Note: If there are dependency orders, declare them here. This way
# some things will be built before others.

#foo: baz

# Generates documentation from the code.
doc: doc.create

doc.create:
	$(MAKE) -C doc

release: clean release.create

release.create:
	-mkdir ${release.dir}
	-rm -f ${release.dir}/${version}
	tar --exclude .svn --exclude releases -cvz -f ${release.dir}/${version} .
