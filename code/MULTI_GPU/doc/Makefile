# Generate documentation from the code.
all: api

api:
	doxygen Doxyfile

clean:
	rm -rf api

publish: clean api
	rsync -qrzPl --perms --chmod=ugo+X,ugo+r -e ssh api www.systap.com:/srv/www/htdocs/systap.com/mapgraph/