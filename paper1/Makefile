.PHONY: default, clean, deepclean

MISC = $(patsubst misc/%,%,$(wildcard misc/*))

default:
	cp misc/* .
	python -c "from build import build; build()"
	rm -f $(MISC)
clean:
	cp misc/* .
	python -c "from build import clean; clean()"
	rm -f $(MISC)

deepclean:
	cp misc/* .
	python -c "from build import clean; clean(remove_data=True)"
	rm -f $(MISC)