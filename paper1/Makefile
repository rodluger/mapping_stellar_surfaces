.PHONY: default, clean, deepclean

MISC = $(patsubst misc/%,%,$(wildcard misc/*))

default:
	cp misc/* .
	python -c "from build import build; build()"
	rm $(MISC)
clean:
	python -c "from build import clean; clean()"

deepclean:
	python -c "from build import clean; clean(remove_data=True)"
