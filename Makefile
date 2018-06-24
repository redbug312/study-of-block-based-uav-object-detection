.PHONY: start debug

start:
	mkdir -p output/
	python3 main.py
	bash concatenate.bash

debug:
	python3 -i $(FILE)
