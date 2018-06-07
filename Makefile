.PHONY: start debug

DATA := dataset/EgTest01

start:
	python3 main.py
	#xdg-open panorama.jpg

debug:
	python3 -i $(FILE)

play:
	gst-launch-1.0 multifilesrc location="$(DATA)/frame%05d.jpg" index=0 caps="image/jpeg,framerate=25/1" ! jpegdec ! videoconvert ! videorate ! autovideosink
