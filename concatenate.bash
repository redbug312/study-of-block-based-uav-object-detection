ls output/*.mp4 | sed -e 's/^/file /' | sort -n > order.txt
lib/FFmpeg-n4.0/ffmpeg -f concat -i order.txt -c copy output.mp4
rm order.txt
