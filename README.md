# Study of Block-based UAV-Video Moving Object Detection

Thid is demo part of team 14 presentation, in the course Video Communications (National Taiwan University, 2018 Spring).

> [slides](https://slides.com/redbug312/video-communication-demo#/)

## Build

1. Download FFmpeg (version n4.0) in `lib/` and apply patch
    ```bash
    $ git clone -b n4.0 https://github.com/FFmpeg/FFmpeg.git lib/FFmpeg-n4.0
    $ patch -p0 < lib/ffmpeg-patch
    ```
2. Build FFmpeg with dependent packages
    ```bash
    $ sudo apt install yasm libx264-dev
    $ cd lib/FFmpeg-n4.0
    $ ./configure --prefix=`pwd` --enable-gpl --enable-libx264
    $ make
    $ cd -
    ```
3. Run the program, open the output video
    ```bash
    $ sudo apt install python3-pip
    $ pip3 install numpy opencv-python opencv-contrib-python tqdm
    $ make start
    $ xdg-open output.mp4
    ```

The program will take `dataset/animals_short.mp4` and `output.mp4` as its input and output respectively.
To change the input source video, you have to change the `input_video` variable in `main.py`.

## Structure

```
             ╔═════ main.py ═════╗
▶ frame ╾─┐  ║                   ║
          ├──╫───╼ parser.py     ║
▶  MVs  ╾─┘  ║         ┊         ║
 (1-thread)  ║  homographier.py  ║
             ║         ┊         ║
             ║    detector.py ╾──╫──╼ output ▶
             ║                   ║
             ╚═══════════════════╝
```

- `parser.py`: Parse frame and motion vector informations from `FFmpeg`. It's based on [python-mv](https://github.com/runeksvendsen/python-mv) with some alters for ffmpeg n4.0.
- `homographier.py`: Implement algorithms for estimating homography in frames of UAV-video.
- `detector.py`: Use difference method to detect moving objects.

## Credits

1. Referenced papers
    - A. Hafiane, K. Palaniappan and G. Seetharaman, "[UAV-Video Registration Using Block-Based Features](https://ieeexplore.ieee.org/document/4779192)," IGARSS 2008 - 2008 IEEE International Geoscience and Remote Sensing Symposium, Boston, MA, 2008, pp. II-1104-II-1107.
    - Q. Wei, S. Lao and L. Bai, "[Panorama Stitching, Moving Object Detection and Tracking in UAV Videos](https://ieeexplore.ieee.org/document/8123587/)," 2017 International Conference on Vision, Image and Signal Processing (ICVISP), Osaka, 2017, pp. 46-50
2. Used libraries
    - [FFmpeg](https://ffmpeg.org/), a collection of libraries and tools to process multimedia content such as audio, video, subtitles and related metadata.
    - [Numpy](http://www.numpy.org/), the fundamental package for scientific computing with Python.
    - [OpenCV](https://opencv.org/), the open source computer vision library.
    - [radiant](https://github.com/fzliu/radiant), an advanced image processing library, written primarily in Python.
    - [tqdm](https://github.com/tqdm/tqdm): a fast, extensible progress bar for Python and CLI.
3. Test videos
    - [Drone chasing animals](https://www.youtube.com/watch?v=Fkp7MAF7JzQ) | YouTube
    - [VIVID Tracking Evaluation Web Site](http://vision.cse.psu.edu/data/vividEval/main.html)
