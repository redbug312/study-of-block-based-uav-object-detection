import subprocess as sp
import numpy as np
from tqdm import tqdm

import homographier
import detector
import stitcher
import parser


# TODO: Parse the media info from `$ mediainfo <filepath>`
input_video = {
    'filepath': 'dataset/animals.mp4',
    'frame_shape': (480, 640, 3),
    'frame_count': 16958,
    'frame_rate': 29.970,
}

input_cmd = ['lib/FFmpeg-n4.0/ffmpeg',
             '-flags2', '+export_mvs',
             '-threads', '1',
             '-nostats',
             '-i', input_video['filepath'],
             '-f', 'image2pipe',
             '-pix_fmt', 'rgb24',
             '-vcodec', 'rawvideo', '-']

output_cmd = ['lib/FFmpeg-n4.0/ffmpeg',
              '-y',
              '-f', 'rawvideo',
              '-vcodec', 'rawvideo',
              '-s', '{0[1]}x{0[0]}'.format(input_video['frame_shape']),
              '-pix_fmt', 'rgb24',
              '-r', str(input_video['frame_rate']),
              '-i', '-', '-an',
              '-vcodec', 'libx264',
              '-crf', '20']

output_cmd_to = lambda i: output_cmd + ['output/detect-{:02d}.mp4'.format(i)]

out_params  = {'stdout': sp.PIPE, 'stderr': sp.DEVNULL}
err_params  = {'stdout': sp.DEVNULL, 'stderr': sp.PIPE}
save_params = {'stdout': sp.DEVNULL, 'stderr': sp.PIPE, 'stdin': sp.PIPE}

with sp.Popen(input_cmd, **out_params) as out_proc, sp.Popen(input_cmd, **err_params) as err_proc,\
     sp.Popen(output_cmd_to(0), **save_params) as save_proc, tqdm(total=input_video['frame_count']) as pbar:
    # Unloop first iteration for root frame
    try:
        parser.digest_debug_info_before_mvs(err_proc.stderr)
        image, _ = parser.read_frame_with_mvs(out_proc.stdout, err_proc.stderr, input_video['frame_shape'])
    except IOError as err:
        print(err)
    root_frame = homographier.PanoFrame(image=image)
    homo = homographier.Homographier(root_frame)
    pbar.update()

    adjacent_5frames = [root_frame]
    last_thresh = np.ones(root_frame.img.shape[:2], dtype=np.uint8)

    while True:
        if pbar.n % 1000 == 0:
            # Hot plugged: dangerous but easier to read
            save_proc.__exit__(None, None, None)
            save_proc = sp.Popen(output_cmd_to(pbar.n // 1000), **save_params)

        image, mvs = parser.read_frame_with_mvs(out_proc.stdout, err_proc.stderr, input_video['frame_shape'])
        if image is None:
            break

        next_frame = homographier.PanoFrame(image=image)
        homo.find_homography_to_last(next_frame, mvs=mvs)

        adjacent_5frames.append(next_frame)
        if pbar.n >= 4:
            detected_frame, last_thresh = detector.detect(adjacent_5frames, last_thresh)
            adjacent_5frames = adjacent_5frames[1:]
            try:
                save_proc.stdin.write(detected_frame.img.tostring())
            except IOError as err:
                print(str(err) + '\n' + save_proc.stderr.read().decode('utf-8'))

        pbar.update()

# panorama = stitcher.stitch(homo.pano_frames)
# Image.fromarray(panorama).save('panorama.jpg')
