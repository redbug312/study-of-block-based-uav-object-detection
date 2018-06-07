import cv2
import subprocess as sp
from tqdm import tqdm

import homographier
import detector
import stitcher
import parser


cmd = ['lib/FFmpeg-n4.0/ffmpeg',
       '-flags2', '+export_mvs',
       '-threads', '1',
       '-nostats',
       '-i', '/tmp/a1.mp4',
       '-f', 'image2pipe',
       '-pix_fmt', 'rgb24',
       '-vcodec', 'rawvideo', '-']

out_params = {'stdout': sp.PIPE, 'stderr': sp.DEVNULL}
err_params = {'stdout': sp.DEVNULL, 'stderr': sp.PIPE}

frame_shape = (480, 640, 3)
frame_count = 100

with sp.Popen(cmd, **out_params) as out_proc, sp.Popen(cmd, **err_params) as err_proc,\
     tqdm(total=frame_count) as pbar:
    # Unloop first iteration for root frame
    parser.digest_debug_info_before_mvs(err_proc.stderr)
    image, _ = parser.read_frame_with_mvs(out_proc.stdout, err_proc.stderr, frame_shape)
    root_frame = homographier.PanoFrame(image=image)
    homo = homographier.Homographier(root_frame)
    pbar.update()

    while True:
        image, mvs = parser.read_frame_with_mvs(out_proc.stdout, err_proc.stderr, frame_shape)
        if image is None:
            break

        next_frame = homographier.PanoFrame(image=image)
        homo.find_homography_to_last(next_frame, mvs=mvs)
        pbar.update()

# root_frame = homographier.PanoFrame('dataset/EgTest01/frame00000.jpg')
# homo = homographier.Homographier(root_frame)

# for index in tqdm(range(1, 100)):
#     next_frame = homographier.PanoFrame('dataset/EgTest01/frame{:05d}.jpg'.format(index))
#     homo.find_homography_to_last(next_frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter('detect.mp4', fourcc, 25, root_frame.img.shape[:2][::-1])
for frame in detector.detect(homo.pano_frames):
    writer.write(frame.img)
writer.release()

# panorama = stitcher.stitch(homo.pano_frames)
# cv2.imwrite('panorama.jpg', panorama)
