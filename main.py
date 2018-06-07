from tqdm import tqdm
import cv2

import homographier
import detector
import stitcher


root_frame = homographier.PanoFrame('dataset/EgTest01/frame00000.jpg')
homo = homographier.Homographier(root_frame)

for index in tqdm(range(1, 10)):
    next_frame = homographier.PanoFrame('dataset/EgTest01/frame{:05d}.jpg'.format(index))
    homo.find_homography_to_last(next_frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter('detect.mp4', fourcc, 25, root_frame.img.shape[:2][::-1])
for frame in detector.detect(homo.pano_frames):
    writer.write(frame.img)
writer.release()

# panorama = stitcher.stitch(homo.pano_frames)
# cv2.imwrite('panorama.jpg', panorama)
