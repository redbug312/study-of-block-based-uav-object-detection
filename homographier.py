import cv2
import numpy as np
import lib.pano as pano


class PanoFrame(pano.PanoImage):

    def __init__(self, path=None, image=None):
        super().__init__(path)
        self.img = self.img if self.img is not None else image


class Homographier:

    def __init__(self, root_frame):
        self.pano_frames = list()
        root = PanoFrame(image=root_frame.img)
        root.H = np.identity(3)
        self.pano_frames.append(root)

    def find_homography_to_last(self, new_frame):
        ref_frame = self.pano_frames[-1]
        cur_frame = new_frame

        Ix = np.gradient(cur_frame.img / 255, axis=1)
        Iy = np.gradient(cur_frame.img / 255, axis=0)
        Ixx = np.sum(np.multiply(Ix, Ix), axis=2)
        Ixy = np.sum(np.multiply(Ix, Iy), axis=2)
        Iyy = np.sum(np.multiply(Iy, Iy), axis=2)

        # beltrami = lambda st: 1 + np.linalg.det(st) + np.trace(st)
        beltrami = 1 + np.multiply(Ixx, Iyy) - np.multiply(Ixy, Ixy) + Ixx + Iyy
        np.set_printoptions(threshold=np.nan)

        unique_coordinates = lambda cs: cs[np.unique(np.dot(cs, [[1], [1j]]), return_index=True)[1]]
        src_pts = np.multiply(np.transpose(np.where(beltrami > 1.005)), 1/16).astype('uint8')
        src_pts = np.multiply(unique_coordinates(src_pts), 16)

        candidates = np.lib.stride_tricks.as_strided(
            ref_frame.img,
            shape=(ref_frame.img.shape[0]-16, ref_frame.img.shape[1]-16, 16, 16, ref_frame.img.shape[2]),
            strides=ref_frame.img.strides[:2]+ref_frame.img.strides
        )

        def find_dst(image, block, candidates):
            y, x = block.tolist()
            h, w = image.shape[:2]
            SAD = lambda candidate: np.sum(cv2.absdiff(image[y:y+16, x:x+16], candidate).astype('int'))
            ranges = candidates[max(0,y-8):min(y+9,h), max(0,x-8):min(x+9,w)]
            errors = np.vectorize(SAD, signature='(16,16,3)->()')(ranges)
            return np.array([max(0,y-8), max(0,x-8)]) + \
                   np.unravel_index(errors.argmin(), errors.shape)

        dst_pts = np.vectorize(lambda b: find_dst(cur_frame.img, b, candidates), signature='(2)->(2)')(src_pts)
        cur_frame.H = cv2.findHomography(dst_pts, src_pts, method=cv2.RANSAC)[0]

        self.pano_frames.append(cur_frame)
        return cur_frame

    def find_all_homography(self, frames):
        for frame in frames:
            self.find_homography_to_last(frame)
        return self.pano_frames
