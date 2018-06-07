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

    def find_homography_to_last(self, new_frame, mvs=None):
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
        # src_pts = np.multiply(np.transpose(np.where(beltrami > 1.005)), 1/16).astype('uint8')
        src_pts = np.array(np.where(beltrami > 1.005), dtype=np.uint8).T // 16
        src_pts = np.multiply(unique_coordinates(src_pts), 16) + 8

        candidates = np.lib.stride_tricks.as_strided(
            ref_frame.img,
            shape=(ref_frame.img.shape[0]-16, ref_frame.img.shape[1]-16, 16, 16, ref_frame.img.shape[2]),
            strides=ref_frame.img.strides[:2]+ref_frame.img.strides
        )

        def find_dst(block):
            # if mvs is not None:
            #     mv = mvs[block[0]//16, block[1]//16]
            #     if not np.isnan(mv[0]):
            #         return mv + block
            y, x = block.tolist()
            h, w = cur_frame.img.shape[:2]
            SAD = lambda candidate: np.sum(cv2.absdiff(cur_frame.img[y:y+16, x:x+16], candidate).astype(np.int))
            ranges = candidates[max(0,y-8):min(y+9,h), max(0,x-8):min(x+9,w)]
            errors = np.vectorize(SAD, signature='(16,16,3)->()')(ranges)
            return np.array([max(0,y-8), max(0,x-8)]).astype(np.float) + \
                   np.unravel_index(errors.argmin(), errors.shape)

        dst_pts = np.full(src_pts.shape, np.nan) if mvs is None \
                  else mvs[src_pts.T[0] // 16, src_pts.T[1] // 16] + src_pts
        known_dst_pts = ~np.isnan(dst_pts)[:, 0]
        if not np.all(known_dst_pts):
            dst_pts[~known_dst_pts] = np.vectorize(find_dst, signature='(2)->(2)')(src_pts[~known_dst_pts])
        cur_frame.H = cv2.findHomography(dst_pts, src_pts, method=cv2.RANSAC)[0]
        # print(cur_frame.H)

        self.pano_frames.append(cur_frame)
        return cur_frame

    def find_all_homography(self, frames):
        for frame in frames:
            self.find_homography_to_last(frame)
        return self.pano_frames
