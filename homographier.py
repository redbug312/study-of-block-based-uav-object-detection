import cv2
import numpy as np
import lib.pano as pano


class PanoFrame(pano.PanoImage):

    def __init__(self, path=None, image=None):
        super().__init__(path)
        self.img = self.img if self.img is not None else image


class Homographier:

    def __init__(self, root_frame):
        root = PanoFrame(image=root_frame.img)
        root.H = np.identity(3)
        self.pano_frames = [root]
        self.frame_shape = np.asarray(root.img.shape)
        self.block_shape = np.asarray(self.frame_shape[:2]) // 16

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

        unique_coordinates = lambda cs: cs[np.unique(np.dot(cs, [[1], [1j]]), return_index=True)[1]]
        src_mbs = unique_coordinates(np.column_stack(np.where(beltrami > 1.005)) // 16)
        at_margin = np.any(src_mbs == [0, 0], axis=1) + np.any(src_mbs == self.block_shape - 1, axis=1)
        src_mbs = src_mbs[~at_margin]

        candidates = np.lib.stride_tricks.as_strided(
            ref_frame.img,
            shape=(self.frame_shape[0]-15, self.frame_shape[1]-15, 16, 16, self.frame_shape[2]),
            strides=ref_frame.img.strides[:2]+ref_frame.img.strides
        )

        def find_mv(pt):
            SAD = lambda candidate: np.sum(cv2.absdiff(cur_frame.img[pt[0]:pt[0]+16, pt[1]:pt[1]+16], candidate))
            mv = np.zeros(2, dtype=np.int)
            # Three step search algorithm
            for step in [4, 2, 1]:
                # Python detects exceeded slicing end
                ranges = candidates[max(0, pt[0]+mv[0]-step) : pt[0]+mv[0]+step+1 : step,
                                    max(0, pt[1]+mv[1]-step) : pt[1]+mv[1]+step+1 : step]
                errors = np.vectorize(SAD, signature='(16,16,3)->()')(ranges)
                mv += step * np.subtract(np.unravel_index(errors.argmin(), errors.shape),
                                         [pt[0] + mv[0] >= step, pt[1] + mv[1] >= step])
            return mv

        src_pts = src_mbs * 16 + 8
        dst_mvs = np.full(src_mbs.shape, np.nan) if mvs is None \
                  else mvs[src_mbs.T[0], src_mbs.T[1]]
        known_dst_mvs = ~np.isnan(dst_mvs)[:, 0]
        if not np.all(known_dst_mvs):
            dst_mvs[~known_dst_mvs] = np.vectorize(find_mv, signature='(2)->(2)')(src_pts[~known_dst_mvs] - 8)

        cur_frame.H = cv2.findHomography(src_pts, src_pts + dst_mvs, method=cv2.RANSAC)[0]
        self.pano_frames.append(cur_frame)
        return cur_frame

    def find_all_homography(self, frames):
        for frame in frames:
            self.find_homography_to_last(frame)
        return self.pano_frames
