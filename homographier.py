import cv2
import numpy as np
import lib.pano as pano


def evaluate_PSNR(ref_frame, cur_frame):
    dims = ref_frame.img.shape[:2][::-1]
    H_inv = np.linalg.inv(cur_frame.H)
    mask = cv2.warpPerspective(np.ones(dims), H_inv, dims) == 1

    img = cv2.warpPerspective(cur_frame.img, H_inv, dims)
    ref = ref_frame.img

    err = img.astype(np.float) - ref.astype(np.float)
    mse = np.sum(err[mask] ** 2) / np.sum(mask)
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


class PanoFrame(pano.PanoImage):

    def __init__(self, path=None, image=None, SIFT=False):
        super().__init__(path)
        self.img = self.img if self.img is not None else image
        if SIFT:
            self.keypts, self.descs = self.extract_SIFT_features()

    def extract_SIFT_features(self):
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=1024, edgeThreshold=40)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        keypts, descs = detector.detectAndCompute(img_gray, None)
        return keypts, descs.astype(np.float32)

    def extract_PF_blocks(self, block_shape):
        Ix = np.gradient(self.img / 255, axis=1)
        Iy = np.gradient(self.img / 255, axis=0)
        Ixx = np.sum(np.multiply(Ix, Ix), axis=2)
        Ixy = np.sum(np.multiply(Ix, Iy), axis=2)
        Iyy = np.sum(np.multiply(Iy, Iy), axis=2)

        # beltrami = lambda st: 1 + np.linalg.det(st) + np.trace(st)
        beltrami = 1 + np.multiply(Ixx, Iyy) - np.multiply(Ixy, Ixy) + Ixx + Iyy

        unique_coordinates = lambda cs: cs[np.unique(np.dot(cs, [[1], [1j]]), return_index=True)[1]]
        src_mbs = unique_coordinates(np.column_stack(np.where(beltrami > 1.01)) // 16)
        at_margin = np.any(src_mbs == [0, 0], axis=1) + np.any(src_mbs == block_shape - 1, axis=1)
        src_mbs = src_mbs[~at_margin]
        return src_mbs


class Homographier:

    def __init__(self, root_frame):
        root_frame.H = np.identity(3)
        self.last_frame = root_frame
        self.frame_shape = np.asarray(root_frame.img.shape)
        self.block_shape = np.asarray(self.frame_shape[:2]) // 16

    def match_homography_to_last(self, new_frame):
        ref_frame = self.last_frame
        cur_frame = new_frame

        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
        matches = matcher.knnMatch(cur_frame.descs, ref_frame.descs, k=2)
        best_matches = [m1 for m1, m2 in matches if m1.distance < 0.8 * m2.distance]

        # all matches
        src_pts = np.array([cur_frame.keypts[m.queryIdx].pt for m in best_matches])
        dst_pts = np.array([ref_frame.keypts[m.trainIdx].pt for m in best_matches])

        # get feature correspondences
        cur_frame.H = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)[0]
        self.last_frame = cur_frame
        return cur_frame

    def find_homography_to_last(self, new_frame, mvs=None):
        ref_frame = self.last_frame
        cur_frame = new_frame

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

        src_mbs = new_frame.extract_PF_blocks(self.block_shape)
        src_pts = src_mbs * 16 + 8
        dst_mvs = np.full(src_mbs.shape, np.nan) if mvs is None else \
                  mvs[src_mbs.T[0], src_mbs.T[1]]
        known_dst_mvs = ~np.isnan(dst_mvs)[:, 0]
        if not np.all(known_dst_mvs):
            dst_mvs[~known_dst_mvs] = np.vectorize(find_mv, signature='(2)->(2)')(src_pts[~known_dst_mvs] - 8)

        cur_frame.H = cv2.findHomography(src_pts, src_pts + dst_mvs, method=cv2.RANSAC)[0]
        self.last_frame = cur_frame
        return cur_frame
