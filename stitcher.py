import numpy as np
import lib.pano as pano


def stitch(pano_frames):
    H = np.identity(3)
    for frame in pano_frames:
        H = np.dot(H, frame.H)
        frame.H = np.linalg.inv(H)
    return pano.registerPanoramas([pano_frames], 'planar')[0].astype(np.uint8)
