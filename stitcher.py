import numpy as np
import lib.pano as pano


def stitch(pano_frames):
    return pano.registerPanoramas([pano_frames], 'planar')[0].astype(np.uint8)
