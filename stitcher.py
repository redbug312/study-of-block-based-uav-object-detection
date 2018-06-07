import lib.pano as pano


def stitch(pano_frames):
    return pano.registerPanoramas([pano_frames], 'planar', 'linear')[0]
