import string
import numpy as np


def digest_debug_info_before_mvs(fd_mvs):
    while True:
        stderr = fd_mvs.readline()
        if stderr == b'NEWFRAME;num=0\n':
            break


def read_frame_with_mvs(fd_frame, fd_mvs, frame_shape):
    try:
        stdout = fd_frame.read(np.product(frame_shape))
        image = np.frombuffer(stdout, dtype='uint8').reshape(frame_shape)
    except ValueError:
        return None, None

    mvs_dump = b''
    while True:
        stderr = fd_mvs.readline()
        if stderr[:9] != b'NEWFRAME;' and stderr != b'END;\n':
            mvs_dump = mvs_dump + stderr
        else:
            break
    mvs = parse_mvs_dump(mvs_dump.decode('utf-8'))

    return image, mvs


def parse_mvs_dump(mvs_dump):
    mvs = np.full((30, 40, 2), np.nan)
    mvs_dump = iter(mvs_dump.split('\n'))
    filter_letters = str.maketrans('', '', string.ascii_letters + '=')
    parse_digits = lambda l: l.translate(filter_letters).split(';')

    while True:
        try:
            line = next(mvs_dump)
            assert(line[:11] == 'MACROBLOCK;')
        except (StopIteration, AssertionError):
            break
        else:
            mbx, mby = map(int, parse_digits(line)[1:3])

        try:
            line = next(mvs_dump)
            assert(line[:12] == 'VECTORBLOCK;')
        except (StopIteration, AssertionError):
            break
        else:
            direction, mv_count = map(int, parse_digits(line)[1:3])
            if direction == 0:
                mvs[mby, mbx] = [0, 0]

        for i in range(int(mv_count)):
            try:
                line = next(mvs_dump)
                assert(line[:7] == 'VECTOR;')
            except (StopIteration, AssertionError):
                break
            else:
                if direction == 1:
                    continue
                mx, my = map(float, parse_digits(line)[3:5])
                mvs[mby, mbx] += np.divide([my, mx], mv_count * 4)
    return mvs
