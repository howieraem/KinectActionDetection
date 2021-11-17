from utils.misc import deprecated


def world_coords_to_image(coords_3d, focal_length, width, height):
    assert len(coords_3d) == 3, 'Dimension should be 3.'
    u = int(round(focal_length * coords_3d[0] / coords_3d[2] + width / 2))
    v = int(round(focal_length * (-1 * coords_3d[1]) / coords_3d[2] + height / 2))
    return u, v


@deprecated
def world_coords_to_image_cad60(coords_3d):
    """
    Formula given by Cornell 60 Dataset which seems to be wrong for a 640*480 image.
    :param coords_3d:
    :return:
    """
    u = 0.0976862095248 * coords_3d[0] - 0.0006444357104 * coords_3d[1] + 0.0015715946682 * coords_3d[2] \
        + 156.8584456124928
    v = 0.0002153447766 * coords_3d[0] - 0.1184874093530 * coords_3d[1] - 0.0022134485957 * coords_3d[2] \
        + 125.5357201011431
    return int(round(u)), int(round(v))


def reshape_2d_sequence_to_3d(seq):
    assert seq.shape[1] % 3 == 0., 'Not all points in the dimension are 3-dimensional.'
    return seq.reshape(seq.shape[0], int(seq.shape[1] / 3), 3)
