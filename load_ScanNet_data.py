import os
import numpy as np


def load_axis_alignment_mat(meta_file_path):
    metaFile = meta_file_path  # includes axisAlignment info
    assert os.path.exists(metaFile), '%s' % metaFile
    axis_align_matrix = np.identity(4)
    if os.path.isfile(metaFile):
        with open(metaFile) as f:
            lines = f.readlines()

        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                                     for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break

    return axis_align_matrix
