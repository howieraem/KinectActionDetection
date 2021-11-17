"""Loads and serializes skeleton dataset from the original text format."""
from dataset.skeleton_continuous import *
from global_configs import DatasetProtocol


if __name__ == '__main__':
    """Modify class and the corresponding data directory to serialize different datasets."""
    ds = SkeletonDatasetPKUMMDv1('D:/METR4901/dataset/PKU-MMD',
                                 protocol=DatasetProtocol.CROSS_SUBJECT,
                                 is_translated=True, is_edged=False, is_rotated=False)
    ds.serialize('./dataset/')
