# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WHUDataset(CustomDataset):
    """WHU dataset.

    In segmentation map annotation for WHU, 0 stands for background, which
    is not included in CLASSES. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.tif' and ``seg_map_suffix`` is fixed to
    '.tif'.
    """
    CLASSES = ('non-building', 'building')

    PALETTE = [[0, 0, 0], [0, 0, 255]]

    def __init__(self, **kwargs):
        super(WHUDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
        self.label_map = {0: 0, 255: 1}  # 0 non-building, 1 building
        self.custom_classes = True
