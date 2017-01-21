# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
from fast_rcnn.config import cfg

class bib_detect(imdb):
    def __init__(self, source_folder, exclude_mappings):
        imdb.__init__(self, 'bib_detect_' + os.path.basename(source_folder))
        self._source_folder = source_folder
        self._exclude_mappings = exclude_mappings
        self._classes = ('__background__', # always index 0
                         'bib')
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        # Bib specific config options
        self.config = {'cleanup'     : False,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._source_folder), \
                'Path does not exist: {}'.format(self._source_folder)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        label = index.split('-')[0]
        image_path = os.path.join(self._source_folder, 
                label,
                index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        List all files in the folder
        """
        excluded = set()
        if self._exclude_mappings:
            with open(self._exclude_mappings) as f:
                mapping = json.load(f)
                for _type in mapping: # training or testing
                    for fn in mapping[_type]:
                        excluded.add(os.path.basename(mapping[_type][fn]))
        print excluded
        image_index = []
        for root, dirs, files in os.walk(self._source_folder):
            for name in files:
                if name not in excluded:
                    image_index.append(name.split('.')[0])
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'bib_detect', self._label)