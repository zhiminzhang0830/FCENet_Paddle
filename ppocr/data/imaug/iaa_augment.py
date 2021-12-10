# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import imgaug
import imgaug.augmenters as iaa
import random
import cv2

class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{
                k: self.to_tuple_if_list(v)
                for k, v in args['args'].items()
            })
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment():
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }, {
                'type': 'Affine',
                'args': {
                    'rotate': [-10, 10]
                }
            }, {
                'type': 'Resize',
                'args': {
                    'size': [0.5, 3]
                }
            }]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data['image']
        shape = image.shape

        # import time
        # from PIL import Image, ImageDraw
        # import os
        # cur_time = time.time()
        # base_name = os.path.split(data['img_path'])[-1]
        # image = data['image']
        # text_polys = data['polys']
        # for box in text_polys:
        #     pts = np.array(box, np.int32)
        #     cv2.polylines(image, [pts], True, (0, 0, 255), thickness=3)
        # cv2.imwrite('tmp/{}_{}.jpg'.format(base_name[:-4], cur_time), image)
        
        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        
            # image = data['image']
            # text_polys = data['polys']
            # for box in text_polys:
            #     pts = np.array(box, np.int32)
            #     cv2.polylines(image, [pts], True, (0, 0, 255), thickness=3)
            # cv2.imwrite('tmp/{}_{}_flip.jpg'.format(base_name[:-4], cur_time), image)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(
                keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

class RandomGray():
    def __init__(self, p=0.3, **kwargs):
        self.p = p

    def __call__(self, data):
        image = data['image']
        # import remote_pdb as pdb;pdb.set_trace()
        if random.random() < self.p:
            # import os
            # img_name = os.path.split(data['img_path'])[-1]
            # cv2.imwrite('tmp/'+img_name, image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data['image'] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # cv2.imwrite('tmp/'+img_name[:-4]+'_gray.jpg', data['image'])
            data['texts'] = ['2'] * len(data['texts'])
        return data


