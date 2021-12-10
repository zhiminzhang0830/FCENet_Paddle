# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr


def get_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class Head(nn.Layer):
    def __init__(self, in_channels, name_list):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')
        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x


# def get_bias_attr(k, name):
#     stdv = 1.0 / math.sqrt(k * 1.0)
#     initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
#     bias_attr = ParamAttr(initializer=initializer, name=name + "_b_attr")
#     return bias_attr

# class Head(nn.Layer):
#     def __init__(self, in_channels, name_list):
#         super(Head, self).__init__()
#         self.conv1 = nn.Conv2D(
#             in_channels=in_channels,
#             out_channels=in_channels // 4,
#             kernel_size=3,
#             padding=1,
#             weight_attr=ParamAttr(name=name_list[0] + '.w_0'),
#             bias_attr=False)
#         self.conv_bn1 = nn.BatchNorm(
#             num_channels=in_channels // 4,
#             param_attr=ParamAttr(
#                 name=name_list[1] + '.w_0',
#                 initializer=paddle.nn.initializer.Constant(value=1.0)),
#             bias_attr=ParamAttr(
#                 name=name_list[1] + '.b_0',
#                 initializer=paddle.nn.initializer.Constant(value=1e-4)),
#             moving_mean_name=name_list[1] + '.w_1',
#             moving_variance_name=name_list[1] + '.w_2',
#             act='relu')
#         self.conv2 = nn.Conv2DTranspose(
#             in_channels=in_channels // 4,
#             out_channels=in_channels // 4,
#             kernel_size=2,
#             stride=2,
#             weight_attr=ParamAttr(
#                 name=name_list[2] + '.w_0',
#                 initializer=paddle.nn.initializer.KaimingUniform()),
#             bias_attr=get_bias_attr(in_channels // 4, name_list[-1] + "conv2"))
#         self.conv_bn2 = nn.BatchNorm(
#             num_channels=in_channels // 4,
#             param_attr=ParamAttr(
#                 name=name_list[3] + '.w_0',
#                 initializer=paddle.nn.initializer.Constant(value=1.0)),
#             bias_attr=ParamAttr(
#                 name=name_list[3] + '.b_0',
#                 initializer=paddle.nn.initializer.Constant(value=1e-4)),
#             moving_mean_name=name_list[3] + '.w_1',
#             moving_variance_name=name_list[3] + '.w_2',
#             act="relu")
#         self.conv3 = nn.Conv2DTranspose(
#             in_channels=in_channels // 4,
#             out_channels=1,
#             kernel_size=2,
#             stride=2,
#             weight_attr=ParamAttr(
#                 name=name_list[4] + '.w_0',
#                 initializer=paddle.nn.initializer.KaimingUniform()),
#             bias_attr=get_bias_attr(in_channels // 4, name_list[-1] + "conv3"),
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv_bn1(x)
#         x = self.conv2(x)
#         x = self.conv_bn2(x)
#         x = self.conv3(x)
#         x = F.sigmoid(x)
#         return x


class DBHead(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        binarize_name_list = [
            'conv2d_56', 'batch_norm_47', 'conv2d_transpose_0', 'batch_norm_48',
            'conv2d_transpose_1', 'binarize'
        ]
        thresh_name_list = [
            'conv2d_57', 'batch_norm_49', 'conv2d_transpose_2', 'batch_norm_50',
            'conv2d_transpose_3', 'thresh'
        ]
        self.binarize = Head(in_channels, binarize_name_list)
        self.thresh = Head(in_channels, thresh_name_list)

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        return {'maps': y}


class DBHeadMulti_old(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, num_cls=None, **kwargs):
        super(DBHeadMulti, self).__init__()
        self.k = k

        binarize_name_list = [
            'conv2d_56', 'batch_norm_47', 'conv2d_transpose_0', 'batch_norm_48',
            'conv2d_transpose_1', 'binarize'
        ]
        thresh_name_list = [
            'conv2d_57', 'batch_norm_49', 'conv2d_transpose_2', 'batch_norm_50',
            'conv2d_transpose_3', 'thresh'
        ]

        self.num_cls = num_cls
        if self.num_cls is None:
            self.binarize = Head(in_channels, binarize_name_list)
            self.thresh = Head(in_channels, thresh_name_list)
        else:
            binarize_name_lists = []
            for i in range(self.num_cls):
                if i == 0:
                    binarize_name_lists.append(binarize_name_list)
                    continue
                tmp = []
                for name in binarize_name_list:
                    tmp.append(name + '_' + str(i))
                binarize_name_lists.append(tmp)
            thresh_name_lists = []
            for i in range(self.num_cls):
                if i == 0:
                    thresh_name_lists.append(thresh_name_list)
                    continue
                tmp = []
                for name in thresh_name_list:
                    tmp.append(name + '_' + str(i))
                thresh_name_lists.append(tmp)
            self.binarize = nn.LayerList([Head(in_channels, binarize_name_lists[i]) for i in range(self.num_cls)])
            self.thresh = nn.LayerList([Head(in_channels, thresh_name_lists[i]) for i in range(self.num_cls)])

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        # import remote_pdb as pdb;pdb.set_trace()
        if self.num_cls is None:
            shrink_maps = self.binarize(x)
            if not self.training:
                return {'maps': shrink_maps}

            threshold_maps = self.thresh(x)
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
            return {'maps': y}
        else:
            # import remote_pdb as pdb;pdb.set_trace()
            shrink_maps = [self.binarize[i](x) for i in range(self.num_cls)]
            
            if not self.training:
                shrink_maps = paddle.concat(shrink_maps, axis=1)
                return {'maps': shrink_maps}

            threshold_maps = [self.thresh[i](x) for i in range(self.num_cls)]
            binary_maps = [self.step_function(shrink_map, threshold_map) for shrink_map, threshold_map in zip(shrink_maps, threshold_maps)]

            shrink_maps = paddle.concat(shrink_maps, axis=1)
            threshold_maps = paddle.concat(threshold_maps, axis=1)
            binary_maps = paddle.concat(binary_maps, axis=1)

            y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
            return {'maps': y}


class DBHeadMulti(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, num_cls=2, **kwargs):
        super(DBHeadMulti, self).__init__()
        self.k = k
        self.num_cls = num_cls
        self.heads = nn.LayerList([DBHead(in_channels, k) for _ in range(self.num_cls)])

    def forward(self, x, targets=None):
        outs = [head(x)['maps'] for head in self.heads]
        y = paddle.concat(outs, axis=1)
        return {'maps': y}



