from paddle import nn
import Polygon as plg
from functools import partial
from abc import ABCMeta
import copy
from collections import defaultdict
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.initializer import Normal
import paddle


# from mmcv.runner import BaseModule
# from mmdet.models.builder import HEADS, build_loss
# from .head_mixin import HeadMixin

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def points2polygon(points):
    """Convert k points to 1 polygon.

    Args:
        points (ndarray or list): A ndarray or a list of shape (2k)
            that indicates k points.

    Returns:
        polygon (Polygon): A polygon object.
    """
    if isinstance(points, list):
        points = np.array(points)

    assert isinstance(points, np.ndarray)
    assert (points.size % 2 == 0) and (points.size >= 8)

    point_mat = points.reshape([-1, 2])
    return plg.Polygon(point_mat)


def poly_intersection(poly_det, poly_gt):
    """Calculate the intersection area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        intersection_area (float): The intersection area between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    poly_inter = poly_det & poly_gt
    if len(poly_inter) == 0:
        return 0, poly_inter
    return poly_inter.area(), poly_inter


def poly_union(poly_det, poly_gt):
    """Calculate the union area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        union_area (float): The union area between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    area_det = poly_det.area()
    area_gt = poly_gt.area()
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    return area_det + area_gt - area_inters


def boundary_iou(src, target):
    """Calculate the IOU between two boundaries.

    Args:
       src (list): Source boundary.
       target (list): Target boundary.

    Returns:
       iou (float): The iou between two boundaries.
    """
    assert utils.valid_boundary(src, False)
    assert utils.valid_boundary(target, False)
    src_poly = points2polygon(src)
    target_poly = points2polygon(target)

    return poly_iou(src_poly, target_poly)


def poly_iou(poly_det, poly_gt):
    """Calculate the IOU between two polygons.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        iou (float): The IOU between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    area_union = poly_union(poly_det, poly_gt)
    if area_union == 0:
        return 0.0
    return area_inters / area_union


def poly_nms(polygons, threshold):
    assert isinstance(polygons, list)

    polygons = np.array(sorted(polygons, key=lambda x: x[-1]))

    keep_poly = []
    index = [i for i in range(polygons.shape[0])]

    while len(index) > 0:
        keep_poly.append(polygons[index[-1]].tolist())
        A = polygons[index[-1]][:-1]
        index = np.delete(index, -1)

        iou_list = np.zeros((len(index), ))
        for i in range(len(index)):
            B = polygons[index[i]][:-1]

            iou_list[i] = boundary_iou(A, B)
        remove_index = np.where(iou_list > threshold)
        index = np.delete(index, remove_index)

    return keep_poly


def fcenet_decode(preds,
                  fourier_degree,
                  num_reconstr_points,
                  scale,
                  alpha=1.0,
                  beta=2.0,
                  text_repr_type='poly',
                  score_thr=0.3,
                  nms_thr=0.1):
    """Decoding predictions of FCENet to instances.

    Args:
        preds (list(Tensor)): The head output tensors.
        fourier_degree (int): The maximum Fourier transform degree k.
        num_reconstr_points (int): The points number of the polygon
            reconstructed from predicted Fourier coefficients.
        scale (int): The down-sample scale of the prediction.
        alpha (float) : The parameter to calculate final scores. Score_{final}
                = (Score_{text region} ^ alpha)
                * (Score_{text center region}^ beta)
        beta (float) : The parameter to calculate final score.
        text_repr_type (str):  Boundary encoding type 'poly' or 'quad'.
        score_thr (float) : The threshold used to filter out the final
            candidates.
        nms_thr (float) :  The threshold of nms.

    Returns:
        boundaries (list[list[float]]): The instance boundary and confidence
            list.
    """
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert text_repr_type in ['poly', 'quad']

    cls_pred = preds[0][0]
    tr_pred = cls_pred[0:2].softmax(dim=0).data.cpu().numpy()
    tcl_pred = cls_pred[2:].softmax(dim=0).data.cpu().numpy()

    reg_pred = preds[1][0].permute(1, 2, 0).data.cpu().numpy()
    x_pred = reg_pred[:, :, :2 * fourier_degree + 1]
    y_pred = reg_pred[:, :, 2 * fourier_degree + 1:]

    score_pred = (tr_pred[1]**alpha) * (tcl_pred[1]**beta)
    tr_pred_mask = (score_pred) > score_thr
    tr_mask = fill_hole(tr_pred_mask)

    tr_contours, _ = cv2.findContours(
        tr_mask.astype(np.uint8), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)  # opencv4

    mask = np.zeros_like(tr_mask)
    boundaries = []
    for cont in tr_contours:
        deal_map = mask.copy().astype(np.int8)
        cv2.drawContours(deal_map, [cont], -1, 1, -1)

        score_map = score_pred * deal_map
        score_mask = score_map > 0
        xy_text = np.argwhere(score_mask)
        dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

        x, y = x_pred[score_mask], y_pred[score_mask]
        c = x + y * 1j
        c[:, fourier_degree] = c[:, fourier_degree] + dxy
        c *= scale

        polygons = fourier2poly(c, num_reconstr_points)
        score = score_map[score_mask].reshape(-1, 1)
        polygons = poly_nms(np.hstack((polygons, score)).tolist(), nms_thr)

        boundaries = boundaries + polygons

    boundaries = poly_nms(boundaries, nms_thr)

    if text_repr_type == 'quad':
        new_boundaries = []
        for boundary in boundaries:
            poly = np.array(boundary[:-1]).reshape(-1, 2).astype(np.float32)
            score = boundary[-1]
            points = cv2.boxPoints(cv2.minAreaRect(poly))
            points = np.int0(points)
            new_boundaries.append(points.reshape(-1).tolist() + [score])

    return boundaries

class HeadMixin:
    """The head minxin for dbnet and pannet heads."""

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
            with size 2k+1 with k>=4.
            scale_factor(ndarray): The scale factor of size (4,).

        Returns:
            boundaries (list[list[float]]): The scaled boundaries.
        """
        assert check_argument.is_2dlist(boundaries)
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4

        for b in boundaries:
            sz = len(b)
            check_argument.valid_boundary(b, True)
            b[:sz -
              1] = (np.array(b[:sz - 1]) *
                    (np.tile(scale_factor[:2], int(
                        (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
        return boundaries

    def get_boundary(self, score_maps, img_metas, rescale):
        """Compute text boundaries via post processing.

        Args:
            score_maps (Tensor): The text score map.
            img_metas (dict): The image meta info.
            rescale (bool): Rescale boundaries to the original image resolution
                if true, and keep the score_maps resolution if false.

        Returns:
            results (dict): The result dict.
        """

        assert check_argument.is_type_list(img_metas, dict)
        assert isinstance(rescale, bool)

        score_maps = score_maps.squeeze()
        boundaries = decode(
            decoding_type=self.decoding_type,
            preds=score_maps,
            text_repr_type=self.text_repr_type)
        if rescale:
            boundaries = self.resize_boundary(
                boundaries,
                1.0 / self.downsample_ratio / img_metas[0]['scale_factor'])
        results = dict(boundary_result=boundaries)
        return results

    def loss(self, pred_maps, **kwargs):
        """Compute the loss for text detection.

        Args:
            pred_maps (tensor): The input score maps of NxCxHxW.

        Returns:
            losses (dict): The dict for losses.
        """
        losses = self.loss_module(pred_maps, self.downsample_ratio, **kwargs)
        return losses

class BaseModule(nn.Layer):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

        - ``init_cfg``: the config to control the initialization.
        - ``_params_init_info``: Used to track the parameter
            initialization information.
        - ``init_weights``: The function of parameter
            initialization and recording initialization
            information.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # The `_params_init_info` is used to record the initialization
        # information of the parameters
        # the key should be the obj:`nn.Parameter` of model and the value
        # should be a dict containing
        # - param_name (str): The name of parameter.
        # - init_info (str): The string that describes the initialization.
        # - tmp_mean_value (FloatTensor): The mean of the parameter,
        #       which indicates whether the parameter has been modified.
        # this attribute would be deleted after all parameters is initialized.
        self._params_init_info = defaultdict(dict)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        # check if it is top-level module
        is_top_level_module = len(self._params_init_info) == 0
        if is_top_level_module:
            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]['param_name'] = name
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'

        from ..cnn import initialize
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    def _dump_init_info(self, logger_name):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(
                    'Name of parameter - Initialization information\n')
                for item in list(self._params_init_info.values()):
                    handler.stream.write(
                        f"{item['param_name']} - {item['init_info']} \n")
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for item in list(self._params_init_info.values()):
                print_log(
                    f"{item['param_name']} - {item['init_info']}",
                    logger=logger_name)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


class FCEHead(HeadMixin, BaseModule):
    """The class for implementing FCENet head.
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
    Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, FCEnet tends to be overfitting.
        score_thr (float) : The threshold to filter out the final
            candidates.
        nms_thr (float) : The threshold of nms.
        alpha (float) : The parameter to calculate final scores. Score_{final}
            = (Score_{text region} ^ alpha)
            * (Score{text center region} ^ beta)
        beta (float) :The parameter to calculate final scores.
    """

    def __init__(
        self,
        in_channels,
        scales,
        fourier_degree=5,
        num_sample=50,
        num_reconstr_points=50,
        decoding_type='fcenet',
        score_thr=0.3,
        nms_thr=0.1,
        alpha=1.0,
        beta=1.0,
        text_repr_type='poly',
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type='Normal',
            mean=0,
            std=0.01,
            override=[dict(name='out_conv_cls'),
                      dict(name='out_conv_reg')])):

        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.fourier_degree = fourier_degree
        self.sample_num = num_sample
        self.num_reconstr_points = num_reconstr_points
        self.decoding_type = decoding_type
        # self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.text_repr_type = text_repr_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2

        # self.out_conv_cls = nn.Conv2d(
        #     self.in_channels,
        #     self.out_channels_cls,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1)
        self.out_conv_cls = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=ParamAttr(name='cls_weights', initializer=Normal(
                     mean=paddle.to_tensor(0.), std=paddle.to_tensor(0.01))),
            bias_attr=True)
        # self.out_conv_reg = nn.Conv2d(
        #     self.in_channels,
        #     self.out_channels_reg,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1)
        self.out_conv_reg = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=ParamAttr(name='reg_weights', initializer=Normal(
                     mean=paddle.to_tensor(0.), std=paddle.to_tensor(0.01))),
            bias_attr=True)

    def forward(self, feats, targets=None):
        cls_res, reg_res = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        # import pdb;pdb.set_trace()
        outs = {}
        
        if not self.training:
            for i in range(level_num):
                tr_pred = F.softmax(cls_res[i][:, 0:2, :, :], axis=1)
                tcl_pred = F.softmax(cls_res[i][:, 2:, :, :], axis=1)
                outs['level_{}'.format(i)] = paddle.concat([tr_pred, tcl_pred, reg_res[i]], axis=1)
        else:
            preds = [[cls_res[i], reg_res[i]] for i in range(level_num)]
            outs['levels'] = preds
        return outs

    def forward_single(self, x):
        cls_predict = self.out_conv_cls(x)
        reg_predict = self.out_conv_reg(x)
        return cls_predict, reg_predict

    def get_boundary(self, score_maps, img_metas, rescale):
        assert len(score_maps) == len(self.scales)

        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(
                score_map, scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries)
        return results

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        return fcenet_decode(
            decoding_type=self.decoding_type,
            preds=score_map,
            fourier_degree=self.fourier_degree,
            num_reconstr_points=self.num_reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            text_repr_type=self.text_repr_type,
            score_thr=self.score_thr,
            nms_thr=self.nms_thr)
