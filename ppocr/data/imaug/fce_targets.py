import cv2
import numpy as np
from numpy.fft import fft
from numpy.linalg import norm
import sys


class BaseTextDetTargets:
    """Generate text detector ground truths."""

    def __init__(self):
        pass

    def point2line(self, xs, ys, point_1, point_2):
        """Compute the distance from point to a line. This is adapted from
        https://github.com/MhLiao/DB.

        Args:
            xs (ndarray): The x coordinates of size hxw.
            ys (ndarray): The y coordinates of size hxw.
            point_1 (ndarray): The first point with shape 1x2.
            point_2 (ndarray): The second point with shape 1x2.

        Returns:
            result (ndarray): The distance matrix of size hxw.
        """
        # suppose a triangle with three edge abc with c=point_1 point_2
        # a^2
        a_square = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        # b^2
        b_square = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        # c^2
        c_square = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] -
                                                                  point_2[1])
        # -cosC=(c^2-a^2-b^2)/2(ab)
        neg_cos_c = (
            (c_square - a_square - b_square) /
            (np.finfo(np.float32).eps + 2 * np.sqrt(a_square * b_square)))
        # sinC^2=1-cosC^2
        square_sin = 1 - np.square(neg_cos_c)
        square_sin = np.nan_to_num(square_sin)
        # distance=a*b*sinC/c=a*h/c=2*area/c
        result = np.sqrt(a_square * b_square * square_sin /
                         (np.finfo(np.float32).eps + c_square))
        # set result to minimum edge if C<pi/2
        result[neg_cos_c < 0] = np.sqrt(np.fmin(a_square,
                                                b_square))[neg_cos_c < 0]
        return result

    def polygon_area(self, polygon):
        """Compute the polygon area. Please refer to Green's theorem.
        https://en.wikipedia.org/wiki/Green%27s_theorem. This is adapted from
        https://github.com/MhLiao/DB.

        Args:
            polygon (ndarray): The polygon boundary points.
        """

        polygon = polygon.reshape(-1, 2)
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (
                polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

    def polygon_size(self, polygon):
        """Estimate the height and width of the minimum bounding box of the
        polygon.

        Args:
            polygon (ndarray): The polygon point sequence.

        Returns:
            size (tuple): The height and width of the minimum bounding box.
        """
        poly = polygon.reshape(-1, 2)
        rect = cv2.minAreaRect(poly.astype(np.int32))
        size = rect[1]
        return size

    def generate_kernels(self,
                         img_size,
                         text_polys,
                         shrink_ratio,
                         max_shrink=sys.maxsize,
                         ignore_tags=None):
        """Generate text instance kernels for one shrink ratio.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (list[list[ndarray]]: The list of text polygons.
            shrink_ratio (float): The shrink ratio of kernel.

        Returns:
            text_kernel (ndarray): The text kernel mask of (height, width).
        """
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)
        assert isinstance(shrink_ratio, float)

        h, w = img_size
        text_kernel = np.zeros((h, w), dtype=np.float32)

        for text_ind, poly in enumerate(text_polys):
            instance = poly[0].reshape(-1, 2).astype(np.int32)
            area = plg.Polygon(instance).area()
            peri = cv2.arcLength(instance, True)
            distance = min(
                int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                    0.5), max_shrink)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(instance, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
            shrinked = np.array(pco.Execute(-distance), dtype=object)

            # check shrinked == [] or empty ndarray
            if len(shrinked) == 0 or shrinked.size == 0:
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
            try:
                shrinked = np.array(shrinked[0]).reshape(-1, 2)

            except Exception as e:
                print_log(f'{shrinked} with error {e}')
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
            cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)],
                         text_ind + 1)
        return text_kernel, ignore_tags

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """

        # assert check_argument.is_2dlist(polygons_ignore)

        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            instance = poly.reshape(-1,
                                       2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask

    def generate_targets(self, results):
        raise NotImplementedError

    def __call__(self, results):
        results = self.generate_targets(results)
        return results


class TextSnakeTargets(BaseTextDetTargets):
    """Generate the ground truth targets of TextSnake: TextSnake: A Flexible
    Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]. This was partially adapted from
    https://github.com/princewang1994/TextSnake.pytorch.

    Args:
        orientation_thr (float): The threshold for distinguishing between
            head edge and tail edge among the horizontal and vertical edges
            of a quadrangle.
    """

    def __init__(self,
                 orientation_thr=2.0,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3):

        super().__init__()
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio

    def vector_angle(self, vec1, vec2):
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
        return np.arccos(
            np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def vector_slope(self, vec):
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + 1e-8))

    def vector_sin(self, vec):
        assert len(vec) == 2
        return vec[1] / (norm(vec) + 1e-8)

    def vector_cos(self, vec):
        assert len(vec) == 2
        return vec[0] / (norm(vec) + 1e-8)

    def find_head_tail(self, points, orientation_thr):
        """Find the head edge and tail edge of a text polygon.

        Args:
            points (ndarray): The points composing a text polygon.
            orientation_thr (float): The threshold for distinguishing between
                head edge and tail edge among the horizontal and vertical edges
                of a quadrangle.

        Returns:
            head_inds (list): The indexes of two points composing head edge.
            tail_inds (list): The indexes of two points composing tail edge.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(
                    adjacent_edge_vec[0], adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(pad_points[1:] - poly_center, axis=-1),
                norm(pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / np.max(edge_dist)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[
                    (i + 2):(i + len(score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(
                score_matrix.argmax(), score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(
                        points[2] - points[1]) + self.vector_slope(points[0] -
                                                                   points[3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                    points[vertical_edge_inds[0][1]]) + norm(
                                        points[vertical_edge_inds[1][0]] -
                                        points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(
                points[horizontal_edge_inds[0][0]] -
                points[horizontal_edge_inds[0][1]]) + norm(
                    points[horizontal_edge_inds[1][0]] -
                    points[horizontal_edge_inds[1][1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_inds, tail_inds = self.find_head_tail(points,
                                                   self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline

    def resample_line(self, line, n):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """

        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 0

        length_list = [
            norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
        ]
        total_length = sum(length_list)
        length_cumsum = np.cumsum([0.0] + length_list)
        delta_length = total_length / (float(n) + 1e-8)

        current_edge_ind = 0
        resampled_line = [line[0]]

        for i in range(1, n):
            current_line_len = i * delta_length

            while current_line_len >= length_cumsum[current_edge_ind + 1]:
                current_edge_ind += 1
            current_edge_end_shift = current_line_len - length_cumsum[
                current_edge_ind]
            end_shift_ratio = current_edge_end_shift / length_list[
                current_edge_ind]
            current_point = line[current_edge_ind] + (
                line[current_edge_ind + 1] -
                line[current_edge_ind]) * end_shift_ratio
            resampled_line.append(current_point)

        resampled_line.append(line[-1])
        resampled_line = np.array(resampled_line)

        return resampled_line

    def resample_sidelines(self, sideline1, sideline2, resample_step):
        """Resample two sidelines to be of the same points number according to
        step size.

        Args:
            sideline1 (ndarray): The points composing a sideline of a text
                polygon.
            sideline2 (ndarray): The points composing another sideline of a
                text polygon.
            resample_step (float): The resampled step size.

        Returns:
            resampled_line1 (ndarray): The resampled line 1.
            resampled_line2 (ndarray): The resampled line 2.
        """

        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        length1 = sum([
            norm(sideline1[i + 1] - sideline1[i])
            for i in range(len(sideline1) - 1)
        ])
        length2 = sum([
            norm(sideline2[i + 1] - sideline2[i])
            for i in range(len(sideline2) - 1)
        ])

        total_length = (length1 + length2) / 2
        resample_point_num = max(int(float(total_length) / resample_step), 1)

        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    def draw_center_region_maps(self, top_line, bot_line, center_line,
                                center_region_mask, radius_map, sin_map,
                                cos_map, region_shrink_ratio):
        """Draw attributes on text center region.

        Args:
            top_line (ndarray): The points composing top curved sideline of
                text polygon.
            bot_line (ndarray): The points composing bottom curved sideline
                of text polygon.
            center_line (ndarray): The points composing the center line of text
                instance.
            center_region_mask (ndarray): The text center region mask.
            radius_map (ndarray): The map where the distance from point to
                sidelines will be drawn on for each pixel in text center
                region.
            sin_map (ndarray): The map where vector_sin(theta) will be drawn
                on text center regions. Theta is the angle between tangent
                line and vector (1, 0).
            cos_map (ndarray): The map where vector_cos(theta) will be drawn on
                text center regions. Theta is the angle between tangent line
                and vector (1, 0).
            region_shrink_ratio (float): The shrink ratio of text center.
        """

        assert top_line.shape == bot_line.shape == center_line.shape
        assert (center_region_mask.shape == radius_map.shape == sin_map.shape
                == cos_map.shape)
        assert isinstance(region_shrink_ratio, float)
        for i in range(0, len(center_line) - 1):

            top_mid_point = (top_line[i] + top_line[i + 1]) / 2
            bot_mid_point = (bot_line[i] + bot_line[i + 1]) / 2
            radius = norm(top_mid_point - bot_mid_point) / 2

            text_direction = center_line[i + 1] - center_line[i]
            sin_theta = self.vector_sin(text_direction)
            cos_theta = self.vector_cos(text_direction)

            tl = center_line[i] + (top_line[i] -
                                   center_line[i]) * region_shrink_ratio
            tr = center_line[i + 1] + (
                top_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            br = center_line[i + 1] + (
                bot_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            bl = center_line[i] + (bot_line[i] -
                                   center_line[i]) * region_shrink_ratio
            current_center_box = np.vstack([tl, tr, br, bl]).astype(np.int32)

            cv2.fillPoly(center_region_mask, [current_center_box], color=1)
            cv2.fillPoly(sin_map, [current_center_box], color=sin_theta)
            cv2.fillPoly(cos_map, [current_center_box], color=cos_theta)
            cv2.fillPoly(radius_map, [current_center_box], color=radius)

    def generate_center_mask_attrib_maps(self, img_size, text_polys):
        """Generate text center region mask and geometric attribute maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
            radius_map (ndarray): The distance map from each pixel in text
                center region to top sideline.
            sin_map (ndarray): The sin(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
            cos_map (ndarray): The cos(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)
        radius_map = np.zeros((h, w), dtype=np.float32)
        sin_map = np.zeros((h, w), dtype=np.float32)
        cos_map = np.zeros((h, w), dtype=np.float32)

        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon_points = np.array(text_instance).reshape(-1, 2)

            n = len(polygon_points)
            keep_inds = []
            for i in range(n):
                if norm(polygon_points[i] -
                        polygon_points[(i + 1) % n]) > 1e-5:
                    keep_inds.append(i)
            polygon_points = polygon_points[keep_inds]

            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            if self.vector_slope(center_line[-1] - center_line[0]) > 0.9:
                if (center_line[-1] - center_line[0])[1] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]
            else:
                if (center_line[-1] - center_line[0])[0] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)

            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            self.draw_center_region_maps(resampled_top_line,
                                         resampled_bot_line, center_line,
                                         center_region_mask, radius_map,
                                         sin_map, cos_map,
                                         self.center_region_shrink_ratio)

        return center_region_mask, radius_map, sin_map, cos_map

    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            # assert len(poly) == 1
            text_instance = [[poly[i], poly[i + 1]]
                             for i in range(0, len(poly), 2)]
            polygon = np.array(
                text_instance, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def generate_targets(self, results):
        """Generate the gt targets for TextSnake.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        h, w, _ = results['img_shape']

        gt_text_mask = self.generate_text_region_mask((h, w), polygon_masks)
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)

        (gt_center_region_mask, gt_radius_map, gt_sin_map,
         gt_cos_map) = self.generate_center_mask_attrib_maps((h, w),
                                                             polygon_masks)

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_radius_map': gt_radius_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }
        for key, value in mapping.items():
            value = value if isinstance(value, list) else [value]
            results[key] = BitmapMasks(value, h, w)
            results['mask_fields'].append(key)

        return results


class FCENetTargets:
    """Generate the ground truth targets of FCENet: Fourier Contour Embedding
    for Arbitrary-Shaped Text Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        resample_step (float): The step size for resampling the text center
            line (TCL). It's better not to exceed half of the minimum width.
        center_region_shrink_ratio (float): The shrink ratio of text center
            region.
        level_size_divisors (tuple(int)): The downsample ratio on each level.
        level_proportion_range (tuple(tuple(int))): The range of text sizes
            assigned to each level.
    """

    def __init__(self,
                 fourier_degree=5,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0)),
                 orientation_thr=2.0,
                 **kwargs):

        super().__init__()
        assert isinstance(level_size_divisors, tuple)
        assert isinstance(level_proportion_range, tuple)
        assert len(level_size_divisors) == len(level_proportion_range)
        self.fourier_degree = fourier_degree
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range

        self.orientation_thr = orientation_thr

    def vector_angle(self, vec1, vec2):
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
        return np.arccos(
            np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def resample_line(self, line, n):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """

        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 0

        length_list = [
            norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
        ]
        total_length = sum(length_list)
        length_cumsum = np.cumsum([0.0] + length_list)
        delta_length = total_length / (float(n) + 1e-8)

        current_edge_ind = 0
        resampled_line = [line[0]]

        for i in range(1, n):
            current_line_len = i * delta_length

            while current_line_len >= length_cumsum[current_edge_ind + 1]:
                current_edge_ind += 1
            current_edge_end_shift = current_line_len - length_cumsum[
                current_edge_ind]
            end_shift_ratio = current_edge_end_shift / length_list[
                current_edge_ind]
            current_point = line[current_edge_ind] + (
                line[current_edge_ind + 1] -
                line[current_edge_ind]) * end_shift_ratio
            resampled_line.append(current_point)

        resampled_line.append(line[-1])
        resampled_line = np.array(resampled_line)

        return resampled_line


    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_inds, tail_inds = self.find_head_tail(points,
                                                   self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline

    def find_head_tail(self, points, orientation_thr):
        """Find the head edge and tail edge of a text polygon.

        Args:
            points (ndarray): The points composing a text polygon.
            orientation_thr (float): The threshold for distinguishing between
                head edge and tail edge among the horizontal and vertical edges
                of a quadrangle.

        Returns:
            head_inds (list): The indexes of two points composing head edge.
            tail_inds (list): The indexes of two points composing tail edge.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(
                    adjacent_edge_vec[0], adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(pad_points[1:] - poly_center, axis=-1),
                norm(pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / np.max(edge_dist)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[
                    (i + 2):(i + len(score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(
                score_matrix.argmax(), score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(
                        points[2] - points[1]) + self.vector_slope(points[0] -
                                                                   points[3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                    points[vertical_edge_inds[0][1]]) + norm(
                                        points[vertical_edge_inds[1][0]] -
                                        points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(
                points[horizontal_edge_inds[0][0]] -
                points[horizontal_edge_inds[0][1]]) + norm(
                    points[horizontal_edge_inds[1][0]] -
                    points[horizontal_edge_inds[1][1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def resample_sidelines(self, sideline1, sideline2, resample_step):
        """Resample two sidelines to be of the same points number according to
        step size.

        Args:
            sideline1 (ndarray): The points composing a sideline of a text
                polygon.
            sideline2 (ndarray): The points composing another sideline of a
                text polygon.
            resample_step (float): The resampled step size.

        Returns:
            resampled_line1 (ndarray): The resampled line 1.
            resampled_line2 (ndarray): The resampled line 2.
        """

        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        length1 = sum([
            norm(sideline1[i + 1] - sideline1[i])
            for i in range(len(sideline1) - 1)
        ])
        length2 = sum([
            norm(sideline2[i + 1] - sideline2[i])
            for i in range(len(sideline2) - 1)
        ])

        total_length = (length1 + length2) / 2
        resample_point_num = max(int(float(total_length) / resample_step), 1)

        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2


    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            # assert len(poly) == 1
            polygon_points = poly.reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                    resampled_top_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                    resampled_bot_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def resample_polygon(self, polygon, n=400):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        length = []

        for i in range(len(polygon)):
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]
            length.append(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)

        total_length = sum(length)
        n_on_each_line = (np.array(length) / (total_length + 1e-8)) * n
        n_on_each_line = n_on_each_line.astype(np.int32)
        new_polygon = []

        for i in range(len(polygon)):
            num = n_on_each_line[i]
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]

            if num == 0:
                continue

            dxdy = (p2 - p1) / num
            for j in range(num):
                point = p1 + dxdy * j
                new_polygon.append(point)

        return np.array(new_polygon)

    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def poly2fourier(self, polygon, fourier_degree):
        """Perform Fourier transformation to generate Fourier coefficients ck
        from polygon.

        Args:
            polygon (ndarray): An input polygon.
            fourier_degree (int): The maximum Fourier degree K.
        Returns:
            c (ndarray(complex)): Fourier coefficients.
        """
        points = polygon[:, 0] + polygon[:, 1] * 1j
        c_fft = fft(points) / len(points)
        c = np.hstack((c_fft[-fourier_degree:], c_fft[:fourier_degree + 1]))
        return c

    def clockwise(self, c, fourier_degree):
        """Make sure the polygon reconstructed from Fourier coefficients c in
        the clockwise direction.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon in clockwise point order.
        """
        if np.abs(c[fourier_degree + 1]) > np.abs(c[fourier_degree - 1]):
            return c
        elif np.abs(c[fourier_degree + 1]) < np.abs(c[fourier_degree - 1]):
            return c[::-1]
        else:
            if np.abs(c[fourier_degree + 2]) > np.abs(c[fourier_degree - 2]):
                return c
            else:
                return c[::-1]

    def cal_fourier_signature(self, polygon, fourier_degree):
        """Calculate Fourier signature from input polygon.

        Args:
              polygon (ndarray): The input polygon.
              fourier_degree (int): The maximum Fourier degree K.
        Returns:
              fourier_signature (ndarray): An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 Fourier coefficients.
        """
        resampled_polygon = self.resample_polygon(polygon)
        resampled_polygon = self.normalize_polygon(resampled_polygon)

        fourier_coeff = self.poly2fourier(resampled_polygon, fourier_degree)
        fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)

        real_part = np.real(fourier_coeff).reshape((-1, 1))
        image_part = np.imag(fourier_coeff).reshape((-1, 1))
        fourier_signature = np.hstack([real_part, image_part])

        return fourier_signature

    def generate_fourier_maps(self, img_size, text_polys):
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            fourier_real_map (ndarray): The Fourier coefficient real part maps.
            fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """

        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        k = self.fourier_degree
        real_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)
        imag_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)

        for poly in text_polys:
            # assert len(poly) == 1
            # text_instance = [[poly[i], poly[i + 1]]
            #                  for i in range(0, len(poly), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(poly).reshape((1, -1, 2))
            cv2.fillPoly(mask, polygon.astype(np.int32), 1)
            fourier_coeff = self.cal_fourier_signature(polygon[0], k)
            for i in range(-k, k + 1):
                if i != 0:
                    real_map[i + k, :, :] = mask * fourier_coeff[i + k, 0] + (
                        1 - mask) * real_map[i + k, :, :]
                    imag_map[i + k, :, :] = mask * fourier_coeff[i + k, 1] + (
                        1 - mask) * imag_map[i + k, :, :]
                else:
                    yx = np.argwhere(mask > 0.5)
                    k_ind = np.ones((len(yx)), dtype=np.int64) * k
                    y, x = yx[:, 0], yx[:, 1]
                    real_map[k_ind, y, x] = fourier_coeff[k, 0] - x
                    imag_map[k_ind, y, x] = fourier_coeff[k, 1] - y

        return real_map, imag_map

    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            # assert len(poly) == 1
            # text_instance = [[poly[i], poly[i + 1]]
            #                  for i in range(0, len(poly), 2)]
            polygon = np.array(
                poly, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """

        # assert check_argument.is_2dlist(polygons_ignore)

        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            instance = poly.reshape(-1,
                                       2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask


    def generate_level_targets(self, img_size, text_polys, ignore_polys):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
        """
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        level_maps = []
        for poly in text_polys:
            # assert len(poly) == 1
            # text_instance = [[poly[i], poly[i + 1]]
            #                  for i in range(0, len(poly), 2)]
            polygon = np.array(poly, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append(poly / lv_size_divs[ind])

        for ignore_poly in ignore_polys:
            # assert len(ignore_poly) == 1
            # text_instance = [[ignore_poly[i], ignore_poly[i + 1]]
            #                  for i in range(0, len(ignore_poly), 2)]
            polygon = np.array(ignore_poly, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_ignore_polys[ind].append(
                        ignore_poly / lv_size_divs[ind])

        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)

            text_region = self.generate_text_region_mask(
                level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(text_region)

            center_region = self.generate_center_region_mask(
                level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)

            effective_mask = self.generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])[None]
            current_level_maps.append(effective_mask)

            fourier_real_map, fourier_image_maps = self.generate_fourier_maps(
                level_img_size, lv_text_polys[ind])
            current_level_maps.append(fourier_real_map)
            current_level_maps.append(fourier_image_maps)

            level_maps.append(np.concatenate(current_level_maps))

        return level_maps

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)
        image = results['image']
        polygons = results['polys']
        ignore_tags = results['ignore_tags']
        h, w, _ = image.shape

        # import time
        # from PIL import Image, ImageDraw
        # cur_time = time.time()
        # image = results['image']
        # text_polys = results['polys']
        # img = image[..., ::-1]
        # img = Image.fromarray(img)
        # draw = ImageDraw.Draw(img)
        # for box in text_polys:
        #     draw.polygon(box, outline=(0, 255, 255,), )
        # img.save('tmp/{}_resize_pad.jpg'.format(cur_time))

        polygon_masks = []
        polygon_masks_ignore = []
        for tag, polygon in zip(ignore_tags, polygons):
            if tag is True:
                polygon_masks_ignore.append(polygon)
            else:
                polygon_masks.append(polygon)        

        level_maps = self.generate_level_targets((h, w), polygon_masks,
                                                 polygon_masks_ignore)

        # results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        # import remote_pdb as pdb;pdb.set_trace()
        mapping = {
            'p3_maps': level_maps[0],
            'p4_maps': level_maps[1],
            'p5_maps': level_maps[2]
        }
        for key, value in mapping.items():
            results[key] = value

        return results

    def __call__(self, results):
        results = self.generate_targets(results)
        return results
