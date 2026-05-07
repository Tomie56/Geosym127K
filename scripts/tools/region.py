import os
import json
import logging
import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict 
from datetime import datetime
import sympy as sp
from .drawer import GeometryDrawer
from PIL import Image, ImageDraw, ImageFont

# 共享类型与坐标转换
PointMath = Tuple[float, float]
PointPixel = Tuple[int, int]


def _quant4(p: PointMath) -> PointMath:
    """四位小数量化，解决浮点精度问题"""
    return (float(round(p[0], 4)), float(round(p[1], 4)))


def pixel_to_math(coord: PointPixel) -> PointMath:
    """像素坐标 → 数学坐标"""
    return (float(round(coord[0], 4)), float(round(coord[1], 4)))


def math_to_pixel(coord: PointMath) -> PointPixel:
    """数学坐标 → 像素坐标"""
    return (int(round(coord[0])), int(round(coord[1])))

# 区域提取器配置与实现
@dataclass
class RegionExtractConfig:
    """区域提取器配置参数"""
    thresh: int = 200                 # 二值化阈值
    binarize_mode: str = "fixed"      # 二值化模式：fixed/adaptive
    adaptive_block_size: int = 51     # 自适应阈值窗口（奇数）
    adaptive_C: int = -5              # 自适应阈值常数项
    line_thicken: int = 1             # 线条加粗像素
    line_thicken_iter: int = 1        # 加粗迭代次数
    min_area: int = 100               # 最小区域像素面积
    max_area: Optional[int] = None    # 最大区域像素面积
    remove_border_touching: bool = True  # 移除边界区域
    fill_holes: bool = True           # 填充孔洞
    smooth_edges: bool = True         # 边缘平滑
    smooth_kernel: int = 3            # 平滑核大小
    mask_dilate: int = 1              # 掩码膨胀像素
    area_method: str = "supersample"  # 面积计算方法
    supersample_scale: int = 2        # 超采样倍数
    arc_detect: bool = True           # 圆弧检测开关
    curvature_threshold: float = 0.005  # 曲率阈值
    min_arc_points: int = 10          # 最小圆弧点数
    resample_step: float = 2.0        # 轮廓重采样步长
    arc_convexity_threshold: float = 0.1  # 圆弧凹凸阈值


class RegionExtractor:
    """提取图像中的封闭区域，支持双坐标和圆弧信息"""
    def __init__(self, cfg: RegionExtractConfig):
        self.cfg = cfg
        self._validate_config()

    def _validate_config(self) -> None:
        if self.cfg.binarize_mode not in ["fixed", "adaptive"]:
            raise ValueError(f"无效二值化模式: {self.cfg.binarize_mode}")
        if self.cfg.area_method not in ["pixel", "contour", "supersample"]:
            raise ValueError(f"无效面积方法: {self.cfg.area_method}")
        if self.cfg.adaptive_block_size % 2 == 0:
            raise ValueError("自适应窗口必须为奇数")

    def extract(
        self, 
        img_bgr: np.ndarray,  # 修复img未定义，统一用img_bgr
        meta: Optional[Dict[str, Any]] = None,
        shadow_tags: Optional[List[int]] = None
    ) -> Tuple[
        List[np.ndarray],  # 掩码列表
        Optional[List[np.ndarray]],  # 像素轮廓
        Optional[List[np.ndarray]],  # 数学轮廓
        Optional[List[Dict[str, Any]]],  # 统计信息
        Optional[List[Dict[str, Any]]]   # 多边形详情
    ]:
        shadow_tags = shadow_tags or []
        shadow_tag_set = set(shadow_tags)
        arc_center_math, arc_center_pixel = self._extract_centers(meta)

        # 图像预处理
        white = self._binarize_white(img_bgr)
        white = self._thicken_lines_on_white(white)
        if self.cfg.remove_border_touching:
            white = self._remove_border_white(white)

        # 连通区域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            white, connectivity=8, ltype=cv2.CV_32S
        )

        # 结果容器
        masks, pixel_contours, math_contours = [], [], []
        out_stats, out_polygons = [], []
        H, W = white.shape

        for lab in range(1, num_labels):  # 跳过背景（0）
            x_px, y_px, w_px, h_px, area_cc = stats[lab]
            if not self._is_valid_region(area_cc, x_px, y_px, w_px, h_px, W, H):
                continue

            # 生成掩码
            mask = np.zeros_like(white, dtype=np.uint8)
            mask[labels == lab] = 255
            mask = self._postprocess_mask(mask)
            masks.append(mask)

            # 面积与轮廓提取
            area_est, pixel_contour = self._estimate_area(mask)
            if self.cfg.area_method:
                pixel_contours.append(pixel_contour)

            # 坐标转换
            math_contour = self._pixel_contour_to_math(pixel_contour)
            math_contours.append(math_contour)

            # 阴影标记
            is_shadow = lab in shadow_tag_set

            # 多边形详情（含圆弧）
            if pixel_contour.size > 0:
                polygon_info = self._contour_to_dual_coord_polygon(
                    pixel_contour, lab, is_shadow, arc_center_math, arc_center_pixel
                )
                out_polygons.append(polygon_info)

            # 统计信息
            x_math, y_math = pixel_to_math((x_px, y_px))
            out_stats.append({
                "pixel": {
                    "x": x_px, "y": y_px, "w": w_px, "h": h_px,
                    "area_px_cc": area_cc, "area_px_est": area_est
                },
                "math": {
                    "x": round(x_math, 4), "y": round(y_math, 4),
                    "w": float(w_px), "h": float(h_px)
                },
                "label": lab,
                "is_shadow": is_shadow,
                "has_arc": polygon_info.get("has_arc", False) if out_polygons else False
            })

        return (
            masks,
            pixel_contours if pixel_contours else None,
            math_contours if math_contours else None,
            out_stats if out_stats else None,
            out_polygons if out_polygons else None
        )

    # 区域验证
    def _is_valid_region(self, area_cc: int, x_px: int, y_px: int, w_px: int, h_px: int, W: int, H: int) -> bool:
        if area_cc < self.cfg.min_area:
            return False
        if self.cfg.max_area and area_cc > self.cfg.max_area:
            return False
        if x_px <= 0 or y_px <= 0 or (x_px + w_px) >= W or (y_px + h_px) >= H:
            return False
        return True

    # 提取圆心坐标
    def _extract_centers(self, meta: Optional[Dict[str, Any]]) -> Tuple[Optional[PointMath], Optional[PointPixel]]:
        if not meta:
            return None, None
        if "center_math" in meta:
            center_math = tuple(meta["center_math"])
            return center_math, math_to_pixel(center_math)
        elif "center" in meta:
            center_pixel = tuple(map(int, meta["center"]))
            return pixel_to_math(center_pixel), center_pixel
        return None, None

    # 轮廓转双坐标多边形
    def _contour_to_dual_coord_polygon(self, pixel_contour: np.ndarray, label: int, is_shadow: bool,
                                      arc_center_math: Optional[PointMath], arc_center_pixel: Optional[PointPixel]) -> Dict[str, Any]:
        pixel_contour = pixel_contour.reshape(-1, 2).astype(np.float32)
        if len(pixel_contour) < 3:
            return self._empty_dual_polygon(label, is_shadow, arc_center_math, arc_center_pixel)

        # 重采样与简化
        resampled_pixel = self._resample_contour(pixel_contour, self.cfg.resample_step)
        resampled_math = self._pixel_contour_to_math(resampled_pixel.reshape(-1, 1, 2))
        epsilon = 0.005 * cv2.arcLength(resampled_pixel, closed=True)
        approx_pixel = cv2.approxPolyDP(resampled_pixel, epsilon, closed=True).reshape(-1, 2)
        approx_math = np.array([pixel_to_math((x, y)) for x, y in approx_pixel], dtype=np.float32)

        # 顶点与边
        pixel_vertices = [list(pt) for pt in approx_pixel]
        math_vertices = [list(pt) for pt in approx_math]
        n = len(pixel_vertices)
        pixel_edges, math_edges = [], []
        for i in range(n):
            p_px, q_px = pixel_vertices[i], pixel_vertices[(i+1) % n]
            p_math, q_math = math_vertices[i], math_vertices[(i+1) % n]
            pixel_edges.append([p_px, q_px])
            math_edges.append([p_math, q_math])

        # 圆弧检测
        has_arc, arc_segments, arc_details = False, [], []
        if self.cfg.arc_detect and len(resampled_pixel) >= self.cfg.min_arc_points:
            curvatures = self._calculate_curvatures(resampled_pixel)
            arc_segments = self._detect_arc_segments(curvatures, self.cfg.curvature_threshold, self.cfg.min_arc_points)
            has_arc = len(arc_segments) > 0
            if has_arc and arc_center_math:
                arc_details = self._calculate_arc_details(resampled_math, arc_segments, arc_center_math)

        return {
            "type": "polygon_with_arc" if has_arc else "polygon",
            "n": n,
            "label": label,
            "is_shadow": is_shadow,
            "pixel": {"vertices": pixel_vertices, "edges": pixel_edges},
            "math": {"vertices": math_vertices, "edges": math_edges},
            "has_arc": has_arc,
            "arc_segments": arc_segments,
            "arc_details": arc_details,
            "arc_center_math": arc_center_math,
            "arc_center_pixel": arc_center_pixel
        }

    # 计算圆弧详情
    def _calculate_arc_details(self, resampled_math: np.ndarray, arc_segments: List[Tuple[int, int]],
                              center_math: PointMath) -> List[Dict[str, Any]]:
        details = []
        n = len(resampled_math)
        for start_idx, end_idx in arc_segments:
            start_idx = max(0, min(start_idx, n-1))
            end_idx = max(0, min(end_idx, n-1))
            start_pt = tuple(resampled_math[start_idx])
            end_pt = tuple(resampled_math[end_idx])

            # 半径与角度
            radius = (math.hypot(start_pt[0]-center_math[0], start_pt[1]-center_math[1]) +
                      math.hypot(end_pt[0]-center_math[0], end_pt[1]-center_math[1])) / 2
            angle_start = math.atan2(start_pt[1]-center_math[1], start_pt[0]-center_math[0])
            angle_end = math.atan2(end_pt[1]-center_math[1], end_pt[0]-center_math[0])
            angle_range = abs(angle_end - angle_start)
            if angle_range > math.pi:
                angle_range = 2 * math.pi - angle_range

            # 凹凸性
            is_convex, is_concave = self._determine_convexity(resampled_math, (start_idx+end_idx)//2, center_math, n)
            details.append({
                "start_point": _quant4(start_pt),
                "end_point": _quant4(end_pt),
                "center": _quant4(center_math),
                "radius": round(radius, 4),
                "angle_range": round(angle_range, 4),
                "is_convex": is_convex,
                "is_concave": is_concave
            })
        return details

    # 辅助方法：轮廓重采样、曲率计算等
    def _determine_convexity(self, resampled_math: np.ndarray, mid_idx: int, center_math: PointMath, n: int) -> Tuple[bool, bool]:
        if not (0 < mid_idx < n-1):
            return True, False
        mid_pt = tuple(resampled_math[mid_idx])
        vec_mid = (mid_pt[0]-center_math[0], mid_pt[1]-center_math[1])
        prev_pt, next_pt = tuple(resampled_math[mid_idx-1]), tuple(resampled_math[mid_idx+1])
        vec_tangent = (next_pt[0]-prev_pt[0], next_pt[1]-prev_pt[1])
        vec_normal = (-vec_tangent[1], vec_tangent[0])
        dot = vec_normal[0]*vec_mid[0] + vec_normal[1]*vec_mid[1]
        return dot > self.cfg.arc_convexity_threshold, dot < -self.cfg.arc_convexity_threshold

    def _empty_dual_polygon(self, label: int, is_shadow: bool, arc_center_math: Optional[PointMath],
                           arc_center_pixel: Optional[PointPixel]) -> Dict[str, Any]:
        return {
            "type": "empty", "n": 0, "label": label, "is_shadow": is_shadow,
            "pixel": {"vertices": [], "edges": []}, "math": {"vertices": [], "edges": []},
            "has_arc": False, "arc_segments": [], "arc_details": [],
            "arc_center_math": arc_center_math, "arc_center_pixel": arc_center_pixel
        }

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.cfg.fill_holes:
            mask = self._fill_holes(mask)
        if self.cfg.smooth_edges and self.cfg.smooth_kernel >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.smooth_kernel, self.cfg.smooth_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        if self.cfg.mask_dilate > 0:
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.mask_dilate, self.cfg.mask_dilate))
            mask = cv2.dilate(mask, k2, iterations=1)
        return mask

    def _pixel_contour_to_math(self, pixel_contour: np.ndarray) -> np.ndarray:
        if pixel_contour.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([pixel_to_math((x, y)) for x, y in pixel_contour.reshape(-1, 2).astype(np.int32)], dtype=np.float32)

    def _resample_contour(self, contour: np.ndarray, step: float) -> np.ndarray:
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter < 1e-3:
            return contour
        num_points = max(3, int(perimeter / step))
        resampled, current_dist = [contour[0]], 0.0
        for i in range(1, len(contour)+1):
            pt1, pt2 = contour[i-1], contour[i % len(contour)]
            dist = np.linalg.norm(pt2 - pt1)
            if current_dist + dist < step:
                current_dist += dist
                continue
            ratio = (step - current_dist) / dist
            resampled.append(pt1 + ratio * (pt2 - pt1))
            current_dist = 0.0
        while len(resampled) < num_points:
            resampled.append(resampled[-1])
        resampled = np.array(resampled[:num_points], dtype=np.float32)
        if not np.allclose(resampled[0], resampled[-1]):
            resampled = np.vstack([resampled, resampled[0]])
        return resampled

    def _calculate_curvatures(self, contour: np.ndarray) -> List[float]:
        n, curvatures = len(contour), [0.0]*len(contour)
        for i in range(n):
            pt_prev, pt_curr, pt_next = contour[(i-1)%n], contour[i], contour[(i+1)%n]
            v1, v2 = pt_curr - pt_prev, pt_next - pt_curr
            len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            u1, u2 = v1/len1, v2/len2
            n1, n2 = np.array([-u1[1], u1[0]]), np.array([-u2[1], u2[0]])
            dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
            curvatures[i] = np.arccos(dot) / ((len1 + len2)/2) if (len1 + len2) > 1e-6 else 0.0
        return curvatures

    def _detect_arc_segments(self, curvatures: List[float], threshold: float, min_points: int) -> List[Tuple[int, int]]:
        n, segments, in_arc, start_idx = len(curvatures), [], False, 0
        for i in range(n):
            is_arc = abs(curvatures[i]) >= threshold
            if is_arc and not in_arc:
                in_arc, start_idx = True, i
            elif not is_arc and in_arc:
                end_idx = (i-1) % n
                seg_len = end_idx - start_idx + 1 if end_idx >= start_idx else (end_idx + n - start_idx + 1)
                if seg_len >= min_points:
                    segments.append((start_idx, end_idx))
                in_arc = False
            elif is_arc and in_arc and i == n-1:
                end_idx = n-1
                if (end_idx - start_idx + 1) >= min_points:
                    segments.append((start_idx, end_idx))
                in_arc = False
        return segments

    def _estimate_area(self, mask: np.ndarray) -> Tuple[float, np.ndarray]:
        h, w = mask.shape
        contour = self._largest_contour(mask)
        if self.cfg.area_method == "pixel":
            return float(cv2.countNonZero(mask)), contour
        if self.cfg.area_method == "contour":
            return float(abs(cv2.contourArea(contour, oriented=True))), contour
        # supersample方法
        S = max(2, self.cfg.supersample_scale)
        big = cv2.resize(mask, (w*S, h*S), interpolation=cv2.INTER_AREA)
        return float(big.sum()) / 255.0 / (S*S), contour

    def _binarize_white(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        if self.cfg.binarize_mode == "adaptive":
            blk = self.cfg.adaptive_block_size if self.cfg.adaptive_block_size % 2 else self.cfg.adaptive_block_size + 1
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, self.cfg.adaptive_C)
        _, white = cv2.threshold(gray, self.cfg.thresh, 255, cv2.THRESH_BINARY)
        return white

    def _thicken_lines_on_white(self, white: np.ndarray) -> np.ndarray:
        if self.cfg.line_thicken > 0 and self.cfg.line_thicken_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.line_thicken, self.cfg.line_thicken))
            white = cv2.erode(white, k, iterations=self.cfg.line_thicken_iter)
        return white

    def _remove_border_white(self, white: np.ndarray) -> np.ndarray:
        H, W, pad = white.shape[0], white.shape[1], 1
        white_pad = cv2.copyMakeBorder(white, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        ff_img, mask_ff = white_pad.copy(), np.zeros((H+2*pad+2, W+2*pad+2), np.uint8)
        seeds = []
        for x in range(pad, W+pad):
            seeds.extend([(x, pad), (x, H+pad-1)])
        for y in range(pad, H+pad):
            seeds.extend([(pad, y), (W+pad-1, y)])
        for sx, sy in seeds:
            if ff_img[sy, sx] == 255:
                cv2.floodFill(ff_img, mask_ff, (sx, sy), 128)
        core = ff_img[pad:H+pad, pad:W+pad]
        keep = np.zeros_like(white, dtype=np.uint8)
        keep[core == 255] = 255
        return keep

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((0, 1, 2), dtype=np.int32)
        return contours[np.argmax([cv2.contourArea(c) for c in contours])]

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        inv = cv2.bitwise_not(mask)
        h, w = mask.shape
        flood = inv.copy()
        cv2.floodFill(flood, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)
        return cv2.bitwise_or(mask, cv2.bitwise_not(flood))