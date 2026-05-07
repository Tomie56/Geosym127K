import os
import json
import logging
import math
import random
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set, FrozenSet
from collections import defaultdict
from datetime import datetime
import sympy as sp
import re
from .drawer import GeometryDrawer
from .region import RegionExtractor, RegionExtractConfig
from .shaders import SHADERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 共享类型与坐标转换
PointMath = Tuple[float, float]
PointPixel = Tuple[int, int]
system_font = "DejaVu Sans"

# 坐标转换工具类
class CoordTransformer:
    def __init__(self, scale_factor: float, geom_center_x: float, geom_center_y: float,
                 canvas_center_x: float, canvas_center_y: float):
        self.scale_factor = scale_factor
        self.geom_center_x = geom_center_x
        self.geom_center_y = geom_center_y
        self.canvas_center_x = canvas_center_x
        self.canvas_center_y = canvas_center_y

    def math_to_pixel(self, math_coord: Tuple[float, float]) -> Tuple[int, int]:
        """数学坐标→像素坐标"""
        x, y = math_coord
        x_relative = x - self.geom_center_x
        y_relative = y - self.geom_center_y
        x_scaled = x_relative * self.scale_factor
        y_scaled = y_relative * self.scale_factor
        pixel_x = int(round(self.canvas_center_x + x_scaled))
        pixel_y = int(round(self.canvas_center_y - y_scaled))
        return (pixel_x, pixel_y)
    
    def pixel_to_math(self, pixel_coord: Tuple[int, int]) -> Tuple[float, float]:
        """像素坐标→数学坐标"""
        x, y = pixel_coord
        x_relative = (x - self.canvas_center_x) / self.scale_factor
        y_relative = (self.canvas_center_y - y) / self.scale_factor
        math_x = self.geom_center_x + x_relative
        math_y = self.geom_center_y + y_relative
        return (round(math_x, 4), round(math_y, 4))

COLOR_NAME_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0), 
    "blue": (255, 0, 0), 
    "white": (255, 255, 255), 
    "black": (0, 0, 0), 
    "yellow": (0, 255, 255) 
}

class GeometryAnnotator:
    """几何标注工具：按优先级优化文本位置（无冲突>小offset>大距离>正四方向）"""
    def __init__(self, annotator_config: Dict):
        self.config = self._validate_annotator_config(annotator_config)
        self.point_id_pattern = re.compile(r'^(?!circle_\d+$).+$')
        self.center_point_enabled = self.config.get("center_point", True)

        # if(self.config["line"]["text"]["font_selection"] = "random")
        #     self.point_font =
        #     self.line_font =
        # else:
        #     self.point_font = self.config["point"]["text"]["font"]
        #     self.line_font = self.config["line"]["text"]["font"]
        
        self._random_select_font()

    def _random_select_font(self):
        """
        随机选择字体（适配配置中font为"cv2.FONT_XXX"字符串格式）
        特性：
        1. 线标注候选列表默认包含所有OpenCV内置字体
        2. 自动检查字体有效性，剔除无效字体
        3. 兼容random/fixed选择模式
        """
        # ========== 1. 定义完整的OpenCV字体映射（包含所有内置字体） ==========
        OPENCV_FONT_MAP = {
            # 完整字体名称（匹配配置中的"cv2.FONT_XXX"格式）
            "cv2.FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
            "cv2.FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
            "cv2.FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
            "cv2.FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
            "cv2.FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
            "cv2.FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
            "cv2.FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            "cv2.FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            "cv2.FONT_ITALIC": cv2.FONT_ITALIC,
            # 简写形式（兼容配置）
            "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
            "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
            "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
            "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
            "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
            "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
            "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            "FONT_ITALIC": cv2.FONT_ITALIC
        }

        # ========== 2. 通用工具函数 ==========
        def parse_font(font_str: str) -> int:
            """解析字体字符串为OpenCV常量（容错处理）"""
            if not isinstance(font_str, str):
                return cv2.FONT_HERSHEY_SIMPLEX
            
            # 清理字符串（去除引号、空格）
            clean_str = re.sub(r'["\']|\s', '', font_str.strip())
            # 查找映射，无匹配则返回默认值
            return OPENCV_FONT_MAP.get(clean_str, cv2.FONT_HERSHEY_SIMPLEX)

        def validate_font_candidates(candidates: List[str]) -> Set[int]:
            """验证并过滤字体候选列表，返回有效OpenCV字体常量集合"""
            valid_fonts = set()
            for font_str in candidates:
                font_const = parse_font(font_str)
                # 确保是有效的OpenCV字体常量（排除默认值误判）
                if font_const in OPENCV_FONT_MAP.values():
                    valid_fonts.add(font_const)
            return valid_fonts

        # ========== 3. 获取所有内置字体常量（作为线标注默认候选） ==========
        ALL_BUILTIN_FONTS = list(OPENCV_FONT_MAP.values())

        # ========== 4. 处理点标注字体 ==========
        point_text_cfg = self.config["point"]["text"]
        # 4.1 获取点的候选列表
        point_candidates = point_text_cfg.get(
            "font_candidates",
            ALL_BUILTIN_FONTS
        )
        # 4.2 验证并过滤候选列表
        valid_point_fonts = validate_font_candidates(point_candidates)
        # 4.3 兜底：若无有效字体，使用默认候选
        if not valid_point_fonts:
            valid_point_fonts = {cv2.FONT_HERSHEY_SIMPLEX}
        
        # 4.4 根据选择模式设置字体
        if point_text_cfg.get("font_selection") == "random":
            self.point_font = random.choice(list(valid_point_fonts))
        else:
            self.point_font = parse_font(point_text_cfg.get("font", "cv2.FONT_HERSHEY_SIMPLEX"))

        # ========== 5. 处理线标注字体 ==========
        line_text_cfg = self.config["line"]["text"]
        # 5.1 获取线的候选列表
        line_candidates = line_text_cfg.get(
            "font_candidates",
            ALL_BUILTIN_FONTS
        )
        # 兼容配置中直接传常量的情况（转为字符串再验证）
        if isinstance(line_candidates, list) and len(line_candidates) > 0:
            if not isinstance(line_candidates[0], str):
                line_candidates = [str(f) for f in line_candidates]
        else:
            line_candidates = [line_text_cfg.get("font", "cv2.FONT_HERSHEY_COMPLEX")]
        
        # 5.2 验证并过滤候选列表
        valid_line_fonts = validate_font_candidates(line_candidates)
        # 5.3 兜底：若无有效字体，使用所有内置字体
        if not valid_line_fonts:
            valid_line_fonts = set(ALL_BUILTIN_FONTS)
        
        # 5.4 根据选择模式设置字体
        if line_text_cfg.get("font_selection") == "random":
            self.line_font = random.choice(list(valid_line_fonts))
        else:
            self.line_font = parse_font(line_text_cfg.get("font", "cv2.FONT_HERSHEY_COMPLEX"))

    def _validate_annotator_config(self, cfg: Dict) -> Dict:
        default = {
            "point": {
                "enabled": True,
                "radius": 4,
                "color": (0, 0, 255),
                "text": {
                    "enabled": True,
                    "size": 12,
                    "color": (0, 0, 0),
                    "offset": 8,
                    "edge_margin": 5,
                    "min_pixel_dist": 3,
                    "scale": 0.5,
                    "thickness": 1,
                    "font": cv2.FONT_HERSHEY_SIMPLEX
                }
            },
            "line": {
                "enabled": False,
                "color": (0, 255, 0),
                "width": 2,
                "text": {
                    "enabled": True,
                    "size": 12,
                    "color": (0, 0, 0),
                    "offset": 12,
                    "edge_margin": 5,
                    "min_pixel_dist": 3,
                    "scale": 0.5,
                    "thickness": 1,
                    "font": cv2.FONT_HERSHEY_SIMPLEX
                }
            },
            "perpendicular": {
                "enabled": False,
                "color": (255, 0, 0),
                "width": 2,
                "symbol_size": 10
            }
        }

        def fix_color(color) -> Tuple[int, int, int]:
            if isinstance(color, str) and color.lower() in COLOR_NAME_MAP:
                return COLOR_NAME_MAP[color.lower()]
            if isinstance(color, str):
                try:
                    color_str = color.replace(" ", "").strip('()')
                    channels = list(map(int, color_str.split(',')))[:3]
                except:
                    return (255, 0, 0)
            elif isinstance(color, (list, tuple)):
                channels = list(color)[:3]
            else:
                return (255, 0, 0)
            return tuple(max(0, min(255, int(c))) for c in channels) if len(channels) == 3 else (255, 0, 0)

        config = {
            "enabled": cfg.get("enabled", True),
            "center_point": cfg.get("center_point", True),
            "point": {**default["point"],** cfg.get("point", {})},
            "line": {**default["line"],** cfg.get("line", {})},
            "perpendicular": {**default["perpendicular"],** cfg.get("perpendicular", {})}
        }

        if "bg_color" in config["point"]["text"]:
            del config["point"]["text"]["bg_color"]
        if "bg_color" in config["line"]["text"]:
            del config["line"]["text"]["bg_color"]

        config["point"]["color"] = fix_color(config["point"]["color"])
        config["point"]["text"]["color"] = fix_color(config["point"]["text"]["color"])
        config["line"]["color"] = fix_color(config["line"]["color"])
        config["line"]["text"]["color"] = fix_color(config["line"]["text"]["color"])
        config["perpendicular"]["color"] = fix_color(config["perpendicular"]["color"])

        if not isinstance(config["point"]["text"]["font"], int):
            config["point"]["text"]["font"] = cv2.FONT_HERSHEY_SIMPLEX
        if not isinstance(config["line"]["text"]["font"], int):
            config["line"]["text"]["font"] = cv2.FONT_HERSHEY_SIMPLEX

        return config

    def _parse_math_expr(self, expr_str: str) -> float:
        try:
            return float(sp.sympify(expr_str.replace(" ", "")).evalf())
        except Exception as e:
            logger.warning(f"表达式解析失败: {expr_str}，错误: {e}")
            raise

    def _get_text_pixels(self, text: str, font: int, scale: float, thickness: int, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx, ty = pos
        
        temp_img = np.zeros((text_height + baseline, text_width, 3), dtype=np.uint8)
        cv2.putText(temp_img, text, (0, text_height), font, scale, (255, 255, 255), thickness)
        
        text_mask = np.any(temp_img > 0, axis=2)
        y_coords, x_coords = np.where(text_mask)
        
        text_pixels = set()
        for x, y in zip(x_coords, y_coords):
            abs_x = tx + x
            abs_y = ty - text_height + y
            text_pixels.add((abs_x, abs_y))
        
        return text_pixels
    
    def get_image_contour_pixels(self, img: np.ndarray, min_contour_area: int = 5) -> Set[Tuple[int, int]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 
            0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU 
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_NONE
        )
        
        contour_pixels = set()
        for cnt in contours:
            for point in cnt.reshape(-1, 2):
                x, y = int(point[0]), int(point[1])
                contour_pixels.add((x, y))
        
        return contour_pixels


    def _check_pixel_collision(self, text_pixels: Set[Tuple[int, int]], 
                            existing_text_pixels: List[Set[Tuple[int, int]]],
                            contour_pixels: Set[Tuple[int, int]],
                            img_shape: Tuple[int, int], 
                            edge_margin: int,
                            min_pixel_dist: int = 3) -> bool:
        h, w = img_shape[:2]

        for (x, y) in text_pixels:
            if x < edge_margin or x >= (w - edge_margin) or y < edge_margin or y >= (h - edge_margin):
                return True

        if contour_pixels and (text_pixels & contour_pixels):
            return True

        for exist_pixels in existing_text_pixels:
            if text_pixels & exist_pixels:
                return True

        return False

    def _find_best_text_pos(self, img: np.ndarray, base_px: Tuple[int, int], text: str, 
                        existing_text_pixels: List[Set[Tuple[int, int]]], font: int, 
                        scale: float, thickness: int, 
                        base_offset: int, img_shape: Tuple[int, int], edge_margin: int, 
                        min_pixel_dist: int) -> Tuple[Tuple[int, int], Set[Tuple[int, int]]]:
        candidates = []

        priority_dirs = [
            ("正右", 1, 0), ("正左", -2, 0),
            ("正下", 0, -2), ("正上", 0, 1),
            ("右下", 1, -2), ("左下", -2, -2),
            ("右上", 1, 1), ("左上", -2, 1)
        ]

        offset_levels = [
            base_offset - 8, base_offset - 4, base_offset - 2,
            base_offset + 0, base_offset + 2,
            base_offset + 4
        ]
        offset_levels = [o for o in offset_levels if o >= 3] 
        
        min_contour_area = 5
        contour_pixels = self.get_image_contour_pixels(img, min_contour_area) 
        text_cfg = self.config["point"]["text"]
        min_pixel_dist = text_cfg.get("min_pixel_dist", 3)

        for dir_name, dx_coef, dy_coef in priority_dirs:
            for offset in offset_levels:
                dx = dx_coef * offset
                dy = dy_coef * offset
                tx, ty = base_px[0] + dx, base_px[1] + dy

                text_pixels = self._get_text_pixels(text, font, scale, thickness, (tx, ty))
                has_collision = self._check_pixel_collision(
                    text_pixels, existing_text_pixels, contour_pixels, img_shape, edge_margin, min_pixel_dist
                )
                
                v = 2
                if(dx_coef == 0):
                    if(dy_coef == -1):
                        v = 0
                    else:
                        v = 1
                
                candidates.append({
                    "pos": (tx, ty),
                    "pixels": text_pixels,
                    "has_collision": has_collision,
                    "offset": offset,
                    "is_priority_dir": v,
                })

        candidates.sort(
            key=lambda x: (
                x["has_collision"],
                x["is_priority_dir"],
                -x["offset"]
            )
        )

        best = candidates[0]
        (nx, ny) = best["pos"]
        nx = nx + 3
        return (nx, ny), best["pixels"]

    def draw_annotations(self, img: np.ndarray, data: Dict, transformer: Any) -> np.ndarray:
        if not self.config["enabled"]:
            return img.copy()
            
        img_copy = img.copy()
        existing_text_pixels = []
        non_text_bboxes = []
        points = data.get("points", [])
        lines = data.get("lines", [])
        entities = data.get("entities", [])
        
        circle_center_ids = set()
        for entity in entities:
            if entity.get("type") == "circle":
                center_id = entity.get("center_id")
                if center_id:
                    circle_center_ids.add(center_id)
        
        for point in points:
            if point.get("id") in circle_center_ids:
                point["is_circle_center"] = True
        
        
        valid_points = [p for p in points if self.point_id_pattern.match(p["id"])]
        points_dict = {p["id"]: p for p in valid_points}
        
        existing_text_pixels, non_text_bboxes = self.draw_point_annotations(
            img_copy, valid_points, transformer, existing_text_pixels, non_text_bboxes
        )
        
        existing_text_pixels, non_text_bboxes = self.draw_line_annotations(
            img_copy, lines, points_dict, transformer, existing_text_pixels, non_text_bboxes
        )
        
        self.draw_perpendicular_annotations(
            img_copy, lines, points_dict, transformer, non_text_bboxes, data.get("lines", [])
        )
        
        return img_copy

    def draw_point_annotations(self, img: np.ndarray, points: List[Dict], transformer: Any,
                              existing_text_pixels: List[Set[Tuple[int, int]]], non_text_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[List[Set[Tuple[int, int]]], List[Tuple[int, int, int, int]]]:
        cfg = self.config["point"]
        text_cfg = cfg["text"]
        if not cfg["enabled"] or not text_cfg["enabled"]:
            return existing_text_pixels, non_text_bboxes
        
        img_shape = img.shape
        base_offset = text_cfg["offset"]
        font = self.point_font
        font_scale = text_cfg["scale"]
        font_thickness = text_cfg["thickness"]

        for point in points:
            if self.center_point_enabled == False:
                if point.get("is_center", False):
                    if not point.get("is_circle_center", False):
                        if point["id"].startswith("O"):
                            continue          
            try:
                pid = point["id"]
                x = self._parse_math_expr(point["x"]["expr"])
                y = self._parse_math_expr(point["y"]["expr"])
                px, py = transformer.math_to_pixel((x, y))
                
                # 用cv2绘制点
                point_color = cfg["color"]
                radius = cfg["radius"]
                cv2.circle(img, (px, py), radius, point_color, -1)
                non_text_bboxes.append((px - radius, py - radius, px + radius, py + radius))
                
                # 用cv2绘制文本
                label = point.get("label", pid)
                if label:
                    best_pos, text_pixels = self._find_best_text_pos(
                        img,
                        base_px=(px, py),
                        text=label,
                        existing_text_pixels=existing_text_pixels,
                        font=font,
                        scale=font_scale,
                        thickness=font_thickness,
                        base_offset=base_offset,
                        img_shape=img_shape,
                        edge_margin=text_cfg["edge_margin"],
                        min_pixel_dist=text_cfg["min_pixel_dist"]
                    )
                    
                    cv2.putText(
                        img, label, best_pos, font, font_scale,
                        text_cfg["color"], font_thickness, cv2.LINE_AA
                    )
                    
                    existing_text_pixels.append(text_pixels)
            except Exception as e:
                logger.warning(f"点标注失败 (id: {pid}): {e}")
                continue
        
        return existing_text_pixels, non_text_bboxes

    def draw_line_annotations(self, img: np.ndarray, lines: List[Dict], points: Dict, transformer: Any,
                             existing_text_pixels: List[Set[Tuple[int, int]]], non_text_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[List[Set[Tuple[int, int]]], List[Tuple[int, int, int, int]]]:
        cfg = self.config["line"]
        text_cfg = cfg["text"]
        if not cfg["enabled"] or not text_cfg["enabled"]:
            return existing_text_pixels, non_text_bboxes
        
        img_shape = img.shape
        base_offset = text_cfg["offset"]
        font = self.line_font
        font_scale = text_cfg["scale"]
        font_thickness = text_cfg["thickness"]

        for line in lines:
            if line.get("type") == "perpendicular":
                continue
            
            try:
                start_id = line["start_point_id"]
                end_id = line["end_point_id"]
                if start_id not in points or end_id not in points:
                    logger.debug(f"线标注跳过（无效端点）: {line.get('id')}")
                    continue
                
                start_p = points[start_id]
                end_p = points[end_id]
                s_x = self._parse_math_expr(start_p["x"]["expr"])
                s_y = self._parse_math_expr(start_p["y"]["expr"])
                e_x = self._parse_math_expr(end_p["x"]["expr"])
                e_y = self._parse_math_expr(end_p["y"]["expr"])
                
                s_px, s_py = transformer.math_to_pixel((s_x, s_y))
                e_px, e_py = transformer.math_to_pixel((e_x, e_y))
                
                # 用cv2绘制线段
                line_color = cfg["color"]
                cv2.line(
                    img, (s_px, s_py), (e_px, e_py),
                    line_color, cfg["width"], cv2.LINE_AA
                )
                line_bbox = (
                    min(s_px, e_px) - cfg["width"],
                    min(s_py, e_py) - cfg["width"],
                    max(s_px, e_px) + cfg["width"],
                    max(s_py, e_py) + cfg["width"]
                )
                non_text_bboxes.append(line_bbox)
                
                # 用cv2绘制线文本
                label = line.get("label", line.get("id"))
                if label:
                    mid_px, mid_py = (s_px + e_px) // 2, (s_py + e_py) // 2
                    
                    best_pos, text_pixels = self._find_best_text_pos(
                        img,
                        base_px=(mid_px, mid_py),
                        text=label,
                        existing_text_pixels=existing_text_pixels,
                        font=font,
                        scale=font_scale,
                        thickness=font_thickness,
                        base_offset=base_offset,
                        img_shape=img_shape,
                        edge_margin=text_cfg["edge_margin"],
                        min_pixel_dist=text_cfg["min_pixel_dist"]
                    )
                    
                    cv2.putText(
                        img, label, best_pos, font, font_scale,
                        text_cfg["color"], font_thickness, cv2.LINE_AA
                    )
                    
                    existing_text_pixels.append(text_pixels)
            except Exception as e:
                logger.warning(f"线标注失败 (id: {line.get('id')}): {e}")
                continue
        
        return existing_text_pixels, non_text_bboxes


    def draw_perpendicular_annotations(self, img: np.ndarray, lines: List[Dict], points: Dict, 
                                    transformer: Any, non_text_bboxes: List[Tuple[int, int, int, int]],
                                    all_lines: List[Dict]) -> None:
        cfg = self.config["perpendicular"]
        if not cfg["enabled"]:
            return
        
        square_side = cfg["symbol_size"] 
        line_id_map = {line["id"]: line for line in all_lines}
        line_color = cfg["color"]
        line_width = cfg["width"]

        for line in lines:
            if line.get("type") != "perpendicular":
                continue
            
            try:
                foot_id = line["end_point_id"]
                vert_start_id = line["start_point_id"]
                desc = line.get("description", "")
                
                if foot_id not in points or vert_start_id not in points:
                    logger.debug(f"垂足/垂线起点无效: 垂足{foot_id}，起点{vert_start_id}")
                    continue
                
                vert_start_p = points[vert_start_id]
                vs_x = self._parse_math_expr(vert_start_p["x"]["expr"])
                vs_y = self._parse_math_expr(vert_start_p["y"]["expr"])
                vs_px, vs_py = transformer.math_to_pixel((vs_x, vs_y))
                
                foot_p = points[foot_id]
                foot_x = self._parse_math_expr(foot_p["x"]["expr"])
                foot_y = self._parse_math_expr(foot_p["y"]["expr"])
                foot_px, foot_py = transformer.math_to_pixel((foot_x, foot_y))

                line_match = re.search(r'to line (\w+\d*)', desc)
                if not line_match:
                    dir_host = (1, 0)
                    dir_vert = (0, 1)
                else:
                    host_line_id = line_match.group(1)
                    if host_line_id not in line_id_map:
                        dir_host = (1, 0)
                        dir_vert = (0, 1)
                    else:
                        host_line = line_id_map[host_line_id]
                        h_start_id = host_line["start_point_id"]
                        h_end_id = host_line["end_point_id"]
                        if h_start_id not in points or h_end_id not in points:
                            dir_host = (1, 0)
                            dir_vert = (0, 1)
                        else:
                            h_start_p = points[h_start_id]
                            h_end_p = points[h_end_id]
                            hs_x = self._parse_math_expr(h_start_p["x"]["expr"])
                            hs_y = self._parse_math_expr(h_start_p["y"]["expr"])
                            he_x = self._parse_math_expr(h_end_p["x"]["expr"])
                            he_y = self._parse_math_expr(h_end_p["y"]["expr"])
                            
                            host_vec = (he_x - hs_x, he_y - hs_y)
                            host_len = math.hypot(host_vec[0], host_vec[1])
                            if host_len < 1e-3:
                                dir_host = (1, 0)
                            else:
                                dir_host = (host_vec[0]/host_len, host_vec[1]/host_len)
                            
                            foot_to_hstart = math.hypot(foot_x - hs_x, foot_y - hs_y)
                            foot_to_hend = math.hypot(foot_x - he_x, foot_y - he_y)
                            if foot_to_hend < foot_to_hstart:
                                dir_host = (-dir_host[0], -dir_host[1])
                            
                            vert_vec = (foot_x - vs_x, foot_y - vs_y)
                            vert_len = math.hypot(vert_vec[0], vert_vec[1])
                            if vert_len < 1e-3:
                                dir_vert = (0, 1)
                            else:
                                dir_vert = (vert_vec[0]/vert_len, vert_vec[1]/vert_len)
                            
                            dot_product = dir_host[0] * dir_vert[0] + dir_host[1] * dir_vert[1]
                            if abs(dot_product) > 1e-3:
                                dir_vert = (dir_host[1], -dir_host[0])

                host_end_px = foot_px + dir_host[0] * square_side
                host_end_py = foot_py + dir_host[1] * square_side
                
                vert_end_px = foot_px + dir_vert[0] * square_side
                vert_end_py = foot_py + dir_vert[1] * square_side

                # 用cv2绘制L形垂直符号
                cv2.line(
                    img, (int(foot_px), int(foot_py)), (int(host_end_px), int(host_end_py)),
                    line_color, line_width, cv2.LINE_AA
                )
                cv2.line(
                    img, (int(foot_px), int(foot_py)), (int(vert_end_px), int(vert_end_py)),
                    line_color, line_width, cv2.LINE_AA
                )

                sym_min_px = min(foot_px, host_end_px, vert_end_px) - line_width
                sym_min_py = min(foot_py, host_end_py, vert_end_py) - line_width
                sym_max_px = max(foot_px, host_end_px, vert_end_px) + line_width
                sym_max_py = max(foot_py, host_end_py, vert_end_py) + line_width
                non_text_bboxes.append((int(sym_min_px), int(sym_min_py), int(sym_max_px), int(sym_max_py)))
                
            except Exception as e:
                logger.warning(f"垂线标注失败 (id: {line.get('id')}): {e}")
                continue

class EnhancedDrawer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enhanced_jsons: List[Dict] = [] 
        
        # 路径配置
        self.base_output_dir = self.config["global"]["output_root"]
        self.raw_dir = os.path.join(self.base_output_dir, "images/raw")
        self.shaded_dir = os.path.join(self.base_output_dir, "images/shaded")
        self.annotated_dir = os.path.join(self.base_output_dir, "images/annotated")
        self.json_output_dir = os.path.join(self.base_output_dir, "json/shaded")
        self.shaded_path = ''
        self.success = 0
        self.failure = 0
        
        # 创建必要目录
        for dir_path in [self.raw_dir, self.shaded_dir, self.annotated_dir, self.json_output_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.shaded_jsonl_path = self.config["shader"].get(
            "jsonl_path", 
            os.path.join(self.json_output_dir, f"shaded_{self._get_timestamp()}.jsonl")
        )
        
        # 配置参数
        self.shader_enabled = self.config["shader"].get("enabled", True)
        self.annotator_enabled = self.config.get("annotator", {}).get("enabled", True)
        
        # 匹配阈值
        self.x_attempts = max(1, self.config["shader"].get("x_attempts", 4))
        self.distance_threshold = self.config["shader"].get("distance_threshold", 10) + self.config["drawer"].get("line_width",3)
        self.match_threshold = self.config["shader"].get("match_threshold", 0.9)
        self.match_threshold_arc = self.config["shader"].get("match_threshold_arc", 0.75)
        self.min_sample_points = self.config["shader"].get("min_sample_points", 20)

        self.selected_region_labels = defaultdict(list)
        self.region_extractor = self._init_region_extractor()
        self.drawer = None
        self.processed_results = []
        self.annotator = GeometryAnnotator(self.config.get("annotator", {}))
        
    def set_enhanced_data(self, enhanced_jsons: List[Dict]) -> None:
        """设置增强JSON数据（用于替代从文件加载的场景）"""
        if not isinstance(enhanced_jsons, list):
            raise ValueError("增强JSON数据必须为列表类型")
        self.enhanced_jsons = enhanced_jsons
        logger.info(f"已设置 {len(enhanced_jsons)} 条增强JSON数据")
        
    def _save_jsonl(self) -> None:
        """将收集的所有结果写入JSONL文件"""
        with open(self.shaded_jsonl_path, "w", encoding="utf-8") as f:
            for result in self.processed_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    def _init_region_extractor(self) -> RegionExtractor:
        """初始化区域提取器"""
        extract_cfg = RegionExtractConfig(
            thresh=self.config["region_extractor"]["thresh"],
            binarize_mode=self.config["region_extractor"]["binarize_mode"],
            min_area=self.config["shader"]["min_pixel_area"],
            fill_holes=self.config["region_extractor"]["fill_holes"],
            arc_detect=self.config["region_extractor"]["arc_detect"],
            curvature_threshold=self.config["region_extractor"]["curvature_threshold"]
        )
        return RegionExtractor(extract_cfg)

    def load_enhanced_jsonl(self, jsonl_path: str) -> None:
        """加载enhanced jsonl文件"""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.enhanced_jsons = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"加载 {len(self.enhanced_jsons)} 条enhanced数据")

    @staticmethod
    def _parse_expr(expr_str: str) -> float:
        try:
            expr_str = expr_str.replace("pi", "sp.pi").replace("sqrt", "sp.sqrt")
            return float(eval(expr_str))
        except Exception as e:
            logger.warning(f"表达式解析失败: {expr_str}，错误: {e}")
            return 0.0
        
    def _process_shaded_image(self, img_bgr: np.ndarray, data: Dict, idx: int, attempt: int,
                             geom_params: Dict, raw_save_path: str, annotated_raw_path: str,
                             transformer: CoordTransformer) -> None:
        """处理单张图像的阴影生成与实体匹配"""
        # 1. 提取原始图像的区域
        meta = self._get_meta(data)
        masks, _, _, _, polygons = self.region_extractor.extract(img_bgr, meta)
        if not masks or not polygons:
            logger.warning(f"第{idx}条数据的第{attempt}次尝试无有效区域，跳过")
            return

        # 2. 选择阴影区域，按面积排序，排除已选
        exclude_labels = self.selected_region_labels.get(idx, [])
        n_regions = self.config["shader"]["n_regions"]
        select_n = np.random.randint(n_regions[0], n_regions[1]+1)
        selected_regions = self._select_regions(masks, polygons, select_n, exclude_labels)
        if not selected_regions:
            logger.warning(f"第{idx}条数据的第{attempt}次尝试无可用未选区域，跳过")
            return

        # 3. 记录选中区域标签
        current_selected_labels = [region["label"] for region in selected_regions]
        if idx not in self.selected_region_labels:
            self.selected_region_labels[idx] = []
        self.selected_region_labels[idx].extend(current_selected_labels)

        # 4. 阴影实体初始化
        shadow_entities = []
        original_lines = data.get("lines", [])
        original_arcs = data.get("arcs", [])
        original_points = data.get("points", [])
        
        # 5. 阴影配置
        shaded_img = img_bgr.copy()
        shadow_type = np.random.choice(self.config["shader"]["shadow_types"])
        shader = SHADERS[shadow_type]
        shader_params = self._get_shader_params(shadow_type)
        
        shadow_descriptions_text = []

        for region in selected_regions:
            # 从区域掩码中提取原始轮廓
            raw_contour, _ = self._find_contours_from_mask(region['mask'])
            if len(raw_contour) == 0:
                logger.warning(f"区域 {region['label']} 无有效轮廓，跳过匹配")
                continue
            
            # 调用核心匹配逻辑
            entity_data = self._match_primitives_to_contour(
                raw_contour=raw_contour,
                original_lines=original_lines,
                original_arcs=original_arcs,
                original_points=original_points,
                transformer=transformer
            )
            
            if entity_data["validity"] == False:
                logger.warning(f"区域 {region['label']} 轮廓匹配失败，跳过阴影生成")
                self.failure += 1
                continue
            else:
                self.success += 1
                
            desc_parts = [f"Region {region['label']} is shaded with {shadow_type} pattern"]
            if shadow_type == "solid":
                color = shader_params.get("color", "unknown")
                desc_parts.append(f"in color {color}")
            
            final_desc = " ".join(desc_parts) + "."
            shadow_descriptions_text.append(final_desc)

            # 构建阴影实体信息（与原有格式一致）
            shadow_entity = {
                "type": "shadow",
                "region_label": region["label"],
                "description": final_desc,
                "points": entity_data["points"],
                "lines": entity_data["lines"],
                "arcs": entity_data["arcs"],
                "ordered_loops": entity_data["ordered_loops"],
                "validity": entity_data["validity"],
                "shader_params": {
                    "type": shadow_type,
                    **shader_params
                }
            }
            shadow_entities.append(shadow_entity)
            shaded_img = shader.apply(shaded_img, region["mask"],** shader_params)
            
        # 6. 保存结果（阴影图像、标注、JSON）
        raw_filename = os.path.basename(raw_save_path)
        shaded_filename = f"shaded_{idx}_attempt_{attempt}_{raw_filename}"
        shaded_path = os.path.join(self.shaded_dir, shaded_filename)
        cv2.imwrite(shaded_path, shaded_img)
        
        # 生成标注后的阴影图像
        annotated_shaded_img = self.annotator.draw_annotations(shaded_img, data, transformer)
        annotated_shaded_filename = f"annotated_shaded_{idx}_attempt_{attempt}_{raw_filename}"
        annotated_shaded_path = os.path.join(self.annotated_dir, annotated_shaded_filename)
        cv2.imwrite(annotated_shaded_path, annotated_shaded_img)
        
        logger.info(f"成功标注第{idx}条数据第{attempt}次尝试的阴影图像")

        # 更新并保存结果数据
        new_data = data.copy()
        new_data["entities"] = new_data.get("entities", []) + shadow_entities
        
        original_desc = new_data.get("description", "")
        if shadow_descriptions_text:
            new_data["description"] = original_desc + "\n" + "\n".join(shadow_descriptions_text)
            
        new_data.update({
            "raw_path": raw_save_path,
            "annotated_raw_path": annotated_raw_path,
            "shaded_path": shaded_path,
            "annotated_shaded_path": annotated_shaded_path,
            "shadow_type": shadow_type,
            "shader_enabled": True
        })
        self.processed_results.append(new_data)
        self.shaded_path = shaded_path

        logger.info(f"第{idx}条数据第{attempt}次尝试完成，阴影图像路径：{shaded_path}")

    def _get_meta(self, data: Dict) -> Dict:
        """从JSON数据中提取元信息（几何中心）"""
        points = data.get("points", [])
        if not points:
            return {}
        math_pts = []
        for pt in points:
            try:
                x = float(pt["x"]["expr"])
                y = float(pt["y"]["expr"])
                math_pts.append((x, y))
            except (KeyError, ValueError, TypeError):
                continue
        if not math_pts:
            return {}
        center_math = (
            sum(p[0] for p in math_pts) / len(math_pts),
            sum(p[1] for p in math_pts) / len(math_pts)
        )
        return {"center_math": center_math}

    def _select_regions(
        self, 
        masks: List[np.ndarray], 
        polygons: List[Dict], 
        n: int, 
        exclude_labels: List[int]
    ) -> List[Dict]:
        """选择指定数量的未选区域（按面积排序）"""
        available_regions = []
        for mask, polygon in zip(masks, polygons):
            region_label = polygon.get("label")
            if region_label is not None and region_label not in exclude_labels:
                # 确保区域字典包含 mask、polygon、label 三个关键字
                available_regions.append({
                    "mask": mask,
                    "polygon": polygon,
                    "label": region_label
                })
        
        if not available_regions:
            logger.warning(f"无可用未选区域，返回空列表")
            return []
        
        # 选择不超过可用数量的区域
        select_n = min(n, len(available_regions))
        # 按面积降序排序（优先选择大面积区域，面积字段从 polygon 中获取）
        available_regions_sorted = sorted(
            available_regions,
            key=lambda x: x["polygon"].get("pixel", {}).get("area_px_est", 0),
            reverse=True
        )
        return available_regions_sorted[:select_n]

    def _get_shader_params(self, shadow_type: str) -> Dict:
        """生成阴影参数（高度随机化：强度、颜色、间距、角度）"""
        # 1. 随机强度 (基于配置范围)
        intensity = np.random.uniform(*self.config["shader"]["intensity_range"])

        # 辅助函数：生成随机颜色 (BGR格式)
        # --- [修改点] ---
        # 原来是 randint(0, 256)，现在改为 randint(0, 180)
        # 限制最大值为 180，确保生成的颜色较深，避免与白色背景混淆
        def get_random_color():
            return np.random.randint(0, 180, 3).tolist()
        # ----------------

        if shadow_type == "hatch":
            # 间距增加随机微扰 (±20%)
            base_spacing = self.config["shader"]["hatch_spacing"]
            random_spacing = int(base_spacing * np.random.uniform(0.8, 1.2))
            return {
                "spacing": max(3, random_spacing), # 保证最小间距
                "intensity": intensity
            }

        elif shadow_type == "crosshatch":
            # 间距微扰 + 整体角度随机旋转
            base_spacing = self.config["shader"]["crosshatch_spacing"]
            random_spacing = int(base_spacing * np.random.uniform(0.8, 1.2))
            angle_offset = np.random.randint(0, 90) # 随机旋转偏移
            return {
                "spacing": max(3, random_spacing),
                "angle1": 45 + angle_offset, 
                "angle2": 135 + angle_offset,
                "intensity": intensity
            }

        elif shadow_type == "solid":
            # 完全随机颜色 (使用修改后的深色生成逻辑)
            return {
                "color": get_random_color(), 
                "intensity": intensity
            }

        elif shadow_type == "gradient":
            # 随机起始色、结束色、渐变角度 (使用修改后的深色生成逻辑)
            return {
                "start_color": get_random_color(),
                "end_color": get_random_color(),
                "angle_deg": np.random.randint(0, 360),
                "intensity": intensity
            }

        # 默认参数
        return {"intensity": intensity}

    def _find_contours_from_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """从区域掩码中提取主轮廓（确保轮廓为 (N, 2) 格式）"""
        # 处理掩码格式：确保为单通道灰度图
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # 二值化：确保掩码只有 0（背景）和 255（前景）
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # 提取轮廓：只保留最外层轮廓，避免内部噪声
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("从掩码中未提取到任何轮廓")
            return np.array([]), []
        
        # 选择最长的轮廓作为主轮廓（默认区域的核心轮廓）
        main_contour = max(contours, key=cv2.contourArea)
        # 转换为 (N, 2) 格式（方便后续坐标计算）
        main_contour = main_contour.reshape(-1, 2)
        logger.info(f"提取主轮廓：共 {len(main_contour)} 个像素点")
        return main_contour, contours

    def _create_distance_map(self, raw_contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """为轮廓创建距离图"""
        if len(raw_contour) == 0:
            logger.error("轮廓为空，无法创建距离图")
            return np.array([]), (0, 0)
        
        # 1. 获取轮廓的边界框（确定距离图的尺寸，留出 10 像素边距避免采样点超出）
        min_x, min_y = np.min(raw_contour, axis=0).astype(int)
        max_x, max_y = np.max(raw_contour, axis=0).astype(int)
        width = max_x - min_x + 20
        height = max_y - min_y + 20
        offset = (min_x - 10, min_y - 10)  # 距离图的坐标偏移（全局坐标 → 局部坐标）
        
        # 2. 创建轮廓掩码（仅轮廓区域为白色，其余为黑色）
        mask = np.zeros((height, width), dtype=np.uint8)
        # 将轮廓坐标转换为距离图的局部坐标
        translated_contour = raw_contour - np.array(offset)
        cv2.drawContours(mask, [translated_contour.astype(int)], -1, 255, thickness=1)
        
        # 3. 计算距离图（DIST_L2 = 欧氏距离，精度最高）
        # 注意：distanceTransform 输入要求「背景为 255，前景为 0」，因此反转掩码
        distance_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        logger.info(f"创建距离图：尺寸 {width}x{height}，轮廓边界框 [{min_x}, {min_y}] 到 [{max_x}, {max_y}]")
        return distance_map, offset

    @staticmethod
    def _bresenham_line(
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        用 Bresenham 算法生成直线上的采样点，分段插值优化：
        - 线段两端各10%区域：interpolate_step=1.0（像素）
        - 线段中间80%区域：interpolate_step=0.5（像素）
        """
        x0, y0 = start
        x1, y1 = end

        # 第一步：计算直线总长度（像素，浮点精度）
        total_len = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        if total_len < 1e-6:  # 处理重合点（无有效线段）
            return [(x0, y0)]

        # 第二步：生成直线上的连续浮点坐标（基础骨架，确保分段精准）
        min_step = 0.2
        num_total_steps = int(total_len // min_step) + 1
        if num_total_steps < 2:
            num_total_steps = 2  # 至少保证起点和终点

        # 连续坐标列表（(x, y, 累计长度)）
        continuous_points = []
        for step in range(num_total_steps):
            ratio = step / (num_total_steps - 1)  # 0→1 比例
            x = x0 + ratio * (x1 - x0)
            y = y0 + ratio * (y1 - y0)
            cum_len = ratio * total_len  # 当前点到起点的累计长度
            continuous_points.append((x, y, cum_len))

        # 第三步：分段插值（按长度比例划分区域）
        dense_points = []
        len_10pct = total_len * 0.1  # 两端各10%的长度
        start_mid = len_10pct        # 中间区域起点（10%处）
        end_mid = total_len - len_10pct  # 中间区域终点（90%处）

        # 区域步长配置
        def get_step_by_cum_len(cum_len: float) -> float:
            if cum_len <= start_mid or cum_len >= end_mid:
                return 1.0  # 两端10%：步长1.0
            else:
                return 0.5  # 中间80%：步长0.5

        # 遍历连续点，按区域步长取点（避免重复）
        last_added = None  # 记录上一个添加的点（去重）
        for x, y, cum_len in continuous_points:
            current_step = get_step_by_cum_len(cum_len)
            # 计算当前点与上一个添加点的距离
            if last_added is None:
                # 第一个点（起点）直接添加
                added_x = round(x)
                added_y = round(y)
                dense_points.append((added_x, added_y))
                last_added = (added_x, added_y, cum_len)
            else:
                last_x, last_y, last_cum_len = last_added
                dist_to_last = ((x - last_x)**2 + (y - last_y)**2)**0.5
                # 当距离≥当前区域步长时，添加当前点
                if dist_to_last >= current_step - 1e-6:  # 容差避免浮点误差
                    added_x = round(x)
                    added_y = round(y)
                    # 额外去重（避免相邻点重复）
                    if (added_x, added_y) != (last_x, last_y):
                        dense_points.append((added_x, added_y))
                        last_added = (added_x, added_y, cum_len)

        # 确保终点被添加（避免因步长计算遗漏）
        end_x_round = round(x1)
        end_y_round = round(y1)
        if dense_points[-1] != (end_x_round, end_y_round):
            dense_points.append((end_x_round, end_y_round))

        # 最终去重（确保无重复点）
        dense_points = list(dict.fromkeys(dense_points))

        return dense_points

    def _sample_arc_points(
        self,
        center_px: Tuple[float, float],
        radius_px: float,
        start_px: Tuple[float, float],
        end_px: Tuple[float, float],
        is_complete: bool = False
    ) -> List[Tuple[int, int]]:
        """生成圆弧上的所有像素点（Bresenham圆弧算法，密集覆盖无遗漏）"""
        # 像素级坐标转换（仅取整，不优化）
        cx = int(round(center_px[0]))
        cy = int(round(center_px[1]))
        radius = int(round(radius_px))
        
        # 基础校验
        if radius <= 0:
            logger.error("圆弧半径必须大于 0，无法采样")
            return []
        center = (cx, cy)
        start_angle = 0.0
        end_angle = 0.0
        
        # 1. 计算角度范围（兼容完整圆/非完整弧）
        if is_complete:
            end_angle = 2 * math.pi
            logger.debug(f"完整圆采样：圆心 {center}，半径 {radius}，覆盖全角度")
        else:
            # 计算起止点相对于圆心的角度（[-π, π]）
            start_angle = math.atan2(start_px[1] - center_px[1], start_px[0] - center_px[0])
            end_angle = math.atan2(end_px[1] - center_px[1], end_px[0] - center_px[0])
            logger.debug(f"圆弧采样：圆心 {center}，半径 {radius}，角度范围 [{start_angle:.2f}, {end_angle:.2f}] 弧度")
            
            start_angle = 0 - start_angle
            end_angle = 0 - end_angle
            
            if start_angle < 0:
                start_angle = 2 * math.pi + start_angle
            if end_angle < 0:
                end_angle = 2 * math.pi + end_angle
            if end_angle <= start_angle:
                end_angle += 2 * math.pi  # 确保角度范围正确
            
            logger.debug(f"圆弧采样：圆心 {center}，半径 {radius}，角度范围 [{start_angle:.2f}, {end_angle:.2f}] 弧度")
        
        # 2. Bresenham算法生成所有像素点（8分圆对称扩展，确保无遗漏）
        arc_points: Set[Tuple[int, int]] = set()  # 用集合去重（避免对称点重复）
        
        def add_point(x: int, y: int):
            """添加点并校验是否在角度范围内"""
            if is_complete:
                arc_points.add((x, y))
                return
            # 计算当前点相对于圆心的角度
            point_angle = math.atan2(y - cy, x - cx)
            
            point_angle = 0 - point_angle
            if point_angle < 0:
                point_angle = 2 * math.pi + point_angle
            
            # 校验是否在目标角度范围内（包含边界）
            if start_angle <= point_angle <= end_angle:
                arc_points.add((x, y))
        
        # Bresenham核心逻辑（处理第一分圆，其余分圆对称扩展）
        x, y = 0, radius
        d = 3 - 2 * radius  # 初始决策参数
        
        while x <= y:
            # 8分圆对称点（覆盖整个圆的所有像素）
            add_point(cx + x, cy + y)
            add_point(cx - x, cy + y)
            add_point(cx + x, cy - y)
            add_point(cx - x, cy - y)
            add_point(cx + y, cy + x)
            add_point(cx - y, cy + x)
            add_point(cx + y, cy - x)
            add_point(cx - y, cy - x)
            
            # 更新决策参数和坐标
            if d < 0:
                d += 4 * x + 6
            else:
                d += 4 * (x - y) + 10
                y -= 1
            x += 1
        
        # 转换为有序列表（按角度排序，不影响匹配，仅保持一致性）
        def sort_by_angle(point: Tuple[int, int]) -> float:
            angle = math.atan2(point[1] - cy, point[0] - cx)
            return angle if angle >= 0 else angle + 2 * math.pi
        
        sorted_points = sorted(arc_points, key=sort_by_angle)
        logger.debug(f"圆弧采样完成：共生成 {len(sorted_points)} 个像素点（无重复）")
        return sorted_points

    def _parse_point_coords(
        self, 
        point_id: str, 
        original_points: List[Dict], 
        transformer: CoordTransformer
    ) -> Tuple[float, float]:
        """解析点 ID 对应的像素坐标（数学坐标 → 像素坐标）"""
        try:
            # 1. 查找对应的点
            point = next(p for p in original_points if p.get("id") == point_id)
            # 2. 解析点的数学坐标
            x_math = self._parse_expr(point["x"]["expr"])
            y_math = self._parse_expr(point["y"]["expr"])
            # 3. 转换为像素坐标
            x_px, y_px = transformer.math_to_pixel((x_math, y_math))
            # logger.debug(f"解析点 {point_id}：数学坐标 ({x_math:.2f}, {y_math:.2f}) → 像素坐标 ({x_px:.2f}, {y_px:.2f})")
            return (x_px, y_px)
        except StopIteration:
            logger.error(f"未找到点 ID：{point_id}")
            raise
        except KeyError as e:
            logger.error(f"点 {point_id} 缺少必要字段：{str(e)}")
            raise
        except Exception as e:
            logger.error(f"解析点 {point_id} 坐标失败：{str(e)}")
            raise

    def _calculate_match_score(
        self,
        sample_points: List[Tuple[int, int]],
        distance_map: np.ndarray,
        offset: Tuple[int, int]
    ) -> Tuple[float, int, int]:
        """计算采样点的匹配度（有效点占比）"""
        if not sample_points:
            logger.warning("采样点列表为空，匹配度为 0")
            return 0.0, 0, 0
        if distance_map.size == 0:
            logger.error("距离图为空，无法计算匹配度")
            raise ValueError("距离图为空")
        
        total_points = len(sample_points)
        valid_count = 0
        offset_x, offset_y = offset
        map_height, map_width = distance_map.shape

        # 遍历所有采样点，统计有效点（距离 ≤ 阈值）
        for idx, (x_px, y_px) in enumerate(sample_points):
            # 1. 将采样点的全局坐标转换为距离图的局部坐标
            local_x = x_px - offset_x
            local_y = y_px - offset_y
            
            # 2. 检查局部坐标是否在距离图范围内（避免越界）
            if 0 <= local_x < map_width and 0 <= local_y < map_height:
                # 3. 获取该点到轮廓的最近距离
                min_distance = distance_map[int(local_y), int(local_x)]
                # 4. 判断是否为有效点
                if min_distance <= self.distance_threshold:
                    valid_count += 1
                
                # 打印前 5 个点的详细信息（方便调试）
                # if idx < 5:
                #     logger.debug(f"采样点 {idx+1}：全局坐标 ({x_px}, {y_px}) → 局部坐标 ({local_x:.0f}, {local_y:.0f})，最近距离 {min_distance:.2f}")

        # 计算匹配度（避免除零错误）
        match_score = valid_count / total_points if total_points > 0 else 0.0
        # logger.info(f"匹配度计算完成：总采样点 {total_points}，有效点 {valid_count}，匹配度 {match_score:.2%}（阈值：{self.match_threshold:.2%}）")
        return match_score, valid_count, total_points

    def _match_lines(
        self,
        original_lines: List[Dict],
        original_points: List[Dict],
        transformer: CoordTransformer,
        distance_map: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[str]:
        """匹配所有待匹配直线，返回匹配成功的直线 ID 列表"""
        matched_ids = []
        total_lines = len(original_lines)
        if total_lines == 0:
            logger.info("无待匹配的直线")
            return matched_ids
        
        logger.info(f"\n===== 开始匹配直线（共 {total_lines} 条）=====")
        for line in original_lines:
            line_id = line.get("id", f"UnknownLine_{id(line)}")
            
            try:
                # 1. 解析直线的起止点像素坐标
                start_id = line["start_point_id"]
                end_id = line["end_point_id"]
                start_px = self._parse_point_coords(start_id, original_points, transformer)
                end_px = self._parse_point_coords(end_id, original_points, transformer)
                
                # 2. 生成直线的像素采样点
                sample_points = self._bresenham_line(
                    start=(int(round(start_px[0])), int(round(start_px[1]))),
                    end=(int(round(end_px[0])), int(round(end_px[1])))
                )
                
                if len(sample_points) == 0:
                    logger.warning(f"直线 {line_id} 无法生成采样点，跳过")
                    continue

                # 3. 计算匹配度
                match_score, _, _ = self._calculate_match_score(
                    sample_points=sample_points,
                    distance_map=distance_map,
                    offset=offset
                )
                
                # 4. 判断是否匹配成功
                if match_score >= self.match_threshold:
                    matched_ids.append(line_id)
                    logger.info(f"直线 {line_id} 匹配成功！匹配度 {match_score:.2%} ≥ 阈值 {self.match_threshold:.2%}")
                else:
                    logger.info(f"直线 {line_id} 匹配失败！匹配度 {match_score:.2%} < 阈值 {self.match_threshold:.2%}")
            
            except Exception as e:
                logger.error(f"直线 {line_id} 匹配过程出错：{str(e)}，跳过")
                continue
        
        logger.info(f"\n直线匹配完成：共 {len(matched_ids)} 条匹配成功，ID 列表：{matched_ids}")
        return matched_ids

    def _match_arcs(
        self,
        original_arcs: List[Dict],
        original_points: List[Dict],
        transformer: CoordTransformer,
        distance_map: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[str]:
        """匹配所有待匹配圆弧，返回匹配成功的圆弧 ID 列表"""
        matched_ids = []
        total_arcs = len(original_arcs)
        if total_arcs == 0:
            logger.info("无待匹配的圆弧")
            return matched_ids
        
        logger.info(f"\n===== 开始匹配圆弧（共 {total_arcs} 条）=====")
        for arc in original_arcs:
            arc_id = arc.get("id", f"UnknownArc_{id(arc)}")
            logger.info(f"\n--- 匹配圆弧：{arc_id} ---")
            
            try:
                # 1. 解析圆弧的关键参数
                center_id = arc["center_point_id"]
                start_id = arc["start_point_id"]
                end_id = arc["end_point_id"]
                radius_expr = arc["radius"]["expr"]
                is_complete = arc.get("is_complete", False)
                
                # 2. 解析像素坐标和半径
                center_px = self._parse_point_coords(center_id, original_points, transformer)
                start_px = self._parse_point_coords(start_id, original_points, transformer)
                end_px = self._parse_point_coords(end_id, original_points, transformer)
                radius_math = self._parse_expr(radius_expr)
                radius_px = radius_math * transformer.scale_factor  # 数学单位 → 像素单位
                
                # 3. 生成圆弧的像素采样点
                sample_points = self._sample_arc_points(
                    center_px=center_px,
                    radius_px=radius_px,
                    start_px=start_px,
                    end_px=end_px,
                    is_complete=is_complete
                )
                
                # 4. 计算匹配度
                match_score, _, _ = self._calculate_match_score(
                    sample_points=sample_points,
                    distance_map=distance_map,
                    offset=offset
                )
                
                # 5. 判断是否匹配成功
                if match_score >= self.match_threshold_arc:
                    matched_ids.append(arc_id)
                    logger.info(f"圆弧 {arc_id} 匹配成功！匹配度 {match_score:.2%} ≥ 阈值 {self.match_threshold_arc:.2%}")
                else:
                    logger.info(f"圆弧 {arc_id} 匹配失败！匹配度 {match_score:.2%} < 阈值 {self.match_threshold_arc:.2%}")
            
            except Exception as e:
                logger.error(f"圆弧 {arc_id} 匹配过程出错：{str(e)}，跳过")
                continue
        
        logger.info(f"\n圆弧匹配完成：共 {len(matched_ids)} 条匹配成功，ID 列表：{matched_ids}")
        return matched_ids

    def _extract_matched_point_ids(
        self,
        matched_line_ids: List[str],
        matched_arc_ids: List[str],
        original_lines: List[Dict],
        original_arcs: List[Dict]
    ) -> List[str]:
        """从匹配成功的线/弧中提取关联的点 ID（去重）"""
        matched_point_ids = set() 
        
        # 1. 提取匹配直线关联的点
        for line_id in matched_line_ids:
            try:
                line = next(l for l in original_lines if l.get("id") == line_id)
                matched_point_ids.add(line["start_point_id"])
                matched_point_ids.add(line["end_point_id"])
            except StopIteration:
                logger.warning(f"未找到匹配的直线 {line_id}，跳过其关联点提取")
        
        # 2. 提取匹配圆弧关联的点
        for arc_id in matched_arc_ids:
            try:
                arc = next(a for a in original_arcs if a.get("id") == arc_id)
                # matched_point_ids.add(arc["center_point_id"])
                matched_point_ids.add(arc["start_point_id"])
                matched_point_ids.add(arc["end_point_id"])
            except StopIteration:
                logger.warning(f"未找到匹配的圆弧 {arc_id}，跳过其关联点提取")
        
        # 转换为列表并排序（保持一致性）
        matched_point_list = sorted(list(matched_point_ids))
        logger.info(f"\n提取匹配关联点：共 {len(matched_point_list)} 个，ID 列表：{matched_point_list}")
        return matched_point_list

    def _check_geometric_validity(
            self,
            matched_line_ids: List[str],
            matched_arc_ids: List[str],
            original_lines: List[Dict],
            original_arcs: List[Dict]
        ) -> Dict:
            """
            检验匹配到的边和弧是否能形成一个或多个封闭的、有效的、不相交的环。

            检验规则:
            1. 所有点必须连接且只能连接 2 个基元。
            2. 所有基元必须能被遍历形成一个或多个封闭的环。
            3. 环与环之间必须是不相交的（无共享点）。
            4. 不允许存在孤立的基元。
            5. 一个环必须满足以下条件之一：
                a. 至少包含 3 个边。
                b. 恰好包含 2 个边，且这两个边中至少有一个是圆弧，并且它们连接着同一对顶点。

            Args:
                matched_line_ids: 匹配到的线的ID列表。
                matched_arc_ids: 匹配到的弧的ID列表。
                original_lines: 所有原始线的列表(已经传入minimal_lines)。
                original_arcs: 所有原始弧的列表(已经传入minimal_arcs)。

            Returns:
                一个包含检验结果的字典，新增 `ordered_points` 字段记录环的有序点序列。
            """
            result = {
                "is_valid": True,
                "is_closed": False,
                "num_loops": 0,
                "error_message": "",
                "loop_details": []
            }

            if not matched_line_ids and not matched_arc_ids:
                result["error_message"] = "没有匹配到任何线或弧。"
                result["is_valid"] = False
                return result

            # 1. 数据准备：重构edges结构，支持同一顶点对多基元（关键修复）
            edges: Dict[FrozenSet[str], List[Dict]] = {}  # key: 顶点对，value: 该顶点对的所有基元（线/弧）
            vertices: Set[str] = set()
            primitive_id_map: Dict[str, Dict] = {}  # 所有基元的ID映射（线+弧）

            line_id_map = {line["id"]: line for line in original_lines}
            arc_id_map = {arc["id"]: arc for arc in original_arcs}

            # 添加匹配的线到edges（支持同一顶点对多基元）
            for line_id in matched_line_ids:
                line = line_id_map.get(line_id)
                if not line:
                    result["error_message"] = f"在原始线列表中未找到ID为 '{line_id}' 的线。"
                    result["is_valid"] = False
                    return result
                p1, p2 = line["start_point_id"], line["end_point_id"]
                edge_key = frozenset({p1, p2})
                if edge_key not in edges:
                    edges[edge_key] = []
                line_info = {"id": line_id, "type": "line", "points": edge_key, "vertices": (p1, p2)}
                edges[edge_key].append(line_info)
                vertices.add(p1)
                vertices.add(p2)
                primitive_id_map[line_id] = line

            # 添加匹配的弧到edges（支持同一顶点对多基元）
            for arc_id in matched_arc_ids:
                arc = arc_id_map.get(arc_id)
                if not arc:
                    result["error_message"] = f"在原始弧列表中未找到ID为 '{arc_id}' 的弧。"
                    result["is_valid"] = False
                    return result
                p1, p2 = arc["start_point_id"], arc["end_point_id"]
                edge_key = frozenset({p1, p2})
                if edge_key not in edges:
                    edges[edge_key] = []
                arc_info = {"id": arc_id, "type": "arc", "points": edge_key, "vertices": (p1, p2)}
                edges[edge_key].append(arc_info)
                vertices.add(p1)
                vertices.add(p2)
                primitive_id_map[arc_id] = arc

            # 2. 规则1验证：所有顶点的度数必须为2（关键修复：遍历所有基元统计度数）
            vertex_degree: Dict[str, int] = {v: 0 for v in vertices}
            for edge_key in edges:
                for primitive in edges[edge_key]:
                    p1, p2 = primitive["vertices"]
                    vertex_degree[p1] += 1
                    vertex_degree[p2] += 1

            for vertex, degree in vertex_degree.items():
                if degree != 2:
                    result["error_message"] = f"点 '{vertex}' 的连接数为 {degree}，不符合必须连接 2 个基元的规则。"
                    result["is_valid"] = False
                    return result

            if not vertices:
                result["error_message"] = "没有找到任何有效的顶点。"
                result["is_valid"] = False
                return result

            # 3. 寻找所有连通分量并验证是否为环（关键修复：以基元为单位遍历）
            all_primitives: List[Dict] = []  # 所有匹配的基元（线+弧）
            for edge_key in edges:
                all_primitives.extend(edges[edge_key])
            visited_primitives: Set[str] = set()  # 已访问的基元ID

            while all_primitives:
                # 取第一个未访问的基元作为起点
                start_primitive = None
                for prim in all_primitives:
                    if prim["id"] not in visited_primitives:
                        start_primitive = prim
                        break
                if not start_primitive:
                    break

                # BFS 寻找连通分量（基于基元共享顶点）
                component_primitives: List[Dict] = []
                queue = [start_primitive]
                visited_primitives.add(start_primitive["id"])

                while queue:
                    current_prim = queue.pop(0)
                    component_primitives.append(current_prim)
                    current_p1, current_p2 = current_prim["vertices"]

                    # 寻找与当前基元共享顶点的未访问基元
                    for candidate_prim in all_primitives:
                        if candidate_prim["id"] in visited_primitives:
                            continue
                        cand_p1, cand_p2 = candidate_prim["vertices"]
                        if current_p1 in (cand_p1, cand_p2) or current_p2 in (cand_p1, cand_p2):
                            queue.append(candidate_prim)
                            visited_primitives.add(candidate_prim["id"])

                # 验证连通分量是否为有效环（规则5）
                num_primitives = len(component_primitives)
                is_valid_loop = False

                if num_primitives >= 3:
                    is_valid_loop = True
                elif num_primitives == 2:
                    # 规则5b：同一对顶点 + 至少一个是弧
                    prim1 = component_primitives[0]
                    prim2 = component_primitives[1]
                    if prim1["points"] == prim2["points"]:
                        has_arc = any(prim["type"] == "arc" for prim in component_primitives)
                        if has_arc:
                            is_valid_loop = True

                if not is_valid_loop:
                    prim_ids = [prim["id"] for prim in component_primitives]
                    result["error_message"] = f"检测到无效的环结构。基元 {prim_ids} 无法形成一个有效的环。"
                    result["is_valid"] = False
                    return result

                # -------------------------- 新增：生成环的有序点序列 --------------------------
                def get_ordered_loop_points(component):
                    """辅助函数：根据环的基元列表，生成按遍历顺序的闭环点序列"""
                    if not component:
                        return []
                    
                    # 构建顶点到关联基元的映射（每个顶点对应2个基元）
                    vertex_prim_map = defaultdict(list)
                    for prim in component:
                        p1, p2 = prim["vertices"]
                        vertex_prim_map[p1].append(prim)
                        vertex_prim_map[p2].append(prim)
                    
                    ordered_points = []
                    visited_prim_ids = set()
                    total_prims = len(component)

                    # 1. 确定起始基元和初始方向（优先保留弧的原始方向）
                    start_prim = None
                    for prim in component:
                        if prim["type"] == "arc":  # 弧有固定方向，优先作为起始基元
                            start_prim = prim
                            break
                    if not start_prim:
                        start_prim = component[0]  # 全是线的情况，取第一个基元
                    
                    visited_prim_ids.add(start_prim["id"])
                    original_prim = primitive_id_map[start_prim["id"]]
                    # 按原始基元的起止点确定初始路径方向
                    current_p = original_prim["start_point_id"]
                    next_p = original_prim["end_point_id"]
                    ordered_points.extend([current_p, next_p])

                    # 2. 遍历剩余基元，构建完整路径
                    while len(visited_prim_ids) < total_prims:
                        # 查找包含当前终点next_p且未访问的基元
                        candidate_prims = vertex_prim_map[next_p]
                        next_prim = None
                        for prim in candidate_prims:
                            if prim["id"] not in visited_prim_ids:
                                next_prim = prim
                                break
                        if not next_prim:
                            break  # 已通过有效性验证，此处不应触发

                        visited_prim_ids.add(next_prim["id"])
                        next_original = primitive_id_map[next_prim["id"]]
                        # 确定下一个点（保持基元原始方向）
                        if next_original["start_point_id"] == next_p:
                            new_next_p = next_original["end_point_id"]
                        elif next_original["end_point_id"] == next_p:
                            new_next_p = next_original["start_point_id"]
                        else:
                            new_next_p = next_original["end_point_id"]  # 兜底逻辑

                        ordered_points.append(new_next_p)
                        next_p = new_next_p

                    # 3. 确保闭环（首尾点一致）
                    if ordered_points and ordered_points[0] != ordered_points[-1]:
                        ordered_points.append(ordered_points[0])
                    
                    return ordered_points

                ordered_points = get_ordered_loop_points(component_primitives)
                # ----------------------------------------------------------------------------

                # 记录环信息（新增ordered_points字段）
                result["num_loops"] += 1
                result["loop_details"].append({
                    # "primitives": [prim["id"] for prim in component_primitives],
                    # "primitive_types": [prim["type"] for prim in component_primitives],
                    "ordered_points": ordered_points  # 新增：按顺序排列的闭环点ID列表
                })

            # 4. 规则4验证：所有基元都必须被访问（无孤立基元）
            all_matched_primitive_ids = set(matched_line_ids + matched_arc_ids)
            if visited_primitives != all_matched_primitive_ids:
                isolated_primitives = all_matched_primitive_ids - visited_primitives
                result["error_message"] = f"存在未参与任何环的孤立基元: {isolated_primitives}"
                result["is_valid"] = False
                return result

            # 5. 最终判断
            if result["is_valid"] and result["num_loops"] > 0:
                result["is_closed"] = True
                result["error_message"] = f"成功检测到 {result['num_loops']} 个不相交的有效环。"
                
            # 不方便处理环状图形面积的话，启用
            # if result["num_loops"] != 1:
            #     logger.warning(f"检测到多个环结构：共 {result['num_loops']} 个环，详情：{result['loop_details']}")
            #     result["is_valid"] = False

            return result

    def _match_primitives_to_contour(
        self,
        raw_contour: np.ndarray,
        original_lines: List[Dict],
        original_arcs: List[Dict],
        original_points: List[Dict],
        transformer: CoordTransformer
    ) -> Dict:
        """核心匹配函数：判断待匹配线/弧是否为轮廓的一部分，返回匹配结果"""
        logger.info("=" * 80)
        logger.info("开始基元匹配流程")
        logger.info("=" * 80)
        
        try:
            # 1. 准备工作：创建距离图
            distance_map, offset = self._create_distance_map(raw_contour)
            if distance_map.size == 0:
                raise ValueError("距离图创建失败，无法继续匹配")
            
            minimal_lines = [line for line in original_lines if line.get("is_minimal", False)]
            minimal_arcs = [arc for arc in original_arcs if arc.get("is_minimal", False)]
            
            # 2. 分别匹配直线和圆弧
            matched_lines = self._match_lines(
                original_lines=minimal_lines,
                original_points=original_points,
                transformer=transformer,
                distance_map=distance_map,
                offset=offset
            )
            matched_arcs = self._match_arcs(
                original_arcs=minimal_arcs,
                original_points=original_points,
                transformer=transformer,
                distance_map=distance_map,
                offset=offset
            )
            
            # 3. 提取关联的点
            matched_points = self._extract_matched_point_ids(
                matched_line_ids=matched_lines,
                matched_arc_ids=matched_arcs,
                original_lines=minimal_lines,
                original_arcs=minimal_arcs
            )
            
            # 4. 检验几何有效性
            geometry_check_result = self._check_geometric_validity(
                matched_line_ids=matched_lines,
                matched_arc_ids=matched_arcs,
                original_lines=minimal_lines,
                original_arcs=minimal_arcs
            )
            
            # 5. 整理结果
            result = {
                "points": [{"id": p_id} for p_id in matched_points],
                "lines": [{"id": l_id} for l_id in matched_lines],
                "arcs": [{"id": a_id} for a_id in matched_arcs],
                "ordered_loops": geometry_check_result.get("loop_details", []),
                "validity": geometry_check_result["is_valid"]
            }
            
            logger.info("\n" + "=" * 80)
            logger.info("基元匹配流程结束")
            logger.info(f"最终结果：点 {len(result['points'])} 个，线 {len(result['lines'])} 条，弧 {len(result['arcs'])} 条, 有效性: {geometry_check_result['is_valid']}")
            logger.info("=" * 80)
            return result
        
        except Exception as e:
            logger.error(f"基元匹配流程出错：{str(e)}，返回空结果")
            return {"points": [], "lines": [], "arcs": [], "validity": False}

    def process(self) -> None:
        """处理所有增强JSON数据，生成阴影图像和标注"""
        if not self.enhanced_jsons:
            raise RuntimeError("未加载enhanced数据，无法处理")

        logger.info("=== 开始处理图像 ===")
        drawer_cfg = self.config['drawer']
        self.drawer = GeometryDrawer(drawer_cfg)
        raw_image_paths = self.drawer.batch_draw()

        self.geometry_params_list = self.drawer.get_geometry_params()
        if len(self.geometry_params_list) != len(raw_image_paths):
            logger.warning(f"图形参数与图像数量不匹配：{len(self.geometry_params_list)} vs {len(raw_image_paths)}")

        # 处理每张原始图像
        for idx, raw_img_path in enumerate(raw_image_paths):
            img_bgr = cv2.imread(raw_img_path)
            if img_bgr is None:
                logger.warning(f"第{idx}张原始图像读取失败，路径：{raw_img_path}，跳过")
                continue
            
            if idx >= len(self.geometry_params_list):
                logger.warning(f"第{idx}张图像无转换参数，跳过")
                continue
            geom_params = self.geometry_params_list[idx]
            
            data = self.enhanced_jsons[idx] if idx < len(self.enhanced_jsons) else {}
            
            # 保存原始图像到指定目录
            raw_filename = os.path.basename(raw_img_path)
            raw_save_path = os.path.join(self.raw_dir, raw_filename)
            cv2.imwrite(raw_save_path, img_bgr)
            
            # 处理原始图像的标注
            transformer = CoordTransformer(
                scale_factor=geom_params["scale_factor"],
                geom_center_x=geom_params["geom_center_x"],
                geom_center_y=geom_params["geom_center_y"],
                canvas_center_x=geom_params["canvas_center_x"],
                canvas_center_y=geom_params["canvas_center_y"]
            )
            
            # 生成原始图像的标注版本
            annotated_raw_img = self.annotator.draw_annotations(img_bgr, data, transformer)
            annotated_raw_filename = f"annotated_raw_{raw_filename}"
            annotated_raw_path = os.path.join(self.annotated_dir, annotated_raw_filename)
            cv2.imwrite(annotated_raw_path, annotated_raw_img)
            
            logger.info(f"成功标注第{idx}张图像，原始版本")
            self.processed_results.append({
                **data,
                "raw_path": raw_save_path,
                "annotated_raw_path": annotated_raw_path,
                "shaded_path": None,
                "annotated_shaded_path": None,
                "shader_enabled": False
            })
            
            # 如果shader禁用，跳过阴影处理
            if not self.shader_enabled:
                continue
            
            # 处理阴影
            # self.selected_region_labels = defaultdict(list)
            for attempt in range(self.x_attempts):
                try:
                    self._process_shaded_image(
                        img_bgr=img_bgr,
                        data=data,
                        idx=idx,
                        attempt=attempt,
                        geom_params=geom_params,
                        raw_save_path=raw_save_path,
                        annotated_raw_path=annotated_raw_path,
                        transformer=transformer
                    )
                except Exception as e:
                    logger.error(f"第{idx}条数据的第{attempt}次阴影尝试失败: {str(e)}", exc_info=True)
                    continue

        # 保存所有处理结果到JSONL
        self._save_jsonl()
        logger.info(f"=== 所有图像处理完成 ===")
        logger.info(f"处理结果已保存至：{self.shaded_jsonl_path}")
        logger.info(f"原始图像目录：{self.raw_dir}")
        logger.info(f"阴影图像目录：{self.shaded_dir}")
        logger.info(f"标注图像目录：{self.annotated_dir}")
        logger.info(f"阴影匹配成功率：{(self.success/(self.success+self.failure)):.2%}（成功 {self.success}，失败 {self.failure}）")