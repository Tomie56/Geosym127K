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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .region import RegionExtractor, RegionExtractConfig

# 共享类型与坐标转换
PointMath = Tuple[float, float]
PointPixel = Tuple[int, int]

# =========================
# 阴影器实现
# =========================
class Shader:
    """阴影器基类"""
    name: str = "base"

    def apply(
        self, 
        img_bgr: np.ndarray,
        sel_mask: np.ndarray, 
        math_region: Optional[Dict[str, Any]] = None,** params
    ) -> np.ndarray:
        raise NotImplementedError


class HatchShader(Shader):
    """单方向网纹阴影"""
    name = "hatch"

    def apply(self, img_bgr: np.ndarray, sel_mask: np.ndarray, **params) -> np.ndarray:
        spacing = params.get("spacing", 5)
        angle_deg = params.get("angle_deg", 45)
        intensity = params.get("intensity", 0.5)
        return self._apply_hatch(img_bgr, sel_mask, spacing, angle_deg, intensity)

    def _apply_hatch(self, img_bgr: np.ndarray, mask: np.ndarray, spacing: int, angle: float, intensity: float) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        hatch = self._create_hatch((h, w), spacing, angle)
        return self._blend(img_bgr, hatch, mask, intensity)

    def _create_hatch(self, shape: Tuple[int, int], spacing: int, angle_deg: float) -> np.ndarray:
        h, w = shape
        pad = max(h, w)
        canvas = np.full((h+2*pad, w+2*pad), 255, np.uint8)
        for x in range(0, canvas.shape[1], spacing):
            cv2.line(canvas, (x, 0), (x, canvas.shape[0]-1), 0, 1, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((canvas.shape[1]//2, canvas.shape[0]//2), angle_deg, 1.0)
        rot = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), borderValue=255)
        return cv2.cvtColor(rot[pad:pad+h, pad:pad+w], cv2.COLOR_GRAY2BGR)

    def _blend(self, img_bgr: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        out = img_bgr.copy().astype(np.float32)
        mask = (mask > 0).astype(np.float32)[..., None]
        out = out * (1 - alpha * mask) + overlay.astype(np.float32) * (alpha * mask)
        return np.clip(out, 0, 255).astype(np.uint8)


class CrossHatchShader(Shader):
    """交叉网纹阴影"""
    name = "crosshatch"

    def apply(self, img_bgr: np.ndarray, sel_mask: np.ndarray,** params) -> np.ndarray:
        spacing = params.get("spacing", 6)
        angle1, angle2 = params.get("angle1", 45), params.get("angle2", 135)
        intensity = params.get("intensity", 0.5)
        h, w = img_bgr.shape[:2]
        hatch1 = self._create_hatch((h, w), spacing, angle1)
        hatch2 = self._create_hatch((h, w), spacing, angle2)
        cross = cv2.bitwise_and(hatch1, hatch2)
        return self._blend(img_bgr, cross, sel_mask, intensity)

    def _create_hatch(self, shape: Tuple[int, int], spacing: int, angle_deg: float) -> np.ndarray:
        h, w = shape
        pad = max(h, w)
        canvas = np.full((h+2*pad, w+2*pad), 255, np.uint8)
        for x in range(0, canvas.shape[1], spacing):
            cv2.line(canvas, (x, 0), (x, canvas.shape[0]-1), 0, 1, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((canvas.shape[1]//2, canvas.shape[0]//2), angle_deg, 1.0)
        rot = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), borderValue=255)
        return rot[pad:pad+h, pad:pad+w]

    def _blend(self, img_bgr: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        out = img_bgr.copy().astype(np.float32)
        mask = (mask > 0).astype(np.float32)[..., None]
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR).astype(np.float32)
        out = out * (1 - alpha * mask) + overlay_bgr * (alpha * mask)
        return np.clip(out, 0, 255).astype(np.uint8)


class SolidShader(Shader):
    """纯色阴影"""
    name = "solid"

    def apply(self, img_bgr: np.ndarray, sel_mask: np.ndarray, **params) -> np.ndarray:
        random_color = np.random.randint(0, 256, 3).tolist()
        color = params.get("color", random_color)
        intensity = params.get("intensity", 0.5)
        out = img_bgr.copy().astype(np.float32)
        mask = (sel_mask > 0).astype(np.float32)[..., None]
        overlay = np.full_like(out, color, dtype=np.float32)
        out = out * (1 - intensity * mask) + overlay * (intensity * mask)
        return np.clip(out, 0, 255).astype(np.uint8)


class GradientShader(Shader):
    """渐变阴影"""
    name = "gradient"

    def apply(self, img_bgr: np.ndarray, sel_mask: np.ndarray,** params) -> np.ndarray:
        start_color = params.get("start_color", (200, 200, 200))
        end_color = params.get("end_color", (100, 100, 100))
        angle_deg = params.get("angle_deg", 45)
        intensity = params.get("intensity", 0.5)
        h, w = img_bgr.shape[:2]
        gradient = self._create_gradient((h, w), start_color, end_color, angle_deg)
        return self._blend(img_bgr, gradient, sel_mask, intensity)

    def _create_gradient(self, shape: Tuple[int, int], start: Tuple[int, int, int], end: Tuple[int, int, int], angle_deg: float) -> np.ndarray:
        h, w = shape
        center = (w//2, h//2)
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                dx = x - center[0]
                dy = y - center[1]
                angle = math.atan2(dy, dx) * 180 / math.pi
                dist = min(1.0, math.hypot(dx, dy) / (max(w, h)/2))
                ratio = (math.cos((angle - angle_deg) * math.pi / 180) + 1) / 2 * dist
                gradient[y, x] = [
                    int(start[0] * (1 - ratio) + end[0] * ratio),
                    int(start[1] * (1 - ratio) + end[1] * ratio),
                    int(start[2] * (1 - ratio) + end[2] * ratio)
                ]
        return gradient

    def _blend(self, img_bgr: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        out = img_bgr.copy().astype(np.float32)
        mask = (mask > 0).astype(np.float32)[..., None]
        out = out * (1 - alpha * mask) + overlay.astype(np.float32) * (alpha * mask)
        return np.clip(out, 0, 255).astype(np.uint8)


# 阴影器注册表
SHADERS: Dict[str, Shader] = {
    "hatch": HatchShader(),
    "solid": SolidShader(),
    "gradient": GradientShader(),
    "crosshatch": CrossHatchShader()
}