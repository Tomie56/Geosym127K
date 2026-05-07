import json
import os
import sympy as sp
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import time
import random

class GeometryDrawer:
    def __init__(self, config: Dict):
        """
        初始化绘图器（支持批量处理jsonl、配置线段样式，确保图形居中）
        :param config: 绘图配置，包含：
            - jsonl_path: jsonl文件路径
            - output_dir: 输出图片目录
            - canvas_size: 画布尺寸 (宽, 高)
            - margin: 边距（像素）
            - line_color: 线段颜色（如"#000000"）
            - line_width: 线段粗细（像素）
        """
        # 解析配置
        self.config = self._validate_config(config)
        self.jsonl_path = self.config["jsonl_path"]
        self.output_dir = self.config["output_dir"]
        self.canvas_size = self.config["canvas_size"]  # (宽, 高)
        self.margin = self.config["margin"]
        self.line_color = self._hex_to_bgr(self.config["line_color"])  # 转换为BGR格式
        self.line_width = self.config["line_width"]
        
        # 计算有效绘制区域（扣除边距）和其中心坐标
        self.draw_area_width = self.canvas_size[0] - 2 * self.margin
        self.draw_area_height = self.canvas_size[1] - 2 * self.margin
        self.canvas_center_x = self.margin + self.draw_area_width / 2  # 画布中心X（像素）
        self.canvas_center_y = self.margin + self.draw_area_height / 2  # 画布中心Y（像素）
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载jsonl中的所有几何数据（每行对应一个图形）
        self.all_geometries = self._load_jsonl()
        self.per_geometry_params = []

    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """将十六进制颜色（如"#000000"）转换为cv2的BGR格式"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return (0, 0, 0)  # 默认黑色
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)  # BGR顺序

    def _validate_config(self, config: Dict) -> Dict:
        """验证并补全配置参数"""
        required_keys = ["jsonl_path", "output_dir", "canvas_size", "margin"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置缺少必要参数：{key}")
        
        # 补全默认值（线段默认黑色、粗细2像素）
        return {
            **config,
            "line_color": config.get("line_color", "#000000"),
            "line_width": config.get("line_width", 2)
        }

    def _load_jsonl(self) -> List[Dict]:
        """从jsonl文件加载所有几何数据（严格保证每行一个JSON对象，与图片一一对应）"""
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"jsonl文件不存在：{self.jsonl_path}")
        
        geometries = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行，但不影响后续行索引对应
                try:
                    geom = json.loads(line)
                    # 为每个图形添加行索引，确保与jsonl行对应
                    geom["jsonl_line_num"] = line_num
                    geometries.append(geom)
                except json.JSONDecodeError as e:
                    raise ValueError(f"jsonl文件第{line_num}行格式错误：{str(e)}")
        return geometries
    
    def _standardize_segment(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """标准化线段表示（忽略方向）：按坐标排序起点和终点"""
        # 比较坐标（先x后y），确保起点≤终点，统一线段表示
        if start < end:
            return (start, end)
        else:
            return (end, start)
    
    def _parse_point_coords(self, points: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """解析单个几何数据中的点坐标（符号表达式→数值）"""
        coords = {}
        for point in points:
            pid = point["id"]
            x_expr = sp.sympify(point["x"]["expr"])
            y_expr = sp.sympify(point["y"]["expr"])
            # 计算数值并保留6位小数，平衡精度与性能
            coords[pid] = (round(float(x_expr.evalf()), 6), round(float(y_expr.evalf()), 6))
        return coords

    def _parse_arc_params(self, arcs: List[Dict], point_coords: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[Tuple[float, float], float]]:
        """解析单个几何数据中的圆弧/圆参数：(圆心坐标, 半径)"""
        arc_params = {}
        for arc in arcs:
            arc_id = arc["id"]
            # 解析圆心坐标
            center_pid = arc.get("center_point_id") or arc.get("center_id")  # 兼容不同键名
            if not center_pid or center_pid not in point_coords:
                print(f"警告：圆弧{arc_id}的圆心{center_pid}不存在，跳过该圆弧")
                continue
            center_x, center_y = point_coords[center_pid]
            
            # 解析半径（符号表达式→数值）
            radius_expr = sp.sympify(arc["radius"]["expr"] if "radius" in arc else arc["radius_expr"])
            radius = round(float(radius_expr.evalf()), 6)
            if radius <= 1e-9:  # 避免零半径或负半径
                print(f"警告：圆弧{arc_id}的半径无效（{radius}），跳过该圆弧")
                continue
            
            arc_params[arc_id] = ((center_x, center_y), radius)
        return arc_params

    def _estimate_geometry_bounds(self, point_coords: Dict[str, Tuple[float, float]], arc_params: Dict[str, Tuple[Tuple[float, float], float]]) -> Tuple[float, float, float, float, float, float]:
        """
        估计单个几何数据的边界和中心
        返回：(min_x, max_x, min_y, max_y, center_x, center_y)
        """
        # 1. 收集所有点和圆弧的边界
        all_x = []
        all_y = []
        
        # 添加所有点的坐标
        for x, y in point_coords.values():
            all_x.append(x)
            all_y.append(y)
        
        # 添加所有圆弧的边界（圆心±半径）
        for (cx, cy), r in arc_params.values():
            all_x.extend([cx - r, cx + r])
            all_y.extend([cy - r, cy + r])
        
        # 2. 计算边界（处理无元素的极端情况）
        if not all_x or not all_y:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # 3. 计算几何中心
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        # 4. 扩大5%边界，避免元素贴边
        expand_ratio = 0.05
        width = max_x - min_x
        height = max_y - min_y
        
        expand_x = width * expand_ratio if width > 1e-9 else 1.0
        expand_y = height * expand_ratio if height > 1e-9 else 1.0
        
        return (
            min_x - expand_x,
            max_x + expand_x,
            min_y - expand_y,
            max_y + expand_y,
            center_x,
            center_y
        )

    def _calculate_scaling_factor(self, geometry_width: float, geometry_height: float) -> float:
        """计算缩放因子（确保图形适配有效绘制区域）"""
        if geometry_width < 1e-9 and geometry_height < 1e-9:
            return 10.0  # 处理所有元素重合的特殊情况
        
        # 基于宽高比计算缩放因子，取较小值确保完全显示
        scale_x = self.draw_area_width / geometry_width if geometry_width > 1e-9 else float("inf")
        scale_y = self.draw_area_height / geometry_height if geometry_height > 1e-9 else float("inf")
        return min(scale_x, scale_y)

    def _to_pixel_coords(self, x: float, y: float, geom_center_x: float, geom_center_y: float, scale_factor: float) -> Tuple[int, int]:
        """
        几何坐标→像素坐标（核心：将图形中心与画布中心对齐）
        1. 先将几何坐标转换为以图形中心为原点的坐标
        2. 缩放后平移到画布中心
        3. 处理Y轴翻转（图像坐标系Y轴向下）
        """
        # 转换为以图形中心为原点的坐标
        x_relative = x - geom_center_x
        y_relative = y - geom_center_y
        
        # 缩放
        x_scaled = x_relative * scale_factor
        y_scaled = y_relative * scale_factor
        
        # 平移到画布中心（并处理Y轴翻转）
        pixel_x = int(self.canvas_center_x + x_scaled)
        pixel_y = int(self.canvas_center_y - y_scaled)  # Y轴翻转
        
        return pixel_x, pixel_y

    def _draw_single_geometry(self, geom: Dict, index: int) -> str:
        """绘制单个几何数据（新增线段去重逻辑）"""
        # 解析当前几何数据的元素
        points = geom.get("points", [])
        lines = geom.get("lines", [])
        arcs = geom.get("arcs", [])
        line_num = geom.get("jsonl_line_num", index + 1)  # 关联jsonl行号
        
        # 解析坐标和参数
        point_coords = self._parse_point_coords(points)
        arc_params = self._parse_arc_params(arcs, point_coords)
        
        # 计算几何边界、中心和缩放因子
        min_x, max_x, min_y, max_y, geom_center_x, geom_center_y = self._estimate_geometry_bounds(point_coords, arc_params)
        geometry_width = max_x - min_x
        geometry_height = max_y - min_y
        scale_factor = self._calculate_scaling_factor(geometry_width, geometry_height)
        
        self.per_geometry_params.append({
            "scale_factor": scale_factor,
            "geom_center_x": geom_center_x,
            "geom_center_y": geom_center_y,
            "canvas_center_x": self.canvas_center_x,
            "canvas_center_y": self.canvas_center_y
        })
        
        # 创建画布（白色背景，cv2默认BGR格式）
        canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255  # (高, 宽, 3)
        
        # 绘制圆弧（先画圆弧，避免被线段覆盖）
        for arc in arcs:
            arc_id = arc["id"]
            if arc_id not in arc_params:
                continue
            
            (center_x, center_y), radius = arc_params[arc_id]
            # 转换圆心坐标到像素
            center_pixel = self._to_pixel_coords(center_x, center_y, geom_center_x, geom_center_y, scale_factor)
            # 转换半径到像素
            radius_pixel = int(round(radius * scale_factor))
            if radius_pixel < 1:
                continue  # 过滤过小的圆弧
            
            # 解析圆弧角度（默认画完整圆形，可根据实际字段调整）
            if "angle" in arc:
                angle_expr = sp.sympify(arc["angle"]["expr"])
                angle_rad = float(angle_expr.evalf())
            else:
                angle_rad = 2 * sp.pi  # 默认完整圆形
            
            start_angle_rad = 0.0  # 可根据实际需求从arc中解析start_angle
            end_angle_rad = start_angle_rad + angle_rad
            
            # 弧度转角度（cv2.ellipse使用角度制）
            start_angle = float(start_angle_rad * 180 / sp.pi)
            end_angle = float(end_angle_rad * 180 / sp.pi)
            
            # 绘制圆弧（cv2用ellipse实现，圆是特殊的椭圆）
            # 椭圆参数：中心、长短轴（均为半径）、旋转角度0、起始角度、结束角度
            cv2.ellipse(
                canvas,
                center=center_pixel,
                axes=(radius_pixel, radius_pixel),  # 圆的长短轴相等
                angle=0,  # 不旋转
                startAngle=start_angle,
                endAngle=end_angle,
                color=self.line_color,
                thickness=self.line_width,
                lineType=cv2.LINE_AA  # 抗锯齿
            )
        
        processed_segments = set()  # 记录已处理的线段（标准化后）
        unique_lines = []           # 去重后的线段列表

        for line in lines:
            start_pid = line["start_point_id"]
            end_pid = line["end_point_id"]
            if start_pid not in point_coords or end_pid not in point_coords:
                print(f"警告：jsonl第{line_num}行，线段{line['id']}因端点无效被跳过")
                continue
            
            # 转换线段端点到像素坐标
            start_pixel = self._to_pixel_coords(
                point_coords[start_pid][0], 
                point_coords[start_pid][1], 
                geom_center_x, 
                geom_center_y, 
                scale_factor
            )
            end_pixel = self._to_pixel_coords(
                point_coords[end_pid][0], 
                point_coords[end_pid][1], 
                geom_center_x, 
                geom_center_y, 
                scale_factor
            )
            
            # 标准化线段（忽略方向）
            standardized = self._standardize_segment(start_pixel, end_pixel)
            
            # 检查是否已处理（完全重合）
            if standardized not in processed_segments:
                processed_segments.add(standardized)
                unique_lines.append((start_pixel, end_pixel))
        
        # --------------------------
        # 绘制去重后的线段
        # --------------------------
        for start_pixel, end_pixel in unique_lines:
            cv2.line(
                canvas,
                pt1=start_pixel,
                pt2=end_pixel,
                color=self.line_color,
                thickness=self.line_width,
                lineType=cv2.LINE_AA  # 抗锯齿，替代PIL的joint="round"
            )
        
        # 保存图片
        output_path = os.path.join(self.output_dir, f"geometry_line_{line_num:04d}.png")
        cv2.imwrite(output_path, canvas)
        return output_path
    
    def get_geometry_params(self) -> List[Dict]:
        """返回每个图形的转换参数（与batch_draw返回的图像路径列表索引对应）"""
        return self.per_geometry_params.copy()

    def batch_draw(self) -> List[str]:
        """批量绘制jsonl中的所有几何数据（严格保持与输入行的对应关系）"""
        output_paths = []
        total = len(self.all_geometries)
        print(f"开始批量绘制，共{total}个几何图形（与jsonl行一一对应）...")
        
        for idx, geom in enumerate(self.all_geometries):
            try:
                img_path = self._draw_single_geometry(geom, idx)
                output_paths.append(img_path)
                if (idx + 1) % 10 == 0:  # 每10个输出一次进度
                    print(f"已完成 {idx + 1}/{total} 个图形")
            except Exception as e:
                line_num = geom.get("jsonl_line_num", idx + 1)
                print(f"警告：jsonl第{line_num}行图形绘制失败：{str(e)}，跳过该行")
        
        print(f"批量绘制完成，所有图片保存至：{self.output_dir}")
        return output_paths

    def draw_single(self, geom: Dict) -> Tuple[str, Dict]:
        """公共接口：绘制单条几何数据，返回（图像路径，几何参数），线程安全"""
        # 解析当前几何数据的元素（不依赖类实例的批量数据）
        points = geom.get("points", [])
        lines = geom.get("lines", [])
        arcs = geom.get("arcs", [])
        
        # 从geom中提取enhance_id（核心：基于传入的几何数据中的标识，而非随机生成）
        # 优先使用geom中的enhance_id，缺失则基于base_idx和enhance_idx生成
        enhance_id = geom.get("enhance_id")
        if not enhance_id:
            base_idx = geom.get("base_idx", 0)
            enhance_idx = geom.get("enhance_idx", 0)
            enhance_id = f"base_{base_idx:03d}_enhance_{enhance_idx:03d}"
        
        # 解析坐标和参数
        point_coords = self._parse_point_coords(points)
        arc_params = self._parse_arc_params(arcs, point_coords)
        
        # 计算几何边界、中心和缩放因子（核心参数）
        min_x, max_x, min_y, max_y, geom_center_x, geom_center_y = self._estimate_geometry_bounds(point_coords, arc_params)
        geometry_width = max_x - min_x
        geometry_height = max_y - min_y
        scale_factor = self._calculate_scaling_factor(geometry_width, geometry_height)
        
        # 核心：封装当前单条数据的几何转换参数（不存储到实例变量）
        geom_params = {
            "scale_factor": scale_factor,
            "geom_center_x": geom_center_x,
            "geom_center_y": geom_center_y,
            "canvas_center_x": self.canvas_center_x,
            "canvas_center_y": self.canvas_center_y
        }
        
        # 创建画布（白色背景）
        canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255  # (高, 宽, 3)
        
        # 绘制圆弧（先画圆弧，避免被线段覆盖）
        for arc in arcs:
            arc_id = arc["id"]
            if arc_id not in arc_params:
                continue
            
            (center_x, center_y), radius = arc_params[arc_id]
            center_pixel = self._to_pixel_coords(center_x, center_y, geom_center_x, geom_center_y, scale_factor)
            radius_pixel = int(round(radius * scale_factor))
            if radius_pixel < 1:
                continue
            
            # 解析圆弧角度
            if "angle" in arc:
                angle_expr = sp.sympify(arc["angle"]["expr"])
                angle_rad = float(angle_expr.evalf())
            else:
                angle_rad = 2 * sp.pi
            
            start_angle = float(0 * 180 / sp.pi)
            end_angle = float(angle_rad * 180 / sp.pi)
            
            # 绘制圆弧
            cv2.ellipse(
                canvas,
                center=center_pixel,
                axes=(radius_pixel, radius_pixel),
                angle=0,
                startAngle=start_angle,
                endAngle=end_angle,
                color=self.line_color,
                thickness=self.line_width,
                lineType=cv2.LINE_AA
            )
        
        # 线段去重
        processed_segments = set()
        unique_lines = []
        for line in lines:
            start_pid = line["start_point_id"]
            end_pid = line["end_point_id"]
            if start_pid not in point_coords or end_pid not in point_coords:
                continue
            
            start_pixel = self._to_pixel_coords(
                point_coords[start_pid][0], point_coords[start_pid][1],
                geom_center_x, geom_center_y, scale_factor
            )
            end_pixel = self._to_pixel_coords(
                point_coords[end_pid][0], point_coords[end_pid][1],
                geom_center_x, geom_center_y, scale_factor
            )
            
            standardized = self._standardize_segment(start_pixel, end_pixel)
            if standardized not in processed_segments:
                processed_segments.add(standardized)
                unique_lines.append((start_pixel, end_pixel))
        
        # 绘制线段
        for start_pixel, end_pixel in unique_lines:
            cv2.line(
                canvas,
                pt1=start_pixel,
                pt2=end_pixel,
                color=self.line_color,
                thickness=self.line_width,
                lineType=cv2.LINE_AA
            )
        
        # 生成输出路径（基于enhance_id，确保与全流程命名一致）
        # 格式：{enhance_id}_raw.png（明确标识为原始图）
        img_ext = ".png"
        output_path = os.path.join(self.output_dir, f"{enhance_id}_raw{img_ext}")
        
        # 处理路径冲突（若文件已存在，添加随机后缀避免覆盖）
        if os.path.exists(output_path):
            for i in range(3):  # 最多尝试3次
                output_path = os.path.join(self.output_dir, f"{enhance_id}_raw_{random.randint(100, 999)}{img_ext}")
                if not os.path.exists(output_path):
                    break
        
        # 保存图片
        cv2.imwrite(output_path, canvas)
        
        # 返回（图像路径，几何参数）元组
        return output_path, geom_params