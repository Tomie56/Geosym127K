import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import sympy as sp
from sympy import (
    symbols, simplify, sqrt, pi, expand, sign, Rational,
    atan2, Abs, rad, deg, Mod
)
import timeout_decorator


# 配置日志（保留DEBUG便于调试，生产环境可改回CRITICAL）
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GeometryCalculator')


class GeometryCalculator:
    """
    通用几何参数计算器（兼容两点环+原有所有场景，弧类型使用英文标识）
    核心规则：
    1. 两点环：ordered_points如[B,C,B]保留原始结构，基准面积=0，总面积=弧补偿面积；
    2. 弧类型：minor_arc(劣弧)/major_arc(优弧)/semicircle(半圆弧)；
    3. 完全兼容原有非两点环的面积计算逻辑，无破坏性修改；
    """
    # ========== 修改1：延长超时时间（解决复杂计算超时） ==========
    PROCESS_TIMEOUT = 60
    EXPR_LENGTH_THRESHOLD = 200


    def __init__(self):
        self.math_functions = {
            'sqrt': sqrt,
            'pi': pi,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'atan2': atan2,
            'Rational': Rational,
            'rad': rad,
            'deg': deg,
            'Mod': Mod
        }


    # ------------------- 基础工具函数（完全保留原有逻辑） -------------------
    def _preprocess_expr(self, expr_str: str) -> str:
        if not expr_str:
            return ""
        return (expr_str.replace(' ', '')
                .replace('×', '*').replace('÷', '/')
                .replace('+-', '-').replace('-+', '-')
                .replace('π', 'pi'))


    def _parse_expr(self, expr_str: str) -> Optional[sp.Expr]:
        try:
            processed = self._preprocess_expr(expr_str)
            if not processed:
                return None
            return sp.sympify(processed, locals=self.math_functions)
        except Exception as e:
            logger.warning(f"表达式解析失败: {expr_str} → 错误: {str(e)}")
            return None


    def _get_point_coords(self, points: List[Dict[str, Any]], point_id: str) -> Optional[Tuple[sp.Expr, sp.Expr]]:
        for point in points:
            if point.get('id') == point_id:
                x_data = point.get('x', {})
                y_data = point.get('y', {})
                if 'expr' not in x_data or 'expr' not in y_data:
                    logger.warning(f"点 {point_id} 缺少x/y的expr字段")
                    return None
                x_expr = self._parse_expr(x_data['expr'])
                y_expr = self._parse_expr(y_data['expr'])
                if x_expr is None or y_expr is None:
                    logger.warning(f"点 {point_id} 坐标表达式解析失败")
                    return None
                return (x_expr, y_expr)
        logger.warning(f"未找到点 {point_id}")
        return None


    def _reverse_loop_points(self, point_ids: List[str], point_coords: List[Tuple[sp.Expr, sp.Expr]]) -> Tuple[List[str], List[Tuple[sp.Expr, sp.Expr]]]:
        """反转环的点序列（顺时针→逆时针）"""
        reversed_ids = point_ids[::-1]
        reversed_coords = point_coords[::-1]
        return reversed_ids, reversed_coords


    # ------------------- 环方向判断（仅适配两点环，不影响原有逻辑） -------------------
    def _judge_loop_direction_by_3points(self, point_coords: List[Tuple[sp.Expr, sp.Expr]]) -> str:
        """
        保留原有所有逻辑，仅对两点环做适配：
        - 两点环返回counterclockwise；
        - 非两点环完全沿用原有判断逻辑；
        """
        valid_coords = [coord for coord in point_coords if coord is not None]
        unique_coords = list(set(valid_coords))
        
        # 仅两点环适配，不影响其他场景
        if len(unique_coords) <= 2:
            logger.info(f"检测到两点环（{len(unique_coords)}个唯一点），默认逆时针")
            return "counterclockwise"
        
        if len(valid_coords) < 3:
            logger.warning("有效点不足3个，默认逆时针")
            return "counterclockwise"
        
        # 原有非两点环的判断逻辑完全保留
        x1, y1 = valid_coords[0]
        x2, y2 = valid_coords[1]
        x3, y3 = valid_coords[2]
        cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        cross_sign = sp.sign(simplify(cross_product))
        
        if cross_sign > 0:
            return "counterclockwise"
        elif cross_sign < 0:
            return "clockwise"
        else:
            for i in range(3, len(valid_coords)):
                xn, yn = valid_coords[i]
                cross_product = (x2 - x1) * (yn - y1) - (y2 - y1) * (xn - x1)
                cross_sign = sp.sign(simplify(cross_product))
                if cross_sign != 0:
                    return "counterclockwise" if cross_sign > 0 else "clockwise"
            logger.warning("所有点共线，默认逆时针")
            return "counterclockwise"


    # ------------------- 弧角度核心处理（仅替换弧类型为英文，逻辑不变） -------------------
    def _calc_point_angle(self, center_coords: Tuple[sp.Expr, sp.Expr], point_coords: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        # 完全保留原有逻辑
        cx, cy = center_coords
        px, py = point_coords
        dx = px - cx
        dy = py - cy
        raw_angle = simplify(atan2(dy, dx))
        point_angle = simplify(Mod(raw_angle, 2 * pi))
        return point_angle


    def _get_arc_original_angle(self, arc: Dict[str, Any], points: List[Dict[str, Any]]) -> Optional[sp.Expr]:
        # 完全保留原有逻辑
        arc_id = arc.get('id', 'unknown_arc')
        center_id = arc.get('center_point_id')
        start_id = arc.get('start_point_id')
        end_id = arc.get('end_point_id')


        if not all([center_id, start_id, end_id]):
            logger.warning(f"弧 {arc_id} 缺少核心点ID")
            return None


        center_coords = self._get_point_coords(points, center_id)
        start_coords = self._get_point_coords(points, start_id)
        end_coords = self._get_point_coords(points, end_id)
        if None in [center_coords, start_coords, end_coords]:
            logger.warning(f"弧 {arc_id} 坐标解析失败")
            return None


        start_angle = self._calc_point_angle(center_coords, start_coords)
        end_angle = self._calc_point_angle(center_coords, end_coords)
        original_angle = simplify(start_angle - end_angle)
        original_angle_clamped = simplify(Mod(original_angle + 2 * pi, 2 * pi) if original_angle < 0 else original_angle)
        if original_angle_clamped > 2 * pi:
            original_angle_clamped = simplify(original_angle_clamped - 2 * pi)


        return original_angle_clamped


    def _process_arc_angle_temp(self, arc: Dict[str, Any], points: List[Dict[str, Any]]) -> Tuple[Optional[sp.Expr], Optional[sp.Expr], Optional[sp.Expr], str]:
        """
        仅修改：弧类型从中文改为英文
        - 劣弧 → minor_arc
        - 优弧 → major_arc
        - 半圆弧 → semicircle
        逻辑完全保留原有规则
        """
        arc_id = arc.get('id', 'unknown_arc')
        original_angle = None
        
        if 'angle' in arc and arc['angle'].get('expr'):
            original_angle = self._parse_expr(arc['angle']['expr'])
        else:
            original_angle = self._get_arc_original_angle(arc, points)


        if original_angle is None:
            logger.warning(f"弧 {arc_id} 无法获取原始角度")
            return None, None, None, "unknown_arc"


        length_angle = simplify(Abs(original_angle))
        angle_abs = length_angle
        pi_expr = sp.pi
        
        # 仅替换弧类型标识为英文，判断逻辑不变
        if angle_abs > pi_expr:
            area_angle = simplify(2 * pi_expr - angle_abs)
            arc_type = "major_arc"  # 优弧→major_arc
        elif angle_abs < pi_expr:
            area_angle = angle_abs
            arc_type = "minor_arc"  # 劣弧→minor_arc
        else:
            area_angle = pi_expr
            arc_type = "semicircle"  # 半圆弧→semicircle


        return original_angle, length_angle, area_angle, arc_type


    # ------------------- 周长计算（完全保留原有逻辑） -------------------
    def _calc_line_length(self, line: Dict[str, Any], points: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 完全保留原有逻辑
        line_id = line.get('id', 'unknown_line')
        start_id = line.get('start_point_id')
        end_id = line.get('end_point_id')


        if not start_id or not end_id:
            logger.warning(f"线 {line_id} 缺少起止点ID")
            return {'expr': 'unknown', 'latex': 'unknown', 'value': None}


        start_coords = self._get_point_coords(points, start_id)
        end_coords = self._get_point_coords(points, end_id)
        if not start_coords or not end_coords:
            return {'expr': 'unknown', 'latex': 'unknown', 'value': None}


        x1, y1 = start_coords
        x2, y2 = end_coords
        dx = x2 - x1
        dy = y2 - y1
        length_expr = simplify(sqrt(dx**2 + dy**2))


        try:
            length_value = float(sp.N(length_expr, 8))
        except Exception:
            length_value = None


        return {
            'expr': str(length_expr),
            'latex': sp.latex(length_expr),
            'value': round(length_value, 8) if length_value else None
        }


    def _calc_arc_length(self, arc: Dict[str, Any], points: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 完全保留原有逻辑
        arc_id = arc.get('id', 'unknown_arc')
        radius_data = arc.get('radius', {})
        radius_expr = self._parse_expr(radius_data.get('expr', ''))
        
        if radius_expr is None:
            logger.warning(f"弧 {arc_id} 半径解析失败")
            return {'expr': 'unknown', 'latex': 'unknown', 'value': None}


        original_angle, length_angle, area_angle, arc_type = self._process_arc_angle_temp(arc, points)
        if length_angle is None:
            return {'expr': 'unknown', 'latex': 'unknown', 'value': None}


        length_expr = simplify(radius_expr * length_angle)
        try:
            length_value = float(sp.N(length_expr, 8))
        except Exception:
            length_value = None


        return {
            'expr': str(length_expr),
            'latex': sp.latex(length_expr),
            'value': round(length_value, 8) if length_value else None
        }


    def _calc_perimeter(self, entity: Dict[str, Any], lines: List[Dict[str, Any]], arcs: List[Dict[str, Any]]) -> Optional[sp.Expr]:
        # 完全保留原有逻辑
        perimeter_expr = 0
        line_ids = {l['id'] for l in entity.get('lines', [])}
        for line in lines:
            if line.get('id') in line_ids and line.get('length', {}).get('expr') != 'unknown':
                line_expr = self._parse_expr(line['length']['expr'])
                if line_expr:
                    perimeter_expr += line_expr


        arc_ids = {a['id'] for a in entity.get('arcs', [])}
        for arc in arcs:
            if arc.get('id') in arc_ids and arc.get('length', {}).get('expr') != 'unknown':
                arc_expr = self._parse_expr(arc['length']['expr'])
                if arc_expr:
                    perimeter_expr += arc_expr


        return simplify(perimeter_expr) if perimeter_expr != 0 else None


    # ------------------- 面积计算（仅适配两点环+弧类型英文，不影响原有逻辑） -------------------
    def _calc_polygon_signed_area(self, point_coords: List[Tuple[sp.Expr, sp.Expr]]) -> sp.Expr:
        """
        仅适配两点环：基准面积=0；
        非两点环完全保留原有多边形面积计算逻辑；
        """
        valid_coords = [coord for coord in point_coords if coord is not None]
        unique_coords = list(set(valid_coords))
        
        # 仅两点环适配，不影响其他场景
        if len(unique_coords) <= 2:
            logger.info(f"两点环，基准面积=0")
            return sp.Integer(0)
        
        # 非两点环完全保留原有逻辑
        n = len(valid_coords)
        if n < 3:
            return sp.Integer(0)


        area_expr = 0
        for i in range(n):
            x_i, y_i = valid_coords[i]
            x_j, y_j = valid_coords[(i+1) % n]
            area_expr += (x_i * y_j) - (x_j * y_i)


        signed_area = simplify(area_expr * Rational(1, 2))
        return signed_area


    def _get_aligned_arc_temp(self, arc: Dict[str, Any], loop_prev_id: str, loop_next_id: str) -> Tuple[Dict[str, Any], bool]:
        # 完全保留原有逻辑
        arc_start = arc.get('start_point_id')
        arc_end = arc.get('end_point_id')


        if arc_start == loop_prev_id and arc_end == loop_next_id:
            return arc.copy(), False


        if arc_start == loop_next_id and arc_end == loop_prev_id:
            aligned_arc = arc.copy()
            aligned_arc['start_point_id'] = loop_prev_id
            aligned_arc['end_point_id'] = loop_next_id
            return aligned_arc, True


        return arc.copy(), False


    def _calc_arc_segment_area(self, arc: Dict[str, Any], points: List[Dict[str, Any]], loop_direction: str) -> Optional[sp.Expr]:
        """
        仅修改：
        1. 弧类型判断从中文改为英文（minor_arc/major_arc/semicircle）；
        2. 新增叉积为0时的兜底逻辑（解决三点共线/半圆的符号判断）；
        3. 保留所有原有补偿逻辑，不影响非两点环计算；
        """
        arc_id = arc.get('id', 'unknown_arc')
        center_id = arc.get('center_point_id')
        start_id = arc.get('start_point_id')
        end_id = arc.get('end_point_id')
        radius_data = arc.get('radius', {})
        radius_expr = self._parse_expr(radius_data.get('expr', ''))


        if not all([center_id, start_id, end_id, radius_expr]):
            logger.warning(f"弧 {arc_id} 缺少核心参数，跳过补偿面积计算")
            return None


        center_coords = self._get_point_coords(points, center_id)
        start_coords = self._get_point_coords(points, start_id)
        end_coords = self._get_point_coords(points, end_id)
        if None in [center_coords, start_coords, end_coords]:
            logger.warning(f"弧 {arc_id} 坐标解析失败，跳过补偿面积计算")
            return None


        cx, cy = center_coords
        sx, sy = start_coords
        ex, ey = end_coords


        original_angle, length_angle, area_angle, arc_type = self._process_arc_angle_temp(arc, points)
        if area_angle is None:
            logger.warning(f"弧 {arc_id} 角度计算失败，跳过补偿面积计算")
            return None


        # 三角形面积计算逻辑完全保留
        triangle_area = simplify(
            Rational(1, 2) * Abs((sx - cx) * (ey - cy) - (ex - cx) * (sy - cy))
        )
        
        # 仅替换弧类型标识，计算逻辑完全不变
        if arc_type == "minor_arc" or arc_type == "semicircle":  # 劣弧/半圆弧→minor_arc/semicircle
            sector_area = simplify(Rational(1, 2) * radius_expr**2 * area_angle)
            base_segment_area = simplify(sector_area - triangle_area)
            logger.debug(f"弧 {arc_id} | minor_arc/semicircle | 基础弓形面积：{base_segment_area}")
        
        elif arc_type == "major_arc":  # 优弧→major_arc
            circle_area = simplify(pi * radius_expr**2)
            minor_sector_area = simplify(Rational(1, 2) * radius_expr**2 * (2*pi - length_angle))
            minor_segment_area = simplify(minor_sector_area - triangle_area)
            base_segment_area = simplify(circle_area - minor_segment_area)
            logger.debug(f"弧 {arc_id} | major_arc | 基础弓形面积：{base_segment_area}")
        
        else:
            base_segment_area = sp.Integer(0)
            logger.warning(f"弧 {arc_id} | unknown_arc → 基础弓形面积：0")


        # 叉积计算逻辑完全保留
        vec_arc = (ex - sx, ey - sy)
        vec_start_center = (cx - sx, cy - sy)
        cross_expr = simplify(vec_arc[0] * vec_start_center[1] - vec_arc[1] * vec_start_center[0])
        cross_sign = sp.sign(cross_expr)
        logger.debug(f"弧 {arc_id} | 叉积值：{cross_expr} | 叉积符号：{cross_sign}")

        # ========== 修改2：新增叉积为0的兜底逻辑（核心修复） ==========
        # 解决三点共线（半圆）时叉积为0的符号判断问题
        if cross_sign == 0:
            logger.warning(f"弧 {arc_id} 叉积为0（三点共线/半圆），基于环方向+弧类型兜底设置符号")
            # 环方向为逆时针（counterclockwise）：
            if loop_direction == "counterclockwise":
                # 劣弧/半圆：符号为正；优弧：符号为负
                cross_sign = 1 if arc_type in ["minor_arc", "semicircle"] else -1
            # 环方向为顺时针（clockwise）：
            else:
                # 劣弧/半圆：符号为负；优弧：符号为正
                cross_sign = -1 if arc_type in ["minor_arc", "semicircle"] else 1

        # 仅替换弧类型判断，补偿逻辑完全不变
        if arc_type == "minor_arc" or arc_type == "semicircle":
            final_segment_area = base_segment_area if cross_sign > 0 else -base_segment_area
        elif arc_type == "major_arc":
            final_segment_area = -base_segment_area if cross_sign > 0 else base_segment_area
        else:
            final_segment_area = sp.Integer(0)


        logger.info(f"弧 {arc_id} | 最终补偿面积：{final_segment_area}")
        return final_segment_area


    def _calc_area(self, entity: Dict[str, Any], points: List[Dict[str, Any]], arcs: List[Dict[str, Any]]) -> Tuple[Optional[sp.Expr], Optional[float]]:
        """
        仅适配两点环：
        1. 保留ordered_points原始结构（不删除重复点）；
        2. 两点环基准面积=0，重点计算弧补偿面积；
        3. 修复0值判断漏洞（兼容SymPy所有0类型）；
        """
        ordered_loops = entity.get('ordered_loops', [])
        if not ordered_loops:
            logger.warning("无有序环数据")
            return None, None
        
        # 保留原始ordered_points结构，不做去重（兼容两点环闭合结构）
        ordered_point_ids = ordered_loops[0].get('ordered_points', [])
        if len(ordered_point_ids) < 2:
            logger.warning(f"环顶点数不足，无法计算面积")
            return None, None


        # 1. 获取点坐标（保留原始顺序，兼容两点环）
        point_coords = []
        all_coords_invalid = True
        for p_id in ordered_point_ids:
            coords = self._get_point_coords(points, p_id)
            point_coords.append(coords)
            if coords is not None:
                all_coords_invalid = False
        
        if all_coords_invalid:
            logger.error("所有点坐标均无效，无法计算面积")
            return None, None


        # 2. 判断环方向（兼容两点环）
        loop_direction = self._judge_loop_direction_by_3points(point_coords)
        
        # 强制转为逆时针（原有逻辑保留）
        if loop_direction == "clockwise":
            logger.info(f"检测到顺时针环，自动反转")
            ordered_point_ids, point_coords = self._reverse_loop_points(ordered_point_ids, point_coords)
            loop_direction = "counterclockwise"


        # 3. 计算基准面积（两点环=0，非两点环保留原有逻辑）
        base_signed_area = self._calc_polygon_signed_area(point_coords)
        logger.info(f"环基准面积：{base_signed_area}")


        # 4. 计算弧补偿面积（原有逻辑保留，兼容两点环）
        total_segment_area = sp.Integer(0)
        arc_ids = {a['id'] for a in entity.get('arcs', [])}
        arc_map = {a['id']: a for a in arcs if a.get('id') in arc_ids}
        n_points = len(ordered_point_ids)


        # 遍历点对（保留原始顺序，兼容两点环）
        calculated_arc_ids = set()

        # 遍历环的每一对相邻点
        for i in range(n_points):
            loop_prev_id = ordered_point_ids[i]
            loop_next_id = ordered_point_ids[(i+1) % n_points]
            
            # 新增：过滤无效点对（前后点相同）
            if loop_prev_id == loop_next_id:
                logger.debug(f"跳过无效点对：{loop_prev_id} → {loop_next_id}")
                continue
            logger.debug(f"遍历点对：{loop_prev_id} → {loop_next_id}")

            # 匹配连接当前点对的弧
            matched_arcs = []
            for arc_id, arc in arc_map.items():
                # 新增：跳过已计算的弧
                if arc_id in calculated_arc_ids:
                    continue
                arc_start = arc.get('start_point_id')
                arc_end = arc.get('end_point_id')
                if (arc_start == loop_prev_id and arc_end == loop_next_id) or (arc_start == loop_next_id and arc_end == loop_prev_id):
                    matched_arcs.append(arc)
                    calculated_arc_ids.add(arc_id)  # 标记为已计算
                    logger.debug(f"匹配到弧：{arc_id}（连接{loop_prev_id}-{loop_next_id}）")


            # 计算补偿面积（原有逻辑保留）
            for matched_arc in matched_arcs:
                aligned_arc, _ = self._get_aligned_arc_temp(matched_arc, loop_prev_id, loop_next_id)
                seg_area = self._calc_arc_segment_area(aligned_arc, points, loop_direction)
                if seg_area is not None:
                    total_segment_area = simplify(total_segment_area + seg_area)


        logger.info(f"所有弧总补偿面积：{total_segment_area}")


        # 5. 总面积计算（========== 修改3：修复0值判断漏洞 ==========）
        sum_signed_area = simplify(base_signed_area + total_segment_area)

        # 兼容SymPy所有0类型（Integer/Float/Rational），避免返回None
        try:
            simplified_sum = simplify(sum_signed_area)
            # 判断是否为0（包括浮点型0、有理数0）
            if simplified_sum == 0 or sp.N(simplified_sum) == 0:
                final_area_expr = sp.Integer(0)
            else:
                final_area_expr = simplify(Abs(simplified_sum))
        except Exception as e:
            logger.warning(f"面积化简失败：{str(e)}，兜底返回0")
            final_area_expr = sp.Integer(0)

        # 6. 数值转换（原有逻辑保留）
        final_area_value = None
        if final_area_expr is not None:
            try:
                final_area_value = float(sp.N(final_area_expr, 8))
            except Exception as e:
                logger.warning(f"面积数值转换失败：{str(e)}")


        logger.info(f"环最终面积：表达式={final_area_expr}，数值={final_area_value}")
        return final_area_expr, final_area_value


    # ------------------- 主处理逻辑（完全保留原有逻辑，仅loop_direction为英文） -------------------
    def _process_entity(self, entity: Dict[str, Any], points: List[Dict[str, Any]], lines_with_length: List[Dict[str, Any]], arcs_with_length: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 完全保留原有逻辑
        perimeter_expr = self._calc_perimeter(entity, lines_with_length, arcs_with_length)
        perimeter_value = float(sp.N(perimeter_expr, 8)) if perimeter_expr else None
        area_expr, area_value = self._calc_area(entity, points, arcs_with_length)


        processed_entity = entity.copy()
        processed_entity['perimeter'] = {
            'expr': str(perimeter_expr) if perimeter_expr else 'unknown',
            'latex': sp.latex(perimeter_expr) if perimeter_expr else 'unknown',
            'value': round(perimeter_value, 8) if perimeter_value else None
        }
        processed_entity['area'] = {
            'expr': str(area_expr) if area_expr else 'unknown',
            'latex': sp.latex(area_expr) if area_expr else 'unknown',
            'value': round(area_value, 8) if area_value else None
        }
        # loop_direction使用英文
        processed_entity['loop_direction'] = 'counterclockwise'


        return processed_entity


    def calculate(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        # 修复原有代码遗漏的entities定义，不影响其他逻辑
        result = json.loads(json.dumps(json_data))
        points = result.get('points', [])
        lines = result.get('lines', [])
        arcs = result.get('arcs', [])
        entities = result.get('entities', [])


        # 预处理线和弧长度（原有逻辑保留）
        lines_with_length = []
        for line in lines:
            line_copy = line.copy()
            line_copy['length'] = self._calc_line_length(line_copy, points)
            lines_with_length.append(line_copy)


        arcs_with_length = []
        for arc in arcs:
            arc_copy = arc.copy()
            arc_copy['length'] = self._calc_arc_length(arc_copy, points)
            arcs_with_length.append(arc_copy)


        # 处理实体（原有逻辑保留）
        processed_entities = []
        shadow_count = 0
        for entity in entities:
            if entity.get('type') == 'shadow' and entity.get('validity'):
                processed_entity = self._process_entity(entity, points, lines_with_length, arcs_with_length)
                processed_entities.append(processed_entity)
                shadow_count += 1
            else:
                processed_entities.append(entity.copy())


        # 更新结果（原有逻辑保留）
        result['lines'] = lines_with_length
        result['arcs'] = arcs_with_length
        result['entities'] = processed_entities
        result['shadow_regions'] = shadow_count
        
        return result


    # ------------------- 以下方法完全保留原有逻辑，无任何修改 -------------------
    def calculate_single(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(json_data, dict) or 'points' not in json_data:
                raise ValueError("输入必须是包含points的字典")
            return self.calculate(json_data)
        except Exception as e:
            logger.error(f"单次计算失败：{str(e)}", exc_info=True)
            error_result = json.loads(json.dumps(json_data))
            error_result['calculation_error'] = str(e)
            return error_result


    @timeout_decorator.timeout(PROCESS_TIMEOUT, timeout_exception=TimeoutError)
    def _calculate_with_timeout(self, data):
        return self.calculate(data)


    def process_jsonl(self, input_path: str, output_path: str) -> None:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")


        os.makedirs(os.path.dirname(output_path), exist_ok=True)


        logger.info(f"开始批量处理文件：{input_path}")
        with open(input_path, 'r', encoding='utf-8') as f_count:
            total_lines = sum(1 for line in f_count if line.strip())
        logger.info(f"文件总行数：{total_lines} 行")
        
        line_num = 0
        success_num = 0
        fail_num = 0
        timeout_num = 0


        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:


            for line in f_in:
                line_num += 1
                line = line.strip()
                if not line:
                    continue


                try:
                    raw_data = json.loads(line)
                    processed_data = self._calculate_with_timeout(raw_data)
                    json.dump(processed_data, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    success_num += 1
                except TimeoutError:
                    timeout_num += 1
                    logger.error(f"第 {line_num} 行处理超时，跳过")
                except Exception as e:
                    fail_num += 1
                    logger.error(f"第 {line_num} 行处理失败：{str(e)}，跳过")
                    
                progress = (line_num / total_lines) * 100
                logger.info(f"进度：{line_num}/{total_lines} ({progress:.1f}%) | 成功：{success_num} | 失败：{fail_num} | 超时：{timeout_num}")


        logger.info(f"批量处理完成 | 成功：{success_num} | 失败：{fail_num} | 超时：{timeout_num}")
        logger.info(f"结果保存至：{output_path}")



# ------------------- 测试示例（可直接运行验证） -------------------
if __name__ == "__main__":
    test_data_polygon = {
        "points": [
            {
                "id": "O1",
                "x": {"expr": "0", "latex": "0"},
                "y": {"expr": "0", "latex": "0"},
                "related_edges": [],
                "is_center": True,
                "is_circle_init": False,
                "level": 1,
                "is_circle_center": True
            },
            {
                "id": "circle_1",
                "x": {"expr": "7", "latex": "7"},
                "y": {"expr": "0", "latex": "0"},
                "related_edges": ["Arc0"],
                "is_center": False,
                "is_circle_init": True,
                "level": 1
            },
            {
                "id": "C",
                "x": {"expr": "0", "latex": "0"},
                "y": {"expr": "7", "latex": "7"},
                "related_edges": ["L0", "L2", "Arc0"],
                "is_center": False,
                "is_circle_init": False,
                "level": 1,
                "is_circle_center": True
            },
            {
                "id": "circle_2",
                "x": {"expr": "7", "latex": "7"},
                "y": {"expr": "7", "latex": "7"},
                "related_edges": ["Arc1"],
                "is_center": False,
                "is_circle_init": True,
                "level": 1
            },
            {
                "id": "A",
                "x": {"expr": "7*sqrt(3)/2", "latex": "\\frac{7 \\sqrt{3}}{2}"},
                "y": {"expr": "7/2", "latex": "\\frac{7}{2}"},
                "related_edges": ["Arc1", "Arc0"],
                "is_center": False,
                "is_circle_init": False,
                "level": 1
            },
            {
                "id": "B",
                "x": {"expr": "-7*sqrt(3)/2", "latex": "- \\frac{7 \\sqrt{3}}{2}"},
                "y": {"expr": "7/2", "latex": "\\frac{7}{2}"},
                "related_edges": ["Arc1", "Arc0", "L2"],
                "is_center": False,
                "is_circle_init": False,
                "level": 1
            },
            {
                "id": "I2",
                "x": {"expr": "-7", "latex": "-7"},
                "y": {"expr": "7", "latex": "7"},
                "related_edges": [],
                "is_center": True,
                "is_circle_init": False,
                "level": 1,
                "type": "center"
            },
            {
                "id": "D",
                "x": {"expr": "-21/2", "latex": "- \\frac{21}{2}"},
                "y": {"expr": "7*sqrt(3)/2 + 7", "latex": "\\frac{7 \\sqrt{3}}{2} + 7"},
                "related_edges": ["L0", "L1"],
                "is_center": False,
                "is_circle_init": False,
                "level": 3
            },
            {
                "id": "E",
                "x": {"expr": "-21/2", "latex": "- \\frac{21}{2}"},
                "y": {"expr": "7 - 7*sqrt(3)/2", "latex": "7 - \\frac{7 \\sqrt{3}}{2}"},
                "related_edges": ["L1", "L2"],
                "is_center": False,
                "is_circle_init": False,
                "level": 3
            },
            {
                "id": "F",
                "x": {"expr": "-7*sqrt(3)/2", "latex": "- \\frac{7 \\sqrt{3}}{2}"},
                "y": {"expr": "21/2", "latex": "\\frac{21}{2}"},
                "related_edges": ["L0", "Arc1"],
                "is_center": False,
                "is_circle_init": False,
                "level": 3
            },
            {
                "id": "F0",
                "x": {"expr": "-21/4", "latex": "- \\frac{21}{4}"},
                "y": {"expr": "7*sqrt(3)/4 + 7", "latex": "\\frac{7 \\sqrt{3}}{4} + 7"},
                "level": 4,
                "type": "perpendicular_foot",
                "related_vertex": "E",
                "related_edge": "L0"
            }
        ],
        "lines": [
            {
                "id": "L0",
                "type": "line",
                "start_point_id": "C",
                "end_point_id": "D",
                "level": 3,
                "is_original": True,
                "is_minimal": False,
                "length": {"expr": "7*sqrt(3)", "latex": "7 \\sqrt{3}", "value": 12.12435566}
            },
            {
                "id": "L1",
                "type": "line",
                "start_point_id": "D",
                "end_point_id": "E",
                "level": 3,
                "is_original": True,
                "is_minimal": True,
                "length": {"expr": "7*sqrt(3)", "latex": "7 \\sqrt{3}", "value": 12.12435566}
            },
            {
                "id": "L2",
                "type": "line",
                "start_point_id": "E",
                "end_point_id": "C",
                "level": 3,
                "is_original": True,
                "is_minimal": False,
                "length": {"expr": "7*sqrt(3)", "latex": "7 \\sqrt{3}", "value": 12.12435566}
            },
            {
                "id": "PerpL1",
                "type": "perpendicular",
                "start_point_id": "E",
                "end_point_id": "F0",
                "level": 4,
                "description": "Draw perpendicular from point E to line L0 (foot F0 on segment)",
                "is_original": True,
                "is_minimal": False,
                "length": {"expr": "21/2", "latex": "\\frac{21}{2}", "value": 10.5}
            },
            {
                "id": "L_min_1",
                "start_point_id": "C",
                "end_point_id": "F0",
                "level": 4,
                "is_original": False,
                "is_minimal": True,
                "length": {"expr": "7*sqrt(3)/2", "latex": "\\frac{7 \\sqrt{3}}{2}", "value": 6.06217783}
            },
            {
                "id": "L_min_2",
                "start_point_id": "F0",
                "end_point_id": "F",
                "level": 4,
                "is_original": False,
                "is_minimal": True,
                "length": {"expr": "7*sqrt(7 - 4*sqrt(3))/2", "latex": "\\frac{7 \\sqrt{7 - 4 \\sqrt{3}}}{2}", "value": 0.93782217}
            },
            {
                "id": "L_min_3",
                "start_point_id": "F",
                "end_point_id": "D",
                "level": 3,
                "is_original": False,
                "is_minimal": True,
                "length": {"expr": "7*sqrt(4 - 2*sqrt(3))", "latex": "7 \\sqrt{4 - 2 \\sqrt{3}}", "value": 5.12435565}
            },
            {
                "id": "L_min_4",
                "start_point_id": "E",
                "end_point_id": "B",
                "level": 3,
                "is_original": False,
                "is_minimal": True,
                "length": {"expr": "7*sqrt(4 - 2*sqrt(3))", "latex": "7 \\sqrt{4 - 2 \\sqrt{3}}", "value": 5.12435565}
            },
            {
                "id": "L_min_5",
                "start_point_id": "B",
                "end_point_id": "C",
                "level": 1,
                "is_original": False,
                "is_minimal": True,
                "length": {"expr": "7", "latex": "7", "value": 7.0}
            },
            {
                "id": "L_min_6",
                "start_point_id": "E",
                "end_point_id": "I2",
                "level": 3,
                "is_original": False,
                "is_minimal": True,
                "length": {"expr": "7", "latex": "7", "value": 7.0}
            },
            {
                "id": "L_min_7",
                "start_point_id": "I2",
                "end_point_id": "F0",
                "level": 4,
                "is_original": False,
                "is_minimal": True,
                "length": {"expr": "7/2", "latex": "\\frac{7}{2}", "value": 3.5}
            }
        ],
        "arcs": [
            {
                "id": "Arc0",
                "type": "arc",
                "start_point_id": "circle_1",
                "end_point_id": "circle_1",
                "center_point_id": "O1",
                "radius": {"expr": "7", "latex": "7"},
                "angle": {"expr": "2*pi", "latex": "2 \\pi"},
                "is_complete": True,
                "level": 1,
                "is_original": True,
                "is_minimal": False,
                "length": {"expr": "0", "latex": "0", "value": None}
            },
            {
                "id": "Arc1",
                "type": "arc",
                "start_point_id": "circle_2",
                "end_point_id": "circle_2",
                "center_point_id": "C",
                "radius": {"expr": "7", "latex": "7"},
                "angle": {"expr": "2*pi", "latex": "2 \\pi"},
                "is_complete": True,
                "level": 1,
                "is_original": True,
                "is_minimal": False,
                "length": {"expr": "0", "latex": "0", "value": None}
            },
            {
                "id": "Arc_min_1",
                "start_point_id": "circle_1",
                "end_point_id": "A",
                "center_point_id": "O1",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/6"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "7*pi/6", "latex": "\\frac{7 \\pi}{6}", "value": 3.66519143}
            },
            {
                "id": "Arc_min_2",
                "start_point_id": "A",
                "end_point_id": "C",
                "center_point_id": "O1",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/3"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "7*pi/3", "latex": "\\frac{7 \\pi}{3}", "value": 7.33038286}
            },
            {
                "id": "Arc_min_3",
                "start_point_id": "C",
                "end_point_id": "B",
                "center_point_id": "O1",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/3"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "7*pi/3", "latex": "\\frac{7 \\pi}{3}", "value": 7.33038286}
            },
            {
                "id": "Arc_min_4",
                "start_point_id": "B",
                "end_point_id": "circle_1",
                "center_point_id": "O1",
                "radius": {"expr": "7"},
                "angle": {"expr": "7*pi/6"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "35*pi/6", "latex": "\\frac{35 \\pi}{6}", "value": 18.32595715}
            },
            {
                "id": "Arc_min_5",
                "start_point_id": "B",
                "end_point_id": "O1",
                "center_point_id": "C",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/3"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "7*pi/3", "latex": "\\frac{7 \\pi}{3}", "value": 7.33038286}
            },
            {
                "id": "Arc_min_6",
                "start_point_id": "O1",
                "end_point_id": "A",
                "center_point_id": "C",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/3"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "7*pi/3", "latex": "\\frac{7 \\pi}{3}", "value": 7.33038286}
            },
            {
                "id": "Arc_min_7",
                "start_point_id": "A",
                "end_point_id": "circle_2",
                "center_point_id": "C",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/6"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "7*pi/6", "latex": "\\frac{7 \\pi}{6}", "value": 3.66519143}
            },
            {
                "id": "Arc_min_8",
                "start_point_id": "circle_2",
                "end_point_id": "F",
                "center_point_id": "C",
                "radius": {"expr": "7"},
                "angle": {"expr": "5*pi/6"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 3,
                "length": {"expr": "35*pi/6", "latex": "\\frac{35 \\pi}{6}", "value": 18.32595715}
            },
            {
                "id": "Arc_min_9",
                "start_point_id": "F",
                "end_point_id": "I2",
                "center_point_id": "C",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/6"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 3,
                "length": {"expr": "7*pi/6", "latex": "\\frac{7 \\pi}{6}", "value": 3.66519143}
            },
            {
                "id": "Arc_min_10",
                "start_point_id": "I2",
                "end_point_id": "B",
                "center_point_id": "C",
                "radius": {"expr": "7"},
                "angle": {"expr": "pi/6"},
                "is_complete": False,
                "is_original": False,
                "is_minimal": True,
                "level": 1,
                "length": {"expr": "7*pi/6", "latex": "\\frac{7 \\pi}{6}", "value": 3.66519143}
            }
        ],
        "entities": [
            {
                "id": "base_circle",
                "type": "circle",
                "center_id": "O1",
                "start_id": "circle_1",
                "end_id": "circle_1",
                "arcs": "Arc0",
                "radius": {"expr": "7", "latex": "7"},
                "start_angle": {"expr": "0", "latex": "0"},
                "end_angle": {"expr": "2*pi", "latex": "2 \\pi"},
                "is_complete": True,
                "is_base": True,
                "perimeter": {"expr": "14*pi", "latex": "14 \\pi"},
                "area": {"expr": "49*pi", "latex": "49 \\pi"}
            },
            {
                "id": "translated_circle",
                "type": "circle",
                "center_id": "C",
                "start_id": "circle_2",
                "end_id": "circle_2",
                "arcs": "Arc1",
                "radius": {"expr": "7", "latex": "7"},
                "start_angle": {"expr": "0", "latex": "0"},
                "end_angle": {"expr": "2*pi", "latex": "2 \\pi"},
                "is_complete": True,
                "is_base": False,
                "perimeter": {"expr": "14*pi", "latex": "14 \\pi"},
                "area": {"expr": "49*pi", "latex": "49 \\pi"}
            },
            {
                "id": "vertex_on_center_polygon_n3",
                "type": "polygon",
                "center_id": "O3",
                "vertices": ["C", "D", "E"],
                "lines": ["L0", "L1", "L2"],
                "n": 3,
                "side_length": {"expr": "7*sqrt(3)", "latex": "7 \\sqrt{3}"},
                "radius": {"expr": "7", "latex": "7"},
                "inner_radius": {"expr": "7/2", "latex": "\\frac{7}{2}"},
                "rotation": {"expr": "0", "latex": "0"},
                "is_base": False,
                "perimeter": {"expr": "21*sqrt(3)", "latex": "21 \\sqrt{3}"},
                "area": {"expr": "36.75*sqrt(3)", "latex": "36.75 \\sqrt{3}"}
            },
            {
                "type": "shadow",
                "region_label": 6,
                "description": "Region 6 is shaded with hatch pattern.",
                "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "O1"}],
                "lines": [{"id": "L_min_5"}],
                "arcs": [{"id": "Arc_min_2"}, {"id": "Arc_min_5"}, {"id": "Arc_min_6"}],
                "ordered_loops": [{"ordered_points": ["A", "C", "B", "O1", "A"]}],
                "validity": True,
                "shader_params": {"type": "hatch", "spacing": 5, "intensity": 0.7897901737174271},
                "perimeter": {"expr": "7 + 7*pi", "latex": "7 + 7 \\pi", "value": 28.99114856},
                "area": {"expr": "-49*sqrt(3)/4 + 49*pi/2", "latex": "- \\frac{49 \\sqrt{3}}{4} + \\frac{49 \\pi}{2}", "value": 55.75139761}
            },
            {
                "type": "shadow",
                "region_label": 7,
                "description": "Region 7 is shaded with hatch pattern.",
                "points": [{"id": "B"}, {"id": "E"}, {"id": "I2"}],
                "lines": [{"id": "L_min_4"}, {"id": "L_min_6"}],
                "arcs": [{"id": "Arc_min_10"}],
                "ordered_loops": [{"ordered_points": ["I2", "B", "E", "I2"]}],
                "validity": True,
                "shader_params": {"type": "hatch", "spacing": 5, "intensity": 0.7897901737174271},
                "perimeter": {"expr": "7*pi/6 + 7*sqrt(4 - 2*sqrt(3)) + 7", "latex": "\\frac{7 \\pi}{6} + 7 \\sqrt{4 - 2 \\sqrt{3}} + 7", "value": 15.78954709},
                "area": {"expr": "-49/2 + 49*pi/12 + 49*sqrt(3)/4", "latex": "- \\frac{49}{2} + \\frac{49 \\pi}{12} + \\frac{49 \\sqrt{3}}{4}", "value": 9.5457924}
            },
            {
                "type": "shadow",
                "region_label": 8,
                "description": "Region 8 is shaded with hatch pattern.",
                "points": [{"id": "A"}, {"id": "B"}, {"id": "O1"}, {"id": "circle_1"}],
                "lines": [],
                "arcs": [{"id": "Arc_min_1"}, {"id": "Arc_min_4"}, {"id": "Arc_min_5"}, {"id": "Arc_min_6"}],
                "ordered_loops": [{"ordered_points": ["circle_1", "A", "O1", "B", "circle_1"]}],
                "validity": True,
                "shader_params": {"type": "hatch", "spacing": 5, "intensity": 0.7897901737174271},
                "perimeter": {"expr": "35*pi/3", "latex": "\\frac{35 \\pi}{3}", "value": 36.6519143}
            }
        ],
        "description": "Original graphic description: Base shape is a complete circle (center O1) with radius 7. Round 1: applying 'translation' rule. Translation derivation: circle (new center O2) translated up, with distance equal to the radius. Round 2: applying 'vertex_on_center' rule. Vertex-on-center derivation: 3-gon (center O3) with radius equal to base circle radius. Round 3: applying 'side_polygon' rule.\nEnhancement index: 2/5\nNumber of rounds: 1\nRound 1:\nDraw perpendicular from point E to line L0 (foot F0 on segment)\nRegion 6 is shaded with hatch pattern.\nRegion 7 is shaded with hatch pattern.\nRegion 8 is shaded with hatch pattern.",
        "is_base": False,
        "timeout_occurred": False,
        "completed_rounds": 1,
        "base_id": "",
        "enhance_id": "_enhance_001",
        "raw_path": "./results_n2000_mid/images/raw/geometry_line_1107.png",
        "annotated_raw_path": "./results_n2000_mid/images/annotated/annotated_raw_geometry_line_1107.png",
        "shaded_path": "./results_n2000_mid/images/shaded/shaded_1106_attempt_2_geometry_line_1107.png",
        "annotated_shaded_path": "./results_n2000_mid/images/annotated/annotated_shaded_1106_attempt_2_geometry_line_1107.png",
        "shadow_type": "hatch",
        "shader_enabled": True,
        "shadow_regions": 3
    }
    # 初始化计算器
    calculator = GeometryCalculator()

    # 测试Region 6的几何计算
    print("="*60)
    print("测试：自定义复杂几何数据（Region 6）")
    # 执行计算
    result = calculator.calculate_single(test_data_polygon)

    # 精准定位Region 6（筛选type=shadow且region_label=6的实体）
    shadow_region6 = next(
        e for e in result['entities'] 
        if e['type'] == 'shadow' and e['region_label'] == 8
    )

    # 打印Region 6的计算结果
    print(f"Region 6 周长表达式: {shadow_region6['perimeter']['expr']}")
    print(f"Region 6 面积表达式: {shadow_region6['area']['expr']}")
    print(f"Region 6 面积数值: {shadow_region6['area']['value']}")
    print("="*60)