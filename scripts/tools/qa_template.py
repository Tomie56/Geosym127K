import random
import logging
from typing import Dict, List, Any, Optional, Tuple
import sympy as sp
from sympy import simplify, acos, pi, sqrt, cos

logger = logging.getLogger('QAGenerator')

class QAGenerator:
    """几何问题生成器，支持基于level的元素选择和综合难度计算（优化实体/阴影命名）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 长度问题模板（边长度）
        self.length_templates = [
            "What is the length of {edge_name}?",
            "Calculate the length of {edge_name}.",
            "Determine the length of {edge_name} in the figure.",
            "Find the length of {edge_name}.",
            "What is the measurement of {edge_name}?"
        ]
        
        # 周长问题模板（实体周长）- 优化：用直观实体名称替代id
        self.perimeter_templates = [
            "What is the perimeter of the {entity_type} {entity_display_name}?",
            "Calculate the perimeter of the {entity_type} {entity_display_name} in the figure.",
            "Determine the perimeter of {entity_type} {entity_display_name}.",
            "Find the measurement of the perimeter of {entity_type} {entity_display_name}.",
            "What is the total length of the boundary of {entity_type} {entity_display_name}?"
        ]
        
        # 角度问题模板
        self.angle_templates = [
            "What is the measure of angle {angle_name} in radians?",
            "Express the angle at point {vertex_id} between {edge1_name} and {edge2_name} as a radian expression.",
            "Determine the radian measure of the angle formed by {edge1_name} and {edge2_name} at {vertex_id}.",
            "Find the radian expression for angle {angle_name}.",
            "What is the radian value of the angle between {edge1_name} and {edge2_name} at vertex {vertex_id}?"
        ]
        
        # 面积相关模板（优化：实体/阴影用直观名称）
        # 1. 单个实体面积模板
        self.entity_area_templates = [
            "What is the area of the {entity_display_name}?",
            "Calculate the area of the {entity_display_name} in the figure.",
            "Determine the area of {entity_display_name}.",
            "Find the measurement of the area of {entity_display_name}.",
            "What is the numerical value of the area of {entity_display_name}?"
        ]
        
        # 2. 单个阴影区域面积模板 - 优化：用顶点字母组合替代label
        self.shadow_area_templates = [
            "What is the area of the shadow region {shadow_display_name}?",
            "Calculate the area of the shaded region {shadow_display_name} in the figure.",
            "Determine the area of the shadow region {shadow_display_name}.",
            "Find the area of the shadow region {shadow_display_name}.",
            "What is the measurement of the area of the shadow region {shadow_display_name}?"
        ]
        
        # 3. 阴影面积之比模板
        self.shadow_ratio_templates = [
            "What is the ratio of the area of shadow region {shadow_display_name1} to that of shadow region {shadow_display_name2}?",
            "Calculate the ratio of the area of shaded region {shadow_display_name1} to shaded region {shadow_display_name2}.",
            "Determine the simplified ratio of the area of shadow {shadow_display_name1} to shadow {shadow_display_name2}.",
            "Find the ratio of the area of the shadow {shadow_display_name1} to shadow {shadow_display_name2}.",
            "What is the value of the ratio (Area of shadow {shadow_display_name1} : Area of shadow {shadow_display_name2})?"
        ]
        
        # 4. 阴影与实体面积之比模板
        self.shadow_entity_ratio_templates = [
            "What is the ratio of the area of shadow region {shadow_display_name} to the area of {entity_type} {entity_display_name}?",
            "Calculate the ratio of the area of shaded region {shadow_display_name} to the area of {entity_type} {entity_display_name}.",
            "Determine the simplified ratio of the area of shadow {shadow_display_name} to {entity_type} {entity_display_name}.",
            "Find the ratio of the area of shadow {shadow_display_name} to {entity_type} {entity_display_name}.",
            "What is the value of the ratio (Area of shadow {shadow_display_name} : Area of {entity_type} {entity_display_name})?"
        ]
        
        # 表达式复杂度分段阈值（支持后续修改）
        self.EXPR_COMPLEX_THRESHOLD1 = 20  # 简单表达式阈值
        self.EXPR_COMPLEX_THRESHOLD2 = 40  # 中等表达式阈值
        self.EXPR_COMPLEX_THRESHOLD3 = 100  # 较复杂表达式阈值
        
        # 复杂度权重分配（支持后续修改）
        self.GRAPH_WEIGHT = 0.2    # 图复杂度权重
        self.QUESTION_WEIGHT = 0.5  # 问题复杂度权重
        self.ANSWER_WEIGHT = 0.3   # 答案复杂度权重

    # ------------------------------ 新增：命名辅助方法（核心优化） ------------------------------
    def _get_entity_display_name(self, entity: Dict[str, Any]) -> str:
        """生成实体直观名称：支持圆（Circle + 圆心ID）、多边形（类型+顶点字母组合）"""
        entity_type = entity.get('type', 'shape').lower()
        vertices = entity.get('vertices', [])  # 实体顶点ID列表（如["A", "B", "C", "D"]）
        
        # 优先处理圆实体：用「Circle + 圆心ID」命名
        if entity_type == 'circle':
            center_id = entity.get('center_id', '')  # 从实体中提取圆心ID
            if center_id:
                return f"Circle {center_id}"  # 格式：Circle O1、Circle C
            # 无圆心ID时，用原始ID简化版兜底
            entity_id = entity.get('id', 'unknown')
            id_parts = entity_id.split('_')
            key_part = id_parts[-1].capitalize() if len(id_parts) > 1 else entity_id.capitalize()
            return f"Circle {key_part}"
        
        # 处理其他有顶点的实体（如平行四边形、三角形等）
        if vertices and all(isinstance(v, str) for v in vertices):
            # 顶点字母去重并保持原顺序
            unique_vertices = []
            seen = set()
            for v in vertices:
                if v not in seen:
                    seen.add(v)
                    unique_vertices.append(v)
            # 生成名称（首字母大写），如"Parallelogram ABCD"
            vertex_str = "".join(unique_vertices)
            return f"{entity_type.capitalize()} {vertex_str}"
        
        # 无顶点信息的非圆实体：用原始ID的简化版
        entity_id = entity.get('id', 'unknown')
        # 提取ID中的关键部分，如"base_parallelogram_2x4" -> "Parallelogram Base2x4"
        id_parts = entity_id.split('_')
        if len(id_parts) >=3:
            key_part = f"{id_parts[0].capitalize()}{id_parts[-1]}"
        else:
            key_part = entity_id.capitalize()
        return f"{entity_type.capitalize()} {key_part}"

    def _get_shadow_display_name(self, shadow: Dict[str, Any]) -> str:
        """生成阴影直观名称：Shadow + 有序顶点字母组合（基于ordered_loops）"""
        ordered_loops = shadow.get('ordered_loops', [])
        if ordered_loops and isinstance(ordered_loops[0], dict):
            # 取第一个loop的有序顶点，去重并保持原顺序
            ordered_points = ordered_loops[0].get('ordered_points', [])
            if ordered_points:
                unique_points = []
                seen = set()
                for p in ordered_points:
                    if p not in seen and isinstance(p, str):
                        seen.add(p)
                        unique_points.append(p)
                # 生成名称，如"Shadow ADEF"
                point_str = "".join(unique_points)
                return f"Shadow {point_str}"
        # 无ordered_loops时，用region_label生成备用名称
        region_label = shadow.get('region_label', '')
        return f"Shadow {region_label}" if region_label else "Shadow Unknown"

    # ------------------------------ 核心辅助方法（基于level的复杂度计算） ------------------------------
    def _get_element_level(self, element: Dict[str, Any], element_type: str) -> int:
        """获取元素的level值，默认为1"""
        if element_type == 'point':
            return element.get('level', 1)
        elif element_type in ['line', 'edge']:
            return element.get('level', 1)
        elif element_type == 'entity':
            # 实体的level：取关联点的平均level（无则默认1）
            points = self.config.get('points', []) if 'points' not in element else element.get('points', [])
            if points and isinstance(points[0], dict) and 'id' in points[0]:
                point_ids = [p['id'] for p in points]
                total_level = sum(self._get_element_level(p, 'point') for p in self.config.get('points', []) if p['id'] in point_ids)
                return total_level // len(point_ids) if point_ids else 1
            return element.get('level', 1)
        elif element_type == 'shadow':
            # 阴影的level：取关联点的平均level（无则默认1）
            point_ids = [p['id'] for p in element.get('points', []) if isinstance(p, dict) and 'id' in p]
            total_level = sum(self._get_element_level(p, 'point') for p in self.config.get('points', []) if p['id'] in point_ids)
            return total_level // len(point_ids) if point_ids else 1
        return 1

    def _calculate_graph_complexity(self, geo_data: Dict[str, Any]) -> float:
        """计算图的复杂度：基于点、边、实体、阴影的数量和平均level"""
        entities = geo_data.get('entities', [])
        shadows = [e for e in entities if e.get('type') == 'shadow']
        non_shadow_entities = [e for e in entities if e.get('type') != 'shadow']
        
        points = geo_data.get('points', [])
        lines = geo_data.get('lines', [])
        arcs = geo_data.get('arcs', [])
        
        # 基础复杂度：元素数量
        points_count = len(points)
        edges_count = len(lines) + len(arcs)
        entities_count = len(non_shadow_entities)
        shadows_count = len(shadows)
        count_complexity = (points_count * 0.2 + edges_count * 0.2 + 
                           entities_count * 0.3 + shadows_count * 0.3)
        
        # 等级复杂度：各元素平均level
        avg_point_level = sum(self._get_element_level(p, 'point') for p in points) / len(points) if points else 1
        avg_edge_level = (sum(self._get_element_level(l, 'line') for l in lines) + 
                         sum(self._get_element_level(a, 'edge') for a in arcs)) / (len(lines) + len(arcs)) if (lines or arcs) else 1
        avg_entity_level = sum(self._get_element_level(e, 'entity') for e in non_shadow_entities) / len(non_shadow_entities) if non_shadow_entities else 1
        avg_shadow_level = sum(self._get_element_level(s, 'shadow') for s in shadows) / len(shadows) if shadows else 1
        
        level_complexity = (avg_point_level * 0.2 + avg_edge_level * 0.2 + 
                           avg_entity_level * 0.3 + avg_shadow_level * 0.3)
        
        # 综合图复杂度
        return count_complexity * 0.6 + level_complexity * 4.0  # 缩放因子使量级合理

    def _calculate_question_complexity(self, qt_type: str, **kwargs) -> float:
        """计算问题本身的复杂度：基于涉及元素的level"""
        if qt_type == 'length':
            edge_level = kwargs.get('edge_level', 1)
            return edge_level * 2.0
            
        elif qt_type == 'perimeter':
            entity_level = kwargs.get('entity_level', 1)
            return entity_level * 2.5  # 周长问题复杂度略高于基础长度问题
            
        elif qt_type == 'angle':
            point_level = kwargs.get('point_level', 1)
            edge1_level = kwargs.get('edge1_level', 1)
            edge2_level = kwargs.get('edge2_level', 1)
            avg_level = (point_level + edge1_level + edge2_level) / 3.0
            return avg_level * 1.0
            
        elif qt_type == 'entity_area':
            entity_level = kwargs.get('entity_level', 1)
            return entity_level * 3.0
            
        elif qt_type == 'shadow_area':
            shadow_level = kwargs.get('shadow_level', 1)
            return shadow_level * 3.0
            
        elif qt_type == 'shadow_ratio':
            shadow1_level = kwargs.get('shadow1_level', 1)
            shadow2_level = kwargs.get('shadow2_level', 1)
            avg_level = (shadow1_level + shadow2_level) / 2.0
            return avg_level * 3.5  # 比例问题复杂度更高
            
        elif qt_type == 'shadow_entity_ratio':
            shadow_level = kwargs.get('shadow_level', 1)
            entity_level = kwargs.get('entity_level', 1)
            avg_level = (shadow_level + entity_level) / 2.0
            return avg_level * 3.5
            
        return 1.0

    # def _calculate_answer_complexity(self, expr_length: int) -> float:
    #     """计算答案的复杂度：基于表达式长度的分段计算"""
    #     if expr_length <= self.EXPR_COMPLEX_THRESHOLD1:
    #         return expr_length * 0.3  # 简单
    #     elif expr_length <= self.EXPR_COMPLEX_THRESHOLD2:
    #         return expr_length * 0.7  # 中等
    #     elif expr_length <= self.EXPR_COMPLEX_THRESHOLD3:
    #         return expr_length * 0.  # 较复杂
    #     else:
    #         return expr_length * 20.0  # 复杂
    def _calculate_answer_complexity(self, expr_length: int) -> float:
        """
        计算复杂度：基于幂函数的温和递减模型
        锚点设计：
        - Len 1   -> Score 1.0
        - Len 10  -> Score 4.8  (符合 ~5)
        - Len 50  -> Score 11.3 (符合 ~10)
        - Len 200 -> Score 25.0
        """
        # 1. 边界防御
        if expr_length <= 1:
            return 1.0
        # 3. 核心计算
        # 先归一化到 0.0 - 1.0 之间
        normalized_len = (expr_length - 1) / 199.0
        
        # 使用 0.6 次幂
        # 幂 < 1 代表边际递减
        # 幂越接近 1 (如 0.9) 越像直线（增长越匀速）
        # 幂越接近 0 (如 0.1) 越像对数（前期暴涨）
        # 0.6 是一个非常平衡的值，适合"大部分数据在1-30"的场景
        curve = normalized_len ** 0.6
        
        # 映射回 1 - 25
        score = 1.0 + 29.0 * curve
        
        return score

    def _calculate_overall_difficulty(self, graph_complex: float, question_complex: float, answer_complex: float) -> float:
        """计算总体难度：加权求和"""
        return (graph_complex * self.GRAPH_WEIGHT +
                question_complex * self.QUESTION_WEIGHT +
                answer_complex * self.ANSWER_WEIGHT)

    # ------------------------------ 新增辅助方法（实体/阴影选择，添加显示名称） ------------------------------
    def _select_entity(self, geo_data: Dict[str, Any], target_field: str = 'area') -> Optional[Dict[str, Any]]:
        """优先选择level高且包含目标字段（area/perimeter）的实体（非阴影），添加显示名称"""
        entities = [e for e in geo_data.get('entities', []) if e.get('type') != 'shadow']
        if not entities:
            logger.warning("未找到非阴影实体数据")
            return None
        
        # 筛选包含目标字段的实体
        valid_entities = [e for e in entities if target_field in e and isinstance(e[target_field], dict) and 'expr' in e[target_field]]
        if not valid_entities:
            logger.warning(f"未找到包含{target_field}信息的实体")
            return None
        
        # 计算实体level并排序（降序），添加显示名称
        for entity in valid_entities:
            entity['calculated_level'] = self._get_element_level(entity, 'entity')
            entity['display_name'] = self._get_entity_display_name(entity)  # 新增：显示名称
        valid_entities.sort(key=lambda x: -x['calculated_level'])
        
        # 前30%高level实体中随机选择
        top_percent = int(len(valid_entities) * 2) or 1
        top_candidates = valid_entities[:top_percent]
        return random.choice(top_candidates)

    def _select_shadow(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """优先选择level高且包含面积信息的阴影区域，添加显示名称"""
        shadows = [e for e in geo_data.get('entities', []) if e.get('type') == 'shadow']
        if not shadows:
            # logger.warning("未找到阴影区域数据")
            return None
        
        # 筛选包含面积信息的阴影
        valid_shadows = [s for s in shadows if 'area' in s and isinstance(s['area'], dict) and 'expr' in s['area']]
        if not valid_shadows:
            logger.warning("未找到包含面积信息的阴影区域")
            return None
        
        # 计算阴影level并排序（降序），添加显示名称
        for shadow in valid_shadows:
            shadow['calculated_level'] = self._get_element_level(shadow, 'shadow')
            shadow['display_name'] = self._get_shadow_display_name(shadow)  # 新增：显示名称
        valid_shadows.sort(key=lambda x: -x['calculated_level'])
        
        # 前30%高level阴影中随机选择
        top_percent = int(len(valid_shadows) * 2) or 1
        top_candidates = valid_shadows[:top_percent]
        return random.choice(top_candidates)

    def _select_two_shadows(self, geo_data: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """选择两个不同的高level阴影区域（用于比例计算），添加显示名称"""
        shadows = [e for e in geo_data.get('entities', []) if e.get('type') == 'shadow']
        if len(shadows) < 2:
            # logger.warning("阴影区域数量不足2个，无法计算比例")
            return None
        
        # 筛选包含面积信息的阴影
        valid_shadows = [s for s in shadows if 'area' in s and isinstance(s['area'], dict) and 'expr' in s['area']]
        if len(valid_shadows) < 2:
            # logger.warning("包含面积信息的阴影区域数量不足2个")
            return None
        
        # 计算阴影level并排序（降序），添加显示名称
        for shadow in valid_shadows:
            shadow['calculated_level'] = self._get_element_level(shadow, 'shadow')
            shadow['display_name'] = self._get_shadow_display_name(shadow)  # 新增：显示名称
        valid_shadows.sort(key=lambda x: -x['calculated_level'])
        
        # 前50%高level阴影中选择两个不同的
        top_percent = int(len(valid_shadows) * 0.5) or 2
        top_candidates = valid_shadows[:top_percent]
        if len(top_candidates) < 2:
            top_candidates = valid_shadows
        
        # 随机选择两个不同的
        shadow1, shadow2 = random.sample(top_candidates, 2)
        return (shadow1, shadow2)

    # ------------------------------ 原有辅助方法（保持或修改） ------------------------------
    def _simplify_ratio(self, expr1: str, expr2: str) -> Tuple[str, str]:
        """简化两个表达式的比例（基于sympy）"""
        try:
            e1 = sp.sympify(expr1.replace('^', '**'))
            e2 = sp.sympify(expr2.replace('^', '**'))
            ratio = simplify(e1 / e2)
            if ratio.is_Rational:
                numerator = ratio.numerator
                denominator = ratio.denominator
                return (str(numerator), str(denominator))
            else:
                return (expr1, expr2)
        except Exception as e:
            logger.warning(f"比例简化失败: {str(e)}，返回原始表达式")
            return (expr1, expr2)

    def _get_edge_name(self, edge_type: str, edge_data: Dict[str, Any]) -> str:
        if edge_type == 'line':
            start = edge_data['start_point_id']
            end = edge_data['end_point_id']
            return f"Line {start}{end}"  # 优化：简化为Line AB（原Line_A_B）
        elif edge_type == 'arc':
            if edge_data.get('is_complete', False):
                center = edge_data.get('center_point_id', edge_data['id'])
                return f"Circle {center}"
            else:
                start = edge_data['start_point_id']
                end = edge_data['end_point_id']
                return f"Arc {start}{end}"  # 优化：简化为Arc AB（原Arc_A_B）
        return edge_data['id']

    def _get_point_coords(self, points: List[Dict[str, Any]], point_id: str) -> Tuple[sp.Expr, sp.Expr]:
        for point in points:
            if point['id'] == point_id:
                x_expr = sp.sympify(point['x']['expr'].replace('^', '**'))
                y_expr = sp.sympify(point['y']['expr'].replace('^', '**'))
                return (x_expr, y_expr)
        raise ValueError(f"未找到点 {point_id} 的坐标信息")

    def _get_edge_other_vertex(self, edge_data: Dict[str, Any], vertex_id: str) -> str:
        if edge_data['start_point_id'] == vertex_id:
            return edge_data['end_point_id']
        elif edge_data['end_point_id'] == vertex_id:
            return edge_data['start_point_id']
        else:
            raise ValueError(f"边 {self._get_edge_name('line', edge_data)} 不包含顶点 {vertex_id}")

    def _calculate_vector(self, start_point: Tuple[sp.Expr, sp.Expr], end_point: Tuple[sp.Expr, sp.Expr]) -> Tuple[sp.Expr, sp.Expr]:
        return (end_point[0] - start_point[0], end_point[1] - start_point[1])

    def _is_valid_angle(self, angle_rad: sp.Expr) -> bool:
        is_zero = simplify(angle_rad) == 0
        is_straight = simplify(angle_rad - pi) == 0
        return not (is_zero or is_straight)

    def _is_radian_expression(self, angle_rad: sp.Expr) -> bool:
        simplified = simplify(angle_rad)
        if simplified == 0:
            return False
        pi_free = simplify(simplified / pi)
        return pi_free.is_Rational

    # ------------------------------ 优化：元素选择逻辑（优先高level元素） ------------------------------
    def _select_edge(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """优先选择level值高的边"""
        edges = []
        # 处理线段（包含level信息）
        if 'lines' in geo_data:
            for line in geo_data['lines']:
                edge_name = self._get_edge_name('line', line)
                edge_level = self._get_element_level(line, 'line')
                edges.append({
                    'type': 'line',
                    'data': line,
                    'name': edge_name,
                    'level': edge_level
                })
        # 处理圆弧
        if 'arcs' in geo_data:
            for arc in geo_data['arcs']:
                edge_name = self._get_edge_name('arc', arc)
                edge_level = self._get_element_level(arc, 'edge')
                edges.append({
                    'type': 'arc',
                    'data': arc,
                    'name': edge_name,
                    'level': edge_level
                })
        
        if not edges:
            logger.warning("未找到任何线条或圆弧数据")
            return None
        
        # 筛选有长度信息的边
        edges_with_length = [e for e in edges if 'length' in e['data']]
        if not edges_with_length:
            logger.warning("未找到有长度信息的边")
            return None
        
        # 按level降序排序，优先选择高level边
        edges_with_length.sort(key=lambda x: -x['level'])
        
        # 从高level边中随机选择（增加一点随机性）
        top_percent = int(len(edges_with_length) * 0.2) or 1
        top_candidates = edges_with_length[:top_percent]
        return random.choice(top_candidates)

    def _select_angle_edges(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """优先选择包含高level点和边的角度组合"""
        points = geo_data.get('points', [])
        lines = geo_data.get('lines', [])
        line_map = {
            line['id']: {
                'data': line,
                'name': self._get_edge_name('line', line),
                'level': self._get_element_level(line, 'line')
            } for line in lines
        }
        
        valid_combinations = []  # 存储所有有效角度组合
        
        for point in points:
            point_id = point['id']
            point_level = self._get_element_level(point, 'point')
            related_edges = point.get('related_edges', [])
            line_edges = [e for e in related_edges if not e.startswith('Arc') and e in line_map]
            if len(line_edges) < 2:
                continue
            
            from itertools import combinations
            for edge1_id, edge2_id in combinations(line_edges, 2):
                try:
                    edge1 = line_map[edge1_id]
                    edge2 = line_map[edge2_id]
                    edge1_data = edge1['data']
                    edge2_data = edge2['data']
                    
                    # 获取端点坐标
                    vertex1 = self._get_edge_other_vertex(edge1_data, point_id)
                    vertex2 = self._get_edge_other_vertex(edge2_data, point_id)
                    p_coords = self._get_point_coords(points, point_id)
                    v1_coords = self._get_point_coords(points, vertex1)
                    v2_coords = self._get_point_coords(points, vertex2)
                    
                    # 计算角度
                    vec1 = self._calculate_vector(p_coords, v1_coords)
                    vec2 = self._calculate_vector(p_coords, v2_coords)
                    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                    mod1 = sqrt(vec1[0]**2 + vec1[1]** 2)
                    mod2 = sqrt(vec2[0]**2 + vec2[1]** 2)
                    cos_theta = simplify(dot_product / (mod1 * mod2))
                    cos_theta = sp.Piecewise((1, cos_theta > 1), (-1, cos_theta < -1), (cos_theta, True))
                    angle_rad = simplify(acos(cos_theta))
                    
                    if self._is_valid_angle(angle_rad):
                        # 计算角度组合的综合level（点level + 两边level的平均）
                        combo_level = point_level * 0.4 + (edge1['level'] + edge2['level']) * 0.3
                        angle_str = str(angle_rad)
                        valid_combinations.append({
                            'vertex_id': point_id,
                            'vertex_level': point_level,
                            'edge1_id': edge1_id,
                            'edge1_name': edge1['name'],
                            'edge1_level': edge1['level'],
                            'edge2_id': edge2_id,
                            'edge2_name': edge2['name'],
                            'edge2_level': edge2['level'],
                            'vertex1': vertex1,
                            'vertex2': vertex2,
                            'angle_rad': angle_rad,
                            'angle_str': angle_str,
                            'expr_length': len(angle_str),
                            'combo_level': combo_level  # 用于排序的综合level
                        })
                except Exception as e:
                    edge1_name = line_map.get(edge1_id, {}).get('name', edge1_id)
                    edge2_name = line_map.get(edge2_id, {}).get('name', edge2_id)
                    logger.debug(f"计算点 {point_id} 的边 {edge1_name}-{edge2_name} 夹角失败: {str(e)}")
                    continue
        
        if not valid_combinations:
            logger.debug("未找到符合条件的角度")
            return None
        
        # 按综合level降序排序，优先选择高level组合
        valid_combinations.sort(key=lambda x: -x['combo_level'])
        
        # 从高level组合中随机选择
        top_percent = int(len(valid_combinations) * 0.2) or 1
        top_candidates = valid_combinations[:top_percent]
        return random.choice(top_candidates)

    # ------------------------------ 问题生成方法（核心优化：使用直观命名） ------------------------------
    def _generate_length_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        selected = self._select_edge(geo_data)
        if not selected:
            return None
            
        edge_type = selected['type']
        edge_data = selected['data']
        edge_name = selected['name']  # 已优化为Line AB格式
        edge_level = selected['level']
        
        length_info = edge_data['length']
        # 计算长度表达式长度
        expr_str = length_info['expr']
        expr_length = len(expr_str)
        
        # 生成描述（包含直观名称）
        qt_description = f"{edge_type.capitalize()} {edge_name} (Level: {edge_level}, Expr Length: {expr_length})"
        # print(qt_description)
        if 'description' in edge_data:
            qt_description += f" (Desc: {edge_data['description']})"
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='length',
            edge_level=edge_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        question = random.choice(self.length_templates).format(edge_name=edge_name)
        
        return {
            "question": question,
            "gt": length_info,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "length"
        }

    def _generate_perimeter_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """生成实体周长问题（使用直观实体名称）"""
        selected_entity = self._select_entity(geo_data, target_field='perimeter')
        if not selected_entity:
            return None
            
        entity_type = selected_entity['type']
        entity_display_name = selected_entity['display_name']  # 直观名称（如Parallelogram ABCD）
        entity_level = selected_entity['calculated_level']
        perimeter_info = selected_entity['perimeter']
        
        # 计算表达式长度
        expr_str = perimeter_info['expr']
        expr_length = len(expr_str)
        
        # 生成描述（包含直观名称）
        qt_description = f"Perimeter of {entity_display_name} (Level: {entity_level}, Expr Length: {expr_length})"
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='perimeter',
            entity_level=entity_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        question = random.choice(self.perimeter_templates).format(
            entity_type=entity_type,
            entity_display_name=entity_display_name  # 替换为直观名称
        )
        
        return {
            "question": question,
            "gt": perimeter_info,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "perimeter"
        }

    def _generate_angle_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        angle_data = self._select_angle_edges(geo_data)
        if not angle_data:
            return None
        
        # 解析角度数据
        vertex_id = angle_data['vertex_id']
        vertex_level = angle_data['vertex_level']
        edge1_name = angle_data['edge1_name']  # 已优化为Line AB格式
        edge1_level = angle_data['edge1_level']
        edge2_name = angle_data['edge2_name']  # 已优化为Line AB格式
        edge2_level = angle_data['edge2_level']
        vertex1 = angle_data['vertex1']
        vertex2 = angle_data['vertex2']
        angle_rad = angle_data['angle_rad']
        angle_str = angle_data['angle_str']
        expr_length = angle_data['expr_length']
        
        angle_name = f"{vertex1}{vertex_id}{vertex2}"  # 优化：角度名称简化为ABD（原A-B-D）
        gt = {
            'expr': angle_str,
            'latex': sp.latex(angle_rad),
            'is_standard_radian': self._is_radian_expression(angle_rad)
        }
        
        # 生成描述（包含直观名称）
        level_note = f" (Levels: Point={vertex_level}, Edge1={edge1_level}, Edge2={edge2_level})"
        expr_note = f" (Expr Length: {expr_length})"
        qt_description = (
            f"Angle {angle_name} by {edge1_name} & {edge2_name} at {vertex_id}{level_note}{expr_note}"
        )
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='angle',
            point_level=vertex_level,
            edge1_level=edge1_level,
            edge2_level=edge2_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        return {
            "question": random.choice(self.angle_templates).format(
                angle_name=angle_name, vertex_id=vertex_id, edge1_name=edge1_name, edge2_name=edge2_name
            ),
            "gt": gt,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "angle"
        }

    def _generate_entity_area_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """生成单个实体面积问题（使用直观实体名称）"""
        selected_entity = self._select_entity(geo_data, target_field='area')
        if not selected_entity:
            return None
            
        entity_type = selected_entity['type']
        entity_display_name = selected_entity['display_name']  # 直观名称（如Parallelogram ABCD）
        entity_level = selected_entity['calculated_level']
        area_info = selected_entity['area']
        
        # 计算表达式长度
        expr_str = area_info['expr']
        expr_length = len(expr_str)
        
        # 生成描述（包含直观名称）
        qt_description = f"Area of {entity_display_name} (Level: {entity_level}, Expr Length: {expr_length})"
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='entity_area',
            entity_level=entity_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        question = random.choice(self.entity_area_templates).format(
            entity_type=entity_type,
            entity_display_name=entity_display_name  # 替换为直观名称
        )
        
        return {
            "question": question,
            "gt": area_info,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "entity_area"
        }

    def _generate_shadow_area_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """生成单个阴影区域面积问题（使用直观阴影名称）"""
        selected_shadow = self._select_shadow(geo_data)
        if not selected_shadow:
            return None
            
        shadow_display_name = selected_shadow['display_name']  # 直观名称（如Shadow DAEF）
        shadow_type = geo_data.get('shadow_type', 'crosshatch')
        shadow_level = selected_shadow['calculated_level']
        area_info = selected_shadow['area']
        
        # 计算表达式长度
        expr_str = area_info['expr']
        expr_length = len(expr_str)
        
        # 生成描述（包含直观名称）
        qt_description = f"Area of {shadow_display_name} (Type: {shadow_type}, Level: {shadow_level}, Expr Length: {expr_length})"
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='shadow_area',
            shadow_level=shadow_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        question = random.choice(self.shadow_area_templates).format(
            shadow_display_name=shadow_display_name,  # 替换为直观名称
            shadow_type=shadow_type
        )
        
        return {
            "question": question,
            "gt": area_info,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "shadow_area"
        }

    def _generate_shadow_ratio_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """生成两个阴影区域面积之比问题（使用直观阴影名称）"""
        shadow_pair = self._select_two_shadows(geo_data)
        if not shadow_pair:
            return None
            
        shadow1, shadow2 = shadow_pair
        shadow_display_name1 = shadow1['display_name']  # 直观名称（如Shadow DAEF）
        shadow_display_name2 = shadow2['display_name']  # 直观名称（如Shadow ABCD）
        shadow_type = geo_data.get('shadow_type', 'crosshatch')
        shadow1_level = shadow1['calculated_level']
        shadow2_level = shadow2['calculated_level']
        
        # 获取面积表达式并简化比例
        area1_expr = shadow1['area']['expr']
        area2_expr = shadow2['area']['expr']
        simplified1, simplified2 = self._simplify_ratio(area1_expr, area2_expr)
        
        # 计算比例表达式长度（取较长者）
        expr_length = max(len(simplified1), len(simplified2))
        
        # 生成答案数据
        gt = {
            'expr': f"{simplified1} : {simplified2}",
            'latex': f"{sp.latex(sp.sympify(simplified1.replace('^', '**')))} : {sp.latex(sp.sympify(simplified2.replace('^', '**')))}",
            'original_exprs': (area1_expr, area2_expr)
        }
        
        # 生成描述（包含直观名称）
        qt_description = (f"Area ratio of {shadow_display_name1} to {shadow_display_name2} (Type: {shadow_type}, "
                          f"Levels: {shadow1_level}/{shadow2_level}, Expr Length: {expr_length})")
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='shadow_ratio',
            shadow1_level=shadow1_level,
            shadow2_level=shadow2_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        question = random.choice(self.shadow_ratio_templates).format(
            shadow_display_name1=shadow_display_name1,  # 替换为直观名称
            shadow_display_name2=shadow_display_name2,  # 替换为直观名称
            shadow_type=shadow_type
        )
        
        return {
            "question": question,
            "gt": gt,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "shadow_ratio"
        }

    def _generate_shadow_entity_ratio_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """生成阴影区域与实体面积之比问题（使用直观名称）"""
        selected_shadow = self._select_shadow(geo_data)
        selected_entity = self._select_entity(geo_data, target_field='area')
        if not selected_shadow or not selected_entity:
            return None
            
        shadow_display_name = selected_shadow['display_name']  # 直观阴影名称（如Shadow DAEF）
        shadow_type = geo_data.get('shadow_type', 'crosshatch')
        shadow_level = selected_shadow['calculated_level']
        
        entity_type = selected_entity['type']
        entity_display_name = selected_entity['display_name']  # 直观实体名称（如Parallelogram ABCD）
        entity_level = selected_entity['calculated_level']
        
        # 获取面积表达式并简化比例
        shadow_area_expr = selected_shadow['area']['expr']
        entity_area_expr = selected_entity['area']['expr']
        simplified_shadow, simplified_entity = self._simplify_ratio(shadow_area_expr, entity_area_expr)
        
        # 计算比例表达式长度（取较长者）
        expr_length = max(len(simplified_shadow), len(simplified_entity))
        
        # 生成答案数据
        gt = {
            'expr': f"{simplified_shadow} : {simplified_entity}",
            'latex': f"{sp.latex(sp.sympify(simplified_shadow.replace('^', '**')))} : {sp.latex(sp.sympify(simplified_entity.replace('^', '**')))}",
            'original_exprs': (shadow_area_expr, entity_area_expr)
        }
        
        # 生成描述（包含直观名称）
        qt_description = (f"Area ratio of {shadow_display_name} to {entity_display_name} (Shadow Type: {shadow_type}, "
                          f"Levels: Shadow={shadow_level}/Entity={entity_level}, Expr Length: {expr_length})")
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='shadow_entity_ratio',
            shadow_level=shadow_level,
            entity_level=entity_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        question = random.choice(self.shadow_entity_ratio_templates).format(
            shadow_display_name=shadow_display_name,  # 替换为直观阴影名称
            shadow_type=shadow_type,
            entity_type=entity_type,
            entity_display_name=entity_display_name  # 替换为直观实体名称
        )
        
        return {
            "question": question,
            "gt": gt,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "shadow_entity_ratio"
        }

    def _generate_all_shadow_area_questions(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """生成所有阴影区域面积之和的问题"""
        
        # 1. 筛选所有包含面积信息的有效阴影
        shadows = [e for e in geo_data.get('entities', []) if e.get('type') == 'shadow']
        valid_shadows = []
        
        for s in shadows:
            # 检查是否有面积表达式
            if 'area' in s and isinstance(s['area'], dict) and 'expr' in s['area']:
                # 补充计算所需的辅助信息（level和显示名称）
                s['calculated_level'] = self._get_element_level(s, 'shadow')
                s['display_name'] = self._get_shadow_display_name(s)
                valid_shadows.append(s)
        
        if not valid_shadows:
            # logger.warning("图中没有包含面积信息的阴影区域")
            return None

        # 2. 计算面积总和 (使用 SymPy 进行代数求和)
        try:
            total_area_sympy = 0
            shadow_names = []
            shadow_levels = []
            original_exprs = []

            for s in valid_shadows:
                shadow_names.append(s['display_name'])
                shadow_levels.append(s['calculated_level'])
                
                raw_expr = s['area']['expr']
                original_exprs.append(raw_expr)
                
                # 累加表达式 (处理 ^ 为 **)
                term = sp.sympify(raw_expr.replace('^', '**'))
                total_area_sympy += term
            
            # 化简结果
            simplified_total = simplify(total_area_sympy)
            
            # 格式化输出
            total_expr_str = str(simplified_total).replace('**', '^')
            total_latex = sp.latex(simplified_total)
            
        except Exception as e:
            logger.error(f"计算所有阴影面积之和失败: {str(e)}")
            return None

        # 3. 构造问题文本
        # 阴影名称列表字符串，如 "Shadow A, Shadow B and Shadow C"
        if len(shadow_names) > 1:
            names_str = ", ".join(shadow_names[:-1]) + " and " + shadow_names[-1]
        else:
            names_str = shadow_names[0]

        # 专用模板（建议放入 __init__，这里为了方便直接定义）
        all_shadow_templates = [
            "What is the sum of the areas of all shaded regions ({names_str})?",
            "Calculate the total area of all shadow regions in the figure.",
            "Determine the combined area of {names_str}.",
            "Find the sum of the areas of all shadows.",
            "What is the total measurement of the area covered by shadows?"
        ]
        
        question = random.choice(all_shadow_templates).format(names_str=names_str)

        # 4. 计算复杂度
        # 表达式长度
        expr_length = len(total_expr_str)
        
        # 图复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        
        # 问题复杂度：基于平均Level，并根据阴影数量给予额外加权
        avg_shadow_level = sum(shadow_levels) / len(shadow_levels)
        count_penalty = len(valid_shadows) * 0.5  # 每个额外阴影增加0.5复杂度
        question_complex = self._calculate_question_complexity(
            qt_type='shadow_area', # 复用 shadow_area 的基础逻辑
            shadow_level=avg_shadow_level
        ) + count_penalty
        
        # 答案复杂度
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)

        # 5. 构造返回数据
        qt_description = (f"Sum of areas of {len(valid_shadows)} shadows ({names_str}) "
                          f"(Avg Level: {avg_shadow_level:.1f}, Expr Length: {expr_length})")

        gt = {
            'expr': total_expr_str,
            'latex': total_latex,
            'original_exprs': original_exprs, # 记录原始的各个分项，便于调试或追溯
            'count': len(valid_shadows)
        }

        return {
            "question": question,
            "gt": gt,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "annotated_shaded_path": geo_data.get('annotated_shaded_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff,
            "question_type": "all_shadow_area"
        }
    # ------------------------------ 生成入口（支持新问题类型） ------------------------------
    def generate(self, geo_data: Dict[str, Any], num_questions: int, question_types: List[str], type_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        qa_pairs = []
        generated_keys = set()  # 存储已生成问题的唯一标识（用于去重）
        MAX_RETRY = 3  # 单类型问题生成的最大重试次数

        # 支持的所有问题类型
        supported_types = [
            'length', 'perimeter', 'angle', 
            'entity_area', 'shadow_area', 'shadow_ratio', 'shadow_entity_ratio'
        ]
        
        # 筛选有效问题类型
        valid_types = [qt for qt in question_types if qt in supported_types]
        if not valid_types:
            logger.warning("未指定有效问题类型，使用默认类型：length、entity_area、shadow_area")
            valid_types = ['length', 'entity_area', 'shadow_area']
        
        valid_types = ["length", "perimeter"]
        # 处理权重配置，生成目标问题类型列表
        if not type_weights or not isinstance(type_weights, dict):
            target_types = random.choices(valid_types, k=num_questions)
        else:
            weighted_types = []
            weights = []
            for qt in valid_types:
                if qt in type_weights and isinstance(type_weights[qt], (int, float)) and type_weights[qt] > 0:
                    weighted_types.append(qt)
                    weights.append(type_weights[qt])
            if not weighted_types:
                logger.warning("权重配置无效，退化为均等概率")
                target_types = random.choices(valid_types, k=num_questions)
            else:
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                target_types = random.choices(weighted_types, weights=normalized_weights, k=num_questions)
        
        # 生成问题
        for qt in target_types:
            retry = 0
            while retry < MAX_RETRY:
                try:
                # if True:
                    # 生成对应类型的问题
                    if qt == 'length':
                        qa = self._generate_length_question(geo_data)
                    elif qt == 'perimeter':
                        qa = self._generate_perimeter_question(geo_data)
                    elif qt == 'angle':
                        qa = self._generate_angle_question(geo_data)
                    elif qt == 'entity_area':
                        qa = self._generate_entity_area_question(geo_data)
                    elif qt == 'shadow_area':
                        qa = self._generate_shadow_area_question(geo_data)
                    elif qt == 'shadow_ratio':
                        qa = self._generate_shadow_ratio_question(geo_data)
                    elif qt == 'shadow_entity_ratio':
                        qa = self._generate_shadow_entity_ratio_question(geo_data)
                    elif qt == 'all_shadow_area':
                        qa = self._generate_all_shadow_area_questions(geo_data)
                    else:
                        logger.warning(f"不支持的问题类型: {qt}")
                        qa = None

                    if not qa:
                        retry += 1
                        logger.debug(f"第{retry}次重试生成{qt}类型问题（生成失败）")
                        continue

                    # 提取问题的唯一标识（基于直观名称，优化去重）
                    if qt == 'length':
                        edge_name = qa['question']
                        unique_key = f"length_{edge_name}"
                    elif qt == 'perimeter':
                        entity_name = qa['qt_description'].split('Perimeter of ')[1].split(' (')[0]
                        unique_key = f"perimeter_{entity_name}"
                    elif qt == 'angle':
                        angle_name = qa['question'].split('angle ')[1].split()[0]
                        unique_key = f"angle_{angle_name}"
                    elif qt == 'entity_area':
                        entity_name = qa['qt_description'].split('Area of ')[1].split(' (')[0]
                        unique_key = f"entity_area_{entity_name}"
                    elif qt == 'shadow_area':
                        shadow_name = qa['qt_description'].split('Area of ')[1].split(' (')[0]
                        unique_key = f"shadow_area_{shadow_name}"
                    elif qt == 'shadow_ratio':
                        shadow_names = qa['qt_description'].split('Area ratio of ')[1].split(' to ')
                        shadow1 = shadow_names[0].strip()
                        shadow2 = shadow_names[1].split(' (')[0].strip()
                        unique_key = f"shadow_ratio_{min(shadow1, shadow2)}_{max(shadow1, shadow2)}"
                    elif qt == 'shadow_entity_ratio':
                        parts = qa['qt_description'].split('Area ratio of ')[1].split(' to ')
                        shadow_name = parts[0].strip()
                        entity_name = parts[1].split(' (')[0].strip()
                        unique_key = f"shadow_entity_ratio_{shadow_name}_{entity_name}"
                    elif qt == 'all_shadow_area':
                        unique_key = "all_shadow_area_sum"

                    # 检查是否重复
                    if unique_key in generated_keys:
                        retry += 1
                        logger.debug(f"第{retry}次重试生成{qt}类型问题（重复：{unique_key}）")
                        continue

                    # 添加到结果
                    raw_desc = qa.get('description', '')
                    qt_desc = qa.get('qt_description', '')
                    
                    # 1. 简单替换：扫描 entities (含 shadows)，把 ID 和 label 替换成 display_name
                    # 注意：为了简单，这里直接遍历一次所有实体做 replace
                    for entity in geo_data.get('entities', []):
                        e_id = entity.get('id')
                        e_label = entity.get('label') # 比如 "Region 2"
                        
                        # 获取直观名字 (复用已有的方法)
                        if entity.get('type') == 'shadow':
                            display_name = self._get_shadow_display_name(entity)
                        else:
                            display_name = self._get_entity_display_name(entity)
                        
                        # 替换 Label (优先，如 Region 2 -> Triangle ABC)
                        if e_label and e_label in raw_desc:
                            raw_desc = raw_desc.replace(e_label, display_name)
                        if e_label and e_label in qt_desc:
                            qt_desc = qt_desc.replace(e_label, display_name)
                            
                        # 替换 ID (如 entity_3 -> Triangle ABC)
                        if e_id and e_id in raw_desc:
                            raw_desc = raw_desc.replace(e_id, display_name)
                        if e_id and e_id in qt_desc:
                            qt_desc = qt_desc.replace(e_id, display_name)

                    # 2. 简单替换：扫描 lines/arcs，把 ID 替换成 Line AB
                    for line in geo_data.get('lines', []) + geo_data.get('arcs', []):
                        l_id = line.get('id')
                        l_type = 'line' if 'lines' in geo_data and line in geo_data['lines'] else 'arc'
                        display_name = self._get_edge_name(l_type, line) # Line AB
                        
                        if l_id and l_id in raw_desc:
                            raw_desc = raw_desc.replace(l_id, display_name)
                        if l_id and l_id in qt_desc:
                            qt_desc = qt_desc.replace(l_id, display_name)

                    # 更新回 qa 对象，并生成 new_description
                    qa['description'] = raw_desc # 更新原始描述中的ID
                    qa['qt_description'] = qt_desc # 更新生成描述中的ID
                    
                    if raw_desc:
                        qa['new_description'] = f"{qt_desc} | Context: {raw_desc}"
                    else:
                        qa['new_description'] = qt_desc
                    
                    qa_pairs.append(qa)
                    generated_keys.add(unique_key)
                    logger.debug(f"生成{qt}类型问题（唯一标识：{unique_key}，难度：{qa['diff']:.2f}）")
                    break

                except Exception as e:
                    retry += 1
                    logger.error(f"第{retry}次重试生成{qt}类型问题失败: {str(e)}")
                    continue

            if retry >= MAX_RETRY:
                logger.debug(f"生成{qt}类型问题超过最大重试次数（{MAX_RETRY}次），跳过")

        return qa_pairs