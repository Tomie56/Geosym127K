import json
import random
import sympy as sp
import logging
from sympy import symbols, cos, sin, pi, simplify, sqrt, tan, Eq, solve, Min, Max, Ge, Le, Rational, ask, Q
from typing import List, Dict, Tuple, Optional, Union, Set, Generator
import os
import threading
import time
from collections import defaultdict
logger = logging.getLogger('builder')


class RandomGeometryBuilder:
    def __init__(self, base_json: Dict, base_id: str = "", max_points: int = 500, max_lines: int = 300):
        """初始化builder，加载基础图形数据并添加level属性"""
        self.data = base_json.copy()
        self.x, self.y = symbols('x y')
        self.max_points = max_points 
        self.max_lines = max_lines 
        
        self.max_expr_length = 50
        
        # 初始化并补全level属性（默认基础元素level=1）
        self.points = self.data["points"]
        for p in self.points:
            if "level" not in p:
                p["level"] = 1
        
        self.lines = self.data["lines"]
        for l in self.lines:
            if "level" not in l:
                l["level"] = 1
        
        self.arcs = self.data.get("arcs", [])
        for a in self.arcs:
            if "level" not in a:
                a["level"] = 1
        
        self.entities = self.data["entities"]
        
        self.point_id_map = {p["id"]: p for p in self.points}
        self.line_id_map = {l["id"]: l for l in self.lines}
        self.arc_id_map = {a["id"]: a for a in self.arcs}
        self.entity_id_map = {e["id"]: e for e in self.entities}
        
        self.operation_counter = 0
        self.entity_vertices_cache = self._cache_entity_vertices()
        self.entity_lines_cache = self._cache_entity_lines()
        self.entity_arcs_cache = self._cache_entity_arcs()
        self.on_segment_points_cache: Set[str] = set(self._cache_on_segment_points()) 
        self.all_points_cache: Set[str] = set(self.point_id_map.keys())
        self.line_pairs: Set[Tuple[str, str]] = self._build_line_pairs()
        
        self.enhancement_history = []
        
        self.MAX_RETRIES = 4
        self.base_id = base_id
        self.only_circles = self._is_only_circles()

        if len(self.points) >= self.max_points:
            raise ValueError(f"初始点数量({len(self.points)})超过上限({self.max_points})")
        if len(self.lines) >= self.max_lines:
            raise ValueError(f"初始线段数量({len(self.lines)})超过上限({self.max_lines})")

    # ------------------------------ 基础工具方法 ------------------------------
    
    def _is_only_circles(self) -> bool:
        if len(self.entities) == 0:
            return False
        all_circles = all(e["type"] == "circle" for e in self.entities)
        return all_circles

    def _cache_entity_vertices(self) -> Dict[str, List[str]]:
        cache = {}
        for e in self.entities:
            eid = e["id"]
            if "vertices" in e:
                cache[eid] = [c for c in e["vertices"] if c in self.point_id_map]
        return cache

    def _cache_entity_lines(self) -> Dict[str, List[str]]:
        cache = {}
        for e in self.entities:
            eid = e["id"]
            if "lines" in e:
                cache[eid] = [c for c in e["lines"] if c in self.line_id_map]
        return cache

    def _cache_entity_arcs(self) -> Dict[str, List[str]]:
        cache = {}
        for e in self.entities:
            eid = e["id"]
            if "arcs" in e:
                cache[eid] = [c for c in e["arcs"] if c in self.arc_id_map]
        return cache

    def _cache_on_segment_points(self) -> List[str]:
        on_segment = set()
        for line in self.lines:
            on_segment.add(line["start_point_id"])
            on_segment.add(line["end_point_id"])
        for pid in self.point_id_map:
            if pid in on_segment:
                continue
            px, py = self.get_point_coords(pid)
            for line in self.lines:
                s_id, e_id = line["start_point_id"], line["end_point_id"]
                s_x, s_y = self.get_point_coords(s_id)
                e_x, e_y = self.get_point_coords(e_id)
                if self._point_on_segment(px, py, s_x, s_y, e_x, e_y):
                    on_segment.add(pid)
                    break
        return list(on_segment)

    def _build_line_pairs(self) -> Set[Tuple[str, str]]:
        pairs = set()
        for line in self.lines:
            p1, p2 = line["start_point_id"], line["end_point_id"]
            if p1 > p2:
                p1, p2 = p2, p1
            pairs.add((p1, p2))
        return pairs
    
    def _is_expr_too_long(self, pid: str) -> bool:
        p = self.point_id_map[pid]
        x_expr = p["x"]["expr"]
        y_expr = p["y"]["expr"]
        return len(x_expr) > self.max_expr_length or len(y_expr) > self.max_expr_length

    def _is_point_on_any_segment(self, pid: str) -> bool:
        return pid in self.on_segment_points_cache
    
    def _is_point_on_other_edge(self, point_id: str, exclude_line_id: str) -> bool:
        if point_id not in self.point_id_map:
            return False
            
        px, py = self.get_point_coords(point_id)
        
        for line_id, line in self.line_id_map.items():
            if line_id == exclude_line_id:
                continue
                
            s_id, e_id = line["start_point_id"], line["end_point_id"]
            s_x, s_y = self.get_point_coords(s_id)
            e_x, e_y = self.get_point_coords(e_id)
            
            if self._point_on_segment(px, py, s_x, s_y, e_x, e_y):
                return True
        return False

    def _get_unique_id(self, prefix: str) -> str:
        uid = f"{prefix}{self.operation_counter}"
        self.operation_counter += 1
        return uid

    def _parse_expr(self, expr_str: str) -> sp.Expr:
        return sp.sympify(expr_str)

    def _serialize_expr(self, expr: sp.Expr) -> str:
        if expr.has(sp.sin, sp.cos, sp.sqrt, sp.tan):
            expr = simplify(expr)
        return str(expr)

    # ------------------------------ 点和线段操作（含level维护） ------------------------------
    def get_point_coords(self, pid: str) -> Tuple[sp.Expr, sp.Expr]:
        p = self.point_id_map[pid]
        return self._parse_expr(p["x"]["expr"]), self._parse_expr(p["y"]["expr"])

    def add_new_point(self, 
                     x_expr: sp.Expr, 
                     y_expr: sp.Expr, 
                     prefix: str = "P",
                     point_type: Optional[str] = None,
                     related_vertex: Optional[str] = None,
                     related_edge: Optional[str] = None,
                     level: int = 1) -> str:
        
        if len(self.points) >= self.max_points:
            raise OverflowError(f"点数量超过上限({self.max_points})，放弃本轮生成")
        
        # 预处理坐标表达式
        x_simplified = x_expr
        y_simplified = y_expr
        x_str = self._serialize_expr(x_simplified)
        y_str = self._serialize_expr(y_simplified)
        
        # --- 查重逻辑 ---
        for i, p in enumerate(self.points):
            existing_x = self._parse_expr(p["x"]["expr"])
            existing_y = self._parse_expr(p["y"]["expr"])
            
            # 判断坐标是否重合
            x_equal = sp.simplify(existing_x - x_simplified) == 0
            y_equal = sp.simplify(existing_y - y_simplified) == 0
            
            if x_equal and y_equal:
                # 命中重合点！
                old_id = p["id"]
                
                # 判定是否需要“替换”：
                # 如果旧点是特殊点 (以O开头代表圆心，或以circle开头代表初始点)
                # 且 新点是普通点 (prefix通常是P, A, B等，不是O或circle)
                is_old_special = old_id.startswith("O") or old_id.startswith("circle")
                is_new_standard = True

                if is_old_special and is_new_standard:
                    # === 执行替换逻辑 (O1 -> P1) ===
                    
                    # 1. 生成新的标准ID (例如 P3)
                    new_id = self._get_unique_id(prefix)
                    
                    # 2. 更新点本身的属性
                    # ID变了，但保留旧点的部分属性（如是否是圆心），并融合新属性
                    p["id"] = new_id
                    p["level"] = min(p["level"], level) # 提升层级
                    
                    # 如果旧点是圆心，新点虽然叫P，但要在属性里记下它也是圆心
                    if p.get("type") == "center" or old_id.startswith("O"):
                        p["type"] = "center" 
                    elif point_type:
                        p["type"] = point_type
                        
                    if related_vertex: p["related_vertex"] = related_vertex
                    if related_edge: p["related_edge"] = related_edge
                    
                    # 3. 更新辅助映射 (point_id_map)
                    if old_id in self.point_id_map:
                        del self.point_id_map[old_id]
                    self.point_id_map[new_id] = p
                    
                    # 4. 更新缓存 (set)
                    if old_id in self.all_points_cache:
                        self.all_points_cache.remove(old_id)
                        self.all_points_cache.add(new_id)
                    
                    # 5. 【关键】更新全局引用 (Lines, Arcs, Entities)
                    self._update_all_references(old_id, new_id)
                    
                    return new_id

                else:
                    # === 执行普通合并逻辑 (保留旧ID) ===
                    if p["level"] > level:
                        p["level"] = level
                    if point_type and "type" not in p:
                        p["type"] = point_type
                    if related_vertex and "related_vertex" not in p:
                        p["related_vertex"] = related_vertex
                    if related_edge and "related_edge" not in p:
                        p["related_edge"] = related_edge
                    
                    return old_id
        
        # --- 无重合，生成全新点 ---
        pid = self._get_unique_id(prefix)
        new_point = {
            "id": pid,
            "x": {"expr": x_str, "latex": sp.latex(x_simplified)},
            "y": {"expr": y_str, "latex": sp.latex(y_simplified)},
            "level": level
        }
        if point_type: new_point["type"] = point_type
        if related_vertex: new_point["related_vertex"] = related_vertex
        if related_edge: new_point["related_edge"] = related_edge
        
        self.points.append(new_point)
        self.point_id_map[pid] = new_point
        self.all_points_cache.add(pid)
        
        # 更新共线缓存
        px, py = x_simplified, y_simplified
        for line in self.lines:
            s_id, e_id = line["start_point_id"], line["end_point_id"]
            s_x, s_y = self.get_point_coords(s_id)
            e_x, e_y = self.get_point_coords(e_id)
            if self._point_on_segment(px, py, s_x, s_y, e_x, e_y):
                self.on_segment_points_cache.add(pid)
                # 不break，可能在多条线上
        
        # 更新实体顶点缓存
        for eid in self.entity_vertices_cache:
            self.entity_vertices_cache[eid].append(pid)
        
        return pid

    def _update_all_references(self, old_id: str, new_id: str):
        """
        辅助函数：将所有几何元素中对 old_id 的引用更新为 new_id
        """
        # 1. 更新直线
        for line in self.lines:
            if line["start_point_id"] == old_id:
                line["start_point_id"] = new_id
            if line["end_point_id"] == old_id:
                line["end_point_id"] = new_id
        
        # 2. 更新圆/圆弧
        for arc in self.arcs:
            if arc["center_point_id"] == old_id:
                arc["center_point_id"] = new_id
            if arc["start_point_id"] == old_id:
                arc["start_point_id"] = new_id
            if arc["end_point_id"] == old_id:
                arc["end_point_id"] = new_id

        # 3. 更新实体 (Entities) 的 vertices 列表
        for entity in self.entities:
            if "vertices" in entity:
                # 替换列表中的ID
                entity["vertices"] = [new_id if v == old_id else v for v in entity["vertices"]]
            if "center_id" in entity and entity["center_id"] == old_id:
                entity["center_id"] = new_id
        
        # 4. 更新实体缓存
        for eid, vertices in self.entity_vertices_cache.items():
            self.entity_vertices_cache[eid] = [new_id if v == old_id else v for v in vertices]
            
        # 5. 更新线段上点缓存 (如果使用了的话)
        if old_id in self.on_segment_points_cache:
            self.on_segment_points_cache.remove(old_id)
            self.on_segment_points_cache.add(new_id)

    def add_new_line(self, 
                    start_pid: str, 
                    end_pid: str, 
                    prefix: str = "L", 
                    line_type: Optional[str] = None,
                    description: Optional[str] = None,
                    level: int = 1) -> str:  # 新增level参数
        
        if len(self.lines) >= self.max_lines:
            raise OverflowError(f"线段数量超过上限({self.max_lines})，放弃本轮生成")
        
        p1, p2 = start_pid, end_pid
        if p1 > p2:
            p1, p2 = p2, p1
        if (p1, p2) in self.line_pairs:
            for line in self.lines:
                s, e = line["start_point_id"], line["end_point_id"]
                if (s == start_pid and e == end_pid) or (s == end_pid and e == start_pid):
                    if line_type and "type" not in line:
                        line["type"] = line_type
                    if description and "description" not in line:
                        line["description"] = description
                    # 重合线取最高level
                    if line["level"] < level:
                        line["level"] = level
                    return line["id"]
        
        lid = self._get_unique_id(prefix)
        new_line = {
            "id": lid,
            "type": line_type if line_type else "line",
            "start_point_id": start_pid,
            "end_point_id": end_pid,
            "level": level  # 存储level
        }
        if description:
            new_line["description"] = description
        
        self.line_id_map[lid] = new_line
        
        # 立即验证映射是否成功
        if self.line_id_map.get(lid) != new_line:
            raise RuntimeError(f"线段 {lid} 添加到line_id_map失败，映射同步异常")
        self.lines.append(new_line)
        self.line_pairs.add((p1, p2))
        
        self.on_segment_points_cache.add(start_pid)
        self.on_segment_points_cache.add(end_pid)
        
        for eid in self.entity_lines_cache:
            self.entity_lines_cache[eid].append(lid)
        
        return lid

    # ------------------------------ 随机选择方法 ------------------------------
    def _randomly_select_entity(self, entity_types: List[str] = None) -> str:
        candidates = []
        for e in self.entities:
            if not entity_types or e["type"] in entity_types:
                candidates.append(e["id"])
        if not candidates:
            raise ValueError("无符合条件的实体")
        return random.choice(candidates)

    def _randomly_select_points(self, 
                            entity_id: str = None, 
                            count: int = 2, 
                            distinct: bool = True, 
                            filter_isolated: bool = True) -> List[str]:
        if entity_id:
            candidates = self.entity_vertices_cache[entity_id]
        else:
            candidates = list(self.all_points_cache)
        
        if filter_isolated:
            candidates = [pid for pid in candidates if self._is_point_on_any_segment(pid)]

        candidates = [pid for pid in candidates if not self._is_expr_too_long(pid)]
        if not candidates:
            raise ValueError(f"无符合条件的点（已过滤表达式过长的点，上限{self.max_expr_length}字符）")
        
        if count > len(candidates):
            raise ValueError(f"线上点数量不足（需要{count}，现有{len(candidates)}）")
        
        return random.sample(candidates, count)

    def _randomly_select_lines(self, entity_id: str = None, count: int = 2, distinct: bool = True) -> List[str]:
        if entity_id:
            candidates = self.entity_lines_cache[entity_id]
        else:
            candidates = list(self.line_id_map.keys())
        
        if count > len(candidates):
            if entity_id:
                logger.debug(f"实体{entity_id}内线段不足，尝试跨实体选择")
                return self._randomly_select_lines(entity_id=None, count=count, distinct=distinct)
            raise ValueError(f"线段数量不足（需要{count}，现有{len(candidates)}）")
        
        if distinct:
            return random.sample(candidates, count)
        return [random.choice(candidates) for _ in range(count)]

    def _randomly_select_circle_entities(self) -> Tuple[str, str]:
        circle_entities = [e for e in self.entities if e["type"] == "circle"]
        if not circle_entities:
            raise ValueError("无circle类型实体")
        
        valid_circles = []
        for entity in circle_entities:
            for comp_id in entity["arcs"]:
                if comp_id in self.arc_id_map:
                    circle = self.arc_id_map[comp_id]
                    if ("center_id" in circle and "radius_expr" in circle and 
                        circle["center_id"] in self.point_id_map):
                        valid_circles.append((entity["id"], comp_id))
                        break
        
        if not valid_circles:
            raise ValueError("无包含有效圆心和半径的圆形实体")
        return random.choice(valid_circles)

    # ------------------------------ 操作可行性与选择 ------------------------------
    def _is_operation_feasible(self, op_type: str, constraints: Dict) -> bool:
        try:
            if op_type == "connect_points":
                on_segment_points = list(self.on_segment_points_cache)
                return len(on_segment_points) >= 2
            
            elif op_type == "connect_midpoints":
                lines = list(self.line_id_map.keys())
                on_segment_points = list(self.on_segment_points_cache)
                return (len(lines) >= 2) or (len(lines) >= 1 and len(on_segment_points) >= 1)
            
            elif op_type == "draw_perpendicular":
                on_segment_points = list(self.on_segment_points_cache)
                return len(self.line_id_map) >= 1 and len(on_segment_points) >= 1
            
            elif op_type == "draw_diameter":
                target_entity_types = constraints.get("entity_types", ["circle"])
                circle_entities = [e for e in self.entities if e["type"] in target_entity_types]
                if not circle_entities:
                    return False
                
                for entity in circle_entities:
                    for comp_id in entity["arcs"]:
                        if comp_id in self.arc_id_map:
                            circle = self.arc_id_map[comp_id]
                            if ("center_id" in circle and "radius" in circle and 
                                circle["center_id"] in self.point_id_map):
                                radius_expr = circle["radius"]["expr"] if isinstance(circle["radius"], dict) else circle["radius"]
                                if radius_expr:
                                    return True
                return False
            
            else:
                return False
        except Exception as e:
            logger.debug(f"操作{op_type}可行性检查失败: {str(e)}")
            return False

    def _select_feasible_operation_type(self, operation_probs: Dict[str, float], constraints: Dict) -> str:
        adjusted_probs = operation_probs.copy()
        
        draw_diameter_feasible = self._is_operation_feasible("draw_diameter", constraints.get("draw_diameter", {}))
        if draw_diameter_feasible and "draw_diameter" in adjusted_probs:
            diameter_prob = 0.4
            other_total = sum(p for k, p in adjusted_probs.items() if k != "draw_diameter")
            if other_total > 0:
                scale = (1 - diameter_prob) / other_total
                for k in adjusted_probs:
                    if k != "draw_diameter":
                        adjusted_probs[k] *= scale
                adjusted_probs["draw_diameter"] = diameter_prob
            else:
                adjusted_probs["draw_diameter"] = 1.0
            logger.debug(f"检测到可行的圆形实体，提高draw_diameter操作优先级（概率：{diameter_prob}）")

        feasible_types = []
        feasible_probs = []
        
        for op_type, prob in adjusted_probs.items():
            op_constraints = constraints.get(op_type, {})
            if self._is_operation_feasible(op_type, op_constraints):
                feasible_types.append(op_type)
                feasible_probs.append(prob)
        
        if not feasible_types:
            raise ValueError("无可行的操作类型")
        
        total = sum(feasible_probs)
        feasible_probs = [p / total for p in feasible_probs]
        return random.choices(feasible_types, weights=feasible_probs, k=1)[0]

    # ------------------------------ 操作生成 ------------------------------
    def _generate_random_operation(self, op_type: str, op_constraints: Dict) -> Dict:
        if op_type == "connect_points":
            constraints = op_constraints.get("connect_points", {})
            target_entity_types = constraints.get("entity_types", ["polygon", "special_rectangle", "special_triangle", "parallelogram", "trapezoid"])
            allow_isolated = constraints.get("allow_isolated", False)
            
            candidate_points = list(self.on_segment_points_set) if not allow_isolated else list(self.point_id_map.keys())
            if len(candidate_points) < 2:
                raise ValueError("可用的点不足，无法执行connect_points操作")
            
            for retry in range(self.MAX_RETRIES):
                try:
                    # 直接从候选点中随机选择两个不同的点
                    p1_id, p2_id = random.sample(candidate_points, 2)
                    pair = tuple(sorted((p1_id, p2_id)))
                    if pair not in self.connected_pairs_set:
                        logger.debug(f"connect_points重试{retry}次成功：连接点{p1_id}和{p2_id}")
                        entity_id = None
                        if retry < 3: # 只在前几次重试时尝试匹配实体，失败也无妨
                            try:
                                # 简单的启发式：如果两点都属于同一个实体，则选择该实体
                                p1_entities = self._get_entity_ids_for_point(p1_id)
                                p2_entities = self._get_entity_ids_for_point(p2_id)
                                common_entities = p1_entities.intersection(p2_entities)
                                if common_entities:
                                    entity_id = random.choice(list(common_entities))
                            except Exception:
                                pass

                        return {
                            "type": op_type,
                            "entity_id": entity_id,
                            "point_ids": [p1_id, p2_id]
                        }
                    logger.debug(f"connect_points重试{retry}次失败：点对({p1_id},{p2_id})已存在线段")
                except ValueError as e:
                    logger.debug(f"connect_points重试{retry}次失败：{str(e)}")
            
            raise ValueError(f"超过最大重试次数（{self.MAX_RETRIES}次），未找到可连接的线上点对")
        
        elif op_type == "connect_midpoints":
            constraints = op_constraints.get("connect_midpoints", {})
            target_entity_types = constraints.get("entity_types", ["polygon", "composite"])
            
            on_segment_vertices = list(self.on_segment_points_set)
            if not on_segment_vertices:
                raise ValueError("无可用的线上点，无法执行connect_midpoints操作")
            
            entity_id = self._randomly_select_entity(target_entity_types) if (target_entity_types and random.random() < 0.7) else None
            
            lines = self.entity_lines_cache.get(entity_id, list(self.line_id_map.keys())) if entity_id else list(self.line_id_map.keys())
            lines = list(set(lines))
            
            on_segment_vertices = [pid for pid in self.all_points_cache if self._is_point_on_any_segment(pid)]
            
            available_modes = []
            if len(lines) >= 2:
                available_modes.append("two_midpoints")
            if len(lines) >= 1:
                available_modes.append("vertex_and_midpoint")
            if not available_modes:
                raise ValueError("无可用的connect_midpoints模式（线上点不足）")
            
            mode_probs = constraints.get("mode_probs", {"two_midpoints": 0.5, "vertex_and_midpoint": 0.5})
            mode = random.choices(
                available_modes,
                weights=[mode_probs[mode] for mode in available_modes],
                k=1
            )[0]

            if mode == "two_midpoints":
                for retry in range(self.MAX_RETRIES):
                    try:
                        line1_id, line2_id = random.sample(lines, 2)
                        if line1_id == line2_id:
                            continue
                            
                        s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
                        x1, y1 = self.get_point_coords(s1)
                        x2, y2 = self.get_point_coords(e1)
                        mid1_x = (x1 + x2) / 2
                        mid1_y = (y1 + y2) / 2
                        
                        s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
                        x3, y3 = self.get_point_coords(s2)
                        x4, y4 = self.get_point_coords(e2)
                        mid2_x = (x3 + x4) / 2
                        mid2_y = (y3 + y4) / 2
                        
                        temp_mid1_id = f"temp_mid1_{retry}"
                        temp_mid2_id = f"temp_mid2_{retry}"
                        self.point_id_map[temp_mid1_id] = {"x": {"expr": str(mid1_x)}, "y": {"expr": str(mid1_y)}}
                        self.point_id_map[temp_mid2_id] = {"x": {"expr": str(mid2_x)}, "y": {"expr": str(mid2_y)}}
                        
                        mid1_on_other = self._is_point_on_other_edge(temp_mid1_id, line1_id)
                        mid2_on_other = self._is_point_on_other_edge(temp_mid2_id, line2_id)
                        
                        del self.point_id_map[temp_mid1_id]
                        del self.point_id_map[temp_mid2_id]
                        
                        if mid1_on_other or mid2_on_other:
                            logger.debug(f"connect_midpoints重试{retry}次失败：中点在另一条边上")
                            continue
                            
                        logger.debug(f"connect_midpoints（two_midpoints）重试{retry}次成功")
                        return {
                            "type": op_type,
                            "mode": mode,
                            "entity_id": entity_id,
                            "line_ids": [line1_id, line2_id]
                        }
                    except ValueError as e:
                        logger.debug(f"connect_midpoints重试{retry}次失败：{str(e)}")
                    
                    raise ValueError(f"超过最大重试次数，未找到可连接中点的线段对")
            
            else:  # mode == "vertex_and_midpoint"
                allow_endpoint = constraints.get("allow_endpoint", False)
                
                candidate_points = list(self.on_segment_points_set)
                if not candidate_points:
                    raise ValueError("无可用的线上点，无法执行draw_perpendicular操作")

                for retry in range(self.MAX_RETRIES):
                    try:
                        lines = self.entity_lines_cache.get(entity_id, list(self.line_id_map.keys())) if entity_id else list(self.line_id_map.keys())
                        if not lines:
                            continue
                        line_id = random.choice(lines)
                        line = self.line_id_map[line_id]
                        
                        line_id = random.choice(lines)
                        # vertex_id = random.choice(on_segment_vertices)
                        line = self.line_id_map[line_id]
                        vertex_id = random.choice(candidate_points)
                        
                        
                        is_endpoint = (vertex_id == line["start_point_id"]) or (vertex_id == line["end_point_id"])
                        
                        if allow_endpoint or not is_endpoint:
                            s, e = line["start_point_id"], line["end_point_id"]
                            x1, y1 = self.get_point_coords(s)
                            x2, y2 = self.get_point_coords(e)
                            mid_x = (x1 + x2) / 2
                            mid_y = (y1 + y2) / 2
                            
                            temp_mid_id = f"temp_mid_{retry}"
                            self.point_id_map[temp_mid_id] = {"x": {"expr": str(mid_x)}, "y": {"expr": str(mid_y)}}
                            
                            vertex_on_other = self._is_point_on_other_edge(vertex_id, line_id)
                            mid_on_other = self._is_point_on_other_edge(temp_mid_id, line_id)
                            
                            del self.point_id_map[temp_mid_id]
                            
                            if vertex_on_other or mid_on_other:
                                logger.debug(f"connect_midpoints重试{retry}次失败：顶点或中点在另一条边上")
                                continue
                                
                            logger.debug(f"connect_midpoints（vertex_and_midpoint）重试{retry}次成功")
                            return {
                                "type": op_type,
                                "mode": mode,
                                "entity_id": entity_id,
                                "line_id": line_id,
                                "vertex_id": vertex_id
                            }
                        else:
                            logger.debug(f"connect_midpoints重试{retry}次失败：顶点是端点且不允许")
                    except Exception as e:
                        logger.debug(f"connect_midpoints重试{retry}次失败：{str(e)}")
                    
                    raise ValueError(f"超过最大重试次数，未找到可连接的顶点和线段")
            
        elif op_type == "draw_perpendicular":
            constraints = op_constraints.get("draw_perpendicular", {})
            target_entity_types = constraints.get("entity_types", ["polygon", "composite"])
            
            def is_point_on_line(px, py, s_x, s_y, e_x, e_y):
                cross = (e_x - s_x) * (py - s_y) - (e_y - s_y) * (px - s_x)
                return sp.simplify(cross) == 0
            
            def calculate_foot_of_perpendicular(px, py, s_x, s_y, e_x, e_y):
                se_x = e_x - s_x
                se_y = e_y - s_y
                sp_x = px - s_x
                sp_y = py - s_y
                dot_product = sp_x * se_x + sp_y * se_y
                se_sq = se_x**2 + se_y**2
                t = dot_product / se_sq
                foot_x = s_x + t * se_x
                foot_y = s_y + t * se_y
                return sp.simplify(foot_x), sp.simplify(foot_y)
            
            def find_matching_point(target_x, target_y):
                for pid, p in self.point_id_map.items():
                    px = self._parse_expr(p["x"]["expr"])
                    py = self._parse_expr(p["y"]["expr"])
                    if sp.simplify(px == target_x) and sp.simplify(py == target_y):
                        return pid
                return None
            
            for retry in range(self.MAX_RETRIES):
                try:
                    entity_id = self._randomly_select_entity(target_entity_types) if (retry < 5) else None
                    lines = self.entity_lines_cache.get(entity_id, list(self.line_id_map.keys())) if entity_id else list(self.line_id_map.keys())
                    line_id = random.choice(lines)
                    line = self.line_id_map[line_id]
                    
                    point_id = self._randomly_select_points(
                        entity_id=entity_id,
                        count=1,
                        filter_isolated=True
                    )[0]
                    
                    px, py = self.get_point_coords(point_id)
                    s, e = line["start_point_id"], line["end_point_id"]
                    s_x, s_y = self.get_point_coords(s)
                    e_x, e_y = self.get_point_coords(e)
                    
                    if is_point_on_line(px, py, s_x, s_y, e_x, e_y):
                        logger.debug(f"draw_perpendicular重试{retry}次失败：点{point_id}在直线{line_id}（含延长线）上")
                        continue
                    
                    foot_x, foot_y = calculate_foot_of_perpendicular(px, py, s_x, s_y, e_x, e_y)
                    
                    foot_pid = find_matching_point(foot_x, foot_y)
                    if foot_pid:
                        a, b = (point_id, foot_pid) if point_id < foot_pid else (foot_pid, point_id)
                        if (a, b) in self.line_pairs or (b, a) in self.line_pairs:
                            logger.debug(f"draw_perpendicular重试{retry}次失败：垂足{foot_pid}与点{point_id}已存在线段")
                            continue
                    
                    logger.debug(f"draw_perpendicular重试{retry}次成功")
                    return {
                        "type": op_type,
                        "line_id": line_id,
                        "point_id": point_id,
                        "foot_coords": (foot_x, foot_y)
                    }
                except Exception as e:
                    logger.debug(f"draw_perpendicular重试{retry}次失败：{str(e)}")
            
            raise ValueError(f"超过最大重试次数（{self.MAX_RETRIES}次），未找到合适的垂线参数")
        
        elif op_type == "draw_diameter":
            constraints = op_constraints.get("draw_diameter", {})
            target_entity_types = constraints.get("entity_types", ["circle"])
            directions = constraints.get("directions", ["horizontal", "vertical"])
            for retry in range(self.MAX_RETRIES):
                try:
                    entity_id, circle_id = self._randomly_select_circle_entities(target_entity_types)
                    circle = self.arc_id_map[circle_id]
                    if "center_id" in circle and circle["center_id"] in self.point_id_map:
                        direction = random.choice(directions)
                        logger.debug(f"draw_diameter重试{retry}次成功")
                        return {
                            "type": op_type,
                            "entity_id": entity_id,
                            "circle_id": circle_id,
                            "center_id": circle["center_id"],
                            "radius_expr": circle["radius_expr"],
                            "direction": direction
                        }
                except Exception as e:
                    logger.debug(f"draw_diameter重试{retry}次失败：{str(e)}")
            
            raise ValueError(f"超过最大重试次数（{self.MAX_RETRIES}次），未找到有效圆形")
        
        else:
            raise ValueError(f"不支持的操作类型: {op_type}")
        
    def _get_entity_ids_for_point(self, point_id: str) -> set:
        """获取包含指定点的所有实体ID"""
        entity_ids = set()
        for ent_id, vertices in self.entity_vertices_cache.items():
            if point_id in vertices:
                entity_ids.add(ent_id)
        return entity_ids

    # ------------------------------ 交点检测与几何计算（含level处理） ------------------------------
    def _point_on_segment(self, px: sp.Expr, py: sp.Expr, 
                        x1: sp.Expr, y1: sp.Expr, 
                        x2: sp.Expr, y2: sp.Expr) -> bool:
        collinear = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        collinear_simplified = sp.simplify(collinear)
        if not sp.Eq(collinear_simplified, 0):
            return False
        
        x_min = sp.Min(x1, x2)
        x_max = sp.Max(x1, x2)
        y_min = sp.Min(y1, y2)
        y_max = sp.Max(y1, y2)
        
        x_in_range = sp.And(sp.Ge(px, x_min), sp.Le(px, x_max))
        y_in_range = sp.And(sp.Ge(py, y_min), sp.Le(py, y_max))
        
        on_segment = sp.And(x_in_range, y_in_range)
        simplified = sp.simplify(on_segment)
        
        return bool(simplified) if isinstance(simplified, bool) else False

    EPS = 1e-10

    def _eval_value(self, expr: sp.Expr) -> float:
        """
        辅助函数：将符号表达式转换为高精度浮点数，用于快速逻辑判断。
        如果表达式无法求值（如含未知符号），返回0以便后续逻辑处理。
        """
        try:
            # n=20 保证足够的精度以避免边界误判
            val = expr.evalf(n=20)
            if val.is_Number:
                return float(val)
            return 0.0 
        except Exception:
            return 0.0

    def _segment_circle_intersection(self,
                                        x1: sp.Expr, y1: sp.Expr,
                                        x2: sp.Expr, y2: sp.Expr,
                                        cx: sp.Expr, cy: sp.Expr,
                                        r: sp.Expr) -> List[Tuple[sp.Expr, sp.Expr]]:
        """
        计算线段与圆的交点
        优化策略：
        1. 构建符号表达式但不立即 simplify。
        2. 使用 evalf() 算出数值进行判别式 D 和参数 t 的快速筛选。
        3. 筛选通过后，才计算并返回精确的符号坐标。
        """
        intersections = []
        
        # 构建原始表达式树，不立即化简
        dx = x2 - x1
        dy = y2 - y1

        # 1. 计算一元二次方程系数（ax² + bx + c = 0）
        a = dx**2 + dy**2
        b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        c = (x1 - cx)**2 + (y1 - cy)**2 - r**2

        # 2. 线段退化处理：数值判断 a 是否接近 0
        a_val = self._eval_value(a)
        if abs(a_val) < self.EPS:
            # 如果线段是一个点，检查这个点是否在圆上
            c_val = self._eval_value(c)
            if abs(c_val) < self.EPS:
                intersections.append((sp.simplify(x1), sp.simplify(y1)))
            return intersections

        # 3. 判别式分析
        D = b**2 - 4 * a * c
        D_val = self._eval_value(D)

        # 3.1 无实数解（数值判断）
        if D_val < -self.EPS:
            return intersections

        # 4. 计算 t 的候选值（符号表达式）
        t_candidates = []
        if abs(D_val) < self.EPS:  # 切点
            t = -b / (2 * a)
            t_candidates.append(t)
        else:  # 两个解
            sqrt_D = sp.sqrt(D)
            t1 = (-b - sqrt_D) / (2 * a)
            t2 = (-b + sqrt_D) / (2 * a)
            t_candidates = [t1, t2]

        # 5. 校验 t 值有效性
        # 策略：先用数值快速判断是否在 [0, 1] 区间，通过后再保留符号解
        valid_ts = []
        for t in t_candidates:
            t_val = self._eval_value(t)
            # 使用 EPS 处理边界 (0-eps <= t <= 1+eps)
            if -self.EPS <= t_val <= 1.0 + self.EPS:
                valid_ts.append(t)

        if not valid_ts:
            return intersections

        # 6. 计算交点 + 二次校验
        seen = set()
        for t in valid_ts:
            # 计算精确的符号坐标
            x = x1 + t * dx
            y = y1 + t * dy
            
            # 数值二次校验：确保点确实在圆上 (dist^2 - r^2 ≈ 0)
            # 这一步是为了防止极少数情况下的判别式误差
            dist_sq = (x - cx)**2 + (y - cy)**2
            check_val = self._eval_value(dist_sq - r**2)
            
            if abs(check_val) < 1e-5: # 稍微放宽二次校验的容差
                # 仅在最终输出前进行化简
                final_x = sp.simplify(x)
                final_y = sp.simplify(y)
                
                key = (str(final_x), str(final_y))
                if key not in seen:
                    seen.add(key)
                    intersections.append((final_x, final_y))

        return intersections

    def _line_arc_intersection(self, line_id: str, arc_id: str) -> List[Tuple[sp.Expr, sp.Expr, int]]:
        """
        计算线段与完整圆（所有弧均为2π完整圆）的交点，返回(坐标x, 坐标y, 交点level)。
        适配arc实际结构，移除角度范围过滤，保留核心校验与符号计算兼容性。
        """
        try:
            # 1. 获取线段和完整圆的基本信息
            arc = self.arc_id_map[arc_id]
            line = self.line_id_map[line_id]

            # 2. 强化关键参数校验（适配arc实际结构，确保核心参数有效）
            required_arc_params = ["center_point_id", "radius"]  # 对应arc的实际键名
            if not all(param in arc for param in required_arc_params):
                missing = [p for p in required_arc_params if p not in arc]
                logger.warning(f"完整圆 {arc_id} 缺少必要参数: {missing}")
                return []
            
            # 校验radius字典是否包含expr键
            if "expr" not in arc["radius"]:
                logger.warning(f"完整圆 {arc_id} 的radius参数缺少expr字段")
                return []
            
            # 获取圆心ID并校验存在性（对应arc的center_point_id键）
            center_id = arc["center_point_id"]
            if center_id not in self.point_id_map:
                logger.warning(f"完整圆 {arc_id} 的圆心 {center_id} 不存在于点集合中")
                return []
            
            # 3. 计算交点层级（取线段和圆的层级最大值）
            intersection_level = max(line["level"], arc["level"]) + 1

            # 4. 获取并化简所有必要的坐标与表达式（适配arc结构读取半径）
            cx, cy = self.get_point_coords(center_id)
            # 从arc["radius"]["expr"]读取半径表达式，适配实际结构
            radius_expr = arc["radius"]["expr"]
            # 这里 simplify 一次是必要的，以确保半径表达式干净
            radius = sp.simplify(self._parse_expr(radius_expr))
            
            s_id, e_id = line["start_point_id"], line["end_point_id"]
            # 坐标也 simplify 一次，确保后续计算的基础是干净的
            x1 = sp.simplify(self.get_point_coords(s_id)[0])
            y1 = sp.simplify(self.get_point_coords(s_id)[1])
            x2 = sp.simplify(self.get_point_coords(e_id)[0])
            y2 = sp.simplify(self.get_point_coords(e_id)[1])

            # 5. 调用优化后的_segment_circle_intersection计算交点
            # 该函数已完成：线段范围校验、交点去重、圆上二次校验，直接复用
            circle_intersections = self._segment_circle_intersection(
                x1, y1, x2, y2,
                cx, cy, radius
            )
            if not circle_intersections:
                return []

            # 6. 过滤线段端点（不视为新交点，修复符号判断逻辑）
            valid_intersections = []
            
            # 预先计算端点数值，加速判断
            s_x_val = self._eval_value(x1)
            s_y_val = self._eval_value(y1)
            e_x_val = self._eval_value(x2)
            e_y_val = self._eval_value(y2)
            
            for (ix, iy) in circle_intersections:
                # 策略：先用数值快速排斥，如果数值很接近，再用符号判断
                ix_val = self._eval_value(ix)
                iy_val = self._eval_value(iy)
                
                is_start_close = (abs(ix_val - s_x_val) < self.EPS and abs(iy_val - s_y_val) < self.EPS)
                is_end_close = (abs(ix_val - e_x_val) < self.EPS and abs(iy_val - e_y_val) < self.EPS)
                
                if is_start_close or is_end_close:
                    # 如果数值接近，再进行精确的符号检查
                    # 使用 sp.simplify(a - b) == 0 来判断
                    is_s_endpoint = sp.simplify(ix - x1) == 0 and sp.simplify(iy - y1) == 0
                    is_e_endpoint = sp.simplify(ix - x2) == 0 and sp.simplify(iy - y2) == 0
                    
                    if is_s_endpoint or is_e_endpoint:
                        continue # 跳过端点

                valid_intersections.append((ix, iy, intersection_level))

            return valid_intersections

        except KeyError as e:
            logger.warning(f"计算交点时缺少对象: {e}。线段 ID: {line_id}, 完整圆 ID: {arc_id}")
            return []
        except Exception as e:
            logger.error(f"计算线段 {line_id} 与完整圆 {arc_id} 的交点时发生未知错误: {e}", exc_info=True)
            return []

    def detect_new_intersections(self, new_line_id: str) -> List[str]:
        """
        检测一条新线段与场景中所有已有线段和圆弧的交点，并返回新创建的交点 ID 列表。
        """
        new_ips = []
        try:
            # 1. 预计算新线段的所有信息
            new_line = self.line_id_map[new_line_id]
            s_new_id, e_new_id = new_line["start_point_id"], new_line["end_point_id"]
            x1_new, y1_new = self.get_point_coords(s_new_id)
            x2_new, y2_new = self.get_point_coords(e_new_id)
            
            # 计算新线段的轴对齐边界框 (AABB) - 使用数值卫兵优化
            x1_val = self._eval_value(x1_new)
            x2_val = self._eval_value(x2_new)
            y1_val = self._eval_value(y1_new)
            y2_val = self._eval_value(y2_new)

            new_min_x_val = min(x1_val, x2_val)
            new_max_x_val = max(x1_val, x2_val)
            new_min_y_val = min(y1_val, y2_val)
            new_max_y_val = max(y1_val, y2_val)

            # 2. 检测与现有线段的交点
            for line_id, line in self.line_id_map.items():
                if line_id == new_line_id:
                    continue

                # 计算现有线段的 AABB 数值
                s_id, e_id = line["start_point_id"], line["end_point_id"]
                x1, y1 = self.get_point_coords(s_id)
                x2, y2 = self.get_point_coords(e_id)
                
                x1_line_val = self._eval_value(x1)
                x2_line_val = self._eval_value(x2)
                y1_line_val = self._eval_value(y1)
                y2_line_val = self._eval_value(y2)

                min_x_val = min(x1_line_val, x2_line_val)
                max_x_val = max(x1_line_val, x2_line_val)
                min_y_val = min(y1_line_val, y2_line_val)
                max_y_val = max(y1_line_val, y2_line_val)

                # 【核心优化】快速排斥实验：使用数值 AABB 进行判断
                # 使用 EPS 放宽边界，避免浮点误差导致的漏判
                if (max_x_val < new_min_x_val - self.EPS or
                    min_x_val > new_max_x_val + self.EPS or
                    max_y_val < new_min_y_val - self.EPS or
                    min_y_val > new_max_y_val + self.EPS):
                    continue

                # AABB 重叠，需要进行精确的线段相交计算
                intersections = self._line_intersection(new_line_id, line_id)
                for x, y, level in intersections:
                    pid = self.add_new_point(
                        sp.simplify(x), sp.simplify(y), 
                        prefix="I", 
                        point_type="intersection",
                        level=level
                    )
                    if pid:  # 确保 add_new_point 成功创建了点
                        new_ips.append(pid)

            # 3. 检测与现有圆弧的交点
            for arc_id, arc in self.arc_id_map.items():
                # 检查圆弧是否有效
                if "center_point_id" not in arc or "radius" not in arc:
                    continue

                # AABB 重叠，需要进行精确的线-弧相交计算
                intersections = self._line_arc_intersection(new_line_id, arc_id)
                for x, y, level in intersections:
                    pid = self.add_new_point(
                        sp.simplify(x), sp.simplify(y), 
                        prefix="I", 
                        point_type="arc_intersection",
                        level=level
                    )
                    if pid:
                        new_ips.append(pid)

        except Exception as e:
            logger.error(f"在检测新线段 {new_line_id} 的交点时发生错误: {e}", exc_info=True)

        return new_ips

    def _line_intersection(self, line1_id: str, line2_id: str) -> List[Tuple[sp.Expr, sp.Expr, int]]:
        """计算两线段交点，返回(坐标x, 坐标y, 交点level)
        优化策略：
        1. 移除 solve，改用参数方程解析解。
        2. 使用数值卫兵快速判断平行和参数范围。
        """
        s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s1)
        x2, y2 = self.get_point_coords(e1)
        line1_level = self.line_id_map[line1_id]["level"]
        
        s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
        x3, y3 = self.get_point_coords(s2)
        x4, y4 = self.get_point_coords(e2)
        line2_level = self.line_id_map[line2_id]["level"]
        
        # 交点level为两者最大值
        intersection_level = max(line1_level, line2_level) + 1
        
        # 1. 快速端点重合检查（数值优化）
        # 先用数值检查，如果接近再用符号
        p1_val = (self._eval_value(x1), self._eval_value(y1))
        p2_val = (self._eval_value(x2), self._eval_value(y2))
        p3_val = (self._eval_value(x3), self._eval_value(y3))
        p4_val = (self._eval_value(x4), self._eval_value(y4))

        def points_close(v1, v2):
            return abs(v1[0] - v2[0]) < self.EPS and abs(v1[1] - v2[1]) < self.EPS

        if (points_close(p1_val, p3_val) or points_close(p1_val, p4_val) or
            points_close(p2_val, p3_val) or points_close(p2_val, p4_val)):
            return []

        # 2. 计算向量和叉积
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        cross_product = dx1 * dy2 - dx2 * dy1
        cp_val = self._eval_value(cross_product)
        
        # 数值判断平行
        if abs(cp_val) < self.EPS:
            # logger.debug(f"线段{line1_id}与{line2_id}平行或重合，无交点")
            return []
        
        # 3. 计算参数 t 和 s
        delta_x = x3 - x1
        delta_y = y3 - y1
        
        t_num = delta_x * dy2 - delta_y * dx2
        s_num = delta_x * dy1 - delta_y * dx1
        
        t_sym = t_num / cross_product
        s_sym = s_num / cross_product
        
        # 4. 数值卫兵：检查 t, s 是否在有效区间内 [0, 1]
        t_val = self._eval_value(t_sym)
        s_val = self._eval_value(s_sym)
        
        # 使用 EPS 放宽边界判断，包含端点
        t_valid = self.EPS < t_val < 1.0 - self.EPS
        s_valid = self.EPS < s_val < 1.0 - self.EPS
        
        if t_valid and s_valid:
            # 5. 计算并化简交点
            # 使用符号 t 值计算
            x = x1 + t_sym * dx1
            y = y1 + t_sym * dy1
            # 最后再 simplify
            return [(sp.simplify(x), sp.simplify(y), intersection_level)]
        
        return []

    # def _segment_circle_intersection(self,
    #                                     x1: sp.Expr, y1: sp.Expr,
    #                                     x2: sp.Expr, y2: sp.Expr,
    #                                     cx: sp.Expr, cy: sp.Expr,
    #                                     r: sp.Expr) -> List[Tuple[sp.Expr, sp.Expr]]:
    #     """
    #     计算线段与圆的交点
    #     """
    #     intersections = []
    #     dx = sp.simplify(x2 - x1)
    #     dy = sp.simplify(y2 - y1)

    #     # 1. 计算一元二次方程系数（ax² + bx + c = 0）
    #     a = sp.simplify(dx**2 + dy**2)
    #     b = sp.simplify(2 * (dx * (x1 - cx) + dy * (y1 - cy)))
    #     c = sp.simplify((x1 - cx)**2 + (y1 - cy)**2 - r**2)

    #     # 2. 线段退化处理（符号等式判断）
    #     if sp.simplify(sp.Eq(a, 0)):
    #         if sp.simplify(sp.Eq(c, 0)):
    #             intersections.append((x1, y1))
    #         return intersections

    #     # 3. 判别式分析（无solve，纯化简判断）
    #     D = sp.simplify(b**2 - 4 * a * c)
    #     sqrt_D = sp.sqrt(D)

    #     # 3.1 无实数解（D<0，符号化简判断）
    #     if sp.simplify(D < 0):
    #         return intersections

    #     # 4. 计算t值解析解（线段参数方程：t∈[0,1]）
    #     t_candidates = []
    #     if sp.simplify(sp.Eq(D, 0)):  # 切点：唯一解
    #         t = sp.simplify(-b / (2 * a))
    #         t_candidates.append(t)
    #     else:  # 两个解（求根公式直接推导）
    #         t1 = sp.simplify((-b - sqrt_D) / (2 * a))
    #         t2 = sp.simplify((-b + sqrt_D) / (2 * a))
    #         t_candidates = [t1, t2]

    #     # 5. 校验t值有效性（实数+[0,1]区间，无solve）
    #     def is_valid_t(t: sp.Expr) -> bool:
    #         """校验t是否为实数且在[0,1]区间（纯符号化简判断）"""
    #         # 实数判断：虚部为0（符号场景下避免is_real的None返回）
    #         is_real = sp.simplify(sp.im(t)) == 0
    #         # 区间判断：0≤t≤1（符号不等式化简）
    #         in_range = sp.simplify(sp.And(t >= 0, t <= 1))
    #         # 合并判断（化简后为True/False或恒成立的符号表达式）
    #         return sp.simplify(is_real and in_range)

    #     valid_ts = [t for t in t_candidates if is_valid_t(t)]
    #     if not valid_ts:
    #         return intersections

    #     # 6. 计算交点+二次校验（确保在圆上）
    #     seen = set()
    #     for t in valid_ts:
    #         x = sp.simplify(x1 + t * dx)
    #         y = sp.simplify(y1 + t * dy)
            
    #         # 二次校验：交点到圆心距离=半径（符号等式验证）
    #         dist_sq = sp.simplify((x - cx)**2 + (y - cy)**2)
    #         r_sq = sp.simplify(r**2)
    #         if sp.simplify(sp.Eq(dist_sq, r_sq)):
    #             key = (str(x), str(y))
    #             if key not in seen:
    #                 seen.add(key)
    #                 intersections.append((x, y))

    #     return intersections

    # def _line_arc_intersection(self, line_id: str, arc_id: str) -> List[Tuple[sp.Expr, sp.Expr, int]]:
    #     """
    #     计算线段与完整圆（所有弧均为2π完整圆）的交点，返回(坐标x, 坐标y, 交点level)。
    #     适配arc实际结构，移除角度范围过滤，保留核心校验与符号计算兼容性。
    #     """
    #     try:
    #         # 1. 获取线段和完整圆的基本信息
    #         arc = self.arc_id_map[arc_id]
    #         line = self.line_id_map[line_id]

    #         # 2. 强化关键参数校验（适配arc实际结构，确保核心参数有效）
    #         required_arc_params = ["center_point_id", "radius"]  # 对应arc的实际键名
    #         if not all(param in arc for param in required_arc_params):
    #             missing = [p for p in required_arc_params if p not in arc]
    #             logger.warning(f"完整圆 {arc_id} 缺少必要参数: {missing}")
    #             return []
            
    #         # 校验radius字典是否包含expr键
    #         if "expr" not in arc["radius"]:
    #             logger.warning(f"完整圆 {arc_id} 的radius参数缺少expr字段")
    #             return []
            
    #         # 获取圆心ID并校验存在性（对应arc的center_point_id键）
    #         center_id = arc["center_point_id"]
    #         if center_id not in self.point_id_map:
    #             logger.warning(f"完整圆 {arc_id} 的圆心 {center_id} 不存在于点集合中")
    #             return []
            
    #         # 3. 计算交点层级（取线段和圆的层级最大值）
    #         intersection_level = max(line["level"], arc["level"]) + 1

    #         # 4. 获取并化简所有必要的坐标与表达式（适配arc结构读取半径）
    #         cx, cy = self.get_point_coords(center_id)
    #         # 从arc["radius"]["expr"]读取半径表达式，适配实际结构
    #         radius_expr = arc["radius"]["expr"]
    #         radius = sp.simplify(self._parse_expr(radius_expr))
            
    #         s_id, e_id = line["start_point_id"], line["end_point_id"]
    #         x1 = sp.simplify(self.get_point_coords(s_id)[0])
    #         y1 = sp.simplify(self.get_point_coords(s_id)[1])
    #         x2 = sp.simplify(self.get_point_coords(e_id)[0])
    #         y2 = sp.simplify(self.get_point_coords(e_id)[1])

    #         # 5. 调用优化后的_segment_circle_intersection计算交点
    #         # 该函数已完成：线段范围校验、交点去重、圆上二次校验，直接复用
    #         circle_intersections = self._segment_circle_intersection(
    #             x1, y1, x2, y2,
    #             cx, cy, radius
    #         )
    #         if not circle_intersections:
    #             return []

    #         # 6. 过滤线段端点（不视为新交点，修复符号判断逻辑）
    #         valid_intersections = []
    #         s_point = (x1, y1)
    #         e_point = (x2, y2)
            
    #         for (ix, iy) in circle_intersections:
    #             # 用sp.Eq+化简判断是否为线段端点，避免符号场景下==失效
    #             is_s_endpoint = sp.Eq(sp.simplify(ix), sp.simplify(s_point[0])) and sp.Eq(sp.simplify(iy), sp.simplify(s_point[1]))
    #             is_e_endpoint = sp.Eq(sp.simplify(ix), sp.simplify(e_point[0])) and sp.Eq(sp.simplify(iy), sp.simplify(e_point[1]))
                
    #             if not (sp.simplify(is_s_endpoint) or sp.simplify(is_e_endpoint)):
    #                 valid_intersections.append((ix, iy, intersection_level))

    #         return valid_intersections

    #     except KeyError as e:
    #         logger.warning(f"计算交点时缺少对象: {e}。线段 ID: {line_id}, 完整圆 ID: {arc_id}")
    #         return []
    #     except Exception as e:
    #         logger.error(f"计算线段 {line_id} 与完整圆 {arc_id} 的交点时发生未知错误: {e}", exc_info=True)
    #         return []
    
    def _is_point_on_arc_angle_range(self, px: sp.Expr, py: sp.Expr, cx: sp.Expr, cy: sp.Expr, arc: Dict) -> bool:
        """
        辅助函数：判断一个点是否在圆弧的角度范围内。
        :param px, py: 待判断点的坐标。
        :param cx, cy: 圆弧圆心坐标。
        :param arc: 圆弧对象字典。
        :return: 如果点在角度范围内，返回 True，否则返回 False。
                如果圆弧没有角度信息，则默认返回 True。
        """
        # 如果圆弧没有角度信息，视为整个圆
        if "start_angle" not in arc or "end_angle" not in arc:
            return True

        try:
            # 解析并标准化角度到 [0, 2π)
            start_angle = sp.simplify(self._parse_expr(arc["start_angle"]) % (2 * sp.pi))
            end_angle = sp.simplify(self._parse_expr(arc["end_angle"]) % (2 * sp.pi))

            # 计算点相对于圆心的角度
            dx = px - cx
            dy = py - cy
            point_angle = sp.atan2(dy, dx)  # 返回 [-π, π]
            point_angle = sp.simplify(point_angle % (2 * sp.pi))  # 标准化到 [0, 2π)

            # 判断角度范围
            if sp.simplify(start_angle <= end_angle):
                # 情况1：角度范围不跨 0 度
                in_range = sp.simplify(sp.And(point_angle >= start_angle, point_angle <= end_angle))
            else:
                # 情况2：角度范围跨 0 度 (例如从 350° 到 10°)
                in_range = sp.simplify(sp.Or(point_angle >= start_angle, point_angle <= end_angle))

            # 确保返回的是一个布尔值，而不是 SymPy 表达式
            return bool(in_range)

        except Exception as e:
            logger.warning(f"检查点 {px}, {py} 是否在圆弧 {arc.get('id')} 角度范围内时失败: {e}")
            return False

    def _check_and_add_intersection(self, ix: sp.Expr, iy: sp.Expr, s: str, e: str, cx: sp.Expr, cy: sp.Expr, arc: Dict, level: int, intersections: List):
        """
        辅助函数：检查交点是否有效（在角度范围内且非端点），并添加到列表中。
        """
        # 1. 检查是否在圆弧角度范围内
        if not self._is_point_on_arc_angle_range(ix, iy, cx, cy, arc):
            return

        # 2. 检查是否为线段的端点（避免重复添加）
        is_new_point = True
        s_x, s_y = self.get_point_coords(s)
        e_x, e_y = self.get_point_coords(e)
        if (sp.simplify(ix - s_x) == 0 and sp.simplify(iy - s_y) == 0) or \
        (sp.simplify(ix - e_x) == 0 and sp.simplify(iy - e_y) == 0):
            is_new_point = False

        if is_new_point:
            intersections.append((ix, iy, level))

    # def detect_new_intersections(self, new_line_id: str) -> List[str]:
    #     """
    #     检测一条新线段与场景中所有已有线段和圆弧的交点，并返回新创建的交点 ID 列表。
    #     """
    #     new_ips = []
    #     try:
    #         # 1. 预计算新线段的所有信息
    #         new_line = self.line_id_map[new_line_id]
    #         s_new_id, e_new_id = new_line["start_point_id"], new_line["end_point_id"]
    #         x1_new, y1_new = self.get_point_coords(s_new_id)
    #         x2_new, y2_new = self.get_point_coords(e_new_id)
            
    #         # 计算新线段的轴对齐边界框 (AABB)
    #         new_min_x = sp.simplify(sp.Min(x1_new, x2_new))
    #         new_max_x = sp.simplify(sp.Max(x1_new, x2_new))
    #         new_min_y = sp.simplify(sp.Min(y1_new, y2_new))
    #         new_max_y = sp.simplify(sp.Max(y1_new, y2_new))

    #         # 2. 检测与现有线段的交点
    #         for line_id, line in self.line_id_map.items():
    #             if line_id == new_line_id:
    #                 continue

    #             # 计算现有线段的 AABB
    #             s_id, e_id = line["start_point_id"], line["end_point_id"]
    #             x1, y1 = self.get_point_coords(s_id)
    #             x2, y2 = self.get_point_coords(e_id)
    #             min_x = sp.simplify(sp.Min(x1, x2))
    #             max_x = sp.simplify(sp.Max(x1, x2))
    #             min_y = sp.simplify(sp.Min(y1, y2))
    #             max_y = sp.simplify(sp.Max(y1, y2))

    #             # 【核心优化】快速排斥实验：如果两个 AABB 不重叠，则线段一定不相交
    #             # 使用 sympy.simplify 来评估布尔表达式
    #             if (sp.simplify(max_x < new_min_x) or
    #                 sp.simplify(min_x > new_max_x) or
    #                 sp.simplify(max_y < new_min_y) or
    #                 sp.simplify(min_y > new_max_y)):
    #                 continue

    #             # AABB 重叠，需要进行精确的线段相交计算
    #             intersections = self._line_intersection(new_line_id, line_id)
    #             for x, y, level in intersections:
    #                 pid = self.add_new_point(
    #                     sp.simplify(x), sp.simplify(y), 
    #                     prefix="I", 
    #                     point_type="intersection",
    #                     level=level
    #                 )
    #                 if pid:  # 确保 add_new_point 成功创建了点
    #                     new_ips.append(pid)

    #         # 3. 检测与现有圆弧的交点
    #         for arc_id, arc in self.arc_id_map.items():
    #             # 检查圆弧是否有效
    #             if "center_point_id" not in arc or "radius" not in arc:
    #                 continue
                
    #             # 计算圆弧的 AABB
    #             # cx, cy = self.get_point_coords(arc["center_id"])
    #             # r = self._parse_expr(arc["radius_expr"])
    #             # arc_min_x = sp.simplify(cx - r)
    #             # arc_max_x = sp.simplify(cx + r)
    #             # arc_min_y = sp.simplify(cy - r)
    #             # arc_max_y = sp.simplify(cy + r)

    #             # 快速排斥实验：如果线段的 AABB 和圆弧的 AABB 不重叠，则无交点
    #             # if (sp.simplify(arc_max_x < new_min_x) or
    #             #     sp.simplify(arc_min_x > new_max_x) or
    #             #     sp.simplify(arc_max_y < new_min_y) or
    #             #     sp.simplify(arc_min_y > new_max_y)):
    #             #     continue

    #             # AABB 重叠，需要进行精确的线-弧相交计算
    #             intersections = self._line_arc_intersection(new_line_id, arc_id)
    #             for x, y, level in intersections:
    #                 pid = self.add_new_point(
    #                     sp.simplify(x), sp.simplify(y), 
    #                     prefix="I", 
    #                     point_type="arc_intersection",
    #                     level=level
    #                 )
    #                 if pid:
    #                     new_ips.append(pid)

    #     except Exception as e:
    #         logger.error(f"在检测新线段 {new_line_id} 的交点时发生错误: {e}", exc_info=True)

    #     return new_ips

    # def _line_intersection(self, line1_id: str, line2_id: str) -> List[Tuple[sp.Expr, sp.Expr, int]]:
    #     """计算两线段交点，返回(坐标x, 坐标y, 交点level)"""
    #     s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
    #     x1, y1 = self.get_point_coords(s1)
    #     x2, y2 = self.get_point_coords(e1)
    #     line1_level = self.line_id_map[line1_id]["level"]
        
    #     s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
    #     x3, y3 = self.get_point_coords(s2)
    #     x4, y4 = self.get_point_coords(e2)
    #     line2_level = self.line_id_map[line2_id]["level"]
        
    #     # 交点level为两者最大值
    #     intersection_level = max(line1_level, line2_level) + 1
        
    #     def points_equal(p1: Tuple[sp.Expr, sp.Expr], p2: Tuple[sp.Expr, sp.Expr]) -> bool:
    #         return sp.Eq(p1[0], p2[0]) and sp.Eq(p1[1], p2[1])
        
    #     if (points_equal((x1, y1), (x3, y3)) or points_equal((x1, y1), (x4, y4)) or
    #         points_equal((x2, y2), (x3, y3)) or points_equal((x2, y2), (x4, y4))):
    #         return []
        
    #     dx1 = x2 - x1
    #     dy1 = y2 - y1
    #     dx2 = x4 - x3
    #     dy2 = y4 - y3
        
    #     cross_product = sp.simplify(dx1 * dy2 - dx2 * dy1)
    #     if sp.Eq(cross_product, 0):
    #         logger.debug(f"线段{line1_id}与{line2_id}平行或重合，无交点")
    #         return []
        
    #     t, s = symbols('t s')
    #     eq1 = Eq(x1 + t*(x2 - x1), x3 + s*(x4 - x3))
    #     eq2 = Eq(y1 + t*(y2 - y1), y3 + s*(y4 - y3))
        
    #     solutions = solve((eq1, eq2), (t, s), dict=True)
    #     intersections = []
    #     for sol in solutions:
    #         if t not in sol or s not in sol:
    #             continue
            
    #         t_val = sol[t]
    #         s_val = sol[s]
            
    #         t_valid = Ge(t_val, 0) and Le(t_val, 1)
    #         s_valid = Ge(s_val, 0) and Le(s_val, 1)
            
    #         if t_valid and s_valid:
    #             x = simplify(x1 + t_val*(x2 - x1))
    #             y = simplify(y1 + t_val*(y2 - y1))
    #             intersections.append((x, y, intersection_level))
        
    #     return intersections

    def _calculate_foot_of_perpendicular(self, point_id: str, line_id: str) -> Tuple[sp.Expr, sp.Expr, bool, int]:
        """计算垂足，返回(坐标x, 坐标y, 是否在线段上, 垂足level)"""
        px, py = self.get_point_coords(point_id)
        point_level = self.point_id_map[point_id]["level"]
        
        s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s)
        x2, y2 = self.get_point_coords(e)
        line_level = self.line_id_map[line_id]["level"]
        
        # 垂足level = 点level , 线level max + 1
        foot_level = max(point_level, line_level) + 1
        
        dx = x2 - x1
        dy = y2 - y1
        px_vec = px - x1
        py_vec = py - y1
        
        if sp.Eq(dx**2 + dy**2, 0):
            return (x1, y1, True, foot_level)
        
        t = (px_vec * dx + py_vec * dy) / (dx**2 + dy**2)
        foot_x = x1 + t * dx
        foot_y = y1 + t * dy
        on_segment = Ge(t, 0) and Le(t, 1)
        return (foot_x, foot_y, bool(on_segment), foot_level)

    def _finalize_geometry(self, final_geo: Dict):
        """
        对增强后的几何图形进行最终整理（所有计算基于SymPy）。
        生成所有由相邻点组成的最小线段和弧，并正确打上标记。
        - 如果最小单元已存在（如原始线段），则仅将其 `is_minimal` 设为 True。
        - 如果不存在，则创建新元素，并标记为 `is_original=False`, `is_minimal=True`。
        """
        if not final_geo:
            return

        # 1. 准备工作：建立ID到对象的映射，方便快速查找
        point_map = {p["id"]: p for p in final_geo["points"]}
        
        # 建立现有线段和弧的快速查找映射，键为其特征（如端点）
        existing_lines_map = {}
        for line in final_geo.get("lines", []):
            key = tuple(sorted((line["start_point_id"], line["end_point_id"])))
            existing_lines_map[key] = line

        existing_arcs_map = {}
        for arc in final_geo.get("arcs", []):
            # 弧的特征由起点、终点和圆心共同决定
            key = (arc["start_point_id"], arc["end_point_id"], arc["center_point_id"])
            existing_arcs_map[key] = arc

        new_lines = []
        new_arcs = []

        # 确保所有点都有level，如果没有则默认为0
        for p in final_geo["points"]:
            if "level" not in p:
                p["level"] = 0

        # --- 2. 处理线段，生成最小线段 ---
        for line in final_geo.get("lines", []):
            s_id = line["start_point_id"]
            e_id = line["end_point_id"]

            s_p = point_map[s_id]
            e_p = point_map[e_id]
            sx, sy = sp.sympify(s_p["x"]["expr"]), sp.sympify(s_p["y"]["expr"])
            ex, ey = sp.sympify(e_p["x"]["expr"]), sp.sympify(e_p["y"]["expr"])

            points_on_line = []
            for p_id, p in point_map.items():
                px, py = sp.sympify(p["x"]["expr"]), sp.sympify(p["y"]["expr"])
                if self._is_point_on_segment_for_finalize(px, py, sx, sy, ex, ey):
                    if not sp.Eq(sx, ex):
                        t = (px - sx) / (ex - sx)
                    elif not sp.Eq(sy, ey):
                        t = (py - sy) / (ey - sy)
                    else:
                        t = 0
                    points_on_line.append((t, p_id))

            points_on_line.sort(key=lambda x: x[0])
            sorted_point_ids = []
            seen_t = set()
            for t, p_id in points_on_line:
                t_simplified = str(sp.simplify(t))
                if t_simplified not in seen_t:
                    seen_t.add(t_simplified)
                    sorted_point_ids.append(p_id)

            # 生成最小线段
            for i in range(len(sorted_point_ids) - 1):
                start_pid = sorted_point_ids[i]
                end_pid = sorted_point_ids[i + 1]
                level = max(point_map[start_pid]["level"], point_map[end_pid]["level"])
                
                # 生成用于查找的键
                line_key = tuple(sorted((start_pid, end_pid)))

                # 检查这个最小单元是否已存在
                if line_key in existing_lines_map:
                    # 如果存在，直接更新其 is_minimal 属性
                    existing_line = existing_lines_map[line_key]
                    existing_line["is_minimal"] = True
                    # 同时确保 level 是最新的（取最大值）
                    if "level" in existing_line:
                        existing_line["level"] = max(existing_line["level"], level)
                    else:
                        existing_line["level"] = level
                else:
                    # 如果不存在，则创建新的最小线段
                    new_line_id = f"L_min_{len(new_lines) + 1}"
                    new_line = {
                        "id": new_line_id,
                        "start_point_id": start_pid,
                        "end_point_id": end_pid,
                        "level": level,
                        "is_original": False,
                        "is_minimal": True,
                    }
                    new_lines.append(new_line)
                    # 同时更新查找映射，以便后续检查
                    existing_lines_map[line_key] = new_line

        # --- 3. 处理弧，生成最小弧段 ---
        for arc in final_geo.get("arcs", []):
            s_id = arc["start_point_id"]
            e_id = arc["end_point_id"]
            c_id = arc["center_point_id"]
            radius_expr = sp.sympify(arc["radius"]["expr"])
            is_complete = arc.get("is_complete", False)

            c_p = point_map[c_id]
            cx, cy = sp.sympify(c_p["x"]["expr"]), sp.sympify(c_p["y"]["expr"])

            points_on_arc = []
            for p_id, p in point_map.items():
                px, py = sp.sympify(p["x"]["expr"]), sp.sympify(p["y"]["expr"])
                dist_sq = (px - cx)**2 + (py - cy)**2
                if sp.simplify(dist_sq - radius_expr**2) == 0:
                    dx = px - cx
                    dy = py - cy
                    angle = sp.atan2(dy, dx)
                    points_on_arc.append((angle, p_id))

            points_on_arc.sort(key=lambda x: x[0])
            sorted_point_ids = [p_id for _, p_id in points_on_arc]

            if is_complete and sorted_point_ids:
                sorted_point_ids.append(sorted_point_ids[0])

            # 生成最小弧段
            for i in range(len(sorted_point_ids) - 1):
                start_pid = sorted_point_ids[i]
                end_pid = sorted_point_ids[i + 1]
                level = max(point_map[start_pid]["level"], point_map[end_pid]["level"])

                # 计算新弧的角度
                start_dx = sp.sympify(point_map[start_pid]["x"]["expr"]) - cx
                start_dy = sp.sympify(point_map[start_pid]["y"]["expr"]) - cy
                end_dx = sp.sympify(point_map[end_pid]["x"]["expr"]) - cx
                end_dy = sp.sympify(point_map[end_pid]["y"]["expr"]) - cy
                start_angle = sp.atan2(start_dy, start_dx)
                end_angle = sp.atan2(end_dy, end_dx)
                
                if start_angle < 0:
                    start_angle = 2 * sp.pi + start_angle
                if end_angle < 0:
                    end_angle = 2 * sp.pi + end_angle
                
                # sub_angle = start_angle - end_angle
                raw_angle = end_angle - start_angle 
                sub_angle = sp.Mod(raw_angle, 2 * sp.pi) 
                if sp.Eq(sub_angle, 0):
                    sub_angle = 2 * sp.pi

                # if sp.simplify(sub_angle) <= 0:
                #     sub_angle += 2 * sp.pi
                
                # 生成用于查找的键
                arc_key = (start_pid, end_pid, c_id)

                # 检查这个最小单元是否已存在
                if arc_key in existing_arcs_map:
                    # 如果存在，直接更新其 is_minimal 属性
                    existing_arc = existing_arcs_map[arc_key]
                    existing_arc["is_minimal"] = True
                    # 同时确保 level 是最新的
                    if "level" in existing_arc:
                        existing_arc["level"] = max(existing_arc["level"], level)
                    else:
                        existing_arc["level"] = level
                else:
                    # 如果不存在，则创建新的最小弧段
                    new_arc_id = f"Arc_min_{len(new_arcs) + 1}"
                    new_arc = {
                        "id": new_arc_id,
                        "start_point_id": start_pid,
                        "end_point_id": end_pid,
                        # "start_angle": {"expr": str(sp.simplify(start_angle))},
                        # "end_angle": {"expr": str(sp.simplify(end_angle))},
                        "center_point_id": c_id,
                        "radius": {"expr": str(radius_expr)},
                        "angle": {"expr": str(sp.simplify(sub_angle))},
                        "is_complete": False,
                        "is_original": False,
                        "is_minimal": True,
                        "level": level,
                    }
                    new_arcs.append(new_arc)
                    # 同时更新查找映射
                    existing_arcs_map[arc_key] = new_arc

            if is_complete and sorted_point_ids:
                sorted_point_ids.pop()

        # --- 4. 整合结果 ---
        final_geo["lines"].extend(new_lines)
        final_geo["arcs"].extend(new_arcs)

        # print(f"Finalization complete. Added {len(new_lines)} minimal lines and {len(new_arcs)} minimal arcs.")

    def _is_point_on_segment_for_finalize(self, px, py, sx, sy, ex, ey) -> bool:
        """
        辅助函数：判断一个点是否在线段上
        """
        # 检查点是否与线段共线（符号计算）
        cross = (px - sx) * (ey - sy) - (py - sy) * (ex - sx)
        if sp.simplify(cross) != 0:
            return False

        # 检查点是否在线段的 bounding rectangle 内（符号计算）
        dot = (px - sx) * (ex - sx) + (py - sy) * (ey - sy)
        if sp.simplify(dot) < 0:
            return False

        squared_length = (ex - sx)**2 + (ey - sy)** 2
        if sp.simplify(dot) > squared_length:
            return False

        return True
    
    def _initialize_geometry_caches(self):
        """
        在每次增强任务开始前，初始化关键的几何关系缓存，以加速后续操作。
        """
        # 1. 初始化 connected_pairs_set：存储所有已连接的点对 (无序)
        self.connected_pairs_set = set()
        for line_id, line in self.line_id_map.items():
            p1 = line["start_point_id"]
            p2 = line["end_point_id"]
            # 使用排序后的元组确保 (A,B) 和 (B,A) 是同一个键
            pair = tuple(sorted((p1, p2)))
            self.connected_pairs_set.add(pair)

        # 2. 初始化 point_to_lines_map：存储每个点关联的所有线段ID
        self.point_to_lines_map = defaultdict(list)
        for line_id, line in self.line_id_map.items():
            p1 = line["start_point_id"]
            p2 = line["end_point_id"]
            self.point_to_lines_map[p1].append(line_id)
            self.point_to_lines_map[p2].append(line_id)

        # 3. 初始化 on_segment_points_set：存储所有在线段上的点的ID
        # 一个点只要在 point_to_lines_map 中有记录，就说明它至少在一条线上
        self.on_segment_points_set = set(self.point_to_lines_map.keys())
        
        logger.debug(f"几何缓存初始化完成。已连接点对: {len(self.connected_pairs_set)}, 线上点: {len(self.on_segment_points_set)}")

    # ------------------------------ 操作执行（含level计算） ------------------------------
    def execute_operation(self, op: Dict) -> Dict:
        op_type = op["type"]
        result = {"operation": op_type, "details": op, "new_elements": [], "new_line": "", "description": ""}
        
        if op_type == "connect_points":
            p1_id, p2_id = op["point_ids"]
            p1_level = self.point_id_map[p1_id]["level"]
            p2_level = self.point_id_map[p2_id]["level"]
            line_level = max(p1_level, p2_level) + 1
            
            desc = f"Connect points {p1_id} and {p2_id} (both on segments)"
            line_id = self.add_new_line(
                p1_id, p2_id, 
                prefix="ConnP", 
                line_type="connection", 
                description=desc,
                level=line_level  # 设置新线level
            )
            new_ips = self.detect_new_intersections(line_id)
            result["new_elements"].append(("line", line_id))
            result["new_elements"].extend([("point", pid) for pid in new_ips])
            result["description"] = desc
        
        elif op_type == "connect_midpoints":
            mode = op["mode"]
            
            if mode == "two_midpoints":
                line1_id, line2_id = op["line_ids"]
                line1_level = self.line_id_map[line1_id]["level"]
                line2_level = self.line_id_map[line2_id]["level"]
                
                # 中点level = 对应边的level
                s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
                x1, y1 = self.get_point_coords(s1)
                x2, y2 = self.get_point_coords(e1)
                mid1_x = (x1 + x2) / 2
                mid1_y = (y1 + y2) / 2
                mid1_id = self.add_new_point(
                    mid1_x, mid1_y, 
                    prefix="M", 
                    point_type="midpoint", 
                    related_edge=line1_id,
                    level=line1_level  # 中点level=边level
                )
                
                s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
                x3, y3 = self.get_point_coords(s2)
                x4, y4 = self.get_point_coords(e2)
                mid2_x = (x3 + x4) / 2
                mid2_y = (y3 + y4) / 2
                mid2_id = self.add_new_point(
                    mid2_x, mid2_y, 
                    prefix="M", 
                    point_type="midpoint", 
                    related_edge=line2_id,
                    level=line2_level  # 中点level=边level
                )
                
                # 新线level = 两中点level最大值 + 1
                line_level = max(self.point_id_map[mid1_id]["level"], self.point_id_map[mid2_id]["level"]) + 1
                desc = f"Connect midpoints {mid1_id} (of line {line1_id}) and {mid2_id} (of line {line2_id})"
                line_id = self.add_new_line(
                    mid1_id, mid2_id, 
                    prefix="MidL", 
                    line_type="midline", 
                    description=desc,
                    level=line_level  # 设置新线level
                )
                new_ips = self.detect_new_intersections(line_id)
                result["new_elements"].extend([("point", mid1_id), ("point", mid2_id)])
                result["new_elements"].append(("line", line_id))
                result["new_elements"].extend([("point", pid) for pid in new_ips])
                result["description"] = desc
            
            else:  # mode == "vertex_and_midpoint"
                line_id = op["line_id"]
                vertex_id = op["vertex_id"]
                line_level = self.line_id_map[line_id]["level"]
                vertex_level = self.point_id_map[vertex_id]["level"]
                
                # 中点level = 边的level
                s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
                x1, y1 = self.get_point_coords(s)
                x2, y2 = self.get_point_coords(e)
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                mid_id = self.add_new_point(
                    mid_x, mid_y, 
                    prefix="M", 
                    point_type="midpoint", 
                    related_edge=line_id,
                    level=line_level  # 中点level=边level
                )
                
                # 新线level = 顶点level 和 中点level 最大值 + 1
                line_level = max(vertex_level, self.point_id_map[mid_id]["level"]) + 1
                desc = f"Connect vertex {vertex_id} with midpoint {mid_id} (of line {line_id})"
                line_id = self.add_new_line(
                    vertex_id, mid_id, 
                    prefix="VMidL", 
                    line_type="vertex_midline", 
                    description=desc,
                    level=line_level  # 设置新线level
                )
                new_ips = self.detect_new_intersections(line_id)
                result["new_elements"].append(("point", mid_id))
                result["new_elements"].append(("line", line_id))
                result["new_elements"].extend([("point", pid) for pid in new_ips])
                result["description"] = desc
        
        elif op_type == "draw_perpendicular":
            target_line_id = op["line_id"]
            start_point_id = op["point_id"]
            allow_on_segment = op.get("allow_on_segment", False)
            
            # 计算垂足及level（点level + 底边level）
            foot_x, foot_y, on_segment, foot_level = self._calculate_foot_of_perpendicular(start_point_id, target_line_id)
            target_line_level = self.line_id_map[target_line_id]["level"]
            start_point_level = self.point_id_map[start_point_id]["level"]
            
            # 垂足level已在计算方法中确定
            foot_id = self.add_new_point(
                foot_x, foot_y, 
                prefix="F",
                point_type="perpendicular_foot",
                related_vertex=start_point_id,
                related_edge=target_line_id,
                level=foot_level  # 设置垂足level
            )
            
            new_elements = [("point", foot_id)]
            # 垂线level = 点level + 底边level
            perp_line_level = max(start_point_level, target_line_level) + 1
            
            if not on_segment:
                s, e = self.line_id_map[target_line_id]["start_point_id"], self.line_id_map[target_line_id]["end_point_id"]
                s_dist = (foot_x - self.get_point_coords(s)[0])**2 + (foot_y - self.get_point_coords(s)[1])** 2
                e_dist = (foot_x - self.get_point_coords(e)[0])**2 + (foot_y - self.get_point_coords(e)[1])** 2
                extend_from_id = s if s_dist < e_dist else e
                
                # 延长线level = 底边level + 垂足level
                extend_line_level = max(target_line_level, foot_level) + 1
                extend_desc = f"Extend line {target_line_id} to perpendicular foot {foot_id}"
                extend_line_id = self.add_new_line(
                    extend_from_id, foot_id,
                    prefix="ExtL",
                    line_type="extension",
                    description=extend_desc,
                    level=extend_line_level  # 设置延长线level
                )
                new_elements.append(("line", extend_line_id))
                desc = f"Draw perpendicular from point {start_point_id} to line {target_line_id} (foot {foot_id} on extension)"
            else:
                desc = f"Draw perpendicular from point {start_point_id} to line {target_line_id} (foot {foot_id} on segment)"
            
            perp_line_id = self.add_new_line(
                start_point_id, foot_id, 
                prefix="PerpL", 
                line_type="perpendicular",
                description=desc,
                level=perp_line_level  # 设置垂线level
            )
            new_elements.append(("line", perp_line_id))
            
            new_ips = self.detect_new_intersections(perp_line_id)
            if not on_segment:
                new_ips.extend(self.detect_new_intersections(extend_line_id))
            new_elements.extend([("point", pid) for pid in new_ips])
            
            result["new_elements"] = new_elements
            result["description"] = desc
        
        elif op_type == "draw_diameter":
            circle_id = op["circle_id"]
            center_id = op["center_id"]
            radius_expr = op["radius_expr"]
            direction = op["direction"]
            center_x, center_y = self.get_point_coords(center_id)
            radius = self._parse_expr(radius_expr)
            circle_level = self.arc_id_map[circle_id]["level"]
            center_level = self.point_id_map[center_id]["level"]
            
            # 直径端点level = 圆心level 和 圆level 最大值 + 1
            end_level = max(center_level, circle_level) + 1
            
            if direction == "horizontal":
                end1_x = center_x + radius
                end1_y = center_y
                end2_x = center_x - radius
                end2_y = center_y
            else:  # vertical
                end1_x = center_x
                end1_y = center_y + radius
                end2_x = center_x
                end2_y = center_y - radius
            
            end1_id = self.add_new_point(
                end1_x, end1_y, 
                prefix="Diam", 
                point_type="diameter_end",
                level=end_level  # 端点level
            )
            end2_id = self.add_new_point(
                end2_x, end2_y, 
                prefix="Diam", 
                point_type="diameter_end",
                level=end_level  # 端点level
            )
            
            # 直径线level = 两端点level最大值 + 1
            diam_line_level = max(self.point_id_map[end1_id]["level"], self.point_id_map[end2_id]["level"]) + 1
            desc = f"Draw {direction} diameter of circle {circle_id} with endpoints {end1_id} and {end2_id}"
            diam_line_id = self.add_new_line(
                end1_id, end2_id, 
                prefix="DiamL", 
                line_type="diameter",
                description=desc,
                level=diam_line_level  # 直径线level
            )
            
            new_ips = self.detect_new_intersections(diam_line_id)
            result["new_elements"].extend([("point", end1_id), ("point", end2_id)])
            result["new_elements"].append(("line", diam_line_id))
            result["new_elements"].extend([("point", pid) for pid in new_ips])
            result["description"] = desc
        
        return result
    
    # ------------------------------ 增强图形生成 ------------------------------
    def generate_enhancements(self, config: Dict) -> List[Dict]:
        if "rounds_distribution" not in config:
            raise ValueError("config 必须包含 rounds_distribution 配置")
        
        rounds_dist = {int(k): int(v) for k, v in config["rounds_distribution"].items()}
        num_enhancements = sum(rounds_dist.values())
        
        run_config = {
            **config,
            "num_enhancements": num_enhancements,
            "single_enhance_timeout": config.get("single_enhance_timeout", 256)
        }
        
        all_results = self.run_rounds(run_config)
        
        enhancements = []
        for result in all_results:
            if not isinstance(result, dict) or "final_geometry" not in result:
                logger.warning(f"跳过无效增强结果: {result}")
                continue
            
            final_geo = result["final_geometry"]
            final_geo["is_base"] = (result["execution_summary"] == [])
            final_geo["timeout_occurred"] = result.get("timeout_occurred", False)
            final_geo["completed_rounds"] = len(result.get("execution_summary", []))
            final_geo["base_id"] = self.base_id
            final_geo["enhance_id"] = f"{self.base_id}_enhance_{result['enhance_idx']:03d}"
            
            
            for line in final_geo.get("lines", []):
                # 使用 setdefault 避免覆盖已存在的键
                line.setdefault("is_original", True)
                line.setdefault("is_minimal", False)
            
            # 标记所有在分割前就存在的弧
            for arc in final_geo.get("arcs", []):
                arc.setdefault("is_original", True)
                arc.setdefault("is_minimal", False)
                
            self._finalize_geometry(final_geo)
            
            enhancements.append(final_geo)
        
        logger.info(f"生成 {len(enhancements)} 个独立增强图形（预期 {num_enhancements} 个）")
        return enhancements

    def run_rounds(self, config: Dict) -> List[Dict]:
        rounds_distribution = config["rounds_distribution"]
        min_ops_per_round = config["min_operations_per_round"]
        max_ops_per_round = config["max_operations_per_round"]
        num_enhancements = config["num_enhancements"]
        single_enhance_timeout = config.get("single_enhance_timeout", 256)
        result_collector = config.get("result_collector")

        try:
            rounds_list = []
            for round_key, count in rounds_distribution.items():
                round_num = int(round_key)
                rounds_list.extend([round_num] * count)
        except ValueError as e:
            raise ValueError(f"轮数分布键必须为整数（如'0','1'）：{str(e)}") from e

        if len(rounds_list) != num_enhancements:
            raise ValueError(f"轮数分布总和（{len(rounds_list)}）与 num_enhancements（{num_enhancements}）不匹配")

        all_enhance_results = []
        original_data = json.loads(json.dumps(self.data))
        original_desc = original_data.get("description", "无原始描述")
        base_seed = config.get("seed", 42)

        for enh_idx, num_rounds in enumerate(rounds_list):
            enh_seed = base_seed + enh_idx * 100
            random.seed(enh_seed)
            self.data = json.loads(json.dumps(original_data))
            self.points = self.data["points"]
            self.lines = self.data["lines"]
            self.arcs = self.data.get("arcs", [])
            self.point_id_map = {p["id"]: p for p in self.points}
            self.line_id_map = {l["id"]: l for l in self.lines}
            self.arc_id_map = {a["id"]: a for a in self.arcs}
            self._initialize_geometry_caches()

            self.line_pairs = self._build_line_pairs() 
            # self.on_segment_points_cache = set(self._cache_on_segment_points()) 
            self.entity_vertices_cache = self._cache_entity_vertices()
            self.entity_lines_cache = self._cache_entity_lines()
            self.entity_arcs_cache = self._cache_entity_arcs() 
                
            self.enhancement_history = []
            self.operation_counter = 0

            enhancement_desc = [
                f"Original graphic description: {original_desc}",
                f"Enhancement index: {enh_idx + 1}/{num_enhancements}",
                f"Number of rounds: {num_rounds}"
            ]

            timeout_occurred = False
            round_results = []
            start_time = time.time()

            if num_rounds == 0:
                enhancement_desc.append("No enhancement.")
                self.data["description"] = "\n".join(enhancement_desc)
                all_enhance_results.append({
                    "final_geometry": self.data,
                    "execution_summary": round_results,
                    "timeout_occurred": timeout_occurred,
                    "enhance_idx": enh_idx
                })
                continue

            try:
                self._select_feasible_operation_type(config["operation_probs"], config["operation_constraints"])
            except ValueError as e:
                logger.warning(f"增强结果{enh_idx + 1}无可行操作，跳过")
                continue

            for round_idx in range(num_rounds):
                if time.time() - start_time > single_enhance_timeout:
                    timeout_occurred = True
                    logger.warning(f"增强结果{enh_idx + 1}超时，已完成{round_idx}/{num_rounds}轮")
                    break

                logger.info(f"增强结果{enh_idx + 1}：执行第 {round_idx + 1}/{num_rounds} 轮操作")
                num_ops = random.randint(min_ops_per_round, max_ops_per_round)
                round_result = {"round": round_idx, "operations": []}
                round_desc = [f"Round {round_idx + 1}:"]

                for op_idx in range(num_ops):
                    if time.time() - start_time > single_enhance_timeout:
                        timeout_occurred = True
                        break

                    for op_retry in range(self.MAX_RETRIES):
                        try:
                            op_type = self._select_feasible_operation_type(
                                config["operation_probs"], config["operation_constraints"]
                            )
                            op = self._generate_random_operation(op_type, config["operation_constraints"].get(op_type, {}))
                            op_result = self.execute_operation(op)
                            logger.debug(f"操作{op_type}成功，新增元素：{op_result['new_elements']}")

                            round_result["operations"].append(op_result)
                            round_desc.append(f"{op_result['description']}")
                            self.enhancement_history.append(op_result)
                            break
                        except OverflowError as e:
                            logger.warning(f"操作触发数量上限：{str(e)}，终止本轮")
                            timeout_occurred = True
                            break
                        except ValueError as e:
                            logger.debug(f"操作重试{op_retry}次失败：{str(e)}")
                    else:
                        logger.warning(f"操作超过最大重试次数，终止生成")
                        timeout_occurred = True
                        break

                    if timeout_occurred:
                        break

                if timeout_occurred:
                    break

                round_results.append(round_result)
                enhancement_desc.extend(round_desc)
                
                self.entity_vertices_cache = self._cache_entity_vertices()
                self.entity_lines_cache = self._cache_entity_lines()
                self.entity_arcs_cache = self._cache_entity_arcs()
                self.point_id_map = {p["id"]: p for p in self.points}  # 包含本轮新增的点
                self.line_id_map = {l["id"]: l for l in self.lines}    # 包含本轮新增的线
                self.arc_id_map = {a["id"]: a for a in self.arcs}      # 包含本轮新增的圆弧

            self._update_composite_entity()
            
            if timeout_occurred:
                enhancement_desc.append(f"\n超时（{single_enhance_timeout}秒），保留部分操作")
            self.data["description"] = "\n".join(enhancement_desc)
            
            result_entry = {
                "final_geometry": self.data,
                "execution_summary": round_results,
                "timeout_occurred": timeout_occurred,
                "enhance_idx": enh_idx
            }
            
            if result_collector is not None and isinstance(result_collector, list):
                result_collector.append(result_entry)
                
            all_enhance_results.append(result_entry)

        return all_enhance_results

    def _update_composite_entity(self):
        composite_id = "enhanced_composite"
        if composite_id not in self.entity_id_map:
            self.entities.append({
                "id": composite_id,
                "type": "composite"
            })
            self.entity_id_map[composite_id] = self.entities[-1]
        
        composite = self.entity_id_map[composite_id]
        
        all_components = []
        
        if "vertices" in composite:
            for p in self.points:
                if p["id"] not in composite["vertices"]:
                    composite["vertices"].append(p["id"])
                    
        if "lines" in composite:
            for l in self.lines:
                if l["id"] not in composite["lines"]:
                    composite["lines"].append(l["id"])
                    
        if "arcs" in composite:
            for a in self.arcs:
                if a["id"] not in composite["arcs"]:
                    composite["arcs"].append(a["id"])
        
        # composite["components"] = all_components

        self.entity_vertices_cache[composite_id] = [
            c for c in all_components if c in self.point_id_map
        ]

    @property
    def base_json(self) -> Dict:
        return {
            "points": self.points.copy(),
            "lines": self.lines.copy(),
            "arcs": self.arcs.copy(),
            "entities": self.entities.copy(),
            "description": self.data.get("description", "")
        }