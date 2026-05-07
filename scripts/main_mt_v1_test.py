import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import random
import logging
import threading
import queue
import time
import copy
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import cv2
cv2.setNumThreads(0)
import sympy as sp
from sympy import symbols, simplify
import traceback 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('MathGeo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MathGeo')

from tools.template import TemplateGenerator
from tools.builder import RandomGeometryBuilder
from tools.drawer import GeometryDrawer 
from tools.shader import EnhancedDrawer
from tools.gt import GeometryCalculator
from tools.qa_template import QAGenerator


class MathGeoPipeline:
    """图形生成与处理全流程管道（支持多线程+超时限制）"""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._init_workspace()
        self._set_random_seeds()
        
        self.thread_num = self.config['global'].get('thread_num', 4)  # 多线程数量（默认4）
        self.task_timeout = 16 * self.config['global'].get('thread_num', 4)  # 防止卡死，时间感觉和样本数、线程数相关
        
        self.base_jsonl_path: Optional[str] = None
        self.enhanced_jsons: List[Dict[str, Any]] = []
        self.raw_image_paths: List[str] = []
        self.shaded_image_paths: List[str] = []
        self.final_image_paths: List[str] = []
        self.shaded_jsonl_path: Optional[str] = None
        self.gt_jsonl_path: Optional[str] = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载并验证总配置文件（补充线程数默认值）"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 验证核心配置项（包含所有可能模块，即使暂时不启用）
            required_sections = ['template', 'builder', 'drawer', 'shader', 'gt', 'annotator', 'qa']
            for section in required_sections:
                if section not in config:
                    logger.warning(f"配置文件缺少模块: {section}，将使用默认配置")
                    config[section] = {}
            # 补充全局线程数默认值
            if 'thread_num' not in config.get('global', {}):
                config['global']['thread_num'] = 4
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {str(e)}")

    def _init_workspace(self) -> None:
        """初始化工作目录结构（包含所有模块所需目录）"""
        out_root = self.config['global']['output_root']
        subdirs = [
            'json/base',          # 基础图形JSON
            'json/enhanced',      # 增强图形JSON
            'json/final',         # 带GT参数的最终JSON
            'images/raw',         # 原始图像
            'images/shaded',      # 着色图像
            'images/annotated',   # 标注图像
            'qa'                  # 问答数据
        ]
        
        for subdir in subdirs:
            dir_path = os.path.join(out_root, subdir)
            os.makedirs(dir_path, exist_ok=True)
        logger.info(f"工作目录初始化完成: {out_root}")

    def _set_random_seeds(self) -> None:
        main_seed = self.config['global'].get('main_seed', 42)
        random.seed(main_seed)
        logger.info(f"主随机种子: {main_seed}")
        
        self.config['template']['seed'] = main_seed + 1
        self.config['builder']['seed'] = main_seed + 2
        self.config['shader']['seed'] = main_seed + 3
        self.config['qa']['seed'] = main_seed + 4
        logger.info("各模块随机种子初始化完成")

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _run_task_with_timeout(self, task_func, *args, timeout: float = None, **kwargs) -> Tuple[str, Any]:
        """
        执行单个任务，添加超时限制
        返回格式：(状态, 结果) -> 状态：success/timeout/error
        """
        result_queue = queue.Queue(maxsize=1)

        def task_wrapper():
            """任务包装器（捕获异常并返回结果）"""
            try:
                result = task_func(*args, **kwargs)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', e))

        task_thread = threading.Thread(target=task_wrapper, daemon=True)
        task_thread.start()
        task_thread.join(timeout=timeout)

        if task_thread.is_alive():
            return ('timeout', None)
        else:
            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return ('error', RuntimeError("任务无返回结果"))

    def run_template(self) -> None:
        """步骤1：生成基础图形JSONL"""
        
        logger.info(f"=== 开始生成基础图形（多线程：{self.thread_num}个，超时：{self.task_timeout}秒） ===")
        template_cfg = self.config['template']
        seed = template_cfg.get('seed')
        n = template_cfg.get('n', 10)

        self.base_jsonl_path = os.path.join(
            template_cfg.get('output_dir', os.path.join(self.config['global']['output_root'], 'json/base')),
            f'base_{self._get_timestamp()}.jsonl'
        )

        task_queue = queue.Queue()
        for i in range(n):
            sample_seed = seed + 42 * i if seed is not None else None
            task_queue.put((i + 1, sample_seed)) 
            
        write_lock = threading.Lock()
        success_count = 0

        def worker():
            nonlocal success_count
            while not task_queue.empty():
                try:
                    sample_idx, sample_seed = task_queue.get_nowait()
                    logger.info(f"线程[{threading.current_thread().name}] 处理基础图形样本 {sample_idx}/{n}（种子：{sample_seed}）")

                    def generate_sample():
                        generator = TemplateGenerator(template_cfg, seed=sample_seed)
                        generator.generate_base_shape()
                        generator.generate_derivations()
                        return generator.export_json()

                    status, result = self._run_task_with_timeout(generate_sample, timeout=self.task_timeout + 40)

                    if status == 'success':
                        with write_lock:
                            with open(self.base_jsonl_path, "a", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False)
                                f.write("\n")
                        logger.info(f"线程[{threading.current_thread().name}] 样本 {sample_idx} 生成成功")
                        success_count += 1
                    elif status == 'timeout':
                        logger.error(f"线程[{threading.current_thread().name}] 样本 {sample_idx} 生成超时（{self.task_timeout}秒），跳过")
                    else:
                        logger.error(f"线程[{threading.current_thread().name}] 样本 {sample_idx} 生成失败：{str(result)}，跳过")
                        
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"线程[{threading.current_thread().name}] 处理任务异常：{str(e)}，跳过")
                finally:
                    task_queue.task_done()

        threads = []
        for i in range(self.thread_num):
            t = threading.Thread(target=worker, name=f"Template-Worker-{i+1}")
            t.start()
            threads.append(t)

        task_queue.join()
        logger.info(f"基础图形生成完成：成功 {success_count}/{n} 个样本")

        if not os.path.exists(self.base_jsonl_path) or os.path.getsize(self.base_jsonl_path) == 0:
            raise RuntimeError("未生成任何有效的基础图形JSONL文件")
        logger.info(f"基础图形JSONL保存至: {self.base_jsonl_path}")

    def run_builder(self) -> None:
        """步骤2：读取基础图形JSONL，生成增强图形（多线程+单次20秒超时）"""
        # self.base_jsonl_path = ""
        if not self.base_jsonl_path or not os.path.exists(self.base_jsonl_path):
            raise RuntimeError("未找到基础图形JSONL文件，无法执行增强步骤")
        
        logger.info(f"=== 开始生成增强图形（多线程：{self.thread_num}个，超时：{self.task_timeout}秒） ===")
        builder_cfg = self.config['builder']
        

        # 构建增强图形输出路径
        output_filename = f'enhanced_{self._get_timestamp()}.jsonl'
        enhanced_jsonl_path = os.path.join(
            builder_cfg.get('output_dir'),
            output_filename
        )
        # 清空输出文件（避免追加旧数据）
        with open(enhanced_jsonl_path, 'w', encoding='utf-8') as f:
            pass

        # 读取所有基础图形（先加载到内存，避免多线程读取文件冲突）
        base_tasks = []
        with open(self.base_jsonl_path, 'r', encoding='utf-8') as base_f:
            for line_num, line in enumerate(base_f, 1):
                line = line.strip()
                if line:
                    task_seed = builder_cfg['seed'] + line_num * 42  # 每个任务独立种子
                    base_tasks.append((line_num, line, task_seed))

        if not base_tasks:
            raise RuntimeError("基础图形JSONL文件为空，无法执行增强步骤")

        # 任务队列：存储 (行号, 基础图形JSON字符串, 任务种子)
        task_queue = queue.Queue()
        for task in base_tasks:
            task_queue.put(task)

        # 线程安全锁（文件写入+结果收集）
        write_lock = threading.Lock()
        self.enhanced_jsons = []
        success_count = 0
        total_tasks = len(base_tasks)

        def worker():
            """线程工作函数：处理单个基础图形的增强"""
            nonlocal success_count
            while not task_queue.empty():
                try:
                    line_num, line, task_seed = task_queue.get_nowait()
                    logger.info(f"线程[{threading.current_thread().name}] 处理基础图形 #{line_num}")
                    
                    partial_results = []

                    # 定义单个增强任务
                    def enhance_task():
                        base_json = json.loads(line)
                        base_builder = RandomGeometryBuilder(base_json)
                        return base_builder.generate_enhancements({
                            "rounds_distribution": builder_cfg.get('rounds_distribution', {1: 1}),
                            "min_operations_per_round": builder_cfg.get('min_operations_per_round', 1),
                            "max_operations_per_round": builder_cfg.get('max_operations_per_round', 1),
                            "allowed_operation_types": builder_cfg.get('allowed_operation_types', 
                                                                    ['connect_points', 'connect_midpoints', 'draw_perpendicular', 'draw_diameter']),
                            "operation_probs": builder_cfg.get('operation_probs', {}), 
                            "operation_constraints": builder_cfg.get('operation_constraints', {}),
                            "seed": task_seed,
                            "_result_collector": partial_results 
                        })

                    # 执行任务（带超时）
                    status, result = self._run_task_with_timeout(enhance_task, timeout=self.task_timeout)
                    
                    final_results = []

                    if status == 'success':
                        final_results = result
                        logger.info(f"线程[{threading.current_thread().name}] 基础图形 #{line_num} 增强成功（{len(final_results)}个结果）")
                        success_count += 1
                    else:
                        if partial_results:
                            final_results = partial_results
                            logger.warning(f"线程[{threading.current_thread().name}] 基础图形 #{line_num} 发生 {status}，但保留 {len(final_results)} 个有效增强结果")
                        else:
                            logger.error(f"线程[{threading.current_thread().name}] 基础图形 #{line_num} {status} 且无任何产出，跳过。原因：{str(result)}")
                        
                    if final_results:
                        with write_lock:
                            with open(enhanced_jsonl_path, 'a', encoding='utf-8') as enhanced_f:
                                for enh_json in final_results:
                                    json.dump(enh_json, enhanced_f, ensure_ascii=False)
                                    enhanced_f.write("\n")
                                    self.enhanced_jsons.append(enh_json)
                        success_count += 1
                    
                    del partial_results

                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"线程[{threading.current_thread().name}] 处理基础图形 #{line_num} 异常：{str(e)}，跳过")
                    
                finally:
                    task_queue.task_done()

        # 启动线程池
        threads = []
        for i in range(self.thread_num):
            t = threading.Thread(target=worker, name=f"Builder-Worker-{i+1}")
            t.start()
            threads.append(t)

        # 等待所有任务完成
        task_queue.join()
        logger.info(f"增强图形生成完成：成功处理 {success_count}/{total_tasks} 个基础图形")

        if not self.enhanced_jsons:
            raise RuntimeError("未生成任何有效的增强图形")
        
        # 更新drawer配置中的输入路径
        self.config['drawer']['jsonl_path'] = enhanced_jsonl_path
        logger.info(f"增强图形JSONL保存至: {enhanced_jsonl_path}（共{len(self.enhanced_jsons)}个增强图形）")

    def run_drawer(self) -> None:
        """步骤3：绘制原始图像"""
        if not self.enhanced_jsons:
            raise RuntimeError("未生成增强图形，无法执行绘图步骤")
        
        logger.info("=== 开始绘制原始图像 ===")
        drawer_cfg = self.config['drawer']
        # annotator_cfg = self.config['annotator']

        drawer = GeometryDrawer(drawer_cfg)
        self.raw_image_paths = drawer.batch_draw()
        logger.info(f"原始图像保存完成（共 {len(self.raw_image_paths)} 张）")

    def run_shader(self) -> None:
        """步骤4：区域着色与标注"""
        # 1. 提取核心配置参数
        shader_config = self.config['shader']
        annotator_config = self.config.get('annotator', {})
        output_root = self.config['global']['output_root']
        shader_enabled = shader_config.get('enabled', False)
        self.config['drawer']['jsonl_path'] = ""

        json_output_dir = os.path.join(output_root, 'json/shaded')
        self.shaded_jsonl_path = os.path.join(
            json_output_dir, 
            f'shaded_{self._get_timestamp()}.jsonl'
        )

        tool_config = {
            "global": {"output_root": output_root},
            "drawer": self.config['drawer'],  # 绘图配置
            "shader": {**shader_config, 
                "enabled": shader_enabled,
                "jsonl_path": self.shaded_jsonl_path
            },
            "annotator": annotator_config,
            "region_extractor": self.config.get("region_extractor", {})
        }

        enhanced_drawer = EnhancedDrawer(tool_config)
        enhanced_jsonl_path = self.config['drawer'].get('jsonl_path')
        enhanced_drawer.load_enhanced_jsonl(enhanced_jsonl_path)
        if enhanced_jsonl_path and os.path.exists(enhanced_jsonl_path):
            # 如果列表为空，才加载，防止重复添加（视你的需求而定）
            if not self.enhanced_jsons: 
                logger.info(f"从文件加载增强数据: {enhanced_jsonl_path}")
                with open(enhanced_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.enhanced_jsons.append(json.loads(line))
        enhanced_drawer.set_enhanced_data(self.enhanced_jsons)

        try:
            enhanced_drawer.process()
        except Exception as e:
            logger.error(f"着色与标注流程执行失败：{str(e)}", exc_info=True)
            raise

        # self.shaded_image_paths = enhanced_drawer.shaded_path

        logger.info(
            f"=== 处理完成 ==="
            # f"\n- 有效图像数量：{len(self.shaded_image_paths)}"
            f"\n- 结果JSONL路径：{self.shaded_jsonl_path}"
        )
          
    def run_gt(self) -> None:
        """步骤5：计算几何参数（基于符号表达式，结果为化简后的表达式）"""
        # 检查 shader 处理结果是否存在
        # self.shaded_jsonl_path = ""
        if not self.shaded_jsonl_path or not os.path.exists(self.shaded_jsonl_path):
            raise RuntimeError("未找到 shader 处理后的 JSONL 文件，无法执行几何参数计算")
        
        logger.info("=== 开始计算几何参数（符号表达式化简） ===")
        calculator = GeometryCalculator()
        
        # 生成输出路径（在原有路径基础上添加 _with_gt 标识）
        gt_output_dir = os.path.join(self.config['global']['output_root'], 'json/final')
        os.makedirs(gt_output_dir, exist_ok=True)
        
        # 提取原始文件名并添加 GT 标识
        shaded_filename = os.path.basename(self.shaded_jsonl_path)
        gt_filename = shaded_filename.replace('shaded_', 'shaded_with_gt_')
        self.gt_jsonl_path = os.path.join(gt_output_dir, gt_filename)
        
        try:
            # 调用计算器批量处理 JSONL 文件
            calculator.process_jsonl(
                input_path=self.shaded_jsonl_path,
                output_path=self.gt_jsonl_path
            )
            
            # 读取处理后的结果更新到内存（可选，根据后续流程需求）
            self.enhanced_jsons = []
            with open(self.gt_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.enhanced_jsons.append(json.loads(line))
            
            logger.info(f"几何参数计算完成，结果保存至: {self.gt_jsonl_path}")
            logger.info(f"共处理 {len(self.enhanced_jsons)} 条数据，每条数据已添加线条/圆弧长度信息")
            
        except Exception as e:
            logger.error(f"几何参数计算流程失败: {str(e)}", exc_info=True)
            raise

    def run_qa(self) -> None:
        """步骤6：基于GT计算结果生成问答数据（兼容新QA配置：权重控制+新参数名）"""
        # 检查GT处理结果路径是否有效
        # self.gt_jsonl_path = ""
        if not hasattr(self, 'gt_jsonl_path') or not self.gt_jsonl_path or not os.path.exists(self.gt_jsonl_path):
            raise RuntimeError("GT计算结果路径无效或不存在，请先执行run_gt生成结果")
        
        logger.info(f"使用GT结果文件: {self.gt_jsonl_path}")

        # 解析qa配置参数（兼容新配置字段）
        qa_config = self.config['qa']
        output_dir = qa_config.get('output_dir', os.path.join(self.config['global']['output_root'], 'qa'))
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        
        num_questions = qa_config.get('num_questions_per_group', 3)
        question_types = qa_config.get('question_types', ['length', 'angle'])
        type_weights = qa_config.get('type_weights', {})

        # 生成输出文件名（基于GT文件名和时间戳）
        gt_basename = os.path.splitext(os.path.basename(self.gt_jsonl_path))[0]
        timestamp = self._get_timestamp()
        qa_output_path = os.path.join(output_dir, f"qa_{gt_basename}.jsonl")

        try:
            # 初始化问答生成器（传入完整配置，便于内部使用权重）
            qa_generator = QAGenerator(qa_config)
            logger.info(f"=== 开始生成问答数据 ===")
            logger.info(f"QA配置参数: {json.dumps(qa_config, ensure_ascii=False, indent=2)}")

            # 处理GT结果JSONL，逐行生成问答
            processed_count = 0
            total_qa = 0
            with open(self.gt_jsonl_path, 'r', encoding='utf-8') as f_in, \
                open(qa_output_path, 'w', encoding='utf-8') as f_out:

                for line_num, line in enumerate(f_in, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # 解析单条带GT参数的几何数据
                        geo_data = json.loads(line)
                        
                        # 生成问答对（传递新配置参数：权重+新问题数量字段）
                        qa_pairs = qa_generator.generate(
                            geo_data=geo_data,
                            num_questions=num_questions,
                            question_types=question_types,
                            type_weights=type_weights  # 新增：传递权重配置
                        )
                        
                        # 写入输出文件（每条问答对一行）
                        for qa in qa_pairs:
                            json.dump(qa, f_out, ensure_ascii=False)
                            f_out.write('\n')
                        
                        processed_count += 1
                        total_qa += len(qa_pairs)
                        if line_num % 10 == 0:
                            logger.info(f"已处理 {line_num} 条几何数据，累计生成 {total_qa} 个问答对")

                    except json.JSONDecodeError:
                        logger.error(f"GT文件第 {line_num} 行JSON格式错误，跳过")
                        continue
                    except Exception as e:
                        logger.error(f"处理第 {line_num} 条数据时生成问答失败: {str(e)}，跳过")
                        continue

            logger.info(f"=== 问答生成完成 ===")
            logger.info(f"有效处理几何数据: {processed_count} 条")
            logger.info(f"生成问答对总数: {total_qa} 个")
            logger.info(f"问答结果保存至: {qa_output_path}")

        except Exception as e:
            logger.error(f"问答生成流程失败: {str(e)}", exc_info=True)
            raise

    def run(self) -> None:
        """执行全流程"""
        try:
            # self.run_template()      # 生成基础图形（多线程+超时）
            # self.run_builder()       # 增强图形（多线程+超时）
            # self.run_drawer()        # 绘制原始图像
            # self.run_shader()        # 区域阴影与标注
            self.run_gt()            # 计算参数
            # self.run_qa()            # 生成问答
            logger.info("=== 全流程执行完成 ===")
        except Exception as e:
            logger.error(f"流程执行失败: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MathGeo图形生成与处理全流程（支持多线程+超时）")
    parser.add_argument(
        "--config",
        type=str,
        default="./scripts/config_easy.json",
        help="总配置文件路径"
    )
    args = parser.parse_args()

    # 启动全流程
    pipeline = MathGeoPipeline(args.config)
    pipeline.run()