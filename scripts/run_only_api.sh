#!/bin/bash
set -euo pipefail  # 开启严格模式，命令失败/变量未定义时脚本退出

# 定义工作目录
WORK_DIR="/mnt/afs/jingjinhao/project/GeoChain/MathGeo"

# 检查输入参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <输入jsonl文件路径>"
    echo "示例: $0 ./data/input.jsonl"
    exit 1
fi

INPUT_JSONL="$1"
# 提取输入文件名（不含路径）用于生成中间文件
BASE_NAME="geosym_easy"

# 定义中间文件路径（避免覆盖原始文件）
FORMATTED_JSONL="${WORK_DIR}/${BASE_NAME}/api/${BASE_NAME}_formatted.jsonl"
QUESTION_JSONL="${WORK_DIR}/${BASE_NAME}/api/${BASE_NAME}_question.jsonl"
ANSWER_JSONL="${WORK_DIR}/${BASE_NAME}/api/${BASE_NAME}_answer.jsonl"
FINAL_OUTPUT="${WORK_DIR}/${BASE_NAME}/output/${BASE_NAME}_qa.jsonl"

# 创建临时目录和输出目录
mkdir -p "${WORK_DIR}/${BASE_NAME}/api" "${WORK_DIR}/${BASE_NAME}/output"

echo "===== 开始处理流程 ====="
echo "输入文件: $INPUT_JSONL"
echo "工作目录: $WORK_DIR"

# 步骤1: 执行format.py
echo -e "\n[1/4] 运行format.py格式化数据..."
cd "$WORK_DIR" || { echo "错误：无法进入工作目录 $WORK_DIR"; exit 1; }
python ./scripts/gen_qa/format.py "$INPUT_JSONL" --output "$FORMATTED_JSONL"

# 步骤2: 执行get_question.py生成问题
# echo -e "\n[2/4] 运行get_question.py生成问题..."
# python ./scripts/gen_qa/get_question.py "$FORMATTED_JSONL" --output "$QUESTION_JSONL"

# # 步骤3: 执行get_answer.py生成答案（修正路径，避免重复MathGeo）
# echo -e "\n[3/4] 运行get_answer.py生成答案..."
# python ./scripts/gen_qa/get_answer.py "$QUESTION_JSONL" --output "$ANSWER_JSONL"

# # 步骤4: 执行convert.py转换为最终问答对
# echo -e "\n[4/4] 运行convert.py生成最终问答对..."
# python ./scripts/gen_qa/convert.py "$ANSWER_JSONL" --output "$FINAL_OUTPUT"

echo -e "\n===== 处理完成 ====="
echo "最终问答对已保存至: $FINAL_OUTPUT"
echo "中间文件路径: ${WORK_DIR}/api"