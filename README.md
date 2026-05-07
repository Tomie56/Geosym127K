# ⚙️ GeoSym: Scalable Symbolically-verifiable Synthesis Engine

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/Tomie0506/GeoSym127K)

## 🌟 What is GeoSym?
**GeoSym** is a neuro-symbolic framework designed for scalable and verifiable multimodal geometric reasoning. Unlike traditional pipelines that rely on heuristic templates or unreliable LLM pseudo-labels, GeoSym anchors every visual topology and intermediate logical step to an arbitrary-precision mathematical manifold. 

This repository contains the **Core Synthesis Engine** of GeoSym. It is a highly flexible and automated geometric data generation pipeline that seamlessly integrates:
1. **Dynamic Topological Evolution** via type-conditional grammar.
2. **Visual-First Grounding** with precision-aligned rendering (including complex shaded regions).
3. **Analytic SymGT Solver** to derive absolute ground truths in exact symbolic forms.

Using this engine, we synthesized **GeoSym127K**, a massive, difficulty-stratified ecosystem featuring 127K solver-verified questions. 
👉 **[Access the GeoSym127K Dataset here](https://huggingface.co/datasets/Tomie0506/GeoSym127K)**.

---

## 1. Synthesize Your Own Geometric Data
This engine is highly customizable, allowing you to generate geometric diagrams and mathematically precise question-answer pairs tailored to your specific needs.

You can quickly generate a sample batch of geometric data using our multi-threaded test script:

```bash
python scripts/main_mt_v1_test.py
```

### 2. Customize Generation Parameters
The engine's complexity and generation rules are entirely controlled via the configuration file. You can adjust hyperparameters such as the maximum derivation depth, base shape types, and shading complexities by editing:

```bash
vim scripts/config.json
```

Key Parameters in config.json:
"target_quantity": 10000,
"derivation_rounds": [2, 4],
"max_enhancement_rounds": 5,
"shadow_target_count": [1, 4]
