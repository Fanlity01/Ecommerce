

# 电商评论多模态情感分析与可信度评估

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

面向电商评论的**多模态情感分析**与**评论可信度评估**完整工程实现。基于 BERT + ResNet18 + 门控融合完成图文情感三分类，并融合文本、图像、用户行为、时间突发度等多源特征构建可信度二分类器。项目包含模型训练、推理管线、FastAPI 接口及前端演示页面，适合本科毕业设计、科研原型验证与快速部署。

📄 对应论文：《面向电商评论的多模态情感分析与可信度评估研究》（初稿修改版）  
🎯 特点：**论文表述与代码实现完全一致**，可直接复现实验并接入系统展示。

---

## ✨ 主要特点

- **多模态情感分析**：BERT‑base‑uncased 文本编码 + ResNet18 图像编码 + 门控融合模块，完成负面/中性/正面三分类。
- **评论可信度评估**：复用情感模型中间表示，融合评分、时间突发度、用户行为、图文一致性等特征，使用 MLP 进行高低可信判别。
- **工程化流程**：数据整理 → 模型训练 → 推理管线 → FastAPI 接口 → 前端调用，闭环完整。
- **实验可复现**：基于 Amazon Reviews 2023 的 All_Beauty 子集，提供训练日志及最佳准确率指标。
- **即插即用**：支持命令行预测与 HTTP 接口调用，前端提供简洁的 HTML 表单页面。

## 1. 项目结构

```text
multimodal_ecommerce_project/
├─ app.py                         # FastAPI 接口
├─ train_sentiment.py             # 训练多模态情感模型
├─ train_credibility.py           # 训练评论可信度模型
├─ predict.py                     # 本地命令行预测
├─ requirements.txt
├─ README.md
├─ examples/
│  ├─ sentiment_train.csv         # 情感训练数据样例
│  ├─ credibility_train.csv       # 可信度训练数据样例
│  └─ images/                     # 训练 / 测试图片目录（自行放图）
└─ src/
   ├─ config.py
   ├─ data/
   │  ├─ multimodal_dataset.py
   │  └─ credibility_dataset.py
   ├─ features/
   │  └─ credibility_features.py
   ├─ models/
   │  ├─ fusion.py
   │  ├─ sentiment_model.py
   │  └─ credibility_model.py
   ├─ services/
   │  └─ pipeline.py
   └─ utils/
      ├─ io.py
      └─ seed.py
```

## 2. 环境安装

```bash
pip install -r requirements.txt
```

## 3. 数据格式

### 3.1 情感分析训练集 `examples/sentiment_train.csv`

需要至少包含下面字段：

- `review_id`：评论 ID
- `text`：评论文本
- `image_path`：图片相对路径或绝对路径
- `label`：情感标签（0=负面，1=中性，2=正面）

### 3.2 可信度训练集 `examples/credibility_train.csv`

建议包含下面字段：

- `review_id`
- `text`
- `image_path`
- `rating`：评分（1~5）
- `timestamp`：评论时间，例如 `2026-04-08 20:10:00`
- `user_id`
- `user_review_count`
- `user_account_days`
- `helpful_votes`
- `verified_purchase`：0/1
- `label`：可信度标签（0=低可信，1=高可信）

## 4. 训练

### 4.1 训练多模态情感模型

```bash
python train_sentiment.py   --train_csv exampless/sentiment_train.csv   --image_root exampless/images   --save_dir outputs/sentiment
```

### 4.2 训练评论可信度模型

```bash
python train_credibility.py   --train_csv exampless/credibility_train.csv   --image_root exampless/images   --sentiment_ckpt outputs/sentiment/best_sentiment.pt   --save_dir outputs/credibility
```

## 5. 预测

```bash
python predict.py   --text "包装很好，物流很快，实物和图片一致"   --image exampless/images/sample.jpg   --rating 5   --timestamp "2026-04-08 20:10:00"   --user_review_count 18   --user_account_days 420   --helpful_votes 6   --verified_purchase 1   --sentiment_ckpt outputs/sentiment/best_sentiment.pt   --credibility_ckpt outputs/credibility/best_credibility.pt
```

## 6. 启动接口

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 7. 说明

这版代码保留了你论文思路中的两个核心模块：

1. **多模态情感分析**：文本编码器 + 图像编码器 + 门控融合分类器  
2. **可信度挖掘**：基于评论内容、用户行为、时间统计和图文一致性特征进行二分类

为了保证代码更容易跑通，我把原文里“LSTM 时间序列 + GNN 图结构”的研究型设计，先落成了一个**工程上更容易训练和复现的版本**。  
后续你如果要继续升级到：
- 时间序列异常检测（LSTM/Informer）
- 图神经网络可信度评估（GCN/GraphSAGE/GAT）
- 更强的跨模态 Transformer 融合

可以在这个项目上直接继续加。





## 🛠 技术栈

| 模块               | 技术选型                                 |
| ------------------ | ---------------------------------------- |
| 文本编码           | BERT‑base‑uncased (Hugging Face Transformers) |
| 图像编码           | ResNet18 (Torchvision)                   |
| 多模态融合         | 门控融合 (Gated Fusion)                  |
| 可信度分类器       | 多层感知机 (MLP)                         |
| 深度学习框架       | PyTorch 1.13+                            |
| Web 接口           | FastAPI + Uvicorn                        |
| 前端演示           | 原生 HTML + JavaScript                   |

---









## 📊 实验结果

实验基于 **Amazon Reviews 2023 (All_Beauty)** 子集，采用评分弱监督生成情感标签，可信度标签结合规则与用户行为构造。

| 模型         | 样本规模 | 最佳验证准确率 | 说明                         |
| ------------ | -------- | -------------- | ---------------------------- |
| 情感分析模型 | 1,000    | 0.895          | 流程可跑通，基线稳定         |
| 情感分析模型 | 10,000   | **0.900**      | 样本增加后效果提升，随后过拟合 |
| 可信度模型   | 1,000    | 0.750          | 小样本下学习不足             |
| 可信度模型   | 10,000   | **0.994**      | 伪标签设置下区分能力极强     |

> ⚠️ 可信度模型的高准确率与当前伪标签构造方式有关，论文中已做谨慎说明。如需进一步泛化验证，建议引入人工标注测试集。

## 📖 引用

若本项目对您的研究有帮助，请引用对应论文：

```
@article{multimodal_credibility_2026,
  title={面向电商评论的多模态情感分析与可信度评估研究},
  author={Your Name},
  journal={本科毕业设计论文},
  year={2026}
}
```

---

## 📝 许可

本项目采用 [MIT License](LICENSE) 开源，欢迎 Star、Fork 与 PR。

---

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Torchvision](https://github.com/pytorch/vision)
- [FastAPI](https://github.com/tiangolo/fastapi)
- Amazon Reviews 2023 数据集提供方
```
