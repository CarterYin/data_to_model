[//]: # (<br />)
<p align="center"> <h1 align="center">从数据到模型的项目记录</h1>
  <p align="center">
    <b> 包含数据预处理和模型训练的代码 </b>
    <br />
    <a href="https://github.com/your-username"><strong> Carter Yin </strong></a>
  </p>

  <p align="center">
    <a href="https://www.python.org">
      <img src="https://img.shields.io/badge/Python-3.13-blueviolet?style=flat&logo=python" alt="Python 3.13">
    </a>
    <a href="https://pandas.pydata.org/">
      <img src="https://img.shields.io/badge/Pandas-lightgrey?style=flat&logo=pandas" alt="Pandas">
    </a>
    <a href="https://scikit-learn.org/">
      <img src="https://img.shields.io/badge/Scikit--learn-orange?style=flat&logo=scikit-learn" alt="Scikit-learn">
    </a>
<br />

### 项目结构与功能
- **数据预处理代码 (`pre_mod`)**
  - 将数据集中“NO”转换为 **0**，“YES”转换为 **1**。
  - 为 `.tsv` 文件添加表头（header），以确保数据对齐和可读性。

- **模型训练代码 (`train_test`)**
  - 分别在眼部、颈动脉、生理以及全部指标上训练模型。
  - 包含一个批处理文件 `run_all_models.py`，能够激活 `yclearning` conda 环境并运行模型训练代码。

- **多模态训练代码（`multi_model`）**
  - **功能要点**
    - **数据预处理**：分别加载和标准化 `physiology`、`artery` 和 `eye` 三种模态的数据。
    - **模型训练与优化**：
      - 对每种模态独立进行超参数搜索，优化目标为 **MAE**。
      - 可选模型包括：`XGBoost`、`LightGBM`、`CatBoost` 和 `RandomForest`。
      - 当 GPU 可用时，启用 GPU 加速训练。
    - **融合权重学习**：
      - 利用训练集进行 5 折 OOF (Out-of-Fold) 预测。
      - 基于预测结果学习各模态的融合权重，权重需满足**非负**和**归一化**（总和为 1）的条件。
    - **最终预测**：
      - 在整个训练集上拟合各模态的最优模型。
      - 对测试集进行预测，并根据学习到的权重进行加权融合。
    - **性能评估与报告**：
      - 计算并输出 **MAE**、**RMSE**、**R²**、**原始相关**和**校正后相关**等指标。
      - 将所有结果保存至 `late_fusion_gpu_report.txt` 文件。
</p>

