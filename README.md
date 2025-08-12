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

</p>

