# multi_model 使用说明

## 命令行

```bash
# 切换到multi_model目录
cd data_to_model/multi_model

# Soft Voting（均匀加权）
python late_fusion_gpu.py --fusion soft --soft-strategy uniform

# Soft Voting（按CV-MAE倒数加权）
python late_fusion_gpu.py --fusion soft --soft-strategy inverse_mae

# 原先的基于OOF学习权重
python late_fusion_gpu.py --fusion learned

# Stacking
python late_fusion_gpu.py --fusion stacking

```


## 命令行生成文件说明
- 每次运行命令，都会生成一个late_fusion_gpu_report.txt文件，我将三个文件都放在了multi_model/report下，都改了名。
- 使用时请注意，如果不及时另存为生成的报告，将会在下次运行时被覆盖。