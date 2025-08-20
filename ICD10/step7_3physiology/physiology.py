#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选生理学相关字段脚本
从test.tsv和train.tsv文件中筛选指定的生理学相关字段
"""

import pandas as pd
import sys

def select_physiology_fields(input_file, output_file):
    """筛选指定的生理学相关字段"""
    print(f"处理文件: {input_file}")
    
    # 读取原始文件
    df = pd.read_csv(input_file, sep='\t')
    print(f"原始文件行数: {len(df)}")
    print(f"原始文件列数: {len(df.columns)}")
    
    # 定义要保留的字段列表
    fields_to_keep = [
        'samples',  # 保留样本ID
        'random_glucose_x10_resurvey3',
        'uric_umoll_resurvey3',
        'fasting_glucose_x10_resurvey3',
        'chol_mmoll_resurvey3',
        'dbp_first_resurvey3',
        'sbp_second_resurvey3',
        'bmd_left_stiffness_index_resurvey3',
        'dbp_mean_resurvey3',
        'heart_rate_mean_resurvey3',
        'body_fat_perc_x10',
        'bmi_x10',
        'standing_height_cm_x10',
        'weight_kg_x10_resurvey3',
        'met_hours_resurvey3',
        # 'age_at_study_date_x100_resurvey3',
        'body_fat_mass_x10'
    ]
    
    # 检查哪些字段存在于原始文件中
    existing_fields = []
    missing_fields = []
    
    for field in fields_to_keep:
        if field in df.columns:
            existing_fields.append(field)
        else:
            missing_fields.append(field)
    
    print(f"找到的字段数量: {len(existing_fields)}")
    print(f"缺失的字段数量: {len(missing_fields)}")
    
    if missing_fields:
        print("缺失的字段:")
        for field in missing_fields:
            print(f"  - {field}")
    
    # 筛选字段
    df_selected = df[existing_fields]
    
    # 保存筛选后的文件
    df_selected.to_csv(output_file, sep='\t', index=False)
    print(f"筛选后文件行数: {len(df_selected)}")
    print(f"筛选后文件列数: {len(df_selected.columns)}")
    print(f"输出文件: {output_file}")
    
    return len(existing_fields), len(missing_fields)

def main():
    """主函数"""
    print("=== 生理学相关字段筛选脚本 ===")
    
    # 处理test文件
    test_input = "test.tsv"
    test_output = "test_physiology.tsv"
    
    # 处理train文件
    train_input = "train.tsv"
    train_output = "train_physiology.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(test_input):
        print(f"错误: 找不到文件 {test_input}")
        sys.exit(1)
    
    if not os.path.exists(train_input):
        print(f"错误: 找不到文件 {train_input}")
        sys.exit(1)
    
    # 处理文件
    test_fields, test_missing = select_physiology_fields(test_input, test_output)
    train_fields, train_missing = select_physiology_fields(train_input, train_output)
    
    # 生成报告
    with open("physiology_fields_selection_report.txt", "w", encoding="utf-8") as f:
        f.write("生理学相关字段筛选报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("处理文件:\n")
        f.write(f"1. {test_input} -> {test_output}\n")
        f.write(f"2. {train_input} -> {train_output}\n\n")
        
        f.write("筛选结果:\n")
        f.write(f"Test文件: 成功筛选 {test_fields} 个字段，缺失 {test_missing} 个字段\n")
        f.write(f"Train文件: 成功筛选 {train_fields} 个字段，缺失 {train_missing} 个字段\n\n")
        
        f.write("筛选的字段包括:\n")
        f.write("- 血糖相关指标 (随机血糖、空腹血糖)\n")
        f.write("- 尿酸指标\n")
        f.write("- 胆固醇指标\n")
        f.write("- 血压指标 (舒张压、收缩压)\n")
        f.write("- 骨密度指标\n")
        f.write("- 心率指标\n")
        f.write("- 体脂相关指标 (体脂百分比、体脂质量)\n")
        f.write("- 身体测量指标 (BMI、身高、体重)\n")
        f.write("- 代谢当量小时数\n")
        # f.write("- 年龄指标\n")
    
    print("\n=== 处理完成 ===")
    print(f"Test文件: {test_output}")
    print(f"Train文件: {train_output}")
    print("报告文件: physiology_fields_selection_report.txt")

if __name__ == "__main__":
    import os
    main() 