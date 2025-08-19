#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选年龄字段脚本
从test.tsv和train.tsv文件中筛选年龄相关字段
"""

import pandas as pd
import sys

def select_age_fields(input_file, output_file):
    """筛选指定的年龄相关字段"""
    print(f"处理文件: {input_file}")
    
    # 读取原始文件
    df = pd.read_csv(input_file, sep='\t')
    print(f"原始文件行数: {len(df)}")
    print(f"原始文件列数: {len(df.columns)}")
    
    # 定义要保留的字段列表
    fields_to_keep = [
        'samples',  # 保留样本ID
        'age_at_study_date_x100_resurvey3'
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
    print("=== 年龄字段筛选脚本 ===")
    
    # 处理test文件
    test_input = "test.tsv"
    test_output = "test_age.tsv"
    
    # 处理train文件
    train_input = "train.tsv"
    train_output = "train_age.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(test_input):
        print(f"错误: 找不到文件 {test_input}")
        sys.exit(1)
    
    if not os.path.exists(train_input):
        print(f"错误: 找不到文件 {train_input}")
        sys.exit(1)
    
    # 处理文件
    test_fields, test_missing = select_age_fields(test_input, test_output)
    train_fields, train_missing = select_age_fields(train_input, train_output)
    
    # 生成报告
    with open("age_fields_selection_report.txt", "w", encoding="utf-8") as f:
        f.write("年龄字段筛选报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("处理文件:\n")
        f.write(f"1. {test_input} -> {test_output}\n")
        f.write(f"2. {train_input} -> {train_output}\n\n")
        
        f.write("筛选结果:\n")
        f.write(f"Test文件: 成功筛选 {test_fields} 个字段，缺失 {test_missing} 个字段\n")
        f.write(f"Train文件: 成功筛选 {train_fields} 个字段，缺失 {train_missing} 个字段\n\n")
        
        f.write("筛选的字段包括:\n")
        f.write("- samples: 样本ID\n")
        f.write("- age_at_study_date_x100_resurvey3: 研究日期年龄（第3次复查，100倍放大）\n")
        f.write("\n注意: 年龄字段已经过100倍放大处理，实际年龄需要除以100\n")
    
    print("\n=== 处理完成 ===")
    print(f"Test文件: {test_output}")
    print(f"Train文件: {train_output}")
    print("报告文件: age_fields_selection_report.txt")

if __name__ == "__main__":
    import os
    main() 