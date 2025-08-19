#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剔除没有疾病诊断信息的个体
对于指定的疾病诊断字段，如果全部为NA，则剔除该样本个体
"""

import pandas as pd
import numpy as np
import os

def filter_diagnosis_data(input_file, output_file):
    """
    剔除没有疾病诊断信息的个体
    
    Parameters:
    input_file (str): 输入TSV文件路径
    output_file (str): 输出TSV文件路径
    """
    
    # 定义需要检查的疾病诊断字段
    diagnosis_fields = [
        'glaucoma_diag',
        'amd_diag', 
        'cataract_diag',
        'ihd_diag',
        'stroke_or_tia_diag3',
        'diabetes_diag3',
        'hypertension_diag3'
    ]
    
    print(f"正在读取数据文件: {input_file}")
    
    # 读取TSV文件
    try:
        df = pd.read_csv(input_file, sep='\t', low_memory=False)
        print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 检查字段是否存在
    missing_fields = [field for field in diagnosis_fields if field not in df.columns]
    if missing_fields:
        print(f"警告：以下字段在数据中不存在: {missing_fields}")
        # 只保留存在的字段
        diagnosis_fields = [field for field in diagnosis_fields if field in df.columns]
        print(f"将使用以下字段进行筛选: {diagnosis_fields}")
    
    if not diagnosis_fields:
        print("错误：没有找到任何指定的诊断字段")
        return
    
    print(f"开始筛选，检查字段: {diagnosis_fields}")
    
    # 显示筛选前的统计信息
    print("\n筛选前的统计信息:")
    for field in diagnosis_fields:
        if field in df.columns:
            na_count = df[field].isna().sum()
            total_count = len(df)
            print(f"{field}: NA值 {na_count}/{total_count} ({na_count/total_count*100:.2f}%)")
    
    # 创建筛选条件：所有诊断字段都为NA的行
    # 使用all()函数检查所有字段是否都为NA
    all_na_mask = df[diagnosis_fields].isna().all(axis=1)
    
    # 统计要剔除的样本数量
    samples_to_remove = all_na_mask.sum()
    total_samples = len(df)
    
    print(f"\n筛选结果:")
    print(f"总样本数: {total_samples}")
    print(f"要剔除的样本数（所有诊断字段都为NA）: {samples_to_remove}")
    print(f"保留的样本数: {total_samples - samples_to_remove}")
    print(f"剔除比例: {samples_to_remove/total_samples*100:.2f}%")
    
    # 执行筛选：保留至少有一个诊断字段不为NA的样本
    df_filtered = df[~all_na_mask]
    
    print(f"\n筛选后的统计信息:")
    for field in diagnosis_fields:
        if field in df_filtered.columns:
            na_count = df_filtered[field].isna().sum()
            total_count = len(df_filtered)
            print(f"{field}: NA值 {na_count}/{total_count} ({na_count/total_count*100:.2f}%)")
    
    # 保存筛选后的数据
    print(f"\n正在保存筛选后的数据到: {output_file}")
    try:
        df_filtered.to_csv(output_file, sep='\t', index=False)
        print(f"成功保存筛选后的数据，共 {len(df_filtered)} 行")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    # 生成筛选报告
    report_file = output_file.replace('.tsv', '_filter_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("疾病诊断信息筛选报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"输入文件: {input_file}\n")
        f.write(f"输出文件: {output_file}\n")
        f.write(f"筛选时间: {pd.Timestamp.now()}\n\n")
        f.write(f"检查的诊断字段: {diagnosis_fields}\n\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"剔除样本数: {samples_to_remove}\n")
        f.write(f"保留样本数: {len(df_filtered)}\n")
        f.write(f"剔除比例: {samples_to_remove/total_samples*100:.2f}%\n\n")
        f.write("各字段NA值统计（筛选后）:\n")
        for field in diagnosis_fields:
            if field in df_filtered.columns:
                na_count = df_filtered[field].isna().sum()
                total_count = len(df_filtered)
                f.write(f"{field}: {na_count}/{total_count} ({na_count/total_count*100:.2f}%)\n")
    
    print(f"筛选报告已保存到: {report_file}")

def main():
    """主函数"""
    # 设置文件路径
    input_file = "preprocessing.tsv"
    output_file = "preprocessing0.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    print("开始执行疾病诊断信息筛选...")
    print("=" * 50)
    
    # 执行筛选
    filter_diagnosis_data(input_file, output_file)
    
    print("=" * 50)
    print("筛选完成！")

if __name__ == "__main__":
    main() 