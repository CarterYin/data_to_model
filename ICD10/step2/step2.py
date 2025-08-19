#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除心血管疾病和糖尿病诊断字段值为1的样本
对于指定的心血管疾病和糖尿病诊断字段，如果值为1，则删除该样本个体
"""

import pandas as pd
import numpy as np
import os

def remove_cardiovascular_diabetes_samples(input_file, output_file):
    """
    删除心血管疾病和糖尿病诊断字段值为1的样本
    
    Parameters:
    input_file (str): 输入TSV文件路径
    output_file (str): 输出TSV文件路径
    """
    
    # 定义需要检查的心血管疾病和糖尿病诊断字段
    disease_fields = [
        'ihd_diag',                    # 缺血性心脏病诊断
        'stroke_or_tia_diag3',         # 中风或短暂性脑缺血发作诊断
        'diabetes_diag3',              # 糖尿病诊断
        'hypertension_diag3'           # 高血压诊断
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
    missing_fields = [field for field in disease_fields if field not in df.columns]
    if missing_fields:
        print(f"警告：以下字段在数据中不存在: {missing_fields}")
        # 只保留存在的字段
        disease_fields = [field for field in disease_fields if field in df.columns]
        print(f"将使用以下字段进行筛选: {disease_fields}")
    
    if not disease_fields:
        print("错误：没有找到任何指定的疾病诊断字段")
        return
    
    print(f"开始筛选，检查字段: {disease_fields}")
    
    # 显示筛选前的统计信息
    print("\n筛选前的统计信息:")
    for field in disease_fields:
        if field in df.columns:
            # 统计各值的数量
            value_counts = df[field].value_counts(dropna=False)
            print(f"\n{field} 字段值分布:")
            for value, count in value_counts.items():
                percentage = count / len(df) * 100
                print(f"  值 {value}: {count} 个样本 ({percentage:.2f}%)")
    
    # 创建筛选条件：任何疾病字段值为1的行
    # 使用any()函数检查是否有任何字段值为1
    has_disease_mask = (df[disease_fields] == 1).any(axis=1)
    
    # 统计要删除的样本数量
    samples_to_remove = has_disease_mask.sum()
    total_samples = len(df)
    
    print(f"\n筛选结果:")
    print(f"总样本数: {total_samples}")
    print(f"要删除的样本数（任何疾病字段值为1）: {samples_to_remove}")
    print(f"保留的样本数: {total_samples - samples_to_remove}")
    print(f"删除比例: {samples_to_remove/total_samples*100:.2f}%")
    
    # 显示具体哪些字段导致了样本被删除
    print(f"\n各字段值为1的样本统计:")
    for field in disease_fields:
        if field in df.columns:
            field_ones = (df[field] == 1).sum()
            print(f"{field}: {field_ones} 个样本值为1")
    
    # 执行筛选：保留所有疾病字段值都不为1的样本
    df_filtered = df[~has_disease_mask]
    
    print(f"\n筛选后的统计信息:")
    for field in disease_fields:
        if field in df_filtered.columns:
            # 统计各值的数量
            value_counts = df_filtered[field].value_counts(dropna=False)
            print(f"\n{field} 字段值分布（筛选后）:")
            for value, count in value_counts.items():
                percentage = count / len(df_filtered) * 100
                print(f"  值 {value}: {count} 个样本 ({percentage:.2f}%)")
    
    # 保存筛选后的数据
    print(f"\n正在保存筛选后的数据到: {output_file}")
    try:
        df_filtered.to_csv(output_file, sep='\t', index=False)
        print(f"成功保存筛选后的数据，共 {len(df_filtered)} 行")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    # 生成筛选报告
    report_file = output_file.replace('.tsv', '_remove_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("心血管疾病和糖尿病样本删除报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"输入文件: {input_file}\n")
        f.write(f"输出文件: {output_file}\n")
        f.write(f"筛选时间: {pd.Timestamp.now()}\n\n")
        f.write(f"检查的疾病诊断字段: {disease_fields}\n\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"删除样本数: {samples_to_remove}\n")
        f.write(f"保留样本数: {len(df_filtered)}\n")
        f.write(f"删除比例: {samples_to_remove/total_samples*100:.2f}%\n\n")
        
        f.write("各字段值为1的样本统计:\n")
        for field in disease_fields:
            if field in df.columns:
                field_ones = (df[field] == 1).sum()
                f.write(f"{field}: {field_ones} 个样本\n")
        
        f.write("\n筛选后各字段值分布:\n")
        for field in disease_fields:
            if field in df_filtered.columns:
                value_counts = df_filtered[field].value_counts(dropna=False)
                f.write(f"\n{field}:\n")
                for value, count in value_counts.items():
                    percentage = count / len(df_filtered) * 100
                    f.write(f"  值 {value}: {count} 个样本 ({percentage:.2f}%)\n")
    
    print(f"筛选报告已保存到: {report_file}")

def main():
    """主函数"""
    # 设置文件路径
    input_file = "preprocessing1.tsv"
    output_file = "preprocessing2.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    print("开始执行心血管疾病和糖尿病样本删除...")
    print("=" * 50)
    
    # 执行筛选
    remove_cardiovascular_diabetes_samples(input_file, output_file)
    
    print("=" * 50)
    print("筛选完成！")

if __name__ == "__main__":
    main() 