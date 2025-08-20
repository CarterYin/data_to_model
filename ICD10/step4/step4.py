#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除id_ethnic_group_id不为1的样本（保留NA值）
对于指定的id_ethnic_group_id字段，只删除值不为1且不为NA的样本个体
保留NA值和值为1的样本
"""

import pandas as pd
import numpy as np
import os

def remove_non_ethnic_group_1_samples_keep_na(input_file, output_file):
    """
    删除id_ethnic_group_id不为1的样本（保留NA值）
    
    Parameters:
    input_file (str): 输入TSV文件路径
    output_file (str): 输出TSV文件路径
    """
    
    # 定义需要检查的字段
    ethnic_field = 'id_ethnic_group_id'
    
    print(f"正在读取数据文件: {input_file}")
    
    # 读取TSV文件
    try:
        df = pd.read_csv(input_file, sep='\t', low_memory=False)
        print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 检查字段是否存在
    if ethnic_field not in df.columns:
        print(f"错误：字段 {ethnic_field} 在数据中不存在")
        return
    
    print(f"开始筛选，检查字段: {ethnic_field}")
    
    # 显示筛选前的统计信息
    print("\n筛选前的统计信息:")
    print(f"{ethnic_field} 字段值分布:")
    value_counts = df[ethnic_field].value_counts(dropna=False)
    for value, count in value_counts.items():
        percentage = count / len(df) * 100
        if pd.isna(value):
            print(f"  NA值: {count} 个样本 ({percentage:.2f}%)")
        else:
            print(f"  值 {value}: {count} 个样本 ({percentage:.2f}%)")
    
    # 统计要删除的样本数量
    # 保留id_ethnic_group_id为1或为NA的样本
    # 删除id_ethnic_group_id不为1且不为NA的样本
    samples_to_keep_mask = (df[ethnic_field] == 1) | (df[ethnic_field].isna())
    samples_to_remove = (~samples_to_keep_mask).sum()
    total_samples = len(df)
    
    print(f"\n筛选结果:")
    print(f"总样本数: {total_samples}")
    print(f"要删除的样本数（id_ethnic_group_id不为1且不为NA）: {samples_to_remove}")
    print(f"保留的样本数（id_ethnic_group_id为1或为NA）: {samples_to_keep_mask.sum()}")
    print(f"删除比例: {samples_to_remove/total_samples*100:.2f}%")
    
    # 显示具体删除和保留的统计
    print(f"\n详细统计:")
    na_count = df[ethnic_field].isna().sum()
    value_1_count = (df[ethnic_field] == 1).sum()
    not_1_and_not_na_count = ((df[ethnic_field] != 1) & (~df[ethnic_field].isna())).sum()
    print(f"保留的NA值样本: {na_count} 个")
    print(f"保留的值为1样本: {value_1_count} 个")
    print(f"删除的非1且非NA样本: {not_1_and_not_na_count} 个")
    
    # 执行筛选：保留id_ethnic_group_id为1或为NA的样本
    df_filtered = df[samples_to_keep_mask]
    
    print(f"\n筛选后的统计信息:")
    print(f"{ethnic_field} 字段值分布（筛选后）:")
    value_counts_after = df_filtered[ethnic_field].value_counts(dropna=False)
    for value, count in value_counts_after.items():
        percentage = count / len(df_filtered) * 100
        if pd.isna(value):
            print(f"  NA值: {count} 个样本 ({percentage:.2f}%)")
        else:
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
    report_file = output_file.replace('.tsv', '_ethnic_filter_keep_na_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("id_ethnic_group_id筛选报告（保留NA值）\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"输入文件: {input_file}\n")
        f.write(f"输出文件: {output_file}\n")
        f.write(f"筛选时间: {pd.Timestamp.now()}\n\n")
        f.write(f"筛选条件: 保留 {ethnic_field} == 1 或 {ethnic_field} 为 NA 的样本\n")
        f.write(f"删除条件: 删除 {ethnic_field} != 1 且 {ethnic_field} 不为 NA 的样本\n\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"删除样本数: {samples_to_remove}\n")
        f.write(f"保留样本数: {len(df_filtered)}\n")
        f.write(f"删除比例: {samples_to_remove/total_samples*100:.2f}%\n\n")
        
        f.write("详细统计:\n")
        f.write(f"保留的NA值样本: {na_count} 个\n")
        f.write(f"保留的值为1样本: {value_1_count} 个\n")
        f.write(f"删除的非1且非NA样本: {not_1_and_not_na_count} 个\n\n")
        
        f.write("筛选前字段值分布:\n")
        for value, count in value_counts.items():
            percentage = count / len(df) * 100
            if pd.isna(value):
                f.write(f"  NA值: {count} 个样本 ({percentage:.2f}%)\n")
            else:
                f.write(f"  值 {value}: {count} 个样本 ({percentage:.2f}%)\n")
        
        f.write("\n筛选后字段值分布:\n")
        for value, count in value_counts_after.items():
            percentage = count / len(df_filtered) * 100
            if pd.isna(value):
                f.write(f"  NA值: {count} 个样本 ({percentage:.2f}%)\n")
            else:
                f.write(f"  值 {value}: {count} 个样本 ({percentage:.2f}%)\n")
    
    print(f"筛选报告已保存到: {report_file}")

def main():
    """主函数"""
    # 设置文件路径
    input_file = "preprocessing3.tsv"
    output_file = "preprocessing4.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    print("开始执行id_ethnic_group_id筛选（保留NA值）...")
    print("=" * 50)
    
    # 执行筛选
    remove_non_ethnic_group_1_samples_keep_na(input_file, output_file)
    
    print("=" * 50)
    print("筛选完成！")

if __name__ == "__main__":
    main()
