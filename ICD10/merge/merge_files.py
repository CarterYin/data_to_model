#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并preprocessing.tsv和icd.tsv文件
将icd.tsv中的ICD10相关字段添加到preprocessing.tsv中
"""

import pandas as pd
import sys
import os

def merge_tsv_files():
    """
    合并preprocessing.tsv和icd.tsv文件
    """
    # 设置文件路径
    preprocessing_file = "preprocessing.tsv"
    icd_file = "icd.tsv"
    output_file = "all.tsv"
    
    print("开始读取文件...")
    
    try:
        # 读取preprocessing.tsv文件
        print("正在读取 preprocessing.tsv...")
        preprocessing_df = pd.read_csv(preprocessing_file, sep='\t', low_memory=False)
        print(f"preprocessing.tsv 形状: {preprocessing_df.shape}")
        
        # 读取icd.tsv文件
        print("正在读取 icd.tsv...")
        icd_df = pd.read_csv(icd_file, sep='\t', low_memory=False)
        print(f"icd.tsv 形状: {icd_df.shape}")
        
        # 获取两个文件的列名
        preprocessing_cols = set(preprocessing_df.columns)
        icd_cols = set(icd_df.columns)
        
        # 找出icd.tsv中新增的列（在icd.tsv中但不在preprocessing.tsv中的列）
        new_cols = icd_cols - preprocessing_cols
        print(f"icd.tsv中新增的列: {sorted(new_cols)}")
        
        # 验证两个文件的samples列是否匹配
        if not preprocessing_df['samples'].equals(icd_df['samples']):
            print("警告: 两个文件的samples列不匹配，将基于samples列进行合并")
            # 基于samples列进行合并
            # 首先提取icd.tsv中的新增列和samples列
            icd_new_data = icd_df[['samples'] + list(new_cols)]
            # 合并数据
            merged_df = pd.merge(preprocessing_df, icd_new_data, on='samples', how='left')
        else:
            print("两个文件的samples列匹配，直接添加新列")
            # 如果samples列匹配，直接添加新列
            merged_df = preprocessing_df.copy()
            for col in new_cols:
                merged_df[col] = icd_df[col]
        
        print(f"合并后的数据形状: {merged_df.shape}")
        
        # 保存合并后的文件
        print(f"正在保存合并后的文件到 {output_file}...")
        merged_df.to_csv(output_file, sep='\t', index=False)
        
        print("文件合并完成！")
        print(f"输出文件: {output_file}")
        print(f"原始文件列数: {len(preprocessing_df.columns)}")
        print(f"合并后列数: {len(merged_df.columns)}")
        print(f"新增列数: {len(new_cols)}")
        
        # 显示新增的列名
        print("\n新增的列:")
        for col in sorted(new_cols):
            print(f"  - {col}")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    merge_tsv_files()
