#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将TSV文件中除第一行和第一列外的其他位置的"NO"替换为"0"
保持第一行（表头）和第一列（samples列）中的"NO"不变
"""

import pandas as pd
import sys
import os

def process_tsv_file(input_file, output_file=None):
    """
    处理TSV文件，将除第一行和第一列外的其他位置的NO替换为0
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    print(f"正在处理文件: {input_file}")
    
    # 读取TSV文件
    try:
        df = pd.read_csv(input_file, sep='\t', dtype=str, na_filter=False)
        print(f"文件读取成功，共有 {len(df)} 行，{len(df.columns)} 列")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return False
    
    # 获取列名
    columns = df.columns.tolist()
    print(f"第一列名称: {columns[0]}")
    
    # 处理除第一列外的其他列
    if len(columns) > 1:
        # 统计替换前的NO数量
        no_count_before_total = 0
        no_count_before_other_cols = 0
        
        # 统计所有列的NO数量
        for col in columns:
            no_count_before_total += (df[col] == 'NO').sum()
        
        # 统计除第一列外其他列的NO数量
        for col in columns[1:]:  # 跳过第一列
            no_count_before_other_cols += (df[col] == 'NO').sum()
        
        print(f"总共找到 {no_count_before_total} 个 'NO'")
        print(f"在除第一列外的其他列中找到 {no_count_before_other_cols} 个 'NO'")
        
        # 将除第一列外的其他列中的"NO"替换为"0"
        for col in columns[1:]:  # 跳过第一列
            df[col] = df[col].replace('NO', '0')
        
        # 统计替换后的情况
        no_count_after_total = 0
        no_count_after_other_cols = 0
        zero_count_after = 0
        
        # 统计所有列的NO数量
        for col in columns:
            no_count_after_total += (df[col] == 'NO').sum()
            
        # 统计除第一列外其他列的NO和0数量
        for col in columns[1:]:  # 跳过第一列
            no_count_after_other_cols += (df[col] == 'NO').sum()
            zero_count_after += (df[col] == '0').sum()
        
        print(f"替换完成:")
        print(f"  - 总共剩余 'NO': {no_count_after_total} 个")
        print(f"  - 除第一列外的其他列中剩余 'NO': {no_count_after_other_cols} 个")
        print(f"  - 除第一列外的其他列中新增 '0' 的数量: {zero_count_after} 个")
        
        # 检查第一列的NO数量（应该保持不变）
        first_col_no = (df[columns[0]] == 'NO').sum()
        print(f"  - 第一列中保留的 'NO': {first_col_no} 个")
    
    else:
        print("文件只有一列，无需处理")
        return True
    
    # 保存文件
    if output_file is None:
        output_file = input_file
    
    try:
        df.to_csv(output_file, sep='\t', index=False, na_rep='')
        print(f"文件保存成功: {output_file}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False

def main():
    # 处理两个主要的TSV文件
    files_to_process = [
        'analysis_result_new_realigned.tsv',
        'analysis_result_new.tsv'
    ]
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            print(f"\n{'='*50}")
            # 创建备份
            backup_file = file_path + '.backup_before_no_processing'
            if not os.path.exists(backup_file):
                import shutil
                shutil.copy2(file_path, backup_file)
                print(f"已创建备份文件: {backup_file}")
            
            # 处理文件
            success = process_tsv_file(file_path)
            if success:
                print(f"✅ {file_path} 处理完成")
            else:
                print(f"❌ {file_path} 处理失败")
        else:
            print(f"⚠️  文件不存在: {file_path}")

if __name__ == "__main__":
    main() 