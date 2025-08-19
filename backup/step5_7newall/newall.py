#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并数据集脚本
将train_ren.tsv和train_all.tsv合并为train_newall.tsv
将test_ren.tsv和test_all.tsv合并为test_newall.tsv
"""

import pandas as pd
import os

def merge_datasets():
    """合并训练集和测试集数据"""
    
    # 设置文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 训练集文件
    train_ren_file = os.path.join(base_dir, "train_ren.tsv")
    train_all_file = os.path.join(base_dir, "train_all.tsv")
    train_merged_file = os.path.join(base_dir, "train_newall.tsv")
    
    # 测试集文件
    test_ren_file = os.path.join(base_dir, "test_ren.tsv")
    test_all_file = os.path.join(base_dir, "test_all.tsv")
    test_merged_file = os.path.join(base_dir, "test_newall.tsv")
    
    print("开始合并数据集...")
    
    # 合并训练集
    print("正在合并训练集...")
    try:
        # 读取人口学数据训练集
        train_ren = pd.read_csv(train_ren_file, sep='\t')
        print(f"人口学训练集: {train_ren.shape[0]} 行, {train_ren.shape[1]} 列")
        
        # 读取全部数据训练集
        train_all = pd.read_csv(train_all_file, sep='\t')
        print(f"全部数据训练集: {train_all.shape[0]} 行, {train_all.shape[1]} 列")
        
        # 基于samples列合并，保持原始顺序
        train_merged = pd.merge(train_ren, train_all, on='samples', how='outer', sort=False)
        print(f"合并后训练集: {train_merged.shape[0]} 行, {train_merged.shape[1]} 列")
        
        # 重新排序以保持原始顺序
        # 首先获取train_ren中samples的顺序
        sample_order = train_ren['samples'].tolist()
        # 然后按照这个顺序重新排列合并后的数据
        train_merged = train_merged.set_index('samples').reindex(sample_order).reset_index()
        
        # 保存合并后的训练集
        train_merged.to_csv(train_merged_file, sep='\t', index=False)
        print(f"训练集已保存到: {train_merged_file}")
        
    except Exception as e:
        print(f"合并训练集时出错: {e}")
        return False
    
    # 合并测试集
    print("正在合并测试集...")
    try:
        # 读取人口学数据测试集
        test_ren = pd.read_csv(test_ren_file, sep='\t')
        print(f"人口学测试集: {test_ren.shape[0]} 行, {test_ren.shape[1]} 列")
        
        # 读取全部数据测试集
        test_all = pd.read_csv(test_all_file, sep='\t')
        print(f"全部数据测试集: {test_all.shape[0]} 行, {test_all.shape[1]} 列")
        
        # 基于samples列合并，保持原始顺序
        test_merged = pd.merge(test_ren, test_all, on='samples', how='outer', sort=False)
        print(f"合并后测试集: {test_merged.shape[0]} 行, {test_merged.shape[1]} 列")
        
        # 重新排序以保持原始顺序
        # 首先获取test_ren中samples的顺序
        sample_order = test_ren['samples'].tolist()
        # 然后按照这个顺序重新排列合并后的数据
        test_merged = test_merged.set_index('samples').reindex(sample_order).reset_index()
        
        # 保存合并后的测试集
        test_merged.to_csv(test_merged_file, sep='\t', index=False)
        print(f"测试集已保存到: {test_merged_file}")
        
    except Exception as e:
        print(f"合并测试集时出错: {e}")
        return False
    
    print("数据集合并完成！")
    return True

def check_merged_data():
    """检查合并后的数据"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_merged_file = os.path.join(base_dir, "train_merged.tsv")
    test_merged_file = os.path.join(base_dir, "test_merged.tsv")
    
    print("\n检查合并后的数据:")
    
    if os.path.exists(train_merged_file):
        train_merged = pd.read_csv(train_merged_file, sep='\t')
        print(f"合并后训练集: {train_merged.shape[0]} 行, {train_merged.shape[1]} 列")
        print(f"列名: {list(train_merged.columns)}")
        
        # 检查是否有重复的samples
        duplicates = train_merged['samples'].duplicated().sum()
        print(f"重复的samples数量: {duplicates}")
        
        # 检查缺失值
        missing_values = train_merged.isnull().sum().sum()
        print(f"总缺失值数量: {missing_values}")
    
    if os.path.exists(test_merged_file):
        test_merged = pd.read_csv(test_merged_file, sep='\t')
        print(f"\n合并后测试集: {test_merged.shape[0]} 行, {test_merged.shape[1]} 列")
        print(f"列名: {list(test_merged.columns)}")
        
        # 检查是否有重复的samples
        duplicates = test_merged['samples'].duplicated().sum()
        print(f"重复的samples数量: {duplicates}")
        
        # 检查缺失值
        missing_values = test_merged.isnull().sum().sum()
        print(f"总缺失值数量: {missing_values}")

if __name__ == "__main__":
    # 执行合并
    success = merge_datasets()
    
    if success:
        # 检查合并后的数据
        check_merged_data()
    else:
        print("数据集合并失败！")
