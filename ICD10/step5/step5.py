#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将健康对照组数据分为训练集和测试集
功能：
1. 删除方差为0的列（所有值都相同的列）
2. 将数据分为训练集和测试集，比例为4:1
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def remove_zero_variance_columns(df):
    """
    删除方差为0的列（即所有值都相同的列）
    
    Parameters:
    df (DataFrame): 输入数据框
    
    Returns:
    tuple: (清理后的数据框, 被删除的列名列表)
    """
    
    print("正在检查方差为0的列...")
    
    # 获取数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 检查非数值列是否也有零方差（所有值都相同）
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    zero_variance_columns = []
    
    # 检查数值列的方差
    for col in numeric_columns:
        if col == 'samples':  # 跳过样本ID列
            continue
        try:
            variance = df[col].var()
            if pd.isna(variance) or variance == 0:
                zero_variance_columns.append(col)
                print(f"  发现零方差数值列: {col}")
        except:
            # 如果计算方差出错，检查唯一值数量
            unique_count = df[col].nunique()
            if unique_count <= 1:
                zero_variance_columns.append(col)
                print(f"  发现零方差数值列: {col} (计算方差出错，但唯一值<=1)")
    
    # 检查分类列是否所有值都相同
    for col in categorical_columns:
        if col == 'samples':  # 跳过样本ID列
            continue
        unique_count = df[col].nunique()
        if unique_count <= 1:
            zero_variance_columns.append(col)
            print(f"  发现零方差分类列: {col}")
    
    if zero_variance_columns:
        print(f"发现 {len(zero_variance_columns)} 个零方差列，将被删除")
        df_cleaned = df.drop(columns=zero_variance_columns)
        print(f"删除后剩余 {len(df_cleaned.columns)} 列")
    else:
        print("没有发现零方差列")
        df_cleaned = df.copy()
    
    return df_cleaned, zero_variance_columns

def split_train_test_data(input_file, train_file, test_file, train_ratio=0.8, random_state=42):
    """
    将数据集分为训练集和测试集
    
    Parameters:
    input_file (str): 输入TSV文件路径
    train_file (str): 训练集输出文件路径
    test_file (str): 测试集输出文件路径
    train_ratio (float): 训练集比例，默认0.8（即4:1）
    random_state (int): 随机种子，确保结果可重现
    """
    
    print(f"正在读取数据文件: {input_file}")
    
    # 读取TSV文件
    try:
        df = pd.read_csv(input_file, sep='\t', low_memory=False)
        print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 删除方差为0的列
    print(f"\n步骤1: 删除方差为0的列")
    print("-" * 30)
    df_cleaned, removed_columns = remove_zero_variance_columns(df)
    df = df_cleaned
    
    # 显示数据集基本信息
    print(f"\n步骤2: 数据集分割")
    print("-" * 30)
    print(f"清理后数据集信息:")
    print(f"总样本数: {len(df)}")
    print(f"特征数量: {len(df.columns)}")
    print(f"训练集比例: {train_ratio:.1%}")
    print(f"测试集比例: {1-train_ratio:.1%}")
    
    # 计算训练集和测试集的样本数
    train_size = int(len(df) * train_ratio)
    test_size = len(df) - train_size
    
    print(f"预期训练集样本数: {train_size}")
    print(f"预期测试集样本数: {test_size}")
    
    # 使用sklearn的train_test_split进行随机分割
    print(f"\n开始分割数据集...")
    
    # 执行训练测试集分割（不使用分层抽样）
    train_df, test_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state, 
        shuffle=True
    )
    
    print(f"实际训练集样本数: {len(train_df)}")
    print(f"实际测试集样本数: {len(test_df)}")
    print(f"训练集比例: {len(train_df)/len(df):.1%}")
    print(f"测试集比例: {len(test_df)/len(df):.1%}")
    
    # 保存训练集
    print(f"\n正在保存训练集到: {train_file}")
    try:
        train_df.to_csv(train_file, sep='\t', index=False)
        print(f"成功保存训练集，共 {len(train_df)} 行")
    except Exception as e:
        print(f"保存训练集时出错: {e}")
        return
    
    # 保存测试集
    print(f"正在保存测试集到: {test_file}")
    try:
        test_df.to_csv(test_file, sep='\t', index=False)
        print(f"成功保存测试集，共 {len(test_df)} 行")
    except Exception as e:
        print(f"保存测试集时出错: {e}")
        return
    
    # 生成分割报告
    report_file = "train_test_split_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("训练测试集分割报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"输入文件: {input_file}\n")
        f.write(f"训练集文件: {train_file}\n")
        f.write(f"测试集文件: {test_file}\n")
        f.write(f"分割时间: {pd.Timestamp.now()}\n\n")
        
        # 添加零方差列删除信息
        f.write(f"零方差列删除:\n")
        f.write(f"  删除的列数: {len(removed_columns)}\n")
        if removed_columns:
            f.write(f"  被删除的列:\n")
            for col in removed_columns:
                f.write(f"    - {col}\n")
        else:
            f.write(f"  没有发现零方差列\n")
        f.write(f"\n")
        f.write(f"分割参数:\n")
        f.write(f"  训练集比例: {train_ratio:.1%}\n")
        f.write(f"  测试集比例: {1-train_ratio:.1%}\n")
        f.write(f"  随机种子: {random_state}\n")
        f.write(f"  是否分层抽样: 否（使用随机分割）\n")
        f.write(f"  是否打乱数据: 是\n\n")
        f.write(f"数据集统计:\n")
        f.write(f"  总样本数: {len(df)}\n")
        f.write(f"  特征数量: {len(df.columns)}\n")
        f.write(f"  训练集样本数: {len(train_df)}\n")
        f.write(f"  测试集样本数: {len(test_df)}\n")
        f.write(f"  训练集比例: {len(train_df)/len(df):.1%}\n")
        f.write(f"  测试集比例: {len(test_df)/len(df):.1%}\n\n")
        
        # 显示一些样本ID的分布情况
        if 'samples' in df.columns:
            f.write("样本ID分布示例:\n")
            f.write("训练集前10个样本ID:\n")
            for i, sample_id in enumerate(train_df['samples'].head(10)):
                f.write(f"  {i+1}. {sample_id}\n")
            f.write("\n测试集前10个样本ID:\n")
            for i, sample_id in enumerate(test_df['samples'].head(10)):
                f.write(f"  {i+1}. {sample_id}\n")
        
        # 显示一些基本统计信息
        f.write("\n数据质量检查:\n")
        f.write(f"  训练集缺失值总数: {train_df.isnull().sum().sum()}\n")
        f.write(f"  测试集缺失值总数: {test_df.isnull().sum().sum()}\n")
        f.write(f"  训练集重复行数: {train_df.duplicated().sum()}\n")
        f.write(f"  测试集重复行数: {test_df.duplicated().sum()}\n")
    
    print(f"分割报告已保存到: {report_file}")
    
    # 显示一些基本统计信息
    print(f"\n数据分割完成！")
    print(f"训练集: {len(train_df)} 个样本 ({len(train_df)/len(df):.1%})")
    print(f"测试集: {len(test_df)} 个样本 ({len(test_df)/len(df):.1%})")
    
    # 显示数据质量信息
    print(f"\n数据质量检查:")
    print(f"训练集缺失值总数: {train_df.isnull().sum().sum()}")
    print(f"测试集缺失值总数: {test_df.isnull().sum().sum()}")
    print(f"训练集重复行数: {train_df.duplicated().sum()}")
    print(f"测试集重复行数: {test_df.duplicated().sum()}")

def main():
    """主函数"""
    # 设置文件路径
    input_file = "preprocessing4.tsv"
    train_file = "train_set.tsv"
    test_file = "test_set.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    print("开始执行训练测试集分割...")
    print("=" * 50)
    
    # 执行分割
    split_train_test_data(input_file, train_file, test_file)
    
    print("=" * 50)
    print("分割完成！")

if __name__ == "__main__":
    main() 