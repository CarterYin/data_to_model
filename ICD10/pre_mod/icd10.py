#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 合并了三个功能：
1. 添加samples列名（重新对齐表头）
2. 将除第一列外的YES转换为1
3. 将除第一列外的NO转换为0
"""

import pandas as pd
import os
import shutil

def realign_tsv_headers(input_file, output_file):
    """
    在第一列添加'samples'列名，原列名整体右移，使数据与列名对齐
    """
    print("🔄 步骤1: 开始重新对齐表头...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # 处理第一行（列名行）
            first_line = infile.readline()
            if first_line:
                # 在原列名前添加 "samples\t"，原列名整体右移
                new_header = "samples\t" + first_line
                outfile.write(new_header)
                print("✅ 列名行已处理：添加了 'samples' 列名，原列名整体右移")
                
                # 显示处理效果
                ICD10_cols = first_line.strip().split('\t')[:5]  # 前5个原列名
                new_cols = new_header.strip().split('\t')[:6]       # 前6个新列名
                
                print(f"📋 原始前5列名: {ICD10_cols}")
                print(f"📋 新的前6列名: {new_cols}")
            
            # 复制所有数据行（保持不变）
            line_count = 0
            for line in infile:
                outfile.write(line)
                line_count += 1
                if line_count % 50000 == 0:  # 每处理50000行显示进度
                    print(f"📊 已处理 {line_count} 行数据...")
            
            print(f"✅ 所有数据行已复制完成，共 {line_count} 行")
        
        print(f"🎉 表头重新对齐完成！")
        print(f"💾 输出文件：{output_file}")
        
        # 验证结果
        print(f"\n🔍 验证对齐效果：")
        with open(output_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            first_data = f.readline().strip()
        
        header_cols = header.split('\t')
        data_cols = first_data.split('\t')
        
        print(f"列名数量: {len(header_cols)}")
        print(f"数据列数量: {len(data_cols)}")
        
        if len(header_cols) == len(data_cols):
            print("✅ 列名与数据完美对齐！")
            print(f"\n📋 对齐效果预览：")
            for i in range(min(5, len(header_cols))):
                print(f"  '{header_cols[i]}' → '{data_cols[i]}'")
            if len(header_cols) > 5:
                print(f"  ... (还有 {len(header_cols) - 5} 列)")
        else:
            print("⚠️  列数不匹配，请检查")
            
        return True
        
    except Exception as e:
        print(f"❌ 表头重新对齐出错: {e}")
        return False

def convert_yes_to_1(input_file, output_file=None):
    """
    将除第一列外的其他列中的YES替换为1
    保持第一列的YES不变
    """
    print("\n🔄 步骤2: 开始转换YES为1...")
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
        # 统计替换前的YES数量
        yes_count_before = 0
        for col in columns[1:]:  # 跳过第一列
            yes_count_before += (df[col] == 'YES').sum()
        
        print(f"在除第一列外的其他列中找到 {yes_count_before} 个 'YES'")
        
        # 将除第一列外的其他列中的"YES"替换为"1"
        for col in columns[1:]:  # 跳过第一列
            df[col] = df[col].replace('YES', '1')
        
        # 统计替换后的情况
        yes_count_after = 0
        one_count_after = 0
        for col in columns[1:]:  # 跳过第一列
            yes_count_after += (df[col] == 'YES').sum()
            one_count_after += (df[col] == '1').sum()
        
        print(f"✅ YES转换完成:")
        print(f"  - 除第一列外的其他列中剩余 'YES': {yes_count_after} 个")
        print(f"  - 除第一列外的其他列中 '1' 的数量: {one_count_after} 个")
        
        # 检查第一列的YES数量（应该保持不变）
        first_col_yes = (df[columns[0]] == 'YES').sum()
        print(f"  - 第一列中保留的 'YES': {first_col_yes} 个")
    
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

def convert_no_to_zero(input_file, output_file=None):
    """
    将除第一列外的其他列中的NO替换为0
    保持第一列的NO不变
    """
    print("\n🔄 步骤3: 开始转换NO为0...")
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
        
        print(f"✅ NO转换完成:")
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

def process_file_to_output(input_file, output_file):
    """
    完整处理单个文件的所有步骤，输出到指定文件
    """
    print(f"\n{'='*60}")
    print(f"开始处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}")
    
    if not os.path.exists(input_file):
        print(f"⚠️  文件不存在: {input_file}")
        return False
    
    # 创建备份文件
    backup_file = input_file + '.ICD10_backup'
    if not os.path.exists(backup_file):
        shutil.copy2(input_file, backup_file)
        print(f"📦 已创建原始备份文件: {backup_file}")
    
    # 定义中间文件名
    step1_file = input_file.replace('.tsv', '_step1_realigned.tsv')
    step2_file = input_file.replace('.tsv', '_step2_yes_converted.tsv')
    
    try:
        # 步骤1: 重新对齐表头
        if not realign_tsv_headers(input_file, step1_file):
            return False
        
        # 步骤2: 转换YES为1
        if not convert_yes_to_1(step1_file, step2_file):
            return False
        
        # 步骤3: 转换NO为0，输出到最终文件
        if not convert_no_to_zero(step2_file, output_file):
            return False
        
        # 清理中间文件
        if os.path.exists(step1_file):
            os.remove(step1_file)
        if os.path.exists(step2_file):
            os.remove(step2_file)
        
        print(f"\n🎉 文件处理完成！")
        print(f"📥 输入文件: {input_file}")
        print(f"📤 输出文件: {output_file}")
        print(f"✅ 所有步骤执行成功")
        print(f"📦 原始文件备份: {backup_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件时发生错误: {e}")
        # 尝试清理中间文件
        for temp_file in [step1_file, step2_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        return False

def process_single_file(input_file):
    """
    完整处理单个文件的所有步骤（保留原有功能以保持兼容性）
    """
    print(f"\n{'='*60}")
    print(f"开始处理文件: {input_file}")
    print(f"{'='*60}")
    
    if not os.path.exists(input_file):
        print(f"⚠️  文件不存在: {input_file}")
        return False
    
    # 创建备份文件
    backup_file = input_file + '.ICD10_backup'
    if not os.path.exists(backup_file):
        shutil.copy2(input_file, backup_file)
        print(f"📦 已创建原始备份文件: {backup_file}")
    
    # 定义中间文件名
    step1_file = input_file.replace('.tsv', '_step1_realigned.tsv')
    step2_file = input_file.replace('.tsv', '_step2_yes_converted.tsv')
    final_file = input_file  # 最终覆盖原文件
    
    try:
        # 步骤1: 重新对齐表头
        if not realign_tsv_headers(input_file, step1_file):
            return False
        
        # 步骤2: 转换YES为1
        if not convert_yes_to_1(step1_file, step2_file):
            return False
        
        # 步骤3: 转换NO为0
        if not convert_no_to_zero(step2_file, final_file):
            return False
        
        # 清理中间文件
        if os.path.exists(step1_file):
            os.remove(step1_file)
        if os.path.exists(step2_file):
            os.remove(step2_file)
        
        print(f"\n🎉 文件 {input_file} 处理完成！")
        print(f"✅ 所有步骤执行成功")
        print(f"📦 原始文件备份: {backup_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件时发生错误: {e}")
        # 尝试清理中间文件
        for temp_file in [step1_file, step2_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        return False

def main():
    """
    主函数：处理ICD10.tsv文件，输出icd.tsv
    """
    print("🚀 数据预处理脚本启动")
    print("功能：1) 添加samples列名 → 2) YES转换为1 → 3) NO转换为0")
    print("输入文件：ICD10.tsv")
    print("输出文件：icd.tsv")
    
    input_file = 'ICD10.tsv'
    output_file = 'icd.tsv'
    
    if process_file_to_output(input_file, output_file):
        print(f"\n🎉 数据预处理完成！")
        print(f"📥 输入文件: {input_file}")
        print(f"📤 输出文件: {output_file}")
    else:
        print(f"\n❌ 数据预处理失败！")

if __name__ == "__main__":
    main()
