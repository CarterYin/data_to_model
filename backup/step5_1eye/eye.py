#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选眼科相关字段脚本
从test.tsv和train.tsv文件中筛选指定的眼科相关字段
"""

import pandas as pd
import sys

def select_eye_fields(input_file, output_file):
    """筛选指定的眼科相关字段"""
    print(f"处理文件: {input_file}")
    
    # 读取原始文件
    df = pd.read_csv(input_file, sep='\t')
    print(f"原始文件行数: {len(df)}")
    print(f"原始文件列数: {len(df.columns)}")
    
    # 定义要保留的字段列表
    fields_to_keep = [
        'samples',  # 保留样本ID
        'ai_axial_length_l',
        'ai_AVR_l',
        'ai_disc_rim_width_I_l',
        'ai_HCDR_l',
        'ai_disc_diameter_horizontal_l',
        'ai_disc_area_l',
        'ai_disc_diameter_vertical_l',
        'ai_disc_rim_area_l',
        'ai_cup_diameter_horizontal_l',
        'ai_cup_area_l',
        'ai_cup_to_disc_area_ratio_l',
        'ai_cup_diameter_vertical_l',
        'ai_disc_rim_width_S_l',
        'ai_mean_tessellation_density_l',
        'ai_disc_rim_width_T_l',
        'ai_myopic_crescent_diameter_l',
        'ai_myopic_crescent_to_disc_area_ratio_l',
        'ai_VCDR_l',
        'ai_disc_rim_width_N_l',
        'ai_axial_length_r',
        'ai_AVR_r',
        'ai_disc_rim_width_I_r',
        'ai_HCDR_r',
        'ai_disc_diameter_horizontal_r',
        'ai_disc_area_r',
        'ai_disc_diameter_vertical_r',
        'ai_disc_rim_area_r',
        'ai_cup_diameter_horizontal_r',
        'ai_cup_area_r',
        'ai_cup_to_disc_area_ratio_r',
        'ai_cup_diameter_vertical_r',
        'ai_disc_rim_width_S_r',
        'ai_mean_tessellation_density_r',
        'ai_disc_rim_width_T_r',
        'ai_myopic_crescent_diameter_r',
        'ai_myopic_crescent_to_disc_area_ratio_r',
        'ai_VCDR_r',
        'ai_disc_rim_width_N_r',
        'ai_CRVE_l',
        'ai_CRAE_l',
        'ai_CRVE_r',
        'ai_CRAE_r',
        'left_eye_x10',
        'left_eye_x100',
        'right_eye_x10',
        'right_eye_x100',
        'wear_glasses'
    ]
    
    # 检查哪些字段存在
    existing_fields = []
    missing_fields = []
    
    for field in fields_to_keep:
        if field in df.columns:
            existing_fields.append(field)
        else:
            missing_fields.append(field)
    
    print(f"存在的字段数量: {len(existing_fields)}")
    print(f"缺失的字段数量: {len(missing_fields)}")
    
    if missing_fields:
        print("缺失的字段:")
        for field in missing_fields:
            print(f"  - {field}")
    
    # 筛选字段
    df_selected = df[existing_fields]
    
    print(f"筛选后文件行数: {len(df_selected)}")
    print(f"筛选后文件列数: {len(df_selected.columns)}")
    
    # 保存文件
    df_selected.to_csv(output_file, sep='\t', index=False)
    print(f"文件已保存为: {output_file}")
    
    return df_selected

def main():
    """主函数"""
    print("开始筛选眼科相关字段...")
    
    # 处理test文件
    print("\n" + "="*50)
    print("处理 test.tsv")
    print("="*50)
    
    test_df = select_eye_fields('test.tsv', 'test_eye.tsv')
    
    # 处理train文件
    print("\n" + "="*50)
    print("处理 train.tsv")
    print("="*50)
    
    train_df = select_eye_fields('train.tsv', 'train_eye.tsv')
    
    # 生成报告
    print("\n" + "="*50)
    print("生成筛选报告")
    print("="*50)
    
    report_file = 'eye_fields_selection_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("眼科字段筛选报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("筛选的字段列表:\n")
        fields_list = [
            'ai_axial_length_l', 'ai_AVR_l', 'ai_disc_rim_width_I_l', 'ai_HCDR_l',
            'ai_disc_diameter_horizontal_l', 'ai_disc_area_l', 'ai_disc_diameter_vertical_l',
            'ai_disc_rim_area_l', 'ai_cup_diameter_horizontal_l', 'ai_cup_area_l',
            'ai_cup_to_disc_area_ratio_l', 'ai_cup_diameter_vertical_l', 'ai_disc_rim_width_S_l',
            'ai_mean_tessellation_density_l', 'ai_disc_rim_width_T_l', 'ai_myopic_crescent_diameter_l',
            'ai_myopic_crescent_to_disc_area_ratio_l', 'ai_VCDR_l', 'ai_disc_rim_width_N_l',
            'ai_axial_length_r', 'ai_AVR_r', 'ai_disc_rim_width_I_r', 'ai_HCDR_r',
            'ai_disc_diameter_horizontal_r', 'ai_disc_area_r', 'ai_disc_diameter_vertical_r',
            'ai_disc_rim_area_r', 'ai_cup_diameter_horizontal_r', 'ai_cup_area_r',
            'ai_cup_to_disc_area_ratio_r', 'ai_cup_diameter_vertical_r', 'ai_disc_rim_width_S_r',
            'ai_mean_tessellation_density_r', 'ai_disc_rim_width_T_r', 'ai_myopic_crescent_diameter_r',
            'ai_myopic_crescent_to_disc_area_ratio_r', 'ai_VCDR_r', 'ai_disc_rim_width_N_r',
            'ai_CRVE_l', 'ai_CRAE_l', 'ai_CRVE_r', 'ai_CRAE_r', 'left_eye_x10', 'left_eye_x100',
            'right_eye_x10', 'right_eye_x100', 'wear_glasses'
        ]
        
        for i, field in enumerate(fields_list, 1):
            f.write(f"{i:2d}. {field}\n")
        
        f.write("\n缺失的字段:\n")
        f.write("- left_correction (不存在于原始文件中)\n")
        f.write("- right_correction (不存在于原始文件中)\n")
        
        f.write("\n处理结果:\n")
        f.write(f"Test文件:\n")
        f.write(f"- 原始列数: {len(pd.read_csv('test.tsv', sep='\t').columns)}\n")
        f.write(f"- 筛选后列数: {len(test_df.columns)}\n")
        f.write(f"- 输出文件: test_eye.tsv\n\n")
        
        f.write(f"Train文件:\n")
        f.write(f"- 原始列数: {len(pd.read_csv('train.tsv', sep='\t').columns)}\n")
        f.write(f"- 筛选后列数: {len(train_df.columns)}\n")
        f.write(f"- 输出文件: train_eye.tsv\n")
    
    print(f"筛选报告已保存为: {report_file}")
    print("\n眼科字段筛选完成！")

if __name__ == "__main__":
    main() 