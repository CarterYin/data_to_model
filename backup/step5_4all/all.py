#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选所有相关字段脚本
从test.tsv和train.tsv文件中筛选指定的所有相关字段
包括眼科、动脉和生理学相关字段
"""

import pandas as pd
import sys

def select_all_fields(input_file, output_file):
    """筛选指定的所有相关字段"""
    print(f"处理文件: {input_file}")
    
    # 读取原始文件
    df = pd.read_csv(input_file, sep='\t')
    print(f"原始文件行数: {len(df)}")
    print(f"原始文件列数: {len(df.columns)}")
    
    # 定义要保留的字段列表
    fields_to_keep = [
        'samples',  # 保留样本ID
        # 眼科相关字段 (AI测量)
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
        'left_correction',
        'left_eye_x100',
        'right_eye_x10',
        'right_correction',
        'right_eye_x100',
        'wear_glasses',
        # 动脉相关字段 (CIMT和PWV)
        'cimt_lplq_count_cca',
        'cimt_lplq',
        'cimt_lplq_count_bif',
        'cimt_lplq_count_eca',
        'cimt_lplq_count_ica',
        'cimt_lplq_distance',
        'cimt_lplq_stenosis',
        'cimt_r_min_imt_120',
        'cimt_l_max_imt_240',
        'cimt_l_max_imt_210',
        'cimt_r_max_imt_150',
        'cimt_r_max_imt_120',
        'cimt_r_min_imt_150',
        'cimt_rplq',
        'cimt_rplq_count_bif',
        'cimt_rplq_count_eca',
        'cimt_rplq_count_ica',
        'cimt_rplq_count_cca',
        'cimt_rplq_combined_count',
        'cimt_rplq_stenosis',
        'cimt_rplq_distance',
        'cimt_rplq_divided_count',
        'cimt_lplq_combined_count',
        'cimt_lplq_excluded_count',
        'cimt_rplq_excluded_count',
        'cimt_lplq_divided_count',
        'cimt_lplq_minpixelsize',
        'cimt_rplq_minpixelsize',
        'cimt_lplq_maxpixelsize',
        'cimt_rplq_maxpixelsize',
        'cimt_l_meanimt_report',
        'cimt_rplq_special_count',
        'cimt_lplq_special_count',
        'cimt_l_mean_imt_240',
        'cimt_r_mean_imt_150',
        'cimt_r_mean_imt_120',
        'cimt_lplq_count',
        'cimt_rplq_count',
        'cimt_plq_score',
        'pwv_no_notch',
        'pwv_heart_rate',
        'pwv_notch_pos',
        'pwv_shoulder_pos',
        # 生理学相关字段
        'random_glucose_x10_resurvey3',
        'uric_umoll_resurvey3',
        'fasting_glucose_x10_resurvey3',
        'chol_mmoll_resurvey3',
        'dbp_first_resurvey3',
        'sbp_second_resurvey3',
        'bmd_left_stiffness_index_resurvey3',
        'dbp_mean_resurvey3',
        'heart_rate_mean_resurvey3',
        'body_fat_perc_x10',
        'bmi_x10',
        'standing_height_cm_x10',
        'weight_kg_x10_resurvey3',
        'met_hours_resurvey3',
        # 'age_at_study_date_x100_resurvey3',
        'body_fat_mass_x10'
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
    print("=== 所有相关字段筛选脚本 ===")
    
    # 处理test文件
    test_input = "test.tsv"
    test_output = "test_all.tsv"
    
    # 处理train文件
    train_input = "train.tsv"
    train_output = "train_all.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(test_input):
        print(f"错误: 找不到文件 {test_input}")
        sys.exit(1)
    
    if not os.path.exists(train_input):
        print(f"错误: 找不到文件 {train_input}")
        sys.exit(1)
    
    # 处理文件
    test_fields, test_missing = select_all_fields(test_input, test_output)
    train_fields, train_missing = select_all_fields(train_input, train_output)
    
    # 生成报告
    with open("all_fields_selection_report.txt", "w", encoding="utf-8") as f:
        f.write("所有相关字段筛选报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("处理文件:\n")
        f.write(f"1. {test_input} -> {test_output}\n")
        f.write(f"2. {train_input} -> {train_output}\n\n")
        
        f.write("筛选结果:\n")
        f.write(f"Test文件: 成功筛选 {test_fields} 个字段，缺失 {test_missing} 个字段\n")
        f.write(f"Train文件: 成功筛选 {train_fields} 个字段，缺失 {train_missing} 个字段\n\n")
        
        f.write("筛选的字段包括:\n")
        f.write("- 眼科相关指标 (AI测量): 视盘、视杯、轴长、血管等\n")
        f.write("- 动脉相关指标 (CIMT和PWV): 颈动脉内膜中层厚度、斑块、脉搏波速度等\n")
        f.write("- 生理学相关指标: 血糖、尿酸、胆固醇、血压、体脂、身体测量等\n")
        f.write("- 左右眼和左右颈动脉的对称性测量\n")
        f.write("- 各种统计指标和评分\n")
    
    print("\n=== 处理完成 ===")
    print(f"Test文件: {test_output}")
    print(f"Train文件: {train_output}")
    print("报告文件: all_fields_selection_report.txt")

if __name__ == "__main__":
    import os
    main() 