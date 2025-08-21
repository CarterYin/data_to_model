#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并的数据预处理脚本
功能：
1. 删除训练集中缺失率超过50%的字段
2. 填充特定字段的缺失值（bmi_x10等）
3. 选择指定的字段
4. 最终缺失值填充
输入：train_set.tsv, test_set.tsv
输出：train.tsv, test.tsv
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

def get_high_missing_fields(train_df, missing_threshold=50.0):
    """
    获取训练集中缺失率超过阈值的字段列表
    
    Parameters:
    train_df (DataFrame): 训练集数据框
    missing_threshold (float): 缺失率阈值，默认50%
    
    Returns:
    list: 缺失率超过阈值的字段列表
    """
    
    print(f"正在分析训练集缺失率超过{missing_threshold}%的字段...")
    
    # 计算每个字段的缺失率
    missing_rates = []
    field_names = []
    
    for column in train_df.columns:
        missing_count = train_df[column].isna().sum()
        missing_rate = missing_count / len(train_df) * 100
        missing_rates.append(missing_rate)
        field_names.append(column)
    
    # 创建缺失率数据框
    missing_df = pd.DataFrame({
        'Field': field_names,
        'Missing_Rate': missing_rates
    })
    
    # 找出缺失率超过阈值的字段
    high_missing_fields = missing_df[missing_df['Missing_Rate'] > missing_threshold]['Field'].tolist()
    
    print(f"发现 {len(high_missing_fields)} 个字段缺失率超过{missing_threshold}%")
    print(f"这些字段将被删除")
    
    return high_missing_fields

def remove_high_missing_fields(train_df, test_df, fields_to_remove):
    """
    从训练集和测试集中删除指定字段
    
    Parameters:
    train_df (DataFrame): 训练集数据框
    test_df (DataFrame): 测试集数据框
    fields_to_remove (list): 要删除的字段列表
    
    Returns:
    tuple: (清理后的训练集, 清理后的测试集)
    """
    
    print(f"\n正在删除高缺失率字段...")
    
    # 删除指定字段
    train_df_cleaned = train_df.drop(columns=fields_to_remove, errors='ignore')
    test_df_cleaned = test_df.drop(columns=fields_to_remove, errors='ignore')
    
    print(f"训练集: 从{len(train_df.columns)}列减少到{len(train_df_cleaned.columns)}列")
    print(f"测试集: 从{len(test_df.columns)}列减少到{len(test_df_cleaned.columns)}列")
    print(f"删除了 {len(train_df.columns) - len(train_df_cleaned.columns)} 个字段")
    
    return train_df_cleaned, test_df_cleaned

def fill_bmi_x10(df):
    """
    填充bmi_x10字段
    如果bmi_x10为空，则使用bmi_calc_resurvey2或bmi_calc_baseline的值乘10后填充
    优先级：bmi_calc_resurvey2 > bmi_calc_baseline
    """
    print("正在填充bmi_x10字段...")
    
    # 检查字段是否存在
    bmi_fields = ['bmi_x10', 'bmi_calc_resurvey2', 'bmi_calc_baseline']
    missing_fields = [field for field in bmi_fields if field not in df.columns]
    
    if missing_fields:
        print(f"警告：以下字段不存在，跳过bmi_x10填充: {missing_fields}")
        return df
    
    # 统计填充前的缺失值数量
    missing_before = df['bmi_x10'].isna().sum()
    print(f"填充前bmi_x10缺失值数量: {missing_before}")
    
    # 创建填充后的DataFrame副本
    df_filled = df.copy()
    
    # 找出bmi_x10为空的索引
    missing_mask = df_filled['bmi_x10'].isna()
    
    # 填充逻辑：优先使用bmi_calc_resurvey2，其次使用bmi_calc_baseline
    for idx in df_filled[missing_mask].index:
        if pd.notna(df_filled.loc[idx, 'bmi_calc_resurvey2']):
            # 使用bmi_calc_resurvey2的值乘10
            df_filled.loc[idx, 'bmi_x10'] = df_filled.loc[idx, 'bmi_calc_resurvey2'] * 10
        elif pd.notna(df_filled.loc[idx, 'bmi_calc_baseline']):
            # 使用bmi_calc_baseline的值乘10
            df_filled.loc[idx, 'bmi_x10'] = df_filled.loc[idx, 'bmi_calc_baseline'] * 10
    
    # 统计填充后的缺失值数量
    missing_after = df_filled['bmi_x10'].isna().sum()
    filled_count = missing_before - missing_after
    print(f"填充后bmi_x10缺失值数量: {missing_after}")
    print(f"成功填充了 {filled_count} 个值")
    
    return df_filled

def fill_standing_height_cm_x10(df):
    """
    填充standing_height_cm_x10字段
    如果standing_height_cm_x10为空，则使用standing_height_mm_resurvey2或standing_height_mm_baseline的值填充
    优先级：standing_height_mm_resurvey2 > standing_height_mm_baseline
    """
    print("正在填充standing_height_cm_x10字段...")
    
    # 检查字段是否存在
    height_fields = ['standing_height_cm_x10', 'standing_height_mm_resurvey2', 'standing_height_mm_baseline']
    missing_fields = [field for field in height_fields if field not in df.columns]
    
    if missing_fields:
        print(f"警告：以下字段不存在，跳过standing_height_cm_x10填充: {missing_fields}")
        return df
    
    # 统计填充前的缺失值数量
    missing_before = df['standing_height_cm_x10'].isna().sum()
    print(f"填充前standing_height_cm_x10缺失值数量: {missing_before}")
    
    # 创建填充后的DataFrame副本
    df_filled = df.copy()
    
    # 找出standing_height_cm_x10为空的索引
    missing_mask = df_filled['standing_height_cm_x10'].isna()
    
    # 填充逻辑：优先使用standing_height_mm_resurvey2，其次使用standing_height_mm_baseline
    for idx in df_filled[missing_mask].index:
        if pd.notna(df_filled.loc[idx, 'standing_height_mm_resurvey2']):
            # 使用standing_height_mm_resurvey2的值
            df_filled.loc[idx, 'standing_height_cm_x10'] = df_filled.loc[idx, 'standing_height_mm_resurvey2']
        elif pd.notna(df_filled.loc[idx, 'standing_height_mm_baseline']):
            # 使用standing_height_mm_baseline的值
            df_filled.loc[idx, 'standing_height_cm_x10'] = df_filled.loc[idx, 'standing_height_mm_baseline']
    
    # 统计填充后的缺失值数量
    missing_after = df_filled['standing_height_cm_x10'].isna().sum()
    filled_count = missing_before - missing_after
    print(f"填充后standing_height_cm_x10缺失值数量: {missing_after}")
    print(f"成功填充了 {filled_count} 个值")
    
    return df_filled

def fill_weight_kg_x10_resurvey3(df):
    """
    填充weight_kg_x10_resurvey3字段
    如果weight_kg_x10_resurvey3为空，则使用weight_kg_x10_resurvey2或weight_kg_x10_baseline的值填充
    优先级：weight_kg_x10_resurvey2 > weight_kg_x10_baseline
    """
    print("正在填充weight_kg_x10_resurvey3字段...")
    
    # 检查字段是否存在
    weight_fields = ['weight_kg_x10_resurvey3', 'weight_kg_x10_resurvey2', 'weight_kg_x10_baseline']
    missing_fields = [field for field in weight_fields if field not in df.columns]
    
    if missing_fields:
        print(f"警告：以下字段不存在，跳过weight_kg_x10_resurvey3填充: {missing_fields}")
        return df
    
    # 统计填充前的缺失值数量
    missing_before = df['weight_kg_x10_resurvey3'].isna().sum()
    print(f"填充前weight_kg_x10_resurvey3缺失值数量: {missing_before}")
    
    # 创建填充后的DataFrame副本
    df_filled = df.copy()
    
    # 找出weight_kg_x10_resurvey3为空的索引
    missing_mask = df_filled['weight_kg_x10_resurvey3'].isna()
    
    # 填充逻辑：优先使用weight_kg_x10_resurvey2，其次使用weight_kg_x10_baseline
    for idx in df_filled[missing_mask].index:
        if pd.notna(df_filled.loc[idx, 'weight_kg_x10_resurvey2']):
            # 使用weight_kg_x10_resurvey2的值
            df_filled.loc[idx, 'weight_kg_x10_resurvey3'] = df_filled.loc[idx, 'weight_kg_x10_resurvey2']
        elif pd.notna(df_filled.loc[idx, 'weight_kg_x10_baseline']):
            # 使用weight_kg_x10_baseline的值
            df_filled.loc[idx, 'weight_kg_x10_resurvey3'] = df_filled.loc[idx, 'weight_kg_x10_baseline']
    
    # 统计填充后的缺失值数量
    missing_after = df_filled['weight_kg_x10_resurvey3'].isna().sum()
    filled_count = missing_before - missing_after
    print(f"填充后weight_kg_x10_resurvey3缺失值数量: {missing_after}")
    print(f"成功填充了 {filled_count} 个值")
    
    return df_filled

def apply_specific_field_filling(df):
    """
    对数据集进行特定字段的缺失值填充
    
    Parameters:
    df (DataFrame): 输入数据框
    
    Returns:
    DataFrame: 填充后的数据框
    """
    
    print(f"\n开始特定字段缺失值填充...")
    
    # 记录填充前的缺失值统计
    target_fields = ['bmi_x10', 'standing_height_cm_x10', 'weight_kg_x10_resurvey3']
    print("填充前缺失值统计:")
    for field in target_fields:
        if field in df.columns:
            missing_count = df[field].isna().sum()
            print(f"  {field}: {missing_count} 个缺失值")
        else:
            print(f"  {field}: 字段不存在")
    
    # 执行填充操作
    df_filled = df.copy()
    
    # 填充bmi_x10
    df_filled = fill_bmi_x10(df_filled)
    
    # 填充standing_height_cm_x10
    df_filled = fill_standing_height_cm_x10(df_filled)
    
    # 填充weight_kg_x10_resurvey3
    df_filled = fill_weight_kg_x10_resurvey3(df_filled)
    
    # 记录填充后的缺失值统计
    print("填充后缺失值统计:")
    for field in target_fields:
        if field in df_filled.columns:
            missing_count = df_filled[field].isna().sum()
            print(f"  {field}: {missing_count} 个缺失值")
        else:
            print(f"  {field}: 字段不存在")
    
    return df_filled

def select_specific_fields(df):
    """
    选择指定字段，删除其余字段
    
    Parameters:
    df (DataFrame): 输入数据框
    
    Returns:
    DataFrame: 选择后的数据框
    """
    
    # 要保留的字段列表
    fields_to_keep = [
        'samples',
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
        'age_at_study_date_x100_resurvey3',
        'body_fat_mass_x10',
        'id_ethnic_group_id',
        'non_local_birth_province',
        'region_code_baseline',
        'is_female_baseline',
        'region_is_urban_baseline',
        'occupation3',
        'occupation_baseline',
        'glaucoma_diag',
        'amd_diag',
        'cataract_diag',
        'diabetes_test',
        'ihd_diag',
        'stroke_or_tia_diag3',
        'diabetes_diag3',
        'hypertension_diag3'
    ]
    
    print(f"\n正在选择指定字段...")
    
    # 检查要保留的字段是否存在
    existing_fields = [field for field in fields_to_keep if field in df.columns]
    missing_fields = [field for field in fields_to_keep if field not in df.columns]
    
    if missing_fields:
        print(f"警告：以下字段不存在于数据集中: {missing_fields}")
        print(f"将只保留存在的字段")
    
    print(f"要保留的字段数量: {len(fields_to_keep)}")
    print(f"实际存在的字段数量: {len(existing_fields)}")
    print(f"缺失的字段数量: {len(missing_fields)}")
    
    # 选择指定字段
    df_selected = df[existing_fields]
    
    print(f"选择后的数据: {len(df_selected)} 行，{len(df_selected.columns)} 列")
    print(f"删除了 {len(df.columns) - len(df_selected.columns)} 个字段")
    
    return df_selected

def fill_final_missing_values(df):
    """按照规则填充最终的缺失值"""
    print(f"\n开始最终缺失值填充...")
    
    # 创建数据副本
    df_filled = df.copy()
    
    # 获取所有列（除了samples）
    columns_to_fill = [col for col in df.columns if col != 'samples']
    
    for col in columns_to_fill:
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            continue
            
        print(f"处理列: {col} (缺失值: {missing_count})")
        
        # 获取非缺失值
        non_missing_values = df[col].dropna()
        
        # 情况1: 列只包含少数几个值（≤10个唯一值）或id_ethnic_group_id列
        if col == 'id_ethnic_group_id' or non_missing_values.nunique() <= 10:
            print(f"  - 使用随机填充（唯一值数量: {non_missing_values.nunique()}）")
            
            # 计算各值的比例
            value_counts = non_missing_values.value_counts()
            probabilities = value_counts / len(non_missing_values)
            
            # 随机填充缺失值
            missing_indices = df[col].isnull()
            random_values = np.random.choice(
                probabilities.index, 
                size=missing_count, 
                p=probabilities.values
            )
            df_filled.loc[missing_indices, col] = random_values
            
        # 情况3: 其他情况，使用均值填充
        else:
            print(f"  - 使用均值填充（唯一值数量: {non_missing_values.nunique()}）")
            
            # 尝试转换为数值类型
            try:
                numeric_values = pd.to_numeric(non_missing_values, errors='coerce')
                if not numeric_values.isna().all():
                    mean_value = numeric_values.mean()
                    df_filled[col].fillna(mean_value, inplace=True)
                    print(f"    - 均值: {mean_value:.4f}")
                else:
                    # 如果无法转换为数值，使用众数
                    mode_value = non_missing_values.mode()[0]
                    df_filled[col].fillna(mode_value, inplace=True)
                    print(f"    - 众数: {mode_value}")
            except:
                # 如果转换失败，使用众数
                mode_value = non_missing_values.mode()[0]
                df_filled[col].fillna(mode_value, inplace=True)
                print(f"    - 众数: {mode_value}")
    
    return df_filled

def verify_final_result(df, filename):
    """验证最终结果"""
    print(f"\n=== {filename} 最终验证 ===")
    
    missing_total = df.isnull().sum().sum()
    print(f"最终缺失值总数: {missing_total}")
    print(f"处理完成: {'是' if missing_total == 0 else '否'}")
    
    if missing_total > 0:
        print("仍有缺失值的列:")
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            print(f"  - {col}: {missing_count}")

def main():
    """主函数"""
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 设置文件路径
    train_input_file = "train_set.tsv"
    test_input_file = "test_set.tsv"
    train_output_file = "train.tsv"
    test_output_file = "test.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(train_input_file):
        print(f"错误：训练集文件 {train_input_file} 不存在")
        return
    
    if not os.path.exists(test_input_file):
        print(f"错误：测试集文件 {test_input_file} 不存在")
        return
    
    print("开始数据预处理...")
    print("=" * 70)
    
    # 1. 读取原始数据
    print("步骤1: 读取原始数据")
    print("-" * 30)
    try:
        train_df = pd.read_csv(train_input_file, sep='\t', low_memory=False)
        test_df = pd.read_csv(test_input_file, sep='\t', low_memory=False)
        print(f"成功读取训练集，共 {len(train_df)} 行，{len(train_df.columns)} 列")
        print(f"成功读取测试集，共 {len(test_df)} 行，{len(test_df.columns)} 列")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 2. 删除高缺失率字段
    print("\n步骤2: 删除高缺失率字段")
    print("-" * 30)
    high_missing_fields = get_high_missing_fields(train_df, missing_threshold=50.0)
    
    if high_missing_fields:
        train_df, test_df = remove_high_missing_fields(train_df, test_df, high_missing_fields)
    else:
        print("没有找到缺失率超过50%的字段，跳过删除步骤")
    
    # 3. 特定字段缺失值填充
    print("\n步骤3: 特定字段缺失值填充")
    print("-" * 30)
    print("处理训练集...")
    train_df = apply_specific_field_filling(train_df)
    
    print("\n处理测试集...")
    test_df = apply_specific_field_filling(test_df)
    
    # 4. 选择指定字段
    print("\n步骤4: 选择指定字段")
    print("-" * 30)
    print("处理训练集...")
    train_df = select_specific_fields(train_df)
    
    print("\n处理测试集...")
    test_df = select_specific_fields(test_df)
    
    # 5. 最终缺失值填充
    print("\n步骤5: 最终缺失值填充")
    print("-" * 30)
    print("处理训练集...")
    train_df = fill_final_missing_values(train_df)
    
    print("\n处理测试集...")
    test_df = fill_final_missing_values(test_df)
    
    # 6. 验证最终结果
    print("\n步骤6: 验证最终结果")
    print("-" * 30)
    verify_final_result(train_df, "训练集")
    verify_final_result(test_df, "测试集")
    
    # 7. 保存最终结果
    print("\n步骤7: 保存最终结果")
    print("-" * 30)
    print(f"正在保存训练集到: {train_output_file}")
    train_df.to_csv(train_output_file, sep='\t', index=False)
    print(f"训练集保存成功！")
    
    print(f"正在保存测试集到: {test_output_file}")
    test_df.to_csv(test_output_file, sep='\t', index=False)
    print(f"测试集保存成功！")
    
    print("=" * 70)
    print("数据预处理完成！")
    print(f"最终训练集: {train_output_file} ({len(train_df)} 行，{len(train_df.columns)} 列)")
    print(f"最终测试集: {test_output_file} ({len(test_df)} 行，{len(test_df.columns)} 列)")

if __name__ == "__main__":
    main()
