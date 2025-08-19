import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
import warnings
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
warnings.filterwarnings('ignore')

# ============================================================================
# 数据加载和预处理函数
# ============================================================================

def load_multimodal_data():
    """加载四个模态的训练和测试数据"""
    print("正在加载四个模态的数据...")
    
    # 定义模态名称
    modalities = ['ren', 'artery', 'eye', 'physiology']
    
    # 加载各模态特征数据
    train_data = {}
    test_data = {}
    
    for modality in modalities:
        print(f"  加载 {modality} 模态数据...")
        train_data[modality] = pd.read_csv(f'train_set_{modality}.tsv', sep='\t')
        test_data[modality] = pd.read_csv(f'test_set_{modality}.tsv', sep='\t')
    
    # 加载年龄标签
    train_age = pd.read_csv('train_set_age.tsv', sep='\t')
    test_age = pd.read_csv('test_set_age.tsv', sep='\t')
    
    print("数据加载完成！")
    return train_data, test_data, train_age, test_age

def preprocess_modality(train_data, test_data, modality_name):
    """预处理单个模态的数据"""
    print(f"  预处理 {modality_name} 模态...")
    
    # 设置样本ID为索引
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    train_df.set_index('samples', inplace=True)
    test_df.set_index('samples', inplace=True)
    
    # 确保训练集和测试集使用相同的特征
    common_features = train_df.columns.intersection(test_df.columns)
    train_df = train_df[common_features]
    test_df = test_df[common_features]
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_modality = scaler.fit_transform(train_df)
    X_test_modality = scaler.transform(test_df)
    
    return X_train_modality, X_test_modality, scaler, common_features.tolist()

def prepare_multimodal_data(train_data, test_data, train_age, test_age):
    """准备多模态数据 - 数据预处理和标准化"""
    print("开始预处理多模态数据...")
    
    modalities = ['ren', 'artery', 'eye', 'physiology']
    
    # 处理年龄标签
    train_age.set_index('samples', inplace=True)
    test_age.set_index('samples', inplace=True)
    
    age_column = 'age_at_study_date_x100_resurvey3'
    y_train = train_age[age_column].values / 100.0
    y_test = test_age[age_column].values / 100.0
    
    # 预处理各模态数据
    X_train_modalities = {}
    X_test_modalities = {}
    scalers = {}
    feature_names = {}
    
    for modality in modalities:
        X_train_mod, X_test_mod, scaler, features = preprocess_modality(
            train_data[modality], test_data[modality], modality
        )
        X_train_modalities[modality] = X_train_mod
        X_test_modalities[modality] = X_test_mod
        scalers[modality] = scaler
        feature_names[modality] = features
    
    print("多模态数据预处理完成！")
    return X_train_modalities, X_test_modalities, y_train, y_test, scalers, feature_names

# ============================================================================
# 校正项计算函数
# ============================================================================

def calculate_correction_term(CA, BA):
    """
    计算校正项 z
    
    参数:
    CA: 日历年龄 (Chronological Age)
    BA: 生物年龄 (Biological Age)
    
    返回:
    z: 校正项
    b: 回归系数
    """
    # 计算平均日历年龄
    MeanCA = np.mean(CA)
    
    # 使用简单线性回归计算BA对CA的回归系数
    # BA = b * CA + intercept
    CA_reshaped = CA.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(CA_reshaped, BA)
    b = reg.coef_[0]
    
    # 计算校正项 z = (CA - MeanCA) * (1 - b)
    z = (CA - MeanCA) * (1 - b)
    
    return z, b

def calculate_corrected_biological_age(CA, BA):
    """
    计算校正后的生物年龄 BAc = BA + z
    
    参数:
    CA: 日历年龄
    BA: 生物年龄
    
    返回:
    BAc: 校正后的生物年龄
    z: 校正项
    b: 回归系数
    """
    z, b = calculate_correction_term(CA, BA)
    BAc = BA + z
    
    return BAc, z, b

# ============================================================================
# 中期融合模型定义
# ============================================================================

class MidFusionBoostingRegressor:
    """中期融合Boosting回归模型（支持XGBoost、CatBoost、LightGBM）"""
    
    def __init__(self, modality_model_type='xgb', fusion_model_type='xgb', 
                 modality_params=None, fusion_params=None):
        """
        初始化中期融合Boosting模型
        
        参数:
        modality_model_type: 单模态模型类型 ('xgb', 'catboost', 'lightgbm')
        fusion_model_type: 融合层模型类型 ('xgb', 'catboost', 'lightgbm')
        modality_params: 单模态模型的参数
        fusion_params: 融合层模型的参数
        """
        self.modality_model_type = modality_model_type
        self.fusion_model_type = fusion_model_type
        self.modality_params = modality_params or {}
        self.fusion_params = fusion_params or {}
        
        # 各模态的模型
        self.modality_models = {}
        
        # 融合层模型
        self.fusion_model = None
        
        # 模态名称
        self.modalities = ['ren', 'artery', 'eye', 'physiology']
    
    def _create_model(self, model_type, params):
        """根据类型创建模型"""
        if model_type == 'xgb':
            return xgb.XGBRegressor(
                random_state=42,
                eval_metric='mae',
                **params
            )
        elif model_type == 'catboost':
            return cb.CatBoostRegressor(
                random_state=42,
                verbose=False,
                **params
            )
        elif model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                random_state=42,
                verbose=-1,
                **params
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def fit(self, X_train_modalities, y_train):
        """训练中期融合模型"""
        print(f"开始训练中期融合{self.modality_model_type.upper()}模型...")
        
        # Step 1: 训练各模态的模型
        modality_predictions = []
        
        for modality in self.modalities:
            print(f"  训练 {modality} 模态{self.modality_model_type.upper()}...")
            
            # 创建模型
            model = self._create_model(self.modality_model_type, self.modality_params)
            
            # 训练模型
            model.fit(X_train_modalities[modality], y_train)
            self.modality_models[modality] = model
            
            # 获取预测结果
            pred = model.predict(X_train_modalities[modality])
            modality_predictions.append(pred)
        
        # Step 2: 准备融合层输入
        # 将各模态的预测结果作为融合层的输入特征
        fusion_features = np.column_stack(modality_predictions)
        
        print(f"  训练融合层{self.fusion_model_type.upper()}...")
        
        # Step 3: 训练融合层模型
        self.fusion_model = self._create_model(self.fusion_model_type, self.fusion_params)
        self.fusion_model.fit(fusion_features, y_train)
        
        print(f"中期融合{self.modality_model_type.upper()}模型训练完成！")
        
        return self
    
    def predict(self, X_test_modalities):
        """使用中期融合模型进行预测"""
        # Step 1: 各模态预测
        modality_predictions = []
        
        for modality in self.modalities:
            pred = self.modality_models[modality].predict(X_test_modalities[modality])
            modality_predictions.append(pred)
        
        # Step 2: 融合层预测
        fusion_features = np.column_stack(modality_predictions)
        final_prediction = self.fusion_model.predict(fusion_features)
        
        return final_prediction
    
    def get_modality_predictions(self, X_modalities):
        """获取各模态的预测结果（用于分析）"""
        predictions = {}
        for modality in self.modalities:
            predictions[modality] = self.modality_models[modality].predict(X_modalities[modality])
        return predictions

def evaluate_models_on_test(X_train_modalities, X_test_modalities, y_train, y_test):
    """快速评估几个主要的Boosting模型"""
    print("快速评估三种主要Boosting模型...")
    test_results = {}
    
    # 定义三种基本配置
    model_configs = {
        'XGBoost_Fast': {
            'modality_model_type': 'xgb',
            'fusion_model_type': 'xgb',
            'modality_params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'fusion_params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
        },
        'CatBoost_Fast': {
            'modality_model_type': 'catboost',
            'fusion_model_type': 'catboost',
            'modality_params': {'iterations': 100, 'depth': 6, 'learning_rate': 0.1},
            'fusion_params': {'iterations': 50, 'depth': 3, 'learning_rate': 0.1}
        },
        'LightGBM_Fast': {
            'modality_model_type': 'lightgbm',
            'fusion_model_type': 'lightgbm',
            'modality_params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'fusion_params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
        }
    }
    
    for name, config in model_configs.items():
        print(f"正在评估 {name}...")
        
        # 创建并训练模型
        model = MidFusionBoostingRegressor(
            modality_model_type=config['modality_model_type'],
            fusion_model_type=config['fusion_model_type'],
            modality_params=config['modality_params'],
            fusion_params=config['fusion_params']
        )
        
        # 训练模型
        model.fit(X_train_modalities, y_train)
        
        # 在测试集上进行预测
        y_test_pred = model.predict(X_test_modalities)
        
        # 计算测试集上的评估指标
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_correlation, _ = stats.pearsonr(y_test, y_test_pred)
        
        # 计算校正后的生物年龄与日历年龄的相关性
        BAc, z, b = calculate_corrected_biological_age(y_test, y_test_pred)
        corrected_correlation, _ = stats.pearsonr(y_test, BAc)
        
        # 获取各模态的预测结果（用于分析）
        modality_predictions = model.get_modality_predictions(X_test_modalities)
        
        # 保存测试结果
        test_results[name] = {
            'mse': test_mse,
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'correlation': test_correlation,
            'predictions': y_test_pred,
            'corrected_predictions': BAc,
            'correction_term': z,
            'regression_coefficient': b,
            'corrected_correlation': corrected_correlation,
            'modality_predictions': modality_predictions,
            'config': config
        }
        
        print(f"  {name} 测试集MAE: {test_mae:.4f}")
        print(f"  {name} 测试集R²: {test_r2:.4f}")
        print(f"  {name} 原始相关性: {test_correlation:.4f}")
        print(f"  {name} 校正后相关性: {corrected_correlation:.4f}")
        print(f"  {name} 回归系数b: {b:.4f}")
    
    return test_results

def save_fast_results(test_results, feature_names):
    """保存快速评估结果"""
    
    with open('mid_fusion_boosting_age_fast_report.txt', 'w', encoding='utf-8') as f:
        f.write("年龄预测中期融合Boosting模型快速评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("模型架构说明:\n")
        f.write("-" * 20 + "\n")
        f.write("中期融合策略:\n")
        f.write("1. 第一层: 各模态独立的Boosting模型\n")
        f.write("   - ren模态模型 (XGBoost/CatBoost/LightGBM)\n")
        f.write("   - artery模态模型 (XGBoost/CatBoost/LightGBM)\n")
        f.write("   - eye模态模型 (XGBoost/CatBoost/LightGBM)\n")
        f.write("   - physiology模态模型 (XGBoost/CatBoost/LightGBM)\n")
        f.write("2. 第二层: 融合Boosting模型\n")
        f.write("   - 输入: 各模态模型的预测结果\n")
        f.write("   - 输出: 最终年龄预测\n\n")
        
        f.write("数据信息:\n")
        f.write("-" * 15 + "\n")
        for modality, features in feature_names.items():
            f.write(f"{modality}模态特征数: {len(features)}\n")
        f.write("\n")
        
        f.write("快速评估结果（固定参数配置）\n")
        f.write("-" * 40 + "\n\n")
        
        # 保存每个模型在测试集上的结果
        for name, test_info in test_results.items():
            f.write(f"{name} 测试集结果:\n")
            f.write("-" * 25 + "\n")
            f.write(f"  模型配置:\n")
            f.write(f"    单模态模型: {test_info['config']['modality_model_type']}\n")
            f.write(f"    融合模型: {test_info['config']['fusion_model_type']}\n")
            f.write(f"    单模态参数: {test_info['config']['modality_params']}\n")
            f.write(f"    融合层参数: {test_info['config']['fusion_params']}\n")
            f.write(f"  性能指标:\n")
            f.write(f"    MSE: {test_info['mse']:.4f}\n")
            f.write(f"    RMSE: {test_info['rmse']:.4f}\n")
            f.write(f"    MAE: {test_info['mae']:.4f}\n")
            f.write(f"    R²: {test_info['r2']:.4f}\n")
            f.write(f"    原始相关系数: {test_info['correlation']:.4f}\n")
            f.write(f"    校正后相关系数: {test_info['corrected_correlation']:.4f}\n")
            f.write(f"    回归系数b: {test_info['regression_coefficient']:.4f}\n")
            f.write(f"    校正项z范围: [{np.min(test_info['correction_term']):.4f}, {np.max(test_info['correction_term']):.4f}]\n")
            f.write("\n")
        
        # 选择最佳模型
        best_model_name = min(test_results.items(), key=lambda x: x[1]['mae'])[0]
        
        f.write("最佳模型选择结果:\n")
        f.write("-" * 25 + "\n")
        f.write(f"选择的最佳模型: {best_model_name}\n")
        f.write(f"选择标准: 测试集最小MAE\n")
        f.write(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}\n")
        f.write(f"测试集R²: {test_results[best_model_name]['r2']:.4f}\n")
        f.write(f"原始相关性: {test_results[best_model_name]['correlation']:.4f}\n")
        f.write(f"校正后相关性: {test_results[best_model_name]['corrected_correlation']:.4f}\n\n")
        
        # 添加汇总表格
        f.write("=" * 80 + "\n")
        f.write("快速评估性能汇总表\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'模型名称':<15} {'MAE':<10} {'R²':<10} {'原始相关性':<12} {'校正后相关性':<12}\n")
        f.write("-" * 80 + "\n")
        
        for name in test_results.keys():
            test_info = test_results[name]
            f.write(f"{name:<15} {test_info['mae']:<10.4f} {test_info['r2']:<10.4f} {test_info['correlation']:<12.4f} {test_info['corrected_correlation']:<12.4f}\n")

def main():
    """主函数 - 执行快速的中期融合Boosting年龄预测分析"""
    print("开始年龄预测中期融合Boosting模型快速评估...")
    
    # Step 1: 数据加载
    print("Step 1: 数据加载...")
    train_data, test_data, train_age, test_age = load_multimodal_data()
    
    # Step 2: 数据预处理
    print("Step 2: 多模态数据预处理...")
    X_train_modalities, X_test_modalities, y_train, y_test, scalers, feature_names = prepare_multimodal_data(
        train_data, test_data, train_age, test_age
    )
    
    # Step 3: 快速评估
    print("Step 3: 快速模型评估...")
    test_results = evaluate_models_on_test(X_train_modalities, X_test_modalities, y_train, y_test)
    
    # Step 4: 保存结果
    print("Step 4: 保存结果...")
    save_fast_results(test_results, feature_names)
    
    # 选择最佳模型
    best_model_name = min(test_results.items(), key=lambda x: x[1]['mae'])[0]
    
    print(f"\n快速评估完成！最佳模型: {best_model_name}")
    print(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}")
    print(f"测试集R²: {test_results[best_model_name]['r2']:.4f}")
    print(f"原始相关性: {test_results[best_model_name]['correlation']:.4f}")
    print(f"校正后相关性: {test_results[best_model_name]['corrected_correlation']:.4f}")
    print("详细结果已保存到 mid_fusion_boosting_age_fast_report.txt")
    print("\n注意: 这是快速评估版本，使用固定的参数配置")
    print("如需完整的超参数调优，请运行 mid_fusion_boosting_age.py")

if __name__ == "__main__":
    main()
