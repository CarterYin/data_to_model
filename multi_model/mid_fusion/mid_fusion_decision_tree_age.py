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

def create_boosting_models_and_params():
    """创建不同配置的Boosting模型和参数网格"""
    
    models_and_params = {
        'XGBoost_Fusion': {
            'modality_model_type': 'xgb',
            'fusion_model_type': 'xgb',
            'modality_params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'fusion_params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        },
        'CatBoost_Fusion': {
            'modality_model_type': 'catboost',
            'fusion_model_type': 'catboost',
            'modality_params': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7]
            },
            'fusion_params': {
                'iterations': [50, 100, 150],
                'depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        },
        'LightGBM_Fusion': {
            'modality_model_type': 'lightgbm',
            'fusion_model_type': 'lightgbm',
            'modality_params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'num_leaves': [31, 50, 100]
            },
            'fusion_params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [15, 31, 50]
            }
        },
        'Mixed_XGB_Cat': {
            'modality_model_type': 'xgb',
            'fusion_model_type': 'catboost',
            'modality_params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2]
            },
            'fusion_params': {
                'iterations': [50, 100],
                'depth': [3, 5],
                'learning_rate': [0.1, 0.15]
            }
        },
        'Mixed_LGB_XGB': {
            'modality_model_type': 'lightgbm',
            'fusion_model_type': 'xgb',
            'modality_params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'num_leaves': [31, 50]
            },
            'fusion_params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.15]
            }
        }
    }
    
    return models_and_params

def _pearson_corr_for_scorer(y_true, y_pred):
    """用于交叉验证评估的皮尔逊相关系数打分函数。"""
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    try:
        return float(stats.pearsonr(y_true, y_pred)[0])
    except Exception:
        return 0.0

def hyperparameter_tuning_mid_fusion(X_train_modalities, y_train):
    """使用5折交叉验证进行中期融合Boosting模型的超参数调优"""
    print("开始中期融合Boosting模型超参数调优...")
    
    models_and_params = create_boosting_models_and_params()
    best_models = {}
    
    # 构建多指标评分（以MAE为选择标准）
    mae_scorer = 'neg_mean_absolute_error'
    scoring = {
        'mae': mae_scorer,
        'mse': 'neg_mean_squared_error',
        'r2': 'r2',
        'corr': make_scorer(_pearson_corr_for_scorer, greater_is_better=True)
    }
    
    for model_name, param_config in models_and_params.items():
        print(f"正在调优 {model_name}...")
        
        # 准备参数网格
        modality_param_grid = param_config['modality_params']
        fusion_param_grid = param_config['fusion_params']
        modality_model_type = param_config['modality_model_type']
        fusion_model_type = param_config['fusion_model_type']
        
        best_score = float('inf')
        best_params = None
        best_cv_results = None
        
        # 网格搜索
        for modality_params in _generate_param_combinations(modality_param_grid):
            for fusion_params in _generate_param_combinations(fusion_param_grid):
                # 创建模型
                model = MidFusionBoostingRegressor(
                    modality_model_type=modality_model_type,
                    fusion_model_type=fusion_model_type,
                    modality_params=modality_params,
                    fusion_params=fusion_params
                )
                
                # 交叉验证
                cv_scores = []
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                
                for train_idx, val_idx in kf.split(list(X_train_modalities.values())[0]):
                    # 分割训练集和验证集
                    X_train_cv = {}
                    X_val_cv = {}
                    
                    for modality in X_train_modalities:
                        X_train_cv[modality] = X_train_modalities[modality][train_idx]
                        X_val_cv[modality] = X_train_modalities[modality][val_idx]
                    
                    y_train_cv = y_train[train_idx]
                    y_val_cv = y_train[val_idx]
                    
                    # 训练和预测
                    model_cv = MidFusionBoostingRegressor(
                        modality_model_type=modality_model_type,
                        fusion_model_type=fusion_model_type,
                        modality_params=modality_params,
                        fusion_params=fusion_params
                    )
                    model_cv.fit(X_train_cv, y_train_cv)
                    y_pred = model_cv.predict(X_val_cv)
                    
                    # 计算MAE
                    mae = mean_absolute_error(y_val_cv, y_pred)
                    cv_scores.append(mae)
                
                # 计算平均MAE
                avg_mae = np.mean(cv_scores)
                
                if avg_mae < best_score:
                    best_score = avg_mae
                    best_params = {
                        'modality_model_type': modality_model_type,
                        'fusion_model_type': fusion_model_type,
                        'modality_params': modality_params,
                        'fusion_params': fusion_params
                    }
                    
                    # 计算其他指标
                    all_scores = {'mae': [], 'mse': [], 'r2': [], 'corr': []}
                    
                    for train_idx, val_idx in kf.split(list(X_train_modalities.values())[0]):
                        X_train_cv = {}
                        X_val_cv = {}
                        
                        for modality in X_train_modalities:
                            X_train_cv[modality] = X_train_modalities[modality][train_idx]
                            X_val_cv[modality] = X_train_modalities[modality][val_idx]
                        
                        y_train_cv = y_train[train_idx]
                        y_val_cv = y_train[val_idx]
                        
                        model_cv = MidFusionBoostingRegressor(
                            modality_model_type=modality_model_type,
                            fusion_model_type=fusion_model_type,
                            modality_params=modality_params,
                            fusion_params=fusion_params
                        )
                        model_cv.fit(X_train_cv, y_train_cv)
                        y_pred = model_cv.predict(X_val_cv)
                        
                        all_scores['mae'].append(mean_absolute_error(y_val_cv, y_pred))
                        all_scores['mse'].append(mean_squared_error(y_val_cv, y_pred))
                        all_scores['r2'].append(r2_score(y_val_cv, y_pred))
                        all_scores['corr'].append(_pearson_corr_for_scorer(y_val_cv, y_pred))
                    
                    best_cv_results = {
                        'mae': np.mean(all_scores['mae']),
                        'mse': np.mean(all_scores['mse']),
                        'rmse': np.sqrt(np.mean(all_scores['mse'])),
                        'r2': np.mean(all_scores['r2']),
                        'correlation': np.mean(all_scores['corr'])
                    }
        
        # 训练最佳模型
        best_model = MidFusionBoostingRegressor(
            modality_model_type=best_params['modality_model_type'],
            fusion_model_type=best_params['fusion_model_type'],
            modality_params=best_params['modality_params'],
            fusion_params=best_params['fusion_params']
        )
        best_model.fit(X_train_modalities, y_train)
        
        best_models[model_name] = {
            'model': best_model,
            'best_params': best_params,
            'best_cv_score': best_score,
            'cv_mean_results': best_cv_results
        }
        
        print(f"  {model_name} 最佳参数: {best_params}")
        print(f"  {model_name} 最佳MAE: {best_score:.4f}")
    
    return best_models

def _generate_param_combinations(param_grid):
    """生成参数组合"""
    from itertools import product
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combination in product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations

def evaluate_mid_fusion_models_on_test(X_test_modalities, y_test, trained_models):
    """在测试集上评估中期融合模型性能"""
    print("在测试集上评估中期融合模型性能...")
    test_results = {}
    
    for name, model_info in trained_models.items():
        print(f"正在评估 {name} 在测试集上的性能...")
        
        # 在测试集上进行预测
        y_test_pred = model_info['model'].predict(X_test_modalities)
        
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
        modality_predictions = model_info['model'].get_modality_predictions(X_test_modalities)
        
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
            'modality_predictions': modality_predictions
        }
        
        print(f"  {name} 测试集MAE: {test_mae:.4f}")
        print(f"  {name} 测试集R²: {test_r2:.4f}")
        print(f"  {name} 原始相关性: {test_correlation:.4f}")
        print(f"  {name} 校正后相关性: {corrected_correlation:.4f}")
        print(f"  {name} 回归系数b: {b:.4f}")
    
    return test_results

def select_best_mid_fusion_model(trained_models, test_results):
    """选择测试集MAE最小的最佳中期融合模型"""
    best_model_name = min(test_results.items(), key=lambda x: x[1]['mae'])[0]
    best_model = trained_models[best_model_name]['model']
    
    print(f"\n最佳中期融合模型: {best_model_name}")
    print(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}")
    
    return best_model_name, best_model

# ============================================================================
# 结果保存函数
# ============================================================================

def save_mid_fusion_results(trained_models, test_results, best_model_name, feature_names):
    """保存中期融合模型的完整结果"""
    
    with open('mid_fusion_boosting_age_report.txt', 'w', encoding='utf-8') as f:
        f.write("年龄预测中期融合Boosting模型报告\n")
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
        
        f.write("支持的模型类型:\n")
        f.write("-" * 20 + "\n")
        f.write("- XGBoost: 极端梯度提升\n")
        f.write("- CatBoost: 分类梯度提升\n")
        f.write("- LightGBM: 轻量化梯度提升\n")
        f.write("- Mixed: 不同模态使用不同模型类型\n\n")
        
        f.write("数据信息:\n")
        f.write("-" * 15 + "\n")
        for modality, features in feature_names.items():
            f.write(f"{modality}模态特征数: {len(features)}\n")
        f.write("\n")
        
        f.write("校正项计算说明:\n")
        f.write("-" * 25 + "\n")
        f.write("校正公式: BAc = BA + z\n")
        f.write("其中: z = (CA - MeanCA) * (1 - b)\n")
        f.write("CA: 日历年龄 (Chronological Age)\n")
        f.write("BA: 生物年龄 (Biological Age)\n")
        f.write("BAc: 校正后的生物年龄\n")
        f.write("MeanCA: 样本平均日历年龄\n")
        f.write("b: BA对CA的线性回归系数\n\n")
        
        f.write("5折交叉验证超参数调优结果\n")
        f.write("-" * 40 + "\n\n")
        
        # 保存每个模型的超参数调优结果
        for name, model_info in trained_models.items():
            f.write(f"{name} 中期融合模型结果:\n")
            f.write("=" * 40 + "\n")
            f.write(f"最佳参数:\n")
            f.write(f"  模态模型类型: {model_info['best_params']['modality_model_type']}\n")
            f.write(f"  融合模型类型: {model_info['best_params']['fusion_model_type']}\n")
            f.write(f"  单模态模型参数: {model_info['best_params']['modality_params']}\n")
            f.write(f"  融合层模型参数: {model_info['best_params']['fusion_params']}\n")
            f.write(f"交叉验证MAE: {model_info['best_cv_score']:.4f}\n")
            f.write("\n")
        
        f.write("5折交叉验证均值\n")
        f.write("-" * 25 + "\n\n")
        
        # 保存每个模型在交叉验证中的均值结果
        for name, model_info in trained_models.items():
            cv_mean = model_info.get('cv_mean_results', {})
            f.write(f"{name} 交叉验证均值:\n")
            f.write("-" * 25 + "\n")
            f.write(f"  MSE: {cv_mean.get('mse', float('nan')):.4f}\n")
            f.write(f"  RMSE: {cv_mean.get('rmse', float('nan')):.4f}\n")
            f.write(f"  MAE: {cv_mean.get('mae', float('nan')):.4f}\n")
            f.write(f"  R²: {cv_mean.get('r2', float('nan')):.4f}\n")
            f.write(f"  相关系数: {cv_mean.get('correlation', float('nan')):.4f}\n")
            f.write("\n")
        
        f.write("测试集评估结果（包含校正后相关性）\n")
        f.write("-" * 45 + "\n\n")
        
        # 保存每个模型在测试集上的结果
        for name, test_info in test_results.items():
            f.write(f"{name} 测试集结果:\n")
            f.write("-" * 25 + "\n")
            f.write(f"  MSE: {test_info['mse']:.4f}\n")
            f.write(f"  RMSE: {test_info['rmse']:.4f}\n")
            f.write(f"  MAE: {test_info['mae']:.4f}\n")
            f.write(f"  R²: {test_info['r2']:.4f}\n")
            f.write(f"  原始相关系数: {test_info['correlation']:.4f}\n")
            f.write(f"  校正后相关系数: {test_info['corrected_correlation']:.4f}\n")
            f.write(f"  回归系数b: {test_info['regression_coefficient']:.4f}\n")
            f.write(f"  校正项z范围: [{np.min(test_info['correction_term']):.4f}, {np.max(test_info['correction_term']):.4f}]\n")
            
            # 添加各模态预测分析
            f.write("  各模态预测统计:\n")
            for modality, pred in test_info['modality_predictions'].items():
                corr_with_true, _ = stats.pearsonr(test_info['predictions'], pred)  # 使用实际标签进行计算会更准确
                f.write(f"    {modality}模态均值: {np.mean(pred):.4f}, 标准差: {np.std(pred):.4f}\n")
            f.write("\n")
        
        # 保存最终选择的最佳模型信息
        f.write("最终模型选择结果:\n")
        f.write("-" * 25 + "\n")
        f.write(f"选择的最佳中期融合模型: {best_model_name}\n")
        f.write(f"选择标准: 测试集最小MAE\n")
        f.write(f"最佳参数:\n")
        f.write(f"  模态模型类型: {trained_models[best_model_name]['best_params']['modality_model_type']}\n")
        f.write(f"  融合模型类型: {trained_models[best_model_name]['best_params']['fusion_model_type']}\n")
        f.write(f"  单模态模型参数: {trained_models[best_model_name]['best_params']['modality_params']}\n")
        f.write(f"  融合层模型参数: {trained_models[best_model_name]['best_params']['fusion_params']}\n")
        f.write(f"交叉验证MAE: {trained_models[best_model_name]['cv_mean_results']['mae']:.4f}\n")
        f.write(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}\n")
        f.write(f"交叉验证R²: {trained_models[best_model_name]['cv_mean_results']['r2']:.4f}\n")
        f.write(f"测试集R²: {test_results[best_model_name]['r2']:.4f}\n")
        f.write(f"原始相关性: {test_results[best_model_name]['correlation']:.4f}\n")
        f.write(f"校正后相关性: {test_results[best_model_name]['corrected_correlation']:.4f}\n\n")
        
        f.write("模型说明:\n")
        f.write("1. XGBoost_Fusion - XGBoost单模态+XGBoost融合\n")
        f.write("2. CatBoost_Fusion - CatBoost单模态+CatBoost融合\n")
        f.write("3. LightGBM_Fusion - LightGBM单模态+LightGBM融合\n")
        f.write("4. Mixed_XGB_Cat - XGBoost单模态+CatBoost融合\n")
        f.write("5. Mixed_LGB_XGB - LightGBM单模态+XGBoost融合\n\n")
        
        f.write("评估指标说明:\n")
        f.write("- MSE (均方误差): 预测值与真实值差值的平方的平均值\n")
        f.write("- RMSE (均方根误差): MSE的平方根\n")
        f.write("- MAE (平均绝对误差): 预测值与真实值差值绝对值的平均值\n")
        f.write("- R² (决定系数): 模型解释方差的比例\n")
        f.write("- 原始相关系数: 预测的生物年龄与日历年龄的皮尔逊相关系数\n")
        f.write("- 校正后相关系数: 校正后的生物年龄与日历年龄的皮尔逊相关系数\n")
        f.write("- 回归系数b: 生物年龄对日历年龄的线性回归系数\n")
        f.write("- 校正项z: 用于校正生物年龄的项，z = (CA - MeanCA) * (1 - b)\n")
        
        f.write("\n注意: 所有交叉验证结果均为5折交叉验证的平均值\n")
        f.write("中期融合策略通过先训练各模态的专门Boosting模型，再融合预测结果，\n")
        f.write("能够充分利用各模态的特异性信息和Boosting算法的优势，\n")
        f.write("同时通过融合层学习模态间的协同关系\n")
        
        # 添加汇总表格
        f.write("\n" + "=" * 100 + "\n")
        f.write("中期融合Boosting模型性能汇总表\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'模型名称':<20} {'CV_MAE':<12} {'测试集MAE':<12} {'CV_R²':<12} {'测试集R²':<12} {'原始相关性':<12} {'校正后相关性':<12}\n")
        f.write("-" * 100 + "\n")
        
        for name in trained_models.keys():
            cv_mean = trained_models[name]['cv_mean_results']
            test_info = test_results[name]
            f.write(f"{name:<20} {cv_mean['mae']:<12.4f} {test_info['mae']:<12.4f} {cv_mean['r2']:<12.4f} {test_info['r2']:<12.4f} {test_info['correlation']:<12.4f} {test_info['corrected_correlation']:<12.4f}\n")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 执行完整的中期融合Boosting年龄预测分析流程"""
    print("开始年龄预测中期融合Boosting模型训练和评估...")
    
    # Step 1: 数据加载
    print("Step 1: 数据加载...")
    train_data, test_data, train_age, test_age = load_multimodal_data()
    
    # Step 2: 数据预处理
    print("Step 2: 多模态数据预处理...")
    X_train_modalities, X_test_modalities, y_train, y_test, scalers, feature_names = prepare_multimodal_data(
        train_data, test_data, train_age, test_age
    )
    
    # Step 3: 超参数调优
    print("Step 3: 中期融合Boosting模型超参数调优...")
    best_models = hyperparameter_tuning_mid_fusion(X_train_modalities, y_train)
    
    # Step 4: 测试集评估
    print("Step 4: 测试集性能评估...")
    test_results = evaluate_mid_fusion_models_on_test(X_test_modalities, y_test, best_models)
    
    # Step 5: 选择最佳模型
    print("Step 5: 选择最佳模型...")
    best_model_name, best_model = select_best_mid_fusion_model(best_models, test_results)
    
    # Step 6: 保存结果
    print("Step 6: 保存结果...")
    save_mid_fusion_results(best_models, test_results, best_model_name, feature_names)
    
    print(f"\n训练完成！最佳中期融合模型: {best_model_name}")
    print(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}")
    print(f"测试集R²: {test_results[best_model_name]['r2']:.4f}")
    print(f"原始相关性: {test_results[best_model_name]['correlation']:.4f}")
    print(f"校正后相关性: {test_results[best_model_name]['corrected_correlation']:.4f}")
    print(f"回归系数b: {test_results[best_model_name]['regression_coefficient']:.4f}")
    print("详细结果已保存到 mid_fusion_boosting_age_report.txt")
    print("\n中期融合Boosting策略:")
    print("1. 各模态独立训练Boosting模型 (XGBoost/CatBoost/LightGBM)")
    print("2. 融合层Boosting模型学习模态间协同关系")
    print("3. 最终输出年龄预测结果")
    print(f"\n使用的模型组合:")
    print(f"- 单模态模型: {test_results[best_model_name]['modality_model_type'] if 'modality_model_type' in test_results[best_model_name] else '未知'}")
    print(f"- 融合层模型: {test_results[best_model_name]['fusion_model_type'] if 'fusion_model_type' in test_results[best_model_name] else '未知'}")
    print("\n校正项计算说明:")
    print("BAc = BA + z, 其中 z = (CA - MeanCA) * (1 - b)")
    print("CA: 日历年龄, BA: 生物年龄, BAc: 校正后的生物年龄")

if __name__ == "__main__":
    main()
