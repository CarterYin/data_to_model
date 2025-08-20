import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
from scipy import stats
from scipy.special import softmax
import time
warnings.filterwarnings('ignore')

# ============================================================================
# 数据加载和预处理函数（从simple1_gpu.py导入）
# ============================================================================

def load_multimodal_data():
    """加载多模态训练和测试数据"""
    modalities = ['eye', 'artery', 'physiology', 'ren']
    train_data = {}
    test_data = {}
    
    # 加载各模态数据
    for modality in modalities:
        train_data[modality] = pd.read_csv(f'train_{modality}.tsv', sep='\t')
        test_data[modality] = pd.read_csv(f'test_{modality}.tsv', sep='\t')
    
    # 加载年龄标签
    train_age = pd.read_csv('train_age.tsv', sep='\t')
    test_age = pd.read_csv('test_age.tsv', sep='\t')
    
    return train_data, test_data, train_age, test_age

def prepare_multimodal_data(train_data, test_data, train_age, test_age):
    """准备多模态数据 - 数据预处理和标准化"""
    modalities = ['eye', 'artery', 'physiology', 'ren']
    
    # 准备返回的数据结构
    X_train_dict = {}
    X_test_dict = {}
    scalers = {}
    
    # 设置年龄标签索引
    train_age.set_index('samples', inplace=True)
    test_age.set_index('samples', inplace=True)
    
    # 提取年龄标签（除以100，恢复为实际年龄单位）
    age_column = 'age_at_study_date_x100_resurvey3'
    y_train = train_age[age_column].values / 100.0
    y_test = test_age[age_column].values / 100.0
    
    # 处理每个模态的数据
    for modality in modalities:
        # 设置样本ID为索引
        train_modality = train_data[modality].copy()
        test_modality = test_data[modality].copy()
        
        train_modality.set_index('samples', inplace=True)
        test_modality.set_index('samples', inplace=True)
        
        # 确保训练集和测试集使用相同的特征
        common_features = train_modality.columns.intersection(test_modality.columns)
        train_modality = train_modality[common_features]
        test_modality = test_modality[common_features]
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_dict[modality] = scaler.fit_transform(train_modality)
        X_test_dict[modality] = scaler.transform(test_modality)
        scalers[modality] = scaler
        
        print(f"{modality} 模态 - 训练集特征维度: {X_train_dict[modality].shape}")
    
    return X_train_dict, X_test_dict, y_train, y_test, scalers

# ============================================================================
# 校正项计算函数（从simple1_gpu.py导入）
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
# GPU支持检查函数（从simple1_gpu.py导入）
# ============================================================================

def check_gpu_availability():
    """检查GPU可用性"""
    gpu_available = False
    
    # 检查XGBoost GPU支持
    try:
        import xgboost as xgb
        # 尝试创建GPU模型
        test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=1)
        gpu_available = True
        print("✓ XGBoost GPU支持可用")
    except Exception as e:
        print(f"✗ XGBoost GPU支持不可用: {e}")
    
    # 检查LightGBM GPU支持
    try:
        import lightgbm as lgb
        # 尝试创建GPU模型
        test_model = lgb.LGBMRegressor(device='gpu', n_estimators=1)
        print("✓ LightGBM GPU支持可用")
    except Exception as e:
        print(f"✗ LightGBM GPU支持不可用: {e}")
    
    # 检查CatBoost GPU支持
    try:
        import catboost as cb
        # 尝试创建GPU模型
        test_model = cb.CatBoostRegressor(task_type='GPU', iterations=1)
        print("✓ CatBoost GPU支持可用")
    except Exception as e:
        print(f"✗ CatBoost GPU支持不可用: {e}")
    
    return gpu_available

def create_gpu_models():
    """创建支持GPU的基础模型"""
    gpu_available = check_gpu_availability()
    
    models = {
        'XGBoost': xgb.XGBRegressor(
            random_state=42, 
            n_jobs=-1,
            tree_method='gpu_hist' if gpu_available else 'hist',
            gpu_id=0 if gpu_available else None,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        ),
        'LightGBM': lgb.LGBMRegressor(
            random_state=42, 
            n_jobs=-1, 
            verbose=-1,
            device='gpu' if gpu_available else 'cpu',
            gpu_platform_id=0 if gpu_available else None,
            gpu_device_id=0 if gpu_available else None,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        ),
        'CatBoost': cb.CatBoostRegressor(
            random_state=42, 
            verbose=False, 
            task_type='GPU' if gpu_available else 'CPU',
            devices='0' if gpu_available else None,
            iterations=100,
            depth=6,
            learning_rate=0.1
        ),
        'RandomForest': RandomForestRegressor(
            random_state=42, 
            n_jobs=-1,
            n_estimators=100,
            max_depth=8
        ),
        'GradientBoosting': GradientBoostingRegressor(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    }
    
    return models

# ============================================================================
# PB-MVBoost核心实现
# ============================================================================

class PBMVBoost(BaseEstimator, RegressorMixin):
    """
    Probabilistic Boosting Multi-View Boost (PB-MVBoost) 回归器
    
    这是一个多模态boosting算法，能够处理多个视图/模态的数据，
    通过概率权重动态调整每个模态的贡献。
    """
    
    def __init__(self, n_estimators=50, learning_rate=0.1, base_models=None, 
                 modality_weight_update='adaptive', regularization=0.01,
                 min_modality_weight=0.05, random_state=None, verbose=True):
        """
        初始化PB-MVBoost
        
        参数:
        - n_estimators: boosting轮数
        - learning_rate: 学习率
        - base_models: 基础模型字典，键为模型名，值为模型实例
        - modality_weight_update: 模态权重更新策略 ('adaptive', 'uniform')
        - regularization: 正则化参数
        - min_modality_weight: 模态最小权重
        - random_state: 随机种子
        - verbose: 是否打印详细信息
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_models = base_models or create_gpu_models()
        self.modality_weight_update = modality_weight_update
        self.regularization = regularization
        self.min_modality_weight = min_modality_weight
        self.random_state = random_state
        self.verbose = verbose
        
        # 初始化存储结构
        self.modality_estimators_ = {}  # 存储每个模态的估计器
        self.modality_weights_ = {}     # 存储每个模态的权重历史
        self.feature_importances_ = {}  # 存储特征重要性
        self.training_history_ = []     # 存储训练历史
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_modality_weights(self, modalities):
        """初始化模态权重"""
        n_modalities = len(modalities)
        initial_weight = 1.0 / n_modalities
        
        for modality in modalities:
            self.modality_weights_[modality] = []
            self.modality_estimators_[modality] = []
    
    def _update_modality_weights(self, modality_errors, current_weights):
        """更新模态权重"""
        if self.modality_weight_update == 'uniform':
            # 均匀权重
            n_modalities = len(modality_errors)
            return {mod: 1.0/n_modalities for mod in modality_errors.keys()}
        
        elif self.modality_weight_update == 'adaptive':
            # 基于MAE的自适应权重更新
            # MAE越小，权重越大
            inverse_errors = {}
            for mod, error in modality_errors.items():
                # 添加小常数避免除零
                inverse_errors[mod] = 1.0 / (error + 1e-8)
            
            # 归一化权重
            total_inverse_error = sum(inverse_errors.values())
            new_weights = {}
            
            for mod in modality_errors.keys():
                new_weight = inverse_errors[mod] / total_inverse_error
                # 应用最小权重约束
                new_weight = max(new_weight, self.min_modality_weight)
                new_weights[mod] = new_weight
            
            # 重新归一化以确保权重和为1
            total_weight = sum(new_weights.values())
            for mod in new_weights:
                new_weights[mod] /= total_weight
            
            return new_weights
        
        else:
            raise ValueError(f"未知的权重更新策略: {self.modality_weight_update}")
    
    def _calculate_weighted_residuals(self, y_true, predictions, modality_weights):
        """计算加权残差"""
        # 计算加权预测
        weighted_pred = np.zeros_like(y_true, dtype=float)
        
        for modality, pred in predictions.items():
            weight = modality_weights[modality]
            weighted_pred += weight * pred
        
        # 计算残差
        residuals = y_true - weighted_pred
        return residuals, weighted_pred
    
    def fit(self, X_dict, y):
        """
        训练PB-MVBoost模型
        
        参数:
        - X_dict: 字典，键为模态名，值为对应的特征矩阵
        - y: 目标变量
        """
        modalities = list(X_dict.keys())
        self._initialize_modality_weights(modalities)
        
        if self.verbose:
            print(f"开始训练PB-MVBoost模型")
            print(f"模态数量: {len(modalities)}")
            print(f"模态列表: {modalities}")
            print(f"训练样本数: {len(y)}")
            print(f"Boosting轮数: {self.n_estimators}")
        
        # 初始化模态权重
        current_modality_weights = {mod: 1.0/len(modalities) for mod in modalities}
        
        # 初始化残差
        current_residuals = y.copy()
        
        for round_idx in range(self.n_estimators):
            if self.verbose:
                print(f"\n第 {round_idx + 1}/{self.n_estimators} 轮训练")
            
            round_start_time = time.time()
            round_predictions = {}
            round_errors = {}
            
            # 为每个模态训练模型
            for modality in modalities:
                if self.verbose:
                    print(f"  训练 {modality} 模态...")
                
                # 选择基础模型（循环使用）
                model_names = list(self.base_models.keys())
                base_model_name = model_names[round_idx % len(model_names)]
                base_model = clone(self.base_models[base_model_name])
                
                # 训练模型预测残差
                X_modality = X_dict[modality]
                base_model.fit(X_modality, current_residuals)
                
                # 预测
                pred = base_model.predict(X_modality)
                round_predictions[modality] = pred
                
                # 计算模态误差 (使用MAE)
                modality_error = mean_absolute_error(current_residuals, pred)
                round_errors[modality] = modality_error
                
                # 存储模型
                self.modality_estimators_[modality].append({
                    'model': base_model,
                    'base_model_name': base_model_name,
                    'round': round_idx
                })
                
                if self.verbose:
                    print(f"    使用模型: {base_model_name}")
                    print(f"    MAE: {modality_error:.6f}")
            
            # 更新模态权重
            current_modality_weights = self._update_modality_weights(
                round_errors, current_modality_weights
            )
            
            # 存储当前轮的权重
            for modality in modalities:
                self.modality_weights_[modality].append(current_modality_weights[modality])
            
            # 计算加权残差用于下一轮
            residuals, weighted_pred = self._calculate_weighted_residuals(
                current_residuals, round_predictions, current_modality_weights
            )
            
            # 应用学习率
            update = self.learning_rate * weighted_pred
            current_residuals = current_residuals - update
            
            # 计算整体误差 (同时保留MSE和MAE)
            overall_mse = mean_squared_error(y, y - current_residuals)
            overall_mae = mean_absolute_error(y, y - current_residuals)
            
            round_time = time.time() - round_start_time
            
            # 记录训练历史
            history_entry = {
                'round': round_idx,
                'modality_weights': current_modality_weights.copy(),
                'modality_errors': round_errors.copy(),
                'overall_mse': overall_mse,
                'overall_mae': overall_mae,
                'time': round_time
            }
            self.training_history_.append(history_entry)
            
            if self.verbose:
                print(f"  模态权重: {', '.join([f'{mod}: {w:.3f}' for mod, w in current_modality_weights.items()])}")
                print(f"  整体 MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}")
                print(f"  用时: {round_time:.2f}秒")
        
        if self.verbose:
            print(f"\nPB-MVBoost训练完成!")
        
        return self
    
    def predict(self, X_dict):
        """
        使用训练好的模型进行预测
        
        参数:
        - X_dict: 字典，键为模态名，值为对应的特征矩阵
        
        返回:
        - 预测结果
        """
        modalities = list(X_dict.keys())
        n_samples = len(X_dict[modalities[0]])
        
        # 初始化预测结果
        final_predictions = np.zeros(n_samples)
        
        # 对每一轮的预测进行加权累加
        for round_idx in range(len(self.training_history_)):
            round_predictions = {}
            round_weights = {}
            
            # 获取每个模态在当前轮的预测和权重
            for modality in modalities:
                if round_idx < len(self.modality_estimators_[modality]):
                    model_info = self.modality_estimators_[modality][round_idx]
                    model = model_info['model']
                    
                    # 预测
                    pred = model.predict(X_dict[modality])
                    round_predictions[modality] = pred
                    
                    # 获取权重
                    round_weights[modality] = self.modality_weights_[modality][round_idx]
            
            # 计算当前轮的加权预测
            round_weighted_pred = np.zeros(n_samples)
            for modality, pred in round_predictions.items():
                weight = round_weights[modality]
                round_weighted_pred += weight * pred
            
            # 应用学习率并累加到最终预测
            final_predictions += self.learning_rate * round_weighted_pred
        
        return final_predictions
    
    def get_modality_importance(self):
        """获取模态重要性"""
        modalities = list(self.modality_weights_.keys())
        importance = {}
        
        for modality in modalities:
            # 计算平均权重作为重要性
            avg_weight = np.mean(self.modality_weights_[modality])
            importance[modality] = avg_weight
        
        return importance

# ============================================================================
# 评估函数
# ============================================================================

def evaluate_model(y_true, y_pred, model_name="Model"):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 计算相关系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"\n{model_name} 性能评估:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  相关系数: {correlation:.6f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation
    }

def save_summary_report(pb_mvboost, y_train, y_train_pred, y_test, y_test_pred,
                       y_train_pred_corrected, y_test_pred_corrected,
                       train_metrics, test_metrics, train_metrics_corrected, test_metrics_corrected,
                       modality_importance, training_time, output_file="pb_mvboost_report.txt"):
    """保存模型训练和评估结果摘要报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PB-MVBoost 多模态生物年龄预测模型 - 结果报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 数据集信息
        f.write("1. 数据集信息\n")
        f.write("-" * 50 + "\n")
        f.write(f"训练集样本数: {len(y_train)}\n")
        f.write(f"测试集样本数: {len(y_test)}\n")
        f.write(f"年龄范围 - 训练集: {y_train.min():.1f} - {y_train.max():.1f}\n")
        f.write(f"年龄范围 - 测试集: {y_test.min():.1f} - {y_test.max():.1f}\n\n")
        
        # 模型配置
        f.write("2. 模型配置\n")
        f.write("-" * 50 + "\n")
        f.write(f"Boosting轮数: {pb_mvboost.n_estimators}\n")
        f.write(f"学习率: {pb_mvboost.learning_rate}\n")
        f.write(f"权重更新策略: {pb_mvboost.modality_weight_update}\n")
        f.write(f"正则化参数: {pb_mvboost.regularization}\n")
        f.write(f"最小模态权重: {pb_mvboost.min_modality_weight}\n")
        f.write(f"总训练时间: {training_time:.2f}秒\n\n")
        
        # 模态重要性
        f.write("3. 模态重要性分析\n")
        f.write("-" * 50 + "\n")
        for modality, importance in sorted(modality_importance.items(), 
                                         key=lambda x: x[1], reverse=True):
            f.write(f"{modality}: {importance:.4f}\n")
        f.write("\n")
        
        # 原始预测性能
        f.write("4. 原始预测性能评估\n")
        f.write("-" * 50 + "\n")
        f.write("训练集:\n")
        f.write(f"  MSE: {train_metrics['mse']:.6f}\n")
        f.write(f"  MAE: {train_metrics['mae']:.6f}\n")
        f.write(f"  RMSE: {train_metrics['rmse']:.6f}\n")
        f.write(f"  R²: {train_metrics['r2']:.6f}\n")
        f.write(f"  相关系数: {train_metrics['correlation']:.6f}\n\n")
        
        f.write("测试集:\n")
        f.write(f"  MSE: {test_metrics['mse']:.6f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"  R²: {test_metrics['r2']:.6f}\n")
        f.write(f"  相关系数: {test_metrics['correlation']:.6f}\n\n")
        
        # 校正后预测性能
        f.write("5. 校正后预测性能评估\n")
        f.write("-" * 50 + "\n")
        f.write("训练集:\n")
        f.write(f"  MSE: {train_metrics_corrected['mse']:.6f}\n")
        f.write(f"  MAE: {train_metrics_corrected['mae']:.6f}\n")
        f.write(f"  RMSE: {train_metrics_corrected['rmse']:.6f}\n")
        f.write(f"  R²: {train_metrics_corrected['r2']:.6f}\n")
        f.write(f"  相关系数: {train_metrics_corrected['correlation']:.6f}\n\n")
        
        f.write("测试集:\n")
        f.write(f"  MSE: {test_metrics_corrected['mse']:.6f}\n")
        f.write(f"  MAE: {test_metrics_corrected['mae']:.6f}\n")
        f.write(f"  RMSE: {test_metrics_corrected['rmse']:.6f}\n")
        f.write(f"  R²: {test_metrics_corrected['r2']:.6f}\n")
        f.write(f"  相关系数: {test_metrics_corrected['correlation']:.6f}\n\n")
        
        # 校正参数
        train_z, train_b = calculate_correction_term(y_train, y_train_pred)
        test_z, test_b = calculate_correction_term(y_test, y_test_pred)
        f.write("6. 生物年龄校正参数\n")
        f.write("-" * 50 + "\n")
        f.write(f"训练集校正参数 b: {train_b:.6f}\n")
        f.write(f"测试集校正参数 b: {test_b:.6f}\n\n")
        
        # 训练历史摘要
        f.write("7. 训练历史摘要\n")
        f.write("-" * 50 + "\n")
        if pb_mvboost.training_history_:
            final_round = pb_mvboost.training_history_[-1]
            f.write(f"最终轮数: {final_round['round'] + 1}\n")
            f.write(f"最终MSE: {final_round['overall_mse']:.6f}\n")
            f.write(f"最终MAE: {final_round['overall_mae']:.6f}\n")
            f.write("最终模态权重:\n")
            for mod, weight in final_round['modality_weights'].items():
                f.write(f"  {mod}: {weight:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"结果报告已保存到 {output_file}")

# ============================================================================
# 主训练和评估流程
# ============================================================================

def main():
    """主函数"""
    print("=" * 80)
    print("PB-MVBoost 多模态生物年龄预测模型")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n1. 加载多模态数据...")
    train_data, test_data, train_age, test_age = load_multimodal_data()
    
    # 2. 预处理数据
    print("\n2. 数据预处理...")
    X_train_dict, X_test_dict, y_train, y_test, scalers = prepare_multimodal_data(
        train_data, test_data, train_age, test_age
    )
    
    print(f"训练集样本数: {len(y_train)}")
    print(f"测试集样本数: {len(y_test)}")
    print(f"年龄范围 - 训练集: {y_train.min():.1f} - {y_train.max():.1f}")
    print(f"年龄范围 - 测试集: {y_test.min():.1f} - {y_test.max():.1f}")
    
    # 3. 创建和训练PB-MVBoost模型
    print("\n3. 创建PB-MVBoost模型...")
    pb_mvboost = PBMVBoost(
        n_estimators=30,
        learning_rate=0.1,
        modality_weight_update='adaptive',
        regularization=0.01,
        min_modality_weight=0.05,
        random_state=42,
        verbose=True
    )
    
    print("\n4. 训练模型...")
    start_time = time.time()
    pb_mvboost.fit(X_train_dict, y_train)
    training_time = time.time() - start_time
    print(f"总训练时间: {training_time:.2f}秒")
    
    # 4. 预测
    print("\n5. 进行预测...")
    y_train_pred = pb_mvboost.predict(X_train_dict)
    y_test_pred = pb_mvboost.predict(X_test_dict)
    
    # 5. 应用校正
    print("\n6. 应用生物年龄校正...")
    
    # 训练集校正
    y_train_pred_corrected, train_z, train_b = calculate_corrected_biological_age(
        y_train, y_train_pred
    )
    
    # 测试集校正
    y_test_pred_corrected, test_z, test_b = calculate_corrected_biological_age(
        y_test, y_test_pred
    )
    
    print(f"训练集校正参数 b: {train_b:.6f}")
    print(f"测试集校正参数 b: {test_b:.6f}")
    
    # 6. 评估模型
    print("\n7. 模型评估")
    print("=" * 50)
    
    # 原始预测评估
    train_metrics = evaluate_model(y_train, y_train_pred, "训练集 (原始预测)")
    test_metrics = evaluate_model(y_test, y_test_pred, "测试集 (原始预测)")
    
    # 校正后预测评估
    train_metrics_corrected = evaluate_model(y_train, y_train_pred_corrected, "训练集 (校正后)")
    test_metrics_corrected = evaluate_model(y_test, y_test_pred_corrected, "测试集 (校正后)")
    
    # 7. 模态重要性分析
    print("\n8. 模态重要性分析")
    print("=" * 50)
    modality_importance = pb_mvboost.get_modality_importance()
    for modality, importance in sorted(modality_importance.items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"{modality}: {importance:.4f}")
    
    # 8. 保存结果
    print("\n9. 保存结果...")
    
    # 生成综合报告
    save_summary_report(
        pb_mvboost, y_train, y_train_pred, y_test, y_test_pred,
        y_train_pred_corrected, y_test_pred_corrected,
        train_metrics, test_metrics, train_metrics_corrected, test_metrics_corrected,
        modality_importance, training_time, "pb_mvboost_biological_age_report.txt"
    )
    
    print("\n模型训练和评估完成!")
    print("=" * 80)
    
    return pb_mvboost, train_metrics_corrected, test_metrics_corrected

if __name__ == "__main__":
    model, train_metrics, test_metrics = main()
