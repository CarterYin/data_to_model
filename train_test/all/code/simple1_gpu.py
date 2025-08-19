import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# ============================================================================
# 数据加载和预处理函数
# ============================================================================

def load_data():
    """加载训练和测试数据"""
    # 加载特征数据
    train_features = pd.read_csv('../train_all.tsv', sep='\t')
    test_features = pd.read_csv('../test_all.tsv', sep='\t')
    
    # 加载年龄标签
    train_age = pd.read_csv('../train_age.tsv', sep='\t')
    test_age = pd.read_csv('../test_age.tsv', sep='\t')
    
    return train_features, test_features, train_age, test_age

def prepare_data(train_features, test_features, train_age, test_age):
    """准备训练数据 - 数据预处理和标准化"""
    # 设置样本ID为索引
    train_features.set_index('samples', inplace=True)
    test_features.set_index('samples', inplace=True)
    train_age.set_index('samples', inplace=True)
    test_age.set_index('samples', inplace=True)
    
    # 确保训练集和测试集使用相同的特征
    common_features = train_features.columns.intersection(test_features.columns)
    train_features = train_features[common_features]
    test_features = test_features[common_features]
    
    # 检查年龄标签列
    age_column = 'age_at_study_date_x100_resurvey3'
    
    # 提取年龄标签（除以100，恢复为实际年龄单位）
    y_train = train_age[age_column].values / 100.0
    y_test = test_age[age_column].values / 100.0
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)
    
    return X_train, X_test, y_train, y_test, scaler

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
# 模型定义和训练函数
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

def create_models_with_params_gpu():
    """创建支持GPU的模型和对应的超参数网格"""
    gpu_available = check_gpu_availability()
    
    models_and_params = {
        'XGBoost': {
            'model': xgb.XGBRegressor(
                random_state=42, 
                n_jobs=-1,
                tree_method='gpu_hist' if gpu_available else 'hist',
                gpu_id=0 if gpu_available else None
            ),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(
                random_state=42, 
                n_jobs=-1, 
                verbose=-1,
                device='gpu' if gpu_available else 'cpu',
                gpu_platform_id=0 if gpu_available else None,
                gpu_device_id=0 if gpu_available else None
            ),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        },
        'CatBoost': {
            'model': cb.CatBoostRegressor(
                random_state=42, 
                verbose=False, 
                task_type='GPU' if gpu_available else 'CPU',
                devices='0' if gpu_available else None
            ),
            'params': {
                'iterations': [50, 100, 200],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        },
        'Random_Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [6, 8, 10, None]
            }
        }
    }
    
    print(f"\nGPU加速状态: {'启用' if gpu_available else '未启用'}")
    return models_and_params

def _pearson_corr_for_scorer(y_true, y_pred):
    """用于交叉验证评估的皮尔逊相关系数打分函数。"""
    # 处理方差为0或长度过短导致的异常
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    try:
        return float(stats.pearsonr(y_true, y_pred)[0])
    except Exception:
        return 0.0

def hyperparameter_tuning_gpu(X_train, y_train):
    """Step 3: 使用5折交叉验证进行超参数调优，基于MAE选择最佳模型（GPU加速）

    说明：
    - 仍以MAE作为模型选择标准；
    - 同时计算MSE、RMSE、R²、相关系数在5折上的均值，供后续报告使用；
    - 不再在完整训练集上再次训练最佳模型，直接使用GridSearchCV的best_estimator_。
    """
    models_and_params = create_models_with_params_gpu()
    best_models = {}
    
    # 构建多指标评分（以MAE为选择标准）
    mae_scorer = 'neg_mean_absolute_error'
    scoring = {
        'mae': mae_scorer,
        'mse': 'neg_mean_squared_error',
        'r2': 'r2',
        'corr': make_scorer(_pearson_corr_for_scorer, greater_is_better=True)
    }
    
    print("开始超参数调优（GPU加速）...")
    for name, model_info in models_and_params.items():
        print(f"正在调优 {name}...")
        
        # 使用GridSearchCV进行超参数搜索（基于MAE）
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=5,  # 5折交叉验证
            scoring=scoring,
            refit='mae',  # 以MAE作为最终模型选择与refit标准
            n_jobs=1,  # GPU模式下使用单线程
            verbose=0,
            return_train_score=True
        )
        
        # 训练模型
        grid_search.fit(X_train, y_train)
        
        # 保存最佳模型、详细结果及5折均值指标（最佳参数对应）
        best_idx = grid_search.best_index_
        cv = grid_search.cv_results_
        mse_mean = -float(cv['mean_test_mse'][best_idx])
        cv_mean_results = {
            'mae': -float(cv['mean_test_mae'][best_idx]),
            'mse': mse_mean,
            'rmse': float(np.sqrt(mse_mean)),
            'r2': float(cv['mean_test_r2'][best_idx]),
            'correlation': float(cv['mean_test_corr'][best_idx])
        }
        best_models[name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': -grid_search.best_score_,  # 转换回MAE（正值）
            'cv_results': grid_search.cv_results_,
            'cv_mean_results': cv_mean_results
        }
        
        print(f"  {name} 最佳参数: {grid_search.best_params_}")
        print(f"  {name} 最佳MAE: {-grid_search.best_score_:.4f}")
    
    return best_models

def train_all_optimal_models(X_train, y_train, best_models):
    """Step 4: 在完整训练集上训练每种模型的最优模型"""
    print("Step 4: 在完整训练集上训练每种模型的最优模型...")
    trained_models = {}
    
    for name, model_info in best_models.items():
        print(f"正在训练 {name} 的最优模型...")
        
        # 克隆最佳模型并在完整训练集上训练
        optimal_model = clone(model_info['model'])
        optimal_model.fit(X_train, y_train)
        
        # 在训练集上进行预测
        y_train_pred = optimal_model.predict(X_train)
        
        # 计算训练集上的评估指标
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_correlation, _ = stats.pearsonr(y_train, y_train_pred)
        
        # 保存训练结果
        trained_models[name] = {
            'model': optimal_model,
            'best_params': model_info['best_params'],
            'best_cv_score': model_info['best_cv_score'],
            'train_results': {
                'mse': train_mse,
                'rmse': train_rmse,
                'mae': train_mae,
                'r2': train_r2,
                'correlation': train_correlation,
                'predictions': y_train_pred
            }
        }
        
        print(f"  {name} 训练集MAE: {train_mae:.4f}")
        print(f"  {name} 训练集R²: {train_r2:.4f}")
    
    return trained_models

def evaluate_all_models_on_test(X_test, y_test, trained_models):
    """Step 6: 在测试集上评估所有模型的最优模型性能，包括校正后的相关性"""
    print("Step 6: 在测试集上评估所有模型的最优模型性能...")
    test_results = {}
    
    for name, model_info in trained_models.items():
        print(f"正在评估 {name} 在测试集上的性能...")
        
        # 在测试集上进行预测
        y_test_pred = model_info['model'].predict(X_test)
        
        # 计算测试集上的评估指标
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_correlation, _ = stats.pearsonr(y_test, y_test_pred)
        
        # 计算校正后的生物年龄与日历年龄的相关性
        # CA = y_test (日历年龄), BA = y_test_pred (预测的生物年龄)
        BAc, z, b = calculate_corrected_biological_age(y_test, y_test_pred)
        corrected_correlation, _ = stats.pearsonr(y_test, BAc)
        
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
            'corrected_correlation': corrected_correlation
        }
        
        print(f"  {name} 测试集MAE: {test_mae:.4f}")
        print(f"  {name} 测试集R²: {test_r2:.4f}")
        print(f"  {name} 原始相关性: {test_correlation:.4f}")
        print(f"  {name} 校正后相关性: {corrected_correlation:.4f}")
        print(f"  {name} 回归系数b: {b:.4f}")
    
    return test_results

def select_best_model(trained_models, test_results):
    """选择测试集MAE最小的最佳模型"""
    best_model_name = min(test_results.items(), key=lambda x: x[1]['mae'])[0]
    best_model = trained_models[best_model_name]['model']
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}")
    
    return best_model_name, best_model, trained_models, test_results

# ============================================================================
# 结果保存函数
# ============================================================================

def save_results(trained_models, test_results, best_model_name):
    """保存所有模型的交叉验证均值和测试集完整结果，包括校正后的相关性"""
    # 保存详细报告
    with open('hyperparameter_tuning_gpu_report.txt', 'w', encoding='utf-8') as f:
        f.write("年龄预测模型超参数调优报告（GPU加速版本）\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("GPU加速配置:\n")
        f.write("-" * 20 + "\n")
        f.write("XGBoost: tree_method='gpu_hist' (如果GPU可用)\n")
        f.write("LightGBM: device='gpu' (如果GPU可用)\n")
        f.write("CatBoost: task_type='GPU' (如果GPU可用)\n")
        f.write("Random Forest: CPU only (scikit-learn不支持GPU)\n\n")
        
        f.write("校正项计算说明:\n")
        f.write("-" * 25 + "\n")
        f.write("校正公式: BAc = BA + z\n")
        f.write("其中: z = (CA - MeanCA) * (1 - b)\n")
        f.write("CA: 日历年龄 (Chronological Age)\n")
        f.write("BA: 生物年龄 (Biological Age)\n")
        f.write("BAc: 校正后的生物年龄\n")
        f.write("MeanCA: 样本平均日历年龄\n")
        f.write("b: BA对CA的线性回归系数\n\n")
        
        f.write("Step 3: 5折交叉验证超参数调优结果\n")
        f.write("-" * 50 + "\n\n")
        
        # 保存每个模型的超参数调优结果
        for name, model_info in trained_models.items():
            f.write(f"{name} 超参数调优结果:\n")
            f.write("=" * 40 + "\n")
            f.write(f"最佳参数: {model_info['best_params']}\n")
            f.write(f"交叉验证MAE: {model_info['best_cv_score']:.4f}\n")
            f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("Step 4: 5折交叉验证均值（最佳参数）\n")
        f.write("-" * 40 + "\n\n")
        
        # 保存每个模型在交叉验证中的均值结果（用于替代原训练集结果）
        for name, model_info in trained_models.items():
            cv_mean = model_info.get('cv_mean_results', {})
            f.write(f"{name} 交叉验证均值:\n")
            f.write("-" * 25 + "\n")
            f.write(f"  最佳参数: {model_info['best_params']}\n")
            f.write(f"  MSE: {cv_mean.get('mse', float('nan')):.4f}\n")
            f.write(f"  RMSE: {cv_mean.get('rmse', float('nan')):.4f}\n")
            f.write(f"  MAE: {cv_mean.get('mae', float('nan')):.4f}\n")
            f.write(f"  R²: {cv_mean.get('r2', float('nan')):.4f}\n")
            f.write(f"  相关系数: {cv_mean.get('correlation', float('nan')):.4f}\n")
            f.write("\n")
        
        f.write("Step 6: 测试集评估结果（包含校正后相关性）\n")
        f.write("-" * 45 + "\n\n")
        
        # 保存每个模型在测试集上的结果
        for name, test_info in test_results.items():
            f.write(f"{name} 测试集结果:\n")
            f.write("-" * 25 + "\n")
            f.write(f"  最佳参数: {trained_models[name]['best_params']}\n")
            f.write(f"  MSE: {test_info['mse']:.4f}\n")
            f.write(f"  RMSE: {test_info['rmse']:.4f}\n")
            f.write(f"  MAE: {test_info['mae']:.4f}\n")
            f.write(f"  R²: {test_info['r2']:.4f}\n")
            f.write(f"  原始相关系数: {test_info['correlation']:.4f}\n")
            f.write(f"  校正后相关系数: {test_info['corrected_correlation']:.4f}\n")
            f.write(f"  回归系数b: {test_info['regression_coefficient']:.4f}\n")
            f.write(f"  校正项z范围: [{np.min(test_info['correction_term']):.4f}, {np.max(test_info['correction_term']):.4f}]\n")
            f.write("\n")
        
        # 保存最终选择的最佳模型信息
        f.write("最终模型选择结果:\n")
        f.write("-" * 25 + "\n")
        f.write(f"选择的最佳模型: {best_model_name}\n")
        f.write(f"选择标准: 测试集最小MAE\n")
        f.write(f"最佳参数: {trained_models[best_model_name]['best_params']}\n")
        f.write(f"交叉验证MAE: {trained_models[best_model_name]['cv_mean_results']['mae']:.4f}\n")
        f.write(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}\n")
        f.write(f"交叉验证R²: {trained_models[best_model_name]['cv_mean_results']['r2']:.4f}\n")
        f.write(f"测试集R²: {test_results[best_model_name]['r2']:.4f}\n")
        f.write(f"原始相关性: {test_results[best_model_name]['correlation']:.4f}\n")
        f.write(f"校正后相关性: {test_results[best_model_name]['corrected_correlation']:.4f}\n\n")
        
        f.write("模型说明:\n")
        f.write("1. XGBoost - 梯度提升树 (GPU加速)\n")
        f.write("2. LightGBM - 轻量级梯度提升机 (GPU加速)\n")
        f.write("3. CatBoost - 类别特征梯度提升 (GPU加速)\n")
        f.write("4. Random Forest - 随机森林 (CPU only)\n\n")
        
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
        f.write("GPU加速可以显著提升训练速度，特别是对于大型数据集\n")
        f.write("校正后的生物年龄(BAc)通过添加校正项z来减少与日历年龄的系统性偏差\n")
        
        # 添加汇总表格
        f.write("\n" + "=" * 100 + "\n")
        f.write("模型性能汇总表（包含校正后相关性）\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'模型名称':<15} {'CV_MAE':<12} {'测试集MAE':<12} {'CV_R²':<12} {'测试集R²':<12} {'原始相关性':<12} {'校正后相关性':<12} {'回归系数b':<12}\n")
        f.write("-" * 100 + "\n")
        
        for name in trained_models.keys():
            cv_mean = trained_models[name]['cv_mean_results']
            test_info = test_results[name]
            f.write(f"{name:<15} {cv_mean['mae']:<12.4f} {test_info['mae']:<12.4f} {cv_mean['r2']:<12.4f} {test_info['r2']:<12.4f} {test_info['correlation']:<12.4f} {test_info['corrected_correlation']:<12.4f} {test_info['regression_coefficient']:<12.4f}\n")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 执行完整的年龄预测分析流程（GPU加速版本）"""
    print("开始年龄预测模型训练和评估（GPU加速版本）...")
    
    # Step 1 & 2: 数据加载和预处理
    print("Step 1-2: 数据加载和预处理...")
    train_features, test_features, train_age, test_age = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        train_features, test_features, train_age, test_age
    )
    
    # Step 3: 使用5折交叉验证进行超参数调优（GPU加速），以MAE选最佳参数
    best_models = hyperparameter_tuning_gpu(X_train, y_train)
    
    # 不再在完整训练集上再次训练，直接使用CV选出的最佳模型进行测试
    trained_models = best_models
    
    # Step 5: 使用基于训练集的scaler重新缩放测试集
    print("Step 5: 使用训练集scaler重新缩放测试集...")
    # 注意：scaler已经在prepare_data中基于训练集拟合，这里直接使用
    # X_test已经在prepare_data中被正确缩放
    
    # Step 6: 在测试集上评估所有模型（使用CV选出的最佳模型）
    test_results = evaluate_all_models_on_test(X_test, y_test, trained_models)
    
    # 选择最佳模型
    best_model_name, best_model, trained_models, test_results = select_best_model(trained_models, test_results)
    
    # 保存结果
    save_results(trained_models, test_results, best_model_name)
    
    print(f"\n训练完成！最佳模型: {best_model_name}")
    print(f"测试集MAE: {test_results[best_model_name]['mae']:.4f}")
    print(f"测试集R²: {test_results[best_model_name]['r2']:.4f}")
    print(f"原始相关性: {test_results[best_model_name]['correlation']:.4f}")
    print(f"校正后相关性: {test_results[best_model_name]['corrected_correlation']:.4f}")
    print(f"回归系数b: {test_results[best_model_name]['regression_coefficient']:.4f}")
    print("详细结果已保存到 hyperparameter_tuning_gpu_report.txt")
    print("\n校正项计算说明:")
    print("BAc = BA + z, 其中 z = (CA - MeanCA) * (1 - b)")
    print("CA: 日历年龄, BA: 生物年龄, BAc: 校正后的生物年龄")
    print("MeanCA: 样本平均日历年龄, b: BA对CA的线性回归系数")

if __name__ == "__main__":
    main()
