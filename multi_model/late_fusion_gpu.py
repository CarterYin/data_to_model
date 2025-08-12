import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from scipy import stats

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


# ============================================================================
# 数据加载与预处理（按模态）
# ============================================================================

def _read_modal_files(modal: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取指定模态的训练/测试特征和年龄标签。

    参数:
        modal: 'physiology' | 'artery' | 'eye'
    返回:
        (train_features, test_features, train_age, test_age)
    """
    modal_to_train = {
        'physiology': 'train_set_physiology.tsv',
        'artery': 'train_set_artery.tsv',
        'eye': 'train_set_eye.tsv',
    }
    modal_to_test = {
        'physiology': 'test_set_physiology.tsv',
        'artery': 'test_set_artery.tsv',
        'eye': 'test_set_eye.tsv',
    }

    if modal not in modal_to_train:
        raise ValueError(f"未知模态: {modal}")

    train_features = pd.read_csv(modal_to_train[modal], sep='\t')
    test_features = pd.read_csv(modal_to_test[modal], sep='\t')
    train_age = pd.read_csv('train_set_age.tsv', sep='\t')
    test_age = pd.read_csv('test_set_age.tsv', sep='\t')

    return train_features, test_features, train_age, test_age


def prepare_modal_data(modal: str):
    """准备某个模态的数据：对齐、标准化并返回矩阵与缩放器。"""
    train_features, test_features, train_age, test_age = _read_modal_files(modal)

    # 对齐索引
    train_features.set_index('samples', inplace=True)
    test_features.set_index('samples', inplace=True)
    train_age.set_index('samples', inplace=True)
    test_age.set_index('samples', inplace=True)

    # 对齐训练/测试共有特征
    common_features = train_features.columns.intersection(test_features.columns)
    train_features = train_features[common_features]
    test_features = test_features[common_features]

    age_column = 'age_at_study_date_x100_resurvey3'
    y_train = train_age[age_column].values / 100.0
    y_test = test_age[age_column].values / 100.0

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    return X_train, X_test, y_train, y_test, scaler


# ============================================================================
# 校正项与校正生物年龄
# ============================================================================

def calculate_correction_term(CA: np.ndarray, BA: np.ndarray):
    MeanCA = np.mean(CA)
    reg = LinearRegression()
    reg.fit(CA.reshape(-1, 1), BA)
    b = float(reg.coef_[0])
    z = (CA - MeanCA) * (1 - b)
    return z, b


def calculate_corrected_biological_age(CA: np.ndarray, BA: np.ndarray):
    z, b = calculate_correction_term(CA, BA)
    BAc = BA + z
    return BAc, z, b


# ============================================================================
# 模型构建（决策树家族，支持GPU）
# ============================================================================

def check_gpu_availability() -> bool:
    gpu_available = False
    try:
        _ = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=1)
        gpu_available = True
        print("✓ XGBoost GPU支持可用")
    except Exception as e:
        print(f"✗ XGBoost GPU支持不可用: {e}")

    try:
        _ = lgb.LGBMRegressor(device='gpu', n_estimators=1)
        print("✓ LightGBM GPU支持可用")
    except Exception as e:
        print(f"✗ LightGBM GPU支持不可用: {e}")

    try:
        _ = cb.CatBoostRegressor(task_type='GPU', iterations=1)
        print("✓ CatBoost GPU支持可用")
    except Exception as e:
        print(f"✗ CatBoost GPU支持不可用: {e}")

    return gpu_available


def create_models_with_params_gpu() -> Dict[str, Dict]:
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
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1]
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
                'n_estimators': [100, 200],
                'max_depth': [-1, 6, 8],
                'learning_rate': [0.05, 0.1]
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
                'iterations': [200, 400],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1]
            }
        },
        'Random_Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [200, 400],
                'max_depth': [8, 12, None]
            }
        }
    }
    print(f"\nGPU加速状态: {'启用' if gpu_available else '未启用'}")
    return models_and_params


def _pearson_corr_for_scorer(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    try:
        return float(stats.pearsonr(y_true, y_pred)[0])
    except Exception:
        return 0.0


def tune_and_select_best_model(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """对单一模态进行模型与超参搜索，返回该模态的全局最优模型信息。"""
    models_and_params = create_models_with_params_gpu()

    mae_scorer = 'neg_mean_absolute_error'
    scoring = {
        'mae': mae_scorer,
        'mse': 'neg_mean_squared_error',
        'r2': 'r2',
        'corr': make_scorer(_pearson_corr_for_scorer, greater_is_better=True)
    }

    best_overall = None
    print("开始超参数调优（单模态）...")
    for name, model_info in models_and_params.items():
        print(f"  正在调优 {name} ...")
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=5,
            scoring=scoring,
            refit='mae',
            n_jobs=1,
            verbose=0,
            return_train_score=False
        )
        grid_search.fit(X_train, y_train)

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

        candidate = {
            'algo_name': name,
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': -grid_search.best_score_,  # MAE
            'cv_mean_results': cv_mean_results
        }
        print(f"    {name} 最佳参数: {grid_search.best_params_}")
        print(f"    {name} 最佳MAE: {-grid_search.best_score_:.4f}")

        if (best_overall is None) or (candidate['best_cv_score'] < best_overall['best_cv_score']):
            best_overall = candidate

    print(f"该模态最优: {best_overall['algo_name']}  | CV_MAE={best_overall['best_cv_score']:.4f}")
    return best_overall


# ============================================================================
# 晚期融合：学习权重并融合预测
# ============================================================================

def get_oof_predictions(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """基于交叉验证的OOF预测，用于学习融合权重。"""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # 使用克隆，避免污染原模型
    estimator = clone(model)
    oof_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=1, method='predict')
    return oof_pred.astype(float)


def learn_fusion_weights(oof_pred_dict: Dict[str, np.ndarray], y_train: np.ndarray) -> Dict[str, float]:
    """学习各模态的融合权重（非负、和为1）。"""
    modal_names = list(oof_pred_dict.keys())
    X_oof = np.vstack([oof_pred_dict[m] for m in modal_names]).T  # (N, M)

    # 先尝试正约束回归（若环境不支持positive参数，则退化）
    try:
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X_oof, y_train)
        raw_weights = reg.coef_.astype(float)
    except TypeError:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_oof, y_train)
        raw_weights = np.maximum(reg.coef_.astype(float), 0.0)

    # 归一化为概率分布
    weight_sum = float(np.sum(raw_weights))
    if weight_sum <= 0:
        weights = np.ones_like(raw_weights) / len(raw_weights)
    else:
        weights = raw_weights / weight_sum

    return {m: float(w) for m, w in zip(modal_names, weights)}


# ============================================================================
# 评估与报告
# ============================================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = float(stats.pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0
    return {
        'mse': float(mse),
        'rmse': rmse,
        'mae': float(mae),
        'r2': float(r2),
        'correlation': corr,
    }


def save_late_fusion_report(
    modal_best: Dict[str, Dict],
    fusion_weights: Dict[str, float],
    test_metrics: Dict[str, float],
    corrected_corr: float,
    reg_b: float,
    z_range: Tuple[float, float]
):
    with open('late_fusion_gpu_report.txt', 'w', encoding='utf-8') as f:
        f.write("多模态模型级晚期融合报告（决策树家族，GPU加速）\n")
        f.write("=" * 60 + "\n\n")

        f.write("各模态最优模型：\n")
        f.write("-" * 20 + "\n")
        for modal, info in modal_best.items():
            f.write(f"模态: {modal}\n")
            f.write(f"  最优算法: {info['algo_name']}\n")
            f.write(f"  最佳参数: {info['best_params']}\n")
            f.write(f"  CV_MAE: {info['best_cv_score']:.4f}\n")
            f.write(f"  CV_R²: {info['cv_mean_results']['r2']:.4f}\n")
            f.write(f"  CV_相关: {info['cv_mean_results']['correlation']:.4f}\n\n")

        f.write("融合权重（和为1）：\n")
        f.write("-" * 20 + "\n")
        for modal, w in fusion_weights.items():
            f.write(f"  {modal}: {w:.4f}\n")
        f.write("\n")

        f.write("测试集融合结果：\n")
        f.write("-" * 20 + "\n")
        f.write(f"  MSE: {test_metrics['mse']:.4f}\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"  R²: {test_metrics['r2']:.4f}\n")
        f.write(f"  原始相关系数: {test_metrics['correlation']:.4f}\n")
        f.write(f"  校正后相关系数: {corrected_corr:.4f}\n")
        f.write(f"  回归系数b: {reg_b:.4f}\n")
        f.write(f"  校正项z范围: [{z_range[0]:.4f}, {z_range[1]:.4f}]\n")


# ============================================================================
# 主流程：每个模态独立训练 + 学习融合权重 + 测试集融合
# ============================================================================

def main():
    print("开始多模态模型级晚期融合（决策树家族，GPU加速）...")
    modals = ['physiology', 'artery', 'eye']

    # 1) 为每个模态准备数据
    modal_data = {}
    for m in modals:
        print(f"\n准备模态 {m} 数据...")
        X_train, X_test, y_train, y_test, _ = prepare_modal_data(m)
        modal_data[m] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }

    # 确保所有模态共享同一y（顺序一致）
    # 这里默认以第一个模态的y为全局y
    y_train_global = modal_data[modals[0]]['y_train']
    y_test_global = modal_data[modals[0]]['y_test']

    # 2) 各模态独立调参并选择最优模型
    modal_best = {}
    for m in modals:
        print(f"\n模态 {m}: 模型搜索与选择最优...")
        best = tune_and_select_best_model(modal_data[m]['X_train'], y_train_global)
        modal_best[m] = best

    # 3) 基于OOF预测学习融合权重
    print("\n学习融合权重（基于训练集OOF预测）...")
    oof_preds = {}
    for m in modals:
        model = modal_best[m]['model']
        preds = get_oof_predictions(model, modal_data[m]['X_train'], y_train_global, n_splits=5)
        oof_preds[m] = preds
    fusion_weights = learn_fusion_weights(oof_preds, y_train_global)
    print("融合权重:")
    for m, w in fusion_weights.items():
        print(f"  {m}: {w:.4f}")

    # 4) 在完整训练集上拟合最优模型
    print("\n在完整训练集上重新拟合各模态最优模型...")
    for m in modals:
        estimator = clone(modal_best[m]['model'])
        estimator.fit(modal_data[m]['X_train'], y_train_global)
        modal_best[m]['fitted_model'] = estimator

    # 5) 测试集预测并进行加权融合
    print("\n在测试集上进行预测与融合...")
    modal_test_preds = {}
    for m in modals:
        modal_test_preds[m] = modal_best[m]['fitted_model'].predict(modal_data[m]['X_test']).astype(float)

    # 加权融合
    fused_pred = np.zeros_like(y_test_global, dtype=float)
    for m in modals:
        fused_pred += fusion_weights[m] * modal_test_preds[m]

    # 评估融合结果
    test_metrics = evaluate_predictions(y_test_global, fused_pred)
    BAc, z, b = calculate_corrected_biological_age(y_test_global, fused_pred)
    corrected_corr = float(stats.pearsonr(y_test_global, BAc)[0]) if len(y_test_global) > 1 else 0.0

    # 输出与保存报告
    print("\n测试集融合评估结果：")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  原始相关性: {test_metrics['correlation']:.4f}")
    print(f"  校正后相关性: {corrected_corr:.4f}")

    save_late_fusion_report(
        modal_best=modal_best,
        fusion_weights=fusion_weights,
        test_metrics=test_metrics,
        corrected_corr=corrected_corr,
        reg_b=float(b),
        z_range=(float(np.min(z)), float(np.max(z)))
    )

    print("\n详细结果已保存到 late_fusion_gpu_report.txt")


if __name__ == '__main__':
    main()


