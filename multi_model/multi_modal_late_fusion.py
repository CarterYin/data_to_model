#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采用四个模态（eye, artery, physiology, ren）进行晚期融合:
- eye模态: CatBoost
- artery模态: XGBoost  
- physiology模态: CatBoost
- ren模态: XGBoost

融合策略:
1. Stacking元学习器（增强特征+正则化+交叉验证）
2. Blending（holdout验证集权重学习）
"""

# 标准库
import os
import warnings

# 数据处理库
import pandas as pd
import numpy as np
from scipy import stats

# 机器学习库
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import clone

warnings.filterwarnings('ignore')

class MultiModalLateFusion:
    """多模态晚期融合生物学年龄预测器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.modalities = ['eye', 'artery', 'physiology', 'ren']
        self.base_models = {}
        self.scalers = {}
        self.meta_learner = None
        self.meta_scaler = None
        
        # 为每个模态定义对应的模型类型
        self.model_types = {
            'eye': 'catboost',
            'artery': 'xgboost', 
            'physiology': 'catboost',
            'ren': 'xgboost'
        }
    
    # ========================= 数据加载和预处理方法 =========================
        
    def load_data(self):
        """加载所有模态的训练和测试数据"""
        print("正在加载多模态数据...")
        
        data = {}
        
        for modality in self.modalities:
            # 加载特征数据
            train_features = pd.read_csv(f'train_{modality}.tsv', sep='\t')
            test_features = pd.read_csv(f'test_{modality}.tsv', sep='\t')
            
            data[modality] = {
                'train_features': train_features,
                'test_features': test_features
            }
            
        # 加载年龄标签
        train_age = pd.read_csv('train_age.tsv', sep='\t')
        test_age = pd.read_csv('test_age.tsv', sep='\t')
        
        data['age'] = {
            'train': train_age,
            'test': test_age
        }
        
        return data
    
    def prepare_modality_data(self, train_features, test_features, train_age, test_age):
        """准备单个模态的数据"""
        # 设置样本ID为索引
        train_features = train_features.set_index('samples')
        test_features = test_features.set_index('samples')
        train_age = train_age.set_index('samples')
        test_age = test_age.set_index('samples')
        
        # 确保训练集和测试集使用相同的特征
        common_features = train_features.columns.intersection(test_features.columns)
        train_features = train_features[common_features]
        test_features = test_features[common_features]
        
        # 提取年龄标签（除以100，恢复为实际年龄单位）
        age_column = 'age_at_study_date_x100_resurvey3'
        y_train = train_age[age_column].values / 100.0
        y_test = test_age[age_column].values / 100.0
        
        # 特征标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_features)
        X_test = scaler.transform(test_features)
        
        return X_train, X_test, y_train, y_test, scaler
    
    # ========================= 基础模型训练方法 =========================
    
    def create_base_model(self, model_type):
        """根据模型类型创建基础模型"""
        if model_type == 'catboost':
            return cb.CatBoostRegressor(
                random_state=self.random_state,
                verbose=False,
                task_type='GPU',  # 如果有GPU支持
                devices='0'
            )
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                tree_method='gpu_hist',  # 如果有GPU支持
                gpu_id=0
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def get_hyperparameter_grid(self, model_type):
        """获取不同模型类型的超参数网格"""
        if model_type == 'catboost':
            return {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15]
            }
    
    def train_single_modality(self, modality, X_train, y_train, cv_folds=5):
        """训练单个模态的模型"""
        print(f"正在训练{modality}模态模型（{self.model_types[modality]}）...")
        
        model_type = self.model_types[modality]
        base_model = self.create_base_model(model_type)
        param_grid = self.get_hyperparameter_grid(model_type)
        
        # 使用网格搜索进行超参数优化
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=1,  # GPU模式下使用单线程
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # 直接使用网格搜索中训练好的最佳模型
        from sklearn.model_selection import cross_val_predict
        best_model = grid_search.best_estimator_
        
        # 生成训练集的交叉验证预测
        train_cv_pred = cross_val_predict(
            grid_search.best_estimator_, X_train, y_train, 
            cv=cv_folds, method='predict'
        )
        
        # 保存最佳模型
        self.base_models[modality] = best_model
        
        # 计算交叉验证性能指标
        cv_mae = mean_absolute_error(y_train, train_cv_pred)
        cv_mse = mean_squared_error(y_train, train_cv_pred)
        cv_rmse = np.sqrt(cv_mse)
        cv_r2 = r2_score(y_train, train_cv_pred)
        cv_corr, _ = stats.pearsonr(y_train, train_cv_pred)
        
        cv_metrics = {
            'mae': cv_mae,
            'mse': cv_mse, 
            'rmse': cv_rmse,
            'r2': cv_r2,
            'correlation': cv_corr
        }
        
        print(f"  {modality}模态最佳参数: {grid_search.best_params_}")
        print(f"  {modality}模态交叉验证MAE: {cv_mae:.4f}")
        print(f"  {modality}模态交叉验证R²: {cv_r2:.4f}")
        
        return best_model, grid_search.best_params_, train_cv_pred, cv_metrics
    
    # ========================= 融合策略实现方法 =========================
    
    def generate_base_predictions(self, data):
        """生成所有基础模型的预测结果"""
        print("正在生成基础模型预测...")
        
        base_predictions = {
            'train': {},
            'test': {},
            'cv_metrics': {}  # 新增：保存交叉验证性能指标
        }
        
        for modality in self.modalities:
            print(f"处理{modality}模态...")
            
            # 准备数据
            X_train, X_test, y_train, y_test, scaler = self.prepare_modality_data(
                data[modality]['train_features'],
                data[modality]['test_features'], 
                data['age']['train'],
                data['age']['test']
            )
            
            # 保存scaler
            self.scalers[modality] = scaler
            
            # 训练模型（返回交叉验证预测和性能指标）
            model, best_params, train_cv_pred, cv_metrics = self.train_single_modality(modality, X_train, y_train)
            
            # 生成测试集预测
            test_pred = model.predict(X_test)
            
            # 使用交叉验证预测作为训练集预测结果
            base_predictions['train'][modality] = train_cv_pred
            base_predictions['test'][modality] = test_pred
            base_predictions['cv_metrics'][modality] = cv_metrics
            
            # 计算测试集性能
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"  {modality}模态训练集CV MAE: {cv_metrics['mae']:.4f}, R²: {cv_metrics['r2']:.4f}, 相关系数: {cv_metrics['correlation']:.4f}")
            print(f"  {modality}模态测试集MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        # 保存真实标签
        _, _, y_train, y_test, _ = self.prepare_modality_data(
            data['eye']['train_features'], data['eye']['test_features'],
            data['age']['train'], data['age']['test']
        )
        base_predictions['y_train'] = y_train
        base_predictions['y_test'] = y_test
        
        return base_predictions
    

    
    def _create_enhanced_meta_features(self, base_predictions):
        """创建增强的元学习特征"""
        # 基础预测值
        features = [base_predictions]
        
        # 1. 统计特征
        features.append(np.mean(base_predictions, axis=1, keepdims=True))  # 均值
        features.append(np.std(base_predictions, axis=1, keepdims=True))   # 标准差
        features.append(np.min(base_predictions, axis=1, keepdims=True))   # 最小值
        features.append(np.max(base_predictions, axis=1, keepdims=True))   # 最大值
        
        # 2. 两两模态差异特征
        n_modalities = base_predictions.shape[1]
        for i in range(n_modalities):
            for j in range(i+1, n_modalities):
                diff = base_predictions[:, i] - base_predictions[:, j]
                features.append(diff.reshape(-1, 1))
        
        # 3. 模态置信度特征（与均值的距离）
        mean_pred = np.mean(base_predictions, axis=1, keepdims=True)
        for i in range(n_modalities):
            confidence = np.abs(base_predictions[:, i:i+1] - mean_pred)
            features.append(confidence)
        
        # 4. 排序特征（每个样本中模态预测的排名）
        ranks = np.argsort(np.argsort(base_predictions, axis=1), axis=1)
        features.append(ranks)
        
        return np.column_stack(features)
    
    def meta_learner_fusion(self, base_predictions):
        """改进的元学习器融合策略"""
        print("正在训练增强元学习器...")
        
        # 1. 基础预测特征
        base_train_preds = np.column_stack([
            base_predictions['train'][mod] for mod in self.modalities
        ])
        base_test_preds = np.column_stack([
            base_predictions['test'][mod] for mod in self.modalities
        ])
        
        # 2. 构建增强特征
        X_meta_train = self._create_enhanced_meta_features(base_train_preds)
        X_meta_test = self._create_enhanced_meta_features(base_test_preds)
        
        y_meta_train = base_predictions['y_train']
        
        # 标准化元特征（使用RobustScaler减少异常值影响）
        from sklearn.preprocessing import RobustScaler
        self.meta_scaler = RobustScaler()
        X_meta_train_scaled = self.meta_scaler.fit_transform(X_meta_train)
        X_meta_test_scaled = self.meta_scaler.transform(X_meta_test)
        
        # 使用Stacking方法避免过拟合
        return self._stacking_meta_learner(X_meta_train_scaled, X_meta_test_scaled, y_meta_train)
    
    def _stacking_meta_learner(self, X_meta_train, X_meta_test, y_meta_train):
        """使用Stacking方法的元学习器"""
        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import ElasticNet, Lasso
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.svm import SVR
        
        # 扩展的元学习器候选集（增加正则化）
        meta_models = {
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=self.random_state
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=50, max_depth=5, min_samples_split=10,
                random_state=self.random_state, n_jobs=-1
            )
        }
        
        # 方法1: 传统网格搜索（带正则化）
        best_meta_model = None
        best_meta_score = float('inf')
        best_meta_name = None
        
        # 使用嵌套交叉验证选择最佳元学习器
        kf_outer = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        kf_inner = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for name, model in meta_models.items():
            # 内层交叉验证进行参数调优
            if name == 'Ridge':
                param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
                grid_search = GridSearchCV(model, param_grid, cv=kf_inner, 
                                         scoring='neg_mean_absolute_error', n_jobs=-1)
                grid_search.fit(X_meta_train, y_meta_train)
                model = grid_search.best_estimator_
            
            elif name == 'ElasticNet':
                param_grid = {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}
                grid_search = GridSearchCV(model, param_grid, cv=kf_inner,
                                         scoring='neg_mean_absolute_error', n_jobs=-1)
                grid_search.fit(X_meta_train, y_meta_train)
                model = grid_search.best_estimator_
            
            # 外层交叉验证评估性能
            scores = []
            for train_idx, val_idx in kf_outer.split(X_meta_train):
                X_fold_train, X_fold_val = X_meta_train[train_idx], X_meta_train[val_idx]
                y_fold_train, y_fold_val = y_meta_train[train_idx], y_meta_train[val_idx]
                
                model_clone = clone(model)
                model_clone.fit(X_fold_train, y_fold_train)
                val_pred = model_clone.predict(X_fold_val)
                mae = mean_absolute_error(y_fold_val, val_pred)
                scores.append(mae)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {name}元学习器CV MAE: {avg_score:.4f} (+/- {std_score:.4f})")
            
            if avg_score < best_meta_score:
                best_meta_score = avg_score
                best_meta_model = model
                best_meta_name = name
        
        print(f"选择最佳元学习器: {best_meta_name}")
        
        # 方法2: Stacking交叉验证预测（避免过拟合）
        print("使用Stacking方法生成元学习器预测...")
        
        # 使用cross_val_predict生成训练集的无偏预测
        stacking_train_pred = cross_val_predict(
            best_meta_model, X_meta_train, y_meta_train, 
            cv=5, method='predict'
        )
        
        # 训练最终元学习器
        self.meta_learner = clone(best_meta_model)
        self.meta_learner.fit(X_meta_train, y_meta_train)
        
        # 生成测试集预测
        stacking_test_pred = self.meta_learner.predict(X_meta_test)
        
        return stacking_train_pred, stacking_test_pred, best_meta_name
    
    def blending_fusion(self, base_predictions):
        """Blending融合策略（使用holdout验证集）"""
        print("正在进行Blending融合...")
        
        # 将训练集分为blend_train和blend_holdout
        from sklearn.model_selection import train_test_split
        
        # 准备基础特征
        base_train_preds = np.column_stack([
            base_predictions['train'][mod] for mod in self.modalities
        ])
        y_train = base_predictions['y_train']
        
        # 分割数据：80%用于训练，20%用于调整权重
        X_blend_train, X_blend_holdout, y_blend_train, y_blend_holdout = train_test_split(
            base_train_preds, y_train, test_size=0.2, random_state=self.random_state
        )
        
        # 在holdout集上训练blending权重
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        # 标准化
        scaler = StandardScaler()
        X_blend_holdout_scaled = scaler.fit_transform(X_blend_holdout)
        
        # 训练线性blending模型
        blending_model = LinearRegression()
        blending_model.fit(X_blend_holdout_scaled, y_blend_holdout)
        
        # 获取权重
        blend_weights = blending_model.coef_
        blend_intercept = blending_model.intercept_
        
        print("Blending权重:")
        for i, modality in enumerate(self.modalities):
            print(f"  {modality}: {blend_weights[i]:.4f}")
        print(f"  截距: {blend_intercept:.4f}")
        
        # 应用blending到完整数据
        base_test_preds = np.column_stack([
            base_predictions['test'][mod] for mod in self.modalities
        ])
        
        X_train_scaled = scaler.transform(base_train_preds)
        X_test_scaled = scaler.transform(base_test_preds)
        
        blending_train = blending_model.predict(X_train_scaled)
        blending_test = blending_model.predict(X_test_scaled)
        
        return blending_train, blending_test, blend_weights
    
    # ========================= 评估和结果保存方法 =========================
    
    def evaluate_predictions(self, y_true, y_pred, dataset_name, method_name):
        """评估预测结果"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred) 
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        corr, _ = stats.pearsonr(y_true, y_pred)
        
        print(f"{method_name} {dataset_name}集性能:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")  
        print(f"  R²: {r2:.4f}")
        print(f"  相关系数: {corr:.4f}")
        
        return {
            'mae': mae, 'mse': mse, 'rmse': rmse, 
            'r2': r2, 'correlation': corr
        }
    
    def fit(self, data=None):
        """训练多模态融合模型"""
        print("开始多模态晚期融合训练...")
        
        if data is None:
            data = self.load_data()
        
        # 生成基础模型预测
        base_predictions = self.generate_base_predictions(data)
        
        # Stacking元学习器融合
        print("\n" + "="*50)
        meta_train, meta_test, meta_name = self.meta_learner_fusion(base_predictions)
        
        # Blending融合
        print("\n" + "="*50) 
        blending_train, blending_test, blend_weights = self.blending_fusion(base_predictions)
        
        # 评估两种融合方法
        results = {}
        y_train, y_test = base_predictions['y_train'], base_predictions['y_test']
        
        print("\n" + "="*60)
        print("融合方法性能评估")
        print("="*60)
        
        # Stacking元学习器结果
        results['stacking'] = {
            'train': self.evaluate_predictions(y_train, meta_train, "训练", f"Stacking({meta_name})"),
            'test': self.evaluate_predictions(y_test, meta_test, "测试", f"Stacking({meta_name})"),
            'predictions': {'train': meta_train, 'test': meta_test},
            'meta_model_name': meta_name
        }
        
        # Blending结果
        results['blending'] = {
            'train': self.evaluate_predictions(y_train, blending_train, "训练", "Blending"),
            'test': self.evaluate_predictions(y_test, blending_test, "测试", "Blending"),
            'predictions': {'train': blending_train, 'test': blending_test},
            'blend_weights': blend_weights
        }
        
        # 保存基础模型预测结果
        results['base_predictions'] = base_predictions
        
        self.results = results
        return results


# ========================= 主程序 =========================
    

    
    def save_results(self, filepath='multi_modal_fusion_results.txt'):
        """保存性能汇总表"""
        if not hasattr(self, 'results'):
            print("请先训练模型再保存结果")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("多模态晚期融合生物学年龄预测 - 性能汇总表\n")
            f.write("="*80 + "\n\n")
            
            # 性能汇总表标题
            f.write(f"{'方法':<15} {'数据集':<8} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10} {'相关系数':<10}\n")
            f.write("-"*80 + "\n")
            
            base_preds = self.results['base_predictions']
            y_train, y_test = base_preds['y_train'], base_preds['y_test']
            
            # 基础模型性能
            for modality in self.modalities:
                # 训练集性能（使用交叉验证指标）
                if 'cv_metrics' in base_preds and modality in base_preds['cv_metrics']:
                    train_metrics = base_preds['cv_metrics'][modality]
                    f.write(f"{modality:<15} {'Train_CV':<8} {train_metrics['mae']:<10.4f} {train_metrics['mse']:<10.4f} {train_metrics['rmse']:<10.4f} {train_metrics['r2']:<10.4f} {train_metrics['correlation']:<10.4f}\n")
                else:
                    # 回退到原来的计算方式（兼容性）
                    train_pred = base_preds['train'][modality]
                    train_metrics = self._calculate_all_metrics(y_train, train_pred)
                    f.write(f"{modality:<15} {'Train':<8} {train_metrics['mae']:<10.4f} {train_metrics['mse']:<10.4f} {train_metrics['rmse']:<10.4f} {train_metrics['r2']:<10.4f} {train_metrics['correlation']:<10.4f}\n")
                
                # 测试集性能
                test_pred = base_preds['test'][modality]
                test_metrics = self._calculate_all_metrics(y_test, test_pred)
                f.write(f"{'':<15} {'Test':<8} {test_metrics['mae']:<10.4f} {test_metrics['mse']:<10.4f} {test_metrics['rmse']:<10.4f} {test_metrics['r2']:<10.4f} {test_metrics['correlation']:<10.4f}\n")
                f.write("-"*80 + "\n")
            
            # 融合方法性能
            methods = ['stacking', 'blending']
            method_names = ['Stacking', 'Blending']
            
            for method, name in zip(methods, method_names):
                if method in self.results:
                    result = self.results[method]
                    # 训练集
                    f.write(f"{name:<15} {'Train':<8} {result['train']['mae']:<10.4f} {result['train']['mse']:<10.4f} {result['train']['rmse']:<10.4f} {result['train']['r2']:<10.4f} {result['train']['correlation']:<10.4f}\n")
                    # 测试集
                    f.write(f"{'':<15} {'Test':<8} {result['test']['mae']:<10.4f} {result['test']['mse']:<10.4f} {result['test']['rmse']:<10.4f} {result['test']['r2']:<10.4f} {result['test']['correlation']:<10.4f}\n")
                    f.write("-"*80 + "\n")
        
        print(f"性能汇总表已保存到: {filepath}")
    
    def _calculate_all_metrics(self, y_true, y_pred):
        """计算所有性能指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        corr, _ = stats.pearsonr(y_true, y_pred)
        
        return {
            'mae': mae, 'mse': mse, 'rmse': rmse,
            'r2': r2, 'correlation': corr
        }
    
    # ========================= 主训练方法 =========================


def main():
    """主函数"""
    print("多模态晚期融合生物学年龄预测")
    print("="*50)
    
    # 创建融合器
    fusion_model = MultiModalLateFusion(random_state=42)
    
    # 训练模型
    results = fusion_model.fit()
    
    # 保存结果汇总表
    fusion_model.save_results('multi_modal_fusion_results.txt')
    
    print("\n训练完成！")
    print("性能汇总表已保存到: multi_modal_fusion_results.txt")



if __name__ == "__main__":
    main()
