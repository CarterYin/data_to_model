# 一个从数据到模型的项目记录
## 本仓库包含了数据预处理和模型训练的代码
### 数据预处理代码pre_mod
- 将数据集中NO转化为0
- 将数据集中YES转化为1
- 将.tsv文件添加header使数据对齐
### 模型训练代码train_test
- 分别在眼部，颈动脉，生理以及全部指标上训练模型。
- 一个批处理文件run_all_models.py，能够激活conda环境yclearning并在环境中运行模型代码。

## 一个利用GPT5写的报告

## 一、模型代码总体流程（Pipeline 概览）

总体流程可以分为下列步骤：

1. **数据加载**（`load_data()`）
2. **数据预处理与标准化**（`prepare_data()`）
3. **检测 GPU 支持并构造模型 + 超参数网格**（`check_gpu_availability()`、`create_models_with_params_gpu()`）
4. **超参数调优（GridSearchCV，5 折 CV，以 MAE 为选择标准）**（`hyperparameter_tuning_gpu()`）
5. **（可选）在完整训练集上训练最优模型**（`train_all_optimal_models()` —— 代码中未在主流程里被直接调用）
6. **在测试集上评估各模型**（`evaluate_all_models_on_test()`）
7. **选择最好模型（依据测试集 MAE）并保存报告**（`select_best_model()`、`save_results()`）

```
load_data -> prepare_data -> check_gpu -> create_models -> GridSearchCV(CV) -> evaluate_on_test -> select_best -> save_report
```

---

## 二、逐函数详细说明（含关键代码与解释）

### `load_data()`

**功能**：读取四个 TSV 文件：训练/测试特征、训练/测试年龄标签。

* 输入：无（函数内部使用固定相对路径）。
* 输出：`train_features, test_features, train_age, test_age`（均为 `pandas.DataFrame`）。

**注意点**：

* 路径是相对的（`../train_set_all.tsv` 等），运行前请确保当前工作目录正确。
* 若文件里含有非数值、缺失值、或编码问题，建议加入 `encoding`/`dtype` 控制或捕获异常。

---

### `prepare_data(train_features, test_features, train_age, test_age)`

**功能**：

* 把 `samples` 列设为索引；
* 取训练/测试数据的交集特征（`common_features`），确保训练/测试列对齐；
* 从 `train_age`/`test_age` 中读取年龄列 `age_at_study_date_x100_resurvey3` 并除以 100（恢复为实际年龄）；
* 使用 `StandardScaler()` 基于训练集拟合并转换训练与测试特征。

**输入**：四个 `DataFrame`；
**输出**：`X_train, X_test, y_train, y_test, scaler`。

**关键实现点**：

* `common_features = train_features.columns.intersection(test_features.columns)` 可以避免列错位；
* `scaler.fit_transform(train_features)` 只在训练集上拟合，防止数据泄露（正确）；

**改进建议**：

* 检查并处理缺失值（`dropna` / 插补）；
* 明确 `age_column` 是否存在并在不存在时抛出友好错误；
* 若存在类别特征需做编码（`ColumnTransformer` + `Pipeline`）。

---

### `check_gpu_availability()`

**功能**：尝试检测 XGBoost/LightGBM/CatBoost 的 GPU 支持并打印信息。

**实现细节与潜在问题**：

* 目前实现创建带 GPU 参数的模型对象来检测支持性（例如 `xgb.XGBRegressor(tree_method='gpu_hist')`），但具体是否抛错依赖于库是否编译支持 GPU，有时创建对象不会立刻报错（只有 `fit()` 时才可能出错）。
* 函数 **只在 XGBoost 成功时把 \*\*\*\*`gpu_available=True`**，如果 XGBoost 失败但 LightGBM 或 CatBoost 支持 GPU，函数仍会返回 `False` —— 这是一个小 bug。

**改进建议**：更稳健的检测方式：分别针对每个库单独检测，并返回一个字典或三个布尔标志；或者在一个非常小的数据集上尝试 `fit()` 并捕获 GPU 相关异常以做判断。

示例（更稳健的检测思路）：

```py
# 伪代码示例（不要直接在生产环境里不加 try/except 跑）
def is_xgb_gpu_ok():
    try:
        model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=1)
        X = np.random.rand(2,4); y = np.random.rand(2)
        model.fit(X, y)
        return True
    except Exception:
        return False
```

---

### `create_models_with_params_gpu()`

**功能**：基于 `check_gpu_availability()` 的结果构造四类模型（XGBoost、LightGBM、CatBoost、RandomForest）及它们的超参数网格。

**要点**：

* 若检测到 GPU，可把 `tree_method='gpu_hist'`（XGBoost）、`device='gpu'`（LightGBM）、`task_type='GPU'`（CatBoost）等参数打开；
* `RandomForest` 使用 CPU（sklearn 不支持 GPU）；
* 返回结构 `models_and_params`：每个模型包含 `model` 对象与 `params` 网格。

**建议**：对于大参数空间可考虑用 `RandomizedSearchCV` 或 `optuna`/`skopt` 提高效率；对提升树类建议加入 `early_stopping_rounds` 与 `eval_set` 来避免过拟合并加速调参。

---

### `_pearson_corr_for_scorer(y_true, y_pred)`

**功能**：作为 `make_scorer` 的自定义评分函数，用来计算预测与真实值的皮尔逊相关系数。

**细节**：

* 若样本数小于2或方差为0则返回 0，避免 `pearsonr` 报错；
* 该函数返回的是相关系数（-1..1），在 `GridSearchCV` 的 scoring 字典里用作 `corr` 项。

---

### `hyperparameter_tuning_gpu(X_train, y_train)`

**功能（最关键）**：对每种模型用 **5 折交叉验证** 使用 `GridSearchCV` 搜索最优超参数，并**以 MAE 为最终选择/重拟合 (refit)**。

**关键实现点**：

* `scoring` 字典包含 `'mae'`（`neg_mean_absolute_error`）、`'mse'`（`neg_mean_squared_error`）、`'r2'`、`'corr'`（自定义）；
* `GridSearchCV(..., scoring=scoring, refit='mae', cv=5, return_train_score=True)`：多个评分一起计算，但以 `mae`（取负）作为 `refit` 指标；注意 sklearn 在 `scoring` 使用负值表示 "越大越好" 的规则，因此后面要把 `-grid_search.best_score_` 取为正 MAE；
* `n_jobs=1`：在 GPU 场景下通常限定为 1 以避免多个进程争用单张 GPU；若没有 GPU 可把 `n_jobs` 扩大来并行搜索；
* `grid_search.best_estimator_` 是基于 `refit='mae'` 在整个训练集上重训练后的模型（也就是说 `best_estimator_` 已经 fit 在整个训练集上）。

**保存的结果**：

* `cv_results_`、`best_params_`、`best_score_` 等；代码里还把 CV 上多指标的均值组合成 `cv_mean_results` 并保存到 `best_models`。

**注意**：GridSearch 输出的 `mean_test_mse` 是负值（因为 `scoring='neg_mean_squared_error'`），故用 `-` 取回真实 MSE。

---

### `train_all_optimal_models(X_train, y_train, best_models)`

**原意**：克隆 `best_models` 中的最优 estimator 并在完整训练集上重新 `fit()`（再次训练）以获得训练集指标。

**但主流程说明**：在 `main()` 中作者并未调用 `train_all_optimal_models()`，而是直接将 `best_models` 赋给 `trained_models` 并用于评估。这是合理的，因为 `GridSearchCV(..., refit='mae')` 已经在整个训练集上对 `best_estimator_` 进行过 refit（已训练）。

**因此二选一可行**：

* 方案 A：依赖 `GridSearchCV.best_estimator_`（已 refit），直接使用；
* 方案 B：`clone()` 并再次 `fit()`（有时用于改变训练细节或注入新的回调）。

---

### `evaluate_all_models_on_test(X_test, y_test, trained_models)`

**功能**：对每个已训练模型在测试集上预测并计算指标（MSE, RMSE, MAE, R², Pearson correlation），并把预测数组保存在结果里。

**注意**：

* 使用 `stats.pearsonr(y_test, y_test_pred)` 计算相关系数；当样本很少或常量时 `pearsonr` 会抛错，建议捕获异常。

---

### `select_best_model(trained_models, test_results)`

**功能**：依据测试集的 MAE（越小越好）选出最好的模型并打印。

**返回值**：`best_model_name, best_model, trained_models, test_results`。

---

### `save_results(trained_models, test_results, best_model_name)`

**功能**：把模型配置、CV 平均指标、测试集指标写入 `hyperparameter_tuning_gpu_report.txt`，并生成一个汇总表格。

**注意点**：

* 函数内部把很多指标格式化写入文件，是可复现的好做法；
* 假设 `trained_models[name]` 中包含 `cv_mean_results` 字段（在 `hyperparameter_tuning_gpu()` 中已构造）。

---

## 三、关键细节与常见问题

### 1) 交叉验证中负值评分的含义

* sklearn 中 `neg_mean_absolute_error` / `neg_mean_squared_error` 返回的值是 **负的**（因为所有 scorer 都应当是 "越大越好" 的格式）。
* 因此：`best_score_`（当 refit=mae）通常是负数，代码中用 `-grid_search.best_score_` 把它转换回正的 MAE。


```
GridSearchCV 输出: mean_test_mae = -0.1234  
转换为真实 MAE = - (-0.1234) = 0.1234
```

### 2) `GridSearchCV` 的 `refit='mae'`

* 意味着 GridSearch 在找到最佳超参数后，会使用该最佳参数在 **整个训练集** 上重新训练模型（即 `best_estimator_` 为 refit 后的模型）。
* 但在我修改后没有调用。

### 3) GPU 检测的潜在不确定性（可视化说明）

```
当前实现: try XGBoost with GPU kwargs -> if success set gpu_available=True
问题: XGB 创建实例可能不报错，只有 fit 时才报错；LightGBM/CatBoost 单独失败/成功时返回信息不一致。
```

建议用小样本的实际 `fit()` 测试或分别返回每个库的可用性标志。

---

## 四、可视化（示例代码，运行后可得到图）

下面的代码片段在本地或 notebook 中运行可以生成常见诊断图：预测 vs 真实、残差分布、特征重要性。

### 1) 预测 vs 真实（散点图）

```py
import matplotlib.pyplot as plt

# 假设 y_test, y_pred 已经准备好
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
# 画对角线
mn = min(min(y_test), min(y_pred))
mx = max(max(y_test), max(y_pred))
plt.plot([mn, mx], [mn, mx])
plt.xlabel('真实年龄')
plt.ylabel('预测年龄')
plt.title('Predicted vs True Age')
plt.grid(True)
plt.show()
```

### 2) 残差分布（直方图 + KDE）

```py
residuals = y_test - y_pred
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=30)
plt.xlabel('残差 (True - Pred)')
plt.title('Residual Distribution')
plt.show()
```

### 3) XGBoost / LightGBM 特征重要性（示例）

```py
# XGBoost
xgb.plot_importance(best_model, max_num_features=30)
plt.title('XGBoost Feature Importance')
plt.show()
```

> 注：若要在脚本里保存图片，使用 `plt.savefig('figure.png', dpi=150)`。

---

## 五、常见改进建议（按优先级）

1. **完善 GPU 检测并返回每个库的可用性**（避免只以 xgboost 为准）；
2. **处理缺失值与异常值**，在 `prepare_data()` 中加入 `imputer` 或 `dropna`；
3. **使用 Pipeline 保证预处理与模型打包**（易于复用与部署）；
4. **对大参数空间使用 ****`RandomizedSearchCV`**** 或现代调优库（Optuna）**，并加早停；
5. **对提升树使用 ****`early_stopping_rounds`**** + ****`eval_set`**** 来节省时间并避免过拟合**；
6. **增加日志与异常捕获（try/except）**，输出更可读的调试信息；
7. **固定随机种子并记录运行环境（库版本、是否 GPU）以确保可复现性**；
8. **若样本不平衡或年龄分布极端，考虑分层 CV（按年龄分箱）**。

---

## 六、代码片段：修复 `check_gpu_availability()` 的建议实现（示例）

```py
def check_gpu_availability_more_robust():
    res = {'xgboost': False, 'lightgbm': False, 'catboost': False}
    import numpy as _np
    X = _np.random.rand(4, 10)
    y = _np.random.rand(4)

    try:
        import xgboost as _xgb
        m = _xgb.XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', n_estimators=1)
        m.fit(X, y)
        res['xgboost'] = True
    except Exception:
        res['xgboost'] = False

    try:
        import lightgbm as _lgb
        m = _lgb.LGBMRegressor(device='gpu', n_estimators=1)
        m.fit(X, y)
        res['lightgbm'] = True
    except Exception:
        res['lightgbm'] = False

    try:
        import catboost as _cb
        m = _cb.CatBoostRegressor(task_type='GPU', iterations=1, verbose=False)
        m.fit(X, y)
        res['catboost'] = True
    except Exception:
        res['catboost'] = False

    return res
```

> 提醒：上述方法会尝试真正 `fit()` 一个非常小的模型，因此会检测出是否真正能在当前环境上使用 GPU（也会比较慢）。

---

## 七、快速示例：如何运行与输出说明

```
$ python your_script.py
```

运行后主过程 `main()` 会：

* 在控制台打印训练流程信息；
* 把详细结果写入 `hyperparameter_tuning_gpu_report.txt`（包含 CV 均值表与测试集结果）；
* 在控制台打印最终选择的最佳模型与测试集 MAE / R²。

---

## 八、总结（要点回顾）

* 该脚本实现了一个从数据读取、标准化、按模型进行 CV 调参、在测试集评估并保存报告的完整流水线；
* `GridSearchCV(..., refit='mae')` 已经在训练集上对最佳模型做了 refit，主流程直接使用 `best_estimator_` 是可以的；
* 建议修复 GPU 检测的细节、加入缺失值处理、用 Pipeline 包装预处理与模型，并考虑更高效的调参策略（Randomized / Optuna）。

---


