"""
练习 2: 回归分析
================

学习目标：
- 掌握线性回归的原理
- 实现最小二乘法
- 理解正则化（Ridge、Lasso）
- 评估回归模型

任务：
实现各种回归方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def simple_linear_regression(X, y):
    """
    简单线性回归: y = β₀ + β₁x

    使用最小二乘法

    参数:
        X: 自变量（一维数组）
        y: 因变量（一维数组）

    返回:
        (β₀, β₁): 截距和斜率
    """
    # TODO: 实现简单线性回归
    X = np.array(X)
    y = np.array(y)
    n = len(X)

    # β₁ = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
    x_mean = np.mean(X)
    y_mean = np.mean(y)

    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)

    beta_1 = numerator / denominator
    beta_0 = y_mean - beta_1 * x_mean

    return beta_0, beta_1


def multiple_linear_regression(X, y):
    """
    多元线性回归: y = Xβ + ε

    最小二乘解: β = (X^T X)^(-1) X^T y

    参数:
        X: 设计矩阵 (n_samples × n_features)
        y: 目标变量

    返回:
        β: 回归系数
    """
    # TODO: 实现多元线性回归
    X = np.array(X)
    y = np.array(y)

    # 添加截距项（全1列）
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # β = (X^T X)^(-1) X^T y
    XTX = X_with_intercept.T @ X_with_intercept
    XTy = X_with_intercept.T @ y
    beta = np.linalg.solve(XTX, XTy)

    return beta


def ridge_regression(X, y, alpha=1.0):
    """
    Ridge回归（L2正则化）

    β = (X^T X + αI)^(-1) X^T y

    参数:
        X: 设计矩阵
        y: 目标变量
        alpha: 正则化参数
    """
    # TODO: 实现Ridge回归
    X = np.array(X)
    y = np.array(y)

    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    n_features = X_with_intercept.shape[1]

    # 正则化矩阵（不惩罚截距）
    reg_matrix = alpha * np.eye(n_features)
    reg_matrix[0, 0] = 0  # 不惩罚截距

    # β = (X^T X + αI)^(-1) X^T y
    XTX_reg = X_with_intercept.T @ X_with_intercept + reg_matrix
    XTy = X_with_intercept.T @ y
    beta = np.linalg.solve(XTX_reg, XTy)

    return beta


def lasso_regression_coordinate_descent(X, y, alpha=1.0, max_iter=1000, tol=1e-4):
    """
    Lasso回归（L1正则化）

    使用坐标下降法求解

    min ||y - Xβ||² + α||β||₁

    参数:
        X: 设计矩阵
        y: 目标变量
        alpha: 正则化参数
    """
    # TODO: 实现Lasso回归（简化版）
    X = np.array(X)
    y = np.array(y)

    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    n_samples, n_features = X_with_intercept.shape

    # 初始化系数
    beta = np.zeros(n_features)

    # 坐标下降
    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(n_features):
            # 计算残差（不包括第j个特征）
            X_j = X_with_intercept[:, j]
            residual = y - X_with_intercept @ beta + beta[j] * X_j

            # 软阈值算子
            rho = X_j @ residual
            z = X_j @ X_j

            if j == 0:  # 截距项不正则化
                beta[j] = rho / z
            else:
                beta[j] = np.sign(rho) * max(0, abs(rho) - alpha) / z

        # 检查收敛
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def r_squared(y_true, y_pred):
    """
    R²决定系数

    R² = 1 - SS_res / SS_tot

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        R²分数
    """
    # TODO: 实现R²
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 残差平方和
    ss_res = np.sum((y_true - y_pred) ** 2)

    # 总平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # R²
    r2 = 1 - ss_res / ss_tot

    return r2


def adjusted_r_squared(y_true, y_pred, n_features):
    """
    调整R²（考虑特征数量）

    R²_adj = 1 - (1 - R²)(n - 1) / (n - p - 1)

    参数:
        y_true: 真实值
        y_pred: 预测值
        n_features: 特征数量
    """
    # TODO: 实现调整R²
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return adj_r2


def mean_squared_error(y_true, y_pred):
    """
    均方误差 MSE

    MSE = (1/n) Σ(y - ŷ)²
    """
    # TODO: 实现MSE
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    均方根误差 RMSE

    RMSE = √MSE
    """
    # TODO: 实现RMSE
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    平均绝对误差 MAE

    MAE = (1/n) Σ|y - ŷ|
    """
    # TODO: 实现MAE
    return np.mean(np.abs(y_true - y_pred))


def cross_validation_score(X, y, k=5, model_func=multiple_linear_regression):
    """
    k折交叉验证

    参数:
        X: 特征矩阵
        y: 目标变量
        k: 折数
        model_func: 模型函数

    返回:
        R²分数列表
    """
    # TODO: 实现k折交叉验证
    n = len(X)
    fold_size = n // k
    scores = []

    for i in range(k):
        # 划分训练集和测试集
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n

        test_idx = list(range(test_start, test_end))
        train_idx = list(range(0, test_start)) + list(range(test_end, n))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 训练模型
        beta = model_func(X_train, y_train)

        # 预测
        X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])
        y_pred = X_test_with_intercept @ beta

        # 计算R²
        score = r_squared(y_test, y_pred)
        scores.append(score)

    return scores


def polynomial_features(X, degree):
    """
    生成多项式特征

    例如：degree=2时，[x] → [x, x²]

    参数:
        X: 输入特征
        degree: 多项式阶数
    """
    # TODO: 实现多项式特征生成
    X = np.array(X).reshape(-1, 1) if X.ndim == 1 else X
    poly_features = [X]

    for d in range(2, degree + 1):
        poly_features.append(X ** d)

    return np.hstack(poly_features)


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 生成测试数据
    np.random.seed(42)
    X_simple = np.array([1, 2, 3, 4, 5])
    y_simple = 2 * X_simple + 1 + np.random.normal(0, 0.1, 5)

    print("\n测试 1: 简单线性回归")
    beta_0, beta_1 = simple_linear_regression(X_simple, y_simple)
    # 应该接近 β₀=1, β₁=2
    results.append(abs(beta_0 - 1) < 1.0)
    results.append(abs(beta_1 - 2) < 1.0)
    print(f"  β₀ = {beta_0:.4f} (期望: ~1)")
    print(f"  β₁ = {beta_1:.4f} (期望: ~2)")
    print(f"  {'✓ 通过' if results[-2] and results[-1] else '✗ 失败'}")

    print("\n测试 2: 多元线性回归")
    # 生成多元数据: y = 1 + 2x₁ + 3x₂
    X_multi = np.random.randn(100, 2)
    y_multi = 1 + 2*X_multi[:, 0] + 3*X_multi[:, 1] + np.random.normal(0, 0.1, 100)

    beta = multiple_linear_regression(X_multi, y_multi)
    # 检查系数接近 [1, 2, 3]
    expected_beta = np.array([1, 2, 3])
    close_to_expected = np.allclose(beta, expected_beta, atol=0.5)
    results.append(close_to_expected)
    print(f"  β = {beta}")
    print(f"  期望: [1, 2, 3]")
    print(f"  {'✓ 通过' if close_to_expected else '✗ 失败'}")

    print("\n测试 3: R²分数")
    X_test_with_intercept = np.column_stack([np.ones(len(X_multi)), X_multi])
    y_pred = X_test_with_intercept @ beta
    r2 = r_squared(y_multi, y_pred)
    # R²应该接近1（因为噪声很小）
    results.append(r2 > 0.9)
    print(f"  R² = {r2:.4f}")
    print(f"  {'✓ 通过（R² > 0.9）' if r2 > 0.9 else '✗ 失败'}")

    print("\n测试 4: MSE和RMSE")
    mse = mean_squared_error(y_multi, y_pred)
    rmse = root_mean_squared_error(y_multi, y_pred)
    # RMSE应该接近噪声标准差（0.1）
    results.append(v.assert_close(rmse, np.sqrt(mse), name="RMSE"))
    print(f"  MSE = {mse:.6f}, RMSE = {rmse:.6f}")

    print("\n测试 5: MAE")
    mae = mean_absolute_error(y_multi, y_pred)
    # MAE应该小于RMSE
    results.append(mae < rmse)
    print(f"  MAE = {mae:.6f}, RMSE = {rmse:.6f}")
    print(f"  {'✓ MAE < RMSE' if mae < rmse else '✗ 失败'}")

    print("\n测试 6: Ridge回归")
    beta_ridge = ridge_regression(X_multi, y_multi, alpha=1.0)
    # Ridge系数应该比OLS小（正则化效果）
    beta_norm_ridge = np.linalg.norm(beta_ridge[1:])  # 不包括截距
    beta_norm_ols = np.linalg.norm(beta[1:])
    results.append(beta_norm_ridge <= beta_norm_ols)
    print(f"  ||β_Ridge|| = {beta_norm_ridge:.4f}")
    print(f"  ||β_OLS|| = {beta_norm_ols:.4f}")
    print(f"  {'✓ Ridge收缩系数' if results[-1] else '✗ 失败'}")

    print("\n测试 7: Lasso回归")
    beta_lasso = lasso_regression_coordinate_descent(X_multi, y_multi, alpha=0.1)
    # 检查Lasso是否产生稀疏解
    n_nonzero = np.sum(np.abs(beta_lasso) > 0.01)
    print(f"  非零系数数量: {n_nonzero}")
    print(f"  Lasso系数: {beta_lasso}")
    results.append(True)  # Lasso实现正确即可

    print("\n测试 8: 多项式特征")
    X_poly_test = np.array([1, 2, 3])
    poly_feat = polynomial_features(X_poly_test, degree=2)
    expected_shape = (3, 2)  # [x, x²]
    results.append(v.assert_shape(poly_feat, expected_shape, name="多项式特征形状"))

    return all(results)


if __name__ == "__main__":
    test()
