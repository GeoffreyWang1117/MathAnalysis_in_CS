"""
验证工具 - 用于测试学生的答案
"""

import numpy as np
from typing import Callable, Any, List, Tuple
import matplotlib.pyplot as plt
from io import BytesIO


class Validator:
    """验证器类 - 提供各种验证方法"""

    @staticmethod
    def assert_close(actual, expected, rtol=1e-5, atol=1e-8, name="值"):
        """验证数值是否接近"""
        try:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
            print(f"✓ {name}正确")
            return True
        except AssertionError as e:
            print(f"✗ {name}不正确")
            print(f"  期望: {expected}")
            print(f"  实际: {actual}")
            return False

    @staticmethod
    def assert_array_equal(actual, expected, name="数组"):
        """验证数组是否相等"""
        try:
            np.testing.assert_array_equal(actual, expected)
            print(f"✓ {name}正确")
            return True
        except AssertionError:
            print(f"✗ {name}不正确")
            print(f"  期望: {expected}")
            print(f"  实际: {actual}")
            return False

    @staticmethod
    def assert_shape(actual, expected_shape, name="形状"):
        """验证数组形状"""
        if actual.shape == expected_shape:
            print(f"✓ {name}正确: {expected_shape}")
            return True
        else:
            print(f"✗ {name}不正确")
            print(f"  期望: {expected_shape}")
            print(f"  实际: {actual.shape}")
            return False

    @staticmethod
    def assert_function(func: Callable, test_cases: List[Tuple], name="函数"):
        """
        验证函数输出
        test_cases: [(input, expected_output), ...]
        """
        all_pass = True
        for i, (inputs, expected) in enumerate(test_cases):
            try:
                if isinstance(inputs, tuple):
                    result = func(*inputs)
                else:
                    result = func(inputs)

                if isinstance(expected, (np.ndarray, list)):
                    if not np.allclose(result, expected, rtol=1e-5):
                        print(f"✗ {name}测试用例 {i+1} 失败")
                        print(f"  输入: {inputs}")
                        print(f"  期望: {expected}")
                        print(f"  实际: {result}")
                        all_pass = False
                    else:
                        print(f"✓ {name}测试用例 {i+1} 通过")
                else:
                    if abs(result - expected) > 1e-5:
                        print(f"✗ {name}测试用例 {i+1} 失败")
                        print(f"  输入: {inputs}")
                        print(f"  期望: {expected}")
                        print(f"  实际: {result}")
                        all_pass = False
                    else:
                        print(f"✓ {name}测试用例 {i+1} 通过")
            except Exception as e:
                print(f"✗ {name}测试用例 {i+1} 出错: {e}")
                all_pass = False

        return all_pass

    @staticmethod
    def assert_type(actual, expected_type, name="类型"):
        """验证类型"""
        if isinstance(actual, expected_type):
            print(f"✓ {name}正确")
            return True
        else:
            print(f"✗ {name}不正确")
            print(f"  期望: {expected_type}")
            print(f"  实际: {type(actual)}")
            return False

    @staticmethod
    def check_plot_created():
        """检查是否创建了图形"""
        if plt.get_fignums():
            print("✓ 图形已创建")
            return True
        else:
            print("✗ 未创建图形")
            return False

    @staticmethod
    def assert_matrix_property(matrix, property_name, expected=True):
        """验证矩阵性质"""
        properties = {
            "symmetric": lambda m: np.allclose(m, m.T),
            "orthogonal": lambda m: np.allclose(m @ m.T, np.eye(len(m))),
            "positive_definite": lambda m: np.all(np.linalg.eigvals(m) > 0),
            "invertible": lambda m: abs(np.linalg.det(m)) > 1e-10,
        }

        if property_name not in properties:
            print(f"✗ 未知的矩阵性质: {property_name}")
            return False

        result = properties[property_name](matrix)
        if result == expected:
            print(f"✓ 矩阵{property_name}性质正确")
            return True
        else:
            print(f"✗ 矩阵{property_name}性质不正确")
            return False


def create_test_decorator(func):
    """测试装饰器 - 捕获异常并美化输出"""
    def wrapper(*args, **kwargs):
        try:
            print("\n" + "="*60)
            print("🧪 运行测试...")
            print("="*60)
            result = func(*args, **kwargs)
            print("="*60)
            if result:
                print("🎉 所有测试通过！")
            else:
                print("❌ 部分测试未通过，请检查您的实现")
            print("="*60 + "\n")
            return result
        except Exception as e:
            print(f"\n❌ 测试出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    return wrapper
