"""
MathGym - 数学分析交互式学习平台
"""

__version__ = "1.0.0"
__author__ = "MathGym Team"
__description__ = "Interactive math learning platform for AI+CS students"

from .runner import ExerciseRunner
from .validator import Validator, create_test_decorator
from .utils import *

__all__ = [
    'ExerciseRunner',
    'Validator',
    'create_test_decorator',
]
