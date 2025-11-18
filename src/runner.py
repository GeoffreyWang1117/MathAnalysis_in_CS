#!/usr/bin/env python3
"""
MathGym - 数学分析交互式学习平台
类似Rustlings的数学编程练习系统
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class ExerciseRunner:
    """练习运行器 - 管理和执行数学练习"""

    def __init__(self, exercise_dir: str = "exercises"):
        self.exercise_dir = Path(exercise_dir)
        self.progress_file = Path(".mathgym_progress.json")
        self.progress = self.load_progress()

    def load_progress(self) -> Dict:
        """加载学习进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"completed": [], "last_updated": None}

    def save_progress(self):
        """保存学习进度"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def find_all_exercises(self) -> List[Path]:
        """查找所有练习文件"""
        exercises = []
        for topic_dir in sorted(self.exercise_dir.iterdir()):
            if topic_dir.is_dir():
                for ex_file in sorted(topic_dir.glob("ex*.py")):
                    exercises.append(ex_file)
        return exercises

    def get_next_exercise(self) -> Optional[Path]:
        """获取下一个未完成的练习"""
        exercises = self.find_all_exercises()
        for ex in exercises:
            if str(ex) not in self.progress["completed"]:
                return ex
        return None

    def run_exercise(self, exercise_path: Path) -> Tuple[bool, str]:
        """
        运行单个练习
        返回: (是否通过, 消息)
        """
        try:
            # 动态导入练习模块
            spec = importlib.util.spec_from_file_location("exercise", exercise_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["exercise"] = module
            spec.loader.exec_module(module)

            # 检查是否有TODO标记（未完成）
            with open(exercise_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "# TODO:" in content or "pass  # 请实现" in content:
                    return False, "❌ 练习未完成！请填写所有 TODO 部分"

            # 运行测试
            if hasattr(module, 'test'):
                result = module.test()
                if result:
                    return True, "✅ 测试通过！做得很好！"
                else:
                    return False, "❌ 测试未通过，请检查您的实现"
            else:
                return False, "⚠️  练习文件缺少 test() 函数"

        except Exception as e:
            error_msg = f"❌ 运行错误：\n{traceback.format_exc()}"
            return False, error_msg

    def mark_completed(self, exercise_path: Path):
        """标记练习为已完成"""
        if str(exercise_path) not in self.progress["completed"]:
            self.progress["completed"].append(str(exercise_path))
            self.save_progress()

    def show_progress(self):
        """显示学习进度"""
        exercises = self.find_all_exercises()
        completed = len(self.progress["completed"])
        total = len(exercises)
        percentage = (completed / total * 100) if total > 0 else 0

        print("\n" + "="*60)
        print("📊 学习进度")
        print("="*60)
        print(f"已完成: {completed}/{total} ({percentage:.1f}%)")
        print(f"进度条: [{'█' * int(percentage/2)}{'░' * (50-int(percentage/2))}]")

        # 按主题分类显示
        topics = {}
        for ex in exercises:
            topic = ex.parent.name
            if topic not in topics:
                topics[topic] = {"total": 0, "completed": 0}
            topics[topic]["total"] += 1
            if str(ex) in self.progress["completed"]:
                topics[topic]["completed"] += 1

        print("\n主题进度：")
        for topic, stats in sorted(topics.items()):
            status = f"{stats['completed']}/{stats['total']}"
            print(f"  {topic}: {status}")
        print("="*60 + "\n")

    def run_interactive(self):
        """交互式运行模式（类似Rustlings）"""
        print("\n🎓 欢迎来到 MathGym - 数学分析学习平台！")
        print("=" * 60)
        print("这是一个交互式数学编程学习系统，类似于 Rustlings")
        print("请填写代码中的 TODO 部分，完成所有练习")
        print("=" * 60)

        self.show_progress()

        next_ex = self.get_next_exercise()
        if not next_ex:
            print("🎉 恭喜！您已完成所有练习！")
            return

        print(f"\n📝 下一个练习: {next_ex}")
        print(f"主题: {next_ex.parent.name}")
        print("\n提示: 编辑文件，填写 TODO 部分，然后运行此脚本查看结果")
        print("      使用 'python src/runner.py watch' 自动监控文件变化")
        print("      使用 'python src/runner.py list' 查看所有练习\n")

        # 运行当前练习
        success, message = self.run_exercise(next_ex)
        print(message)

        if success:
            self.mark_completed(next_ex)
            print(f"\n🎊 太棒了！继续下一个练习...")
            self.show_progress()


def main():
    """主函数"""
    runner = ExerciseRunner()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            # 列出所有练习
            exercises = runner.find_all_exercises()
            print("\n📚 所有练习：")
            for ex in exercises:
                status = "✅" if str(ex) in runner.progress["completed"] else "⬜"
                print(f"  {status} {ex}")
            print()

        elif command == "progress":
            # 显示进度
            runner.show_progress()

        elif command == "reset":
            # 重置进度
            runner.progress = {"completed": [], "last_updated": None}
            runner.save_progress()
            print("✅ 进度已重置")

        elif command == "watch":
            # 监控模式（简化版）
            print("⚠️  监控模式需要安装 watchdog 库")
            print("   请运行: pip install watchdog")
            print("   或直接使用: python src/runner.py 运行练习")
        else:
            print(f"未知命令: {command}")
            print("可用命令: list, progress, reset, watch")
    else:
        # 交互式模式
        runner.run_interactive()


if __name__ == "__main__":
    main()
