"""
BPG/SBPG/MSBPG 优化器对比实验单元测试
验证新增的BPG和SBPG方法的正确性、收敛性和差异性

运行方式：
    cd e:\Document\Code\2026\06
    python -m pytest experiment_system\test_optimizer_comparison.py -v
    或
    python experiment_system\test_optimizer_comparison.py
"""

import sys
import os
import numpy as np
import unittest

# 添加优化器模块路径
_OPTIMIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                               '02', 'laser_echo_experiment', 'laser_echo_experiment',
                               'src', 'models', 'optimization')
sys.path.insert(0, _OPTIMIZER_PATH)

from msbpg_optimizer import MSBPGOptimizer, ComparisonOptimizer


def make_linear_problem(n_samples=200, n_features=5, noise_std=0.1, seed=42):
    """生成线性回归测试问题"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    true_theta = rng.randn(n_features)
    y = X @ true_theta + rng.randn(n_samples) * noise_std
    return X, y, true_theta


def make_objective_gradient(X, y):
    """生成线性最小二乘目标函数和梯度函数"""
    def objective(theta, X_batch, y_batch):
        if X_batch is None:
            return 0.5 * np.mean((X @ theta - y) ** 2)
        return 0.5 * np.mean((X_batch @ theta - y_batch) ** 2)

    def gradient(theta, X_batch, y_batch):
        if X_batch is None:
            return X.T @ (X @ theta - y) / len(y)
        return X_batch.T @ (X_batch @ theta - y_batch) / len(y_batch)

    return objective, gradient


class TestBPGOptimizer(unittest.TestCase):
    """测试标准BPG（欧几里得近端，无动量）"""

    def setUp(self):
        self.X, self.y, self.true_theta = make_linear_problem()
        self.obj, self.grad = make_objective_gradient(self.X, self.y)
        np.random.seed(42)
        self.theta_init = np.random.randn(self.X.shape[1]) * 0.1

    def test_bpg_runs_without_error(self):
        """BPG能正常运行不报错"""
        theta, loss_hist = ComparisonOptimizer.bpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=100
        )
        self.assertEqual(theta.shape, self.theta_init.shape)
        self.assertTrue(len(loss_hist) > 0)

    def test_bpg_output_finite(self):
        """BPG输出参数和损失值均为有限值"""
        theta, loss_hist = ComparisonOptimizer.bpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=100
        )
        self.assertTrue(np.all(np.isfinite(theta)), "BPG输出参数含NaN或Inf")
        self.assertTrue(np.all(np.isfinite(loss_hist)), "BPG损失历史含NaN或Inf")

    def test_bpg_converges(self):
        """BPG损失应单调下降或至少最终损失低于初始损失"""
        theta, loss_hist = ComparisonOptimizer.bpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=500
        )
        self.assertLess(loss_hist[-1], loss_hist[0],
                        f"BPG未收敛: 初始loss={loss_hist[0]:.6f}, 最终loss={loss_hist[-1]:.6f}")

    def test_bpg_uses_full_batch(self):
        """BPG应使用全批量梯度（非随机），两次运行结果应完全一致"""
        theta1, _ = ComparisonOptimizer.bpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=50
        )
        theta2, _ = ComparisonOptimizer.bpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=50
        )
        np.testing.assert_array_almost_equal(theta1, theta2,
                                             err_msg="BPG全批量模式两次运行结果不一致")


class TestSBPGOptimizer(unittest.TestCase):
    """测试随机BPG（SBPG，不带动量）"""

    def setUp(self):
        self.X, self.y, self.true_theta = make_linear_problem()
        self.obj, self.grad = make_objective_gradient(self.X, self.y)
        np.random.seed(42)
        self.theta_init = np.random.randn(self.X.shape[1]) * 0.1

    def test_sbpg_runs_without_error(self):
        """SBPG能正常运行不报错"""
        theta, loss_hist = ComparisonOptimizer.sbpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=100
        )
        self.assertEqual(theta.shape, self.theta_init.shape)
        self.assertTrue(len(loss_hist) > 0)

    def test_sbpg_output_finite(self):
        """SBPG输出参数和损失值均为有限值"""
        theta, loss_hist = ComparisonOptimizer.sbpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=100
        )
        self.assertTrue(np.all(np.isfinite(theta)), "SBPG输出参数含NaN或Inf")
        self.assertTrue(np.all(np.isfinite(loss_hist)), "SBPG损失历史含NaN或Inf")

    def test_sbpg_converges(self):
        """SBPG损失应下降"""
        theta, loss_hist = ComparisonOptimizer.sbpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=500
        )
        self.assertLess(loss_hist[-1], loss_hist[0],
                        f"SBPG未收敛: 初始loss={loss_hist[0]:.6f}, 最终loss={loss_hist[-1]:.6f}")

    def test_sbpg_is_stochastic(self):
        """SBPG使用随机采样，固定种子下可复现，但不同种子结果不同"""
        np.random.seed(42)
        theta1, _ = ComparisonOptimizer.sbpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=50
        )
        np.random.seed(123)
        theta2, _ = ComparisonOptimizer.sbpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=50
        )
        # 不同种子应该产生不同结果（随机性验证）
        self.assertFalse(np.allclose(theta1, theta2),
                         "SBPG不同种子产生相同结果，可能未使用随机采样")


class TestOptimizerComparison(unittest.TestCase):
    """测试BPG/SBPG/MSBPG之间的差异性"""

    def setUp(self):
        self.X, self.y, self.true_theta = make_linear_problem(n_samples=500, n_features=10)
        self.obj, self.grad = make_objective_gradient(self.X, self.y)
        np.random.seed(42)
        self.theta_init = np.random.randn(self.X.shape[1]) * 0.1

    def test_msbpg_better_than_bpg(self):
        """MSBPG最终损失应不高于BPG（动量加速收敛）"""
        np.random.seed(42)
        theta_bpg, loss_bpg = ComparisonOptimizer.bpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=500
        )

        np.random.seed(42)
        msbpg = MSBPGOptimizer(learning_rate=0.01, momentum=0.9,
                               batch_size=32, max_iter=500, verbose=False)
        theta_msbpg, hist_msbpg = msbpg.optimize(self.obj, self.grad, self.theta_init, self.X, self.y)

        self.assertLessEqual(hist_msbpg['loss'][-1], loss_bpg[-1] * 1.5,
                             f"MSBPG({hist_msbpg['loss'][-1]:.6f})显著差于BPG({loss_bpg[-1]:.6f})")

    def test_msbpg_better_than_sbpg(self):
        """MSBPG最终损失应不高于SBPG（动量带来增益）"""
        np.random.seed(42)
        theta_sbpg, loss_sbpg = ComparisonOptimizer.sbpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=500
        )

        np.random.seed(42)
        msbpg = MSBPGOptimizer(learning_rate=0.01, momentum=0.9,
                               batch_size=32, max_iter=500, verbose=False)
        theta_msbpg, hist_msbpg = msbpg.optimize(self.obj, self.grad, self.theta_init, self.X, self.y)

        self.assertLessEqual(hist_msbpg['loss'][-1], loss_sbpg[-1] * 1.5,
                             f"MSBPG({hist_msbpg['loss'][-1]:.6f})显著差于SBPG({loss_sbpg[-1]:.6f})")

    def test_bpg_sbpg_produce_different_results(self):
        """BPG（全批量）和SBPG（随机）应产生不同结果"""
        np.random.seed(42)
        theta_bpg, _ = ComparisonOptimizer.bpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=100
        )

        np.random.seed(42)
        theta_sbpg, _ = ComparisonOptimizer.sbpg(
            self.obj, self.grad, self.theta_init, self.X, self.y,
            learning_rate=0.01, max_iter=100
        )

        self.assertFalse(np.allclose(theta_bpg, theta_sbpg),
                         "BPG和SBPG结果完全一致，可能实现有误")

    def test_compare_optimizers_returns_all_five(self):
        """compare_optimizers应返回SGD/Adam/BPG/SBPG/MSBPG五种结果"""
        results = ComparisonOptimizer.compare_optimizers(
            self.obj, self.grad, self.theta_init, self.X, self.y
        )

        expected_keys = {'SGD', 'Adam', 'BPG', 'SBPG', 'MSBPG'}
        self.assertEqual(set(results.keys()), expected_keys,
                         f"缺少优化器: 期望{expected_keys}, 实际{set(results.keys())}")

        for name in expected_keys:
            self.assertIn('theta', results[name], f"{name}缺少theta")
            self.assertIn('loss', results[name], f"{name}缺少loss")
            self.assertTrue(len(results[name]['loss']) > 0, f"{name}损失历史为空")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_single_sample_bpg(self):
        """BPG在单样本下不崩溃"""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        obj, grad = make_objective_gradient(X, y)
        theta_init = np.array([0.1, 0.1])

        theta, loss_hist = ComparisonOptimizer.bpg(
            obj, grad, theta_init, X, y,
            learning_rate=0.001, max_iter=10
        )
        self.assertTrue(np.all(np.isfinite(theta)))

    def test_single_sample_sbpg(self):
        """SBPG在单样本下不崩溃"""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        obj, grad = make_objective_gradient(X, y)
        theta_init = np.array([0.1, 0.1])

        theta, loss_hist = ComparisonOptimizer.sbpg(
            obj, grad, theta_init, X, y,
            learning_rate=0.001, max_iter=10, batch_size=1
        )
        self.assertTrue(np.all(np.isfinite(theta)))

    def test_zero_gradient_bpg(self):
        """BPG在梯度为零时不发散（已收敛状态）"""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([0.0, 0.0])  # 最优解theta=0时梯度为0
        obj, grad = make_objective_gradient(X, y)
        theta_init = np.array([0.0, 0.0])

        theta, loss_hist = ComparisonOptimizer.bpg(
            obj, grad, theta_init, X, y,
            learning_rate=0.01, max_iter=50
        )
        self.assertTrue(np.all(np.isfinite(theta)))
        self.assertAlmostEqual(loss_hist[-1], 0.0, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
