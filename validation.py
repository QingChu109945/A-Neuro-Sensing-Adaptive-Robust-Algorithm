import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .adaptive_threshold import AdaptiveThresholdCalculator, ThresholdResult, StabilityReport, create_adaptive_threshold_calculator

@dataclass
class ValidationResult:
    """验证结果"""
    test_name: str
    passed: bool
    metric: float
    threshold: float
    details: str = ""

class DataValidator:
    """数据验证器"""
    
    def __init__(self, consistency_threshold: float = 0.005,
                 reliability_threshold: float = 0.02,
                 outlier_threshold: float = 3.0,
                 use_adaptive_threshold: bool = True,
                 target_pass_rate: float = 0.95,
                 max_variance: float = 0.05,
                 consecutive_pass_requirement: int = 10):
        self.consistency_threshold = consistency_threshold
        self.reliability_threshold = reliability_threshold
        self.outlier_threshold = outlier_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        
        if use_adaptive_threshold:
            self.adaptive_calculator = create_adaptive_threshold_calculator(
                target_pass_rate=target_pass_rate,
                max_variance=max_variance,
                consecutive_pass_requirement=consecutive_pass_requirement
            )
        else:
            self.adaptive_calculator = None
    
    def validate_consistency(self, data1: np.ndarray, data2: np.ndarray,
                           test_name: str = "Consistency Test") -> ValidationResult:
        """验证数据一致性"""
        if len(data1) != len(data2):
            return ValidationResult(test_name, False, 0.0, self.consistency_threshold,
                                  "Data length mismatch")
        
        diff = np.abs(data1 - data2)
        max_diff = np.max(diff)
        
        passed = max_diff <= self.consistency_threshold
        details = f"Max difference: {max_diff:.6f}, Threshold: {self.consistency_threshold}"
        
        return ValidationResult(test_name, passed, max_diff, self.consistency_threshold, details)
    
    def validate_reliability(self, data: np.ndarray, test_name: str = "Reliability Test") -> ValidationResult:
        """验证数据可靠性（重复性）"""
        if len(data) < 10:
            return ValidationResult(test_name, False, 0.0, self.reliability_threshold,
                                  "Insufficient data for reliability test")
        
        cv = np.std(data) / np.mean(data) if np.mean(data) != 0 else np.inf
        passed = cv <= self.reliability_threshold
        
        details = f"Coefficient of Variation: {cv:.6f}, Threshold: {self.reliability_threshold}"
        
        return ValidationResult(test_name, passed, cv, self.reliability_threshold, details)
    
    def validate_outliers(self, data: np.ndarray, test_name: str = "Outlier Test") -> ValidationResult:
        """检测异常值"""
        mean = np.mean(data)
        std = np.std(data)
        outliers = np.abs(data - mean) > self.outlier_threshold * std
        outlier_ratio = np.mean(outliers)
        
        passed = outlier_ratio <= 0.01
        details = f"Outlier ratio: {outlier_ratio:.6f}, Threshold: 0.01"
        
        return ValidationResult(test_name, passed, outlier_ratio, 0.01, details)
    
    def validate_data_integrity(self, data: Dict[str, np.ndarray], 
                               test_name: str = "Data Integrity Test") -> ValidationResult:
        """验证数据完整性"""
        total_samples = 0
        missing_samples = 0
        
        for sensor_name, sensor_data in data.items():
            total_samples += len(sensor_data)
            if isinstance(sensor_data, np.ndarray):
                missing_samples += np.sum(np.isnan(sensor_data)) + np.sum(np.isinf(sensor_data))
        
        missing_ratio = missing_samples / total_samples if total_samples > 0 else 1.0
        passed = missing_ratio <= 0.001
        
        details = f"Missing ratio: {missing_ratio:.6f}, Total samples: {total_samples}"
        
        return ValidationResult(test_name, passed, missing_ratio, 0.001, details)
    
    def validate_distribution_consistency(self, data1: np.ndarray, data2: np.ndarray,
                                        test_name: str = "Distribution Consistency") -> ValidationResult:
        """验证分布一致性（KS检验）"""
        try:
            stat, p_value = stats.ks_2samp(data1.flatten(), data2.flatten())
            passed = p_value > 0.05
            
            details = f"KS statistic: {stat:.6f}, p-value: {p_value:.6f}"
            
            return ValidationResult(test_name, passed, p_value, 0.05, details)
        except:
            return ValidationResult(test_name, False, 0.0, 0.05, "KS test failed")
    
    def validate_correlation(self, data1: np.ndarray, data2: np.ndarray,
                            test_name: str = "Correlation Test") -> ValidationResult:
        """验证相关性"""
        try:
            corr, p_value = stats.pearsonr(data1.flatten(), data2.flatten())
            passed = abs(corr) > 0.9
            
            details = f"Pearson correlation: {corr:.6f}, p-value: {p_value:.6f}"
            
            return ValidationResult(test_name, passed, corr, 0.9, details)
        except:
            return ValidationResult(test_name, False, 0.0, 0.9, "Correlation test failed")
    
    def validate_temperature_compensation(self, raw_data: np.ndarray, 
                                         compensated_data: np.ndarray,
                                         temperatures: np.ndarray,
                                         test_name: str = "Temperature Compensation Test") -> ValidationResult:
        """验证温度补偿效果"""
        raw_std = np.std(raw_data)
        comp_std = np.std(compensated_data)
        
        improvement = (raw_std - comp_std) / raw_std if raw_std != 0 else 0.0
        passed = improvement > 0.5
        
        details = f"Raw std: {raw_std:.6f}, Compensated std: {comp_std:.6f}, Improvement: {improvement:.6f}"
        
        return ValidationResult(test_name, passed, improvement, 0.5, details)
    
    def validate_filter_performance(self, filtered_data: np.ndarray,
                                   ground_truth: np.ndarray,
                                   test_name: str = "Filter Performance Test") -> ValidationResult:
        """验证滤波性能"""
        rmse = np.sqrt(np.mean((filtered_data - ground_truth)**2))
        passed = rmse < 0.1
        
        details = f"RMSE: {rmse:.6f}, Threshold: 0.1"
        
        return ValidationResult(test_name, passed, rmse, 0.1, details)
    
    def validate_inversion_constraint(self, emissivity: np.ndarray,
                                     reflectivity: np.ndarray,
                                     test_name: str = "Inversion Constraint Test") -> ValidationResult:
        """验证反演物理约束"""
        violations = np.sum(emissivity + reflectivity > 1.01)
        violation_rate = violations / len(emissivity)
        passed = violation_rate == 0.0
        
        details = f"Constraint violations: {violations}, Violation rate: {violation_rate:.6f}"
        
        return ValidationResult(test_name, passed, violation_rate, 0.0, details)
    
    def run_complete_validation(self, data: Dict[str, np.ndarray],
                               reference_data: Dict[str, np.ndarray] = None) -> Tuple[List[ValidationResult], Optional[StabilityReport]]:
        """运行完整验证流程"""
        results = []
        
        results.append(self.validate_data_integrity(data))
        
        adaptive_results = []
        for sensor_name, sensor_data in data.items():
            # 跳过时间戳/1D 派生数组: 它们不是传感器测量, 其归一化指标会
            # 与真实传感器通道量纲不一致, 污染稳定性方差(Result Variance)。
            if sensor_name.endswith('_timestamps'):
                continue
            if isinstance(sensor_data, np.ndarray) and len(sensor_data) > 0:
                if self.use_adaptive_threshold and self.adaptive_calculator:
                    adaptive_result = self.adaptive_calculator.compute_adaptive_threshold(
                        sensor_name, sensor_data
                    )
                    adaptive_results.append(adaptive_result)
                    
                    details = f"{adaptive_result.metric}: {adaptive_result.raw_metric_value:.6f}, "
                    details += f"Threshold: {adaptive_result.computed_threshold:.6f}, "
                    details += f"Reason: {adaptive_result.adjustment_reason}"
                    
                    results.append(ValidationResult(
                        test_name=f"{sensor_name} Reliability",
                        passed=adaptive_result.passed,
                        metric=adaptive_result.normalized_metric_value,
                        threshold=adaptive_result.computed_threshold,
                        details=details
                    ))
                else:
                    results.append(self.validate_reliability(sensor_data, 
                                                            f"{sensor_name} Reliability"))
                
                results.append(self.validate_outliers(sensor_data, 
                                                     f"{sensor_name} Outliers"))
        
        if reference_data:
            for sensor_name in data:
                if sensor_name in reference_data:
                    results.append(self.validate_consistency(
                        data[sensor_name], 
                        reference_data[sensor_name],
                        f"{sensor_name} vs Reference Consistency"
                    ))
                    results.append(self.validate_distribution_consistency(
                        data[sensor_name],
                        reference_data[sensor_name],
                        f"{sensor_name} Distribution Consistency"
                    ))
                    results.append(self.validate_correlation(
                        data[sensor_name],
                        reference_data[sensor_name],
                        f"{sensor_name} Correlation"
                    ))
        
        stability_report = None
        if self.use_adaptive_threshold and self.adaptive_calculator:
            stability_report = self.adaptive_calculator.check_stability()
        
        return results, stability_report
    
    def generate_validation_report(self, results: List[ValidationResult], 
                                  stability_report: StabilityReport = None) -> str:
        """生成验证报告"""
        report = """
========================================
        Data Validation Report
========================================
"""
        
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        
        report += f"\nOverall: {passed_count}/{total_count} tests passed\n"
        report += f"Pass Rate: {(passed_count/total_count*100):.1f}%\n"
        
        if stability_report:
            report += f"\nStability: {'✓ STABLE' if stability_report.stable else '✗ UNSTABLE'}\n"
            report += f"Consecutive Passes: {stability_report.consecutive_pass_count}\n"
            report += f"Long-term Pass Rate: {stability_report.pass_rate:.1%}\n"
        
        report += """
----------------------------------------
Test Details
----------------------------------------
"""
        
        for result in results:
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            report += f"\n{status}: {result.test_name}\n"
            report += f"  Metric: {result.metric:.6f}\n"
            report += f"  Threshold: {result.threshold:.6f}\n"
            report += f"  Details: {result.details}\n"
        
        if stability_report:
            report += """
----------------------------------------
Stability Assessment
----------------------------------------
"""
            report += f"  Consecutive Passes: {stability_report.consecutive_pass_count}\n"
            report += f"  Pass Rate: {stability_report.pass_rate:.1%}\n"
            report += f"  Result Variance: {stability_report.result_variance:.2%}\n"
            report += f"  Overall Stability: {'✓ STABLE' if stability_report.stable else '✗ UNSTABLE'}\n"
            report += "\n  Recommendations:\n"
            for rec in stability_report.recommendations:
                report += f"    {rec}\n"
        
        report += """
========================================
              End of Report
========================================
"""
        
        return report

class FieldLabValidation:
    """外场与实验室数据融合验证"""
    
    def __init__(self):
        self.validator = DataValidator()
    
    def validate_field_lab_consistency(self, field_data: Dict[str, np.ndarray],
                                      lab_data: Dict[str, np.ndarray]) -> Dict[str, ValidationResult]:
        """验证外场与实验室数据一致性"""
        results = {}
        
        for sensor_name in field_data:
            if sensor_name in lab_data:
                field_vals = field_data[sensor_name]
                lab_vals = lab_data[sensor_name]
                
                min_len = min(len(field_vals), len(lab_vals))
                if min_len > 0:
                    results[f"{sensor_name}_consistency"] = \
                        self.validator.validate_consistency(field_vals[:min_len], lab_vals[:min_len])
                    results[f"{sensor_name}_distribution"] = \
                        self.validator.validate_distribution_consistency(field_vals, lab_vals)
                    results[f"{sensor_name}_correlation"] = \
                        self.validator.validate_correlation(field_vals, lab_vals)
        
        return results
    
    def compute_correction_factor(self, field_data: np.ndarray,
                                  lab_data: np.ndarray) -> Tuple[float, float]:
        """计算校正因子"""
        min_len = min(len(field_data), len(lab_data))
        
        if min_len < 2:
            return 1.0, 0.0
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            lab_data[:min_len], field_data[:min_len]
        )
        
        return slope, intercept
    
    def apply_correction(self, data: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        """应用校正"""
        return data * slope + intercept
    
    def validate_dataset_split(self, dataset: Dict[str, np.ndarray],
                              field_ratio: float = 0.8) -> ValidationResult:
        """验证数据集划分比例"""
        total_samples = sum(len(v) for v in dataset.values())
        field_samples = int(total_samples * field_ratio)
        
        return ValidationResult(
            "Dataset Split Validation",
            True,
            field_ratio,
            field_ratio,
            f"Total samples: {total_samples}, Field samples target: {field_samples}"
        )

def create_default_validator() -> DataValidator:
    """创建默认数据验证器"""
    return DataValidator()

def create_field_lab_validator() -> FieldLabValidation:
    """创建外场实验室验证器"""
    return FieldLabValidation()