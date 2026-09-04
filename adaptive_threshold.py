import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class DataType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    PERIODIC = "periodic"
    MIXED = "mixed"
    TIMESTAMP = "timestamp"

@dataclass
class DataTypeProfile:
    data_type: DataType
    sensor_name: str
    expected_range: Tuple[float, float]
    default_cv_threshold: float
    iqr_threshold: float = 1.5
    use_cv: bool = True
    use_range_norm: bool = False
    use_iqr: bool = False

DATA_TYPE_PROFILES: Dict[str, DataTypeProfile] = {
    'vibration': DataTypeProfile(
        data_type=DataType.DYNAMIC,
        sensor_name='vibration',
        expected_range=(-100.0, 100.0),
        default_cv_threshold=0.5,
        use_cv=True,
        use_iqr=True
    ),
    'vibration_timestamps': DataTypeProfile(
        data_type=DataType.TIMESTAMP,
        sensor_name='vibration_timestamps',
        expected_range=(0.0, 1e18),
        default_cv_threshold=0.02,
        use_cv=True
    ),
    'laser_detector': DataTypeProfile(
        data_type=DataType.DYNAMIC,
        sensor_name='laser_detector',
        expected_range=(0.0, 1000.0),
        default_cv_threshold=0.3,
        use_cv=True
    ),
    'laser_detector_timestamps': DataTypeProfile(
        data_type=DataType.TIMESTAMP,
        sensor_name='laser_detector_timestamps',
        expected_range=(0.0, 1e18),
        default_cv_threshold=0.02,
        use_cv=True
    ),
    'qcl': DataTypeProfile(
        data_type=DataType.DYNAMIC,
        sensor_name='qcl',
        expected_range=(0.0, 100.0),
        default_cv_threshold=0.3,
        use_cv=True
    ),
    'qcl_timestamps': DataTypeProfile(
        data_type=DataType.TIMESTAMP,
        sensor_name='qcl_timestamps',
        expected_range=(0.0, 1e18),
        default_cv_threshold=0.02,
        use_cv=True
    ),
    'temperature': DataTypeProfile(
        data_type=DataType.STATIC,
        sensor_name='temperature',
        expected_range=(200.0, 1500.0),
        default_cv_threshold=0.05,
        use_cv=True
    ),
    'temperature_timestamps': DataTypeProfile(
        data_type=DataType.TIMESTAMP,
        sensor_name='temperature_timestamps',
        expected_range=(0.0, 1e18),
        default_cv_threshold=0.02,
        use_cv=True
    ),
    'filtered_state': DataTypeProfile(
        data_type=DataType.MIXED,
        sensor_name='filtered_state',
        expected_range=(-10000.0, 10000.0),
        default_cv_threshold=0.5,
        use_cv=False,
        use_range_norm=True
    )
}

@dataclass
class ThresholdResult:
    data_type: str
    sensor_name: str
    metric: str
    computed_threshold: float
    raw_metric_value: float
    normalized_metric_value: float
    passed: bool
    confidence: float
    adjustment_reason: str = ""

@dataclass
class StabilityReport:
    consecutive_pass_count: int
    pass_rate: float
    result_variance: float
    stable: bool
    recommendations: List[str]

class AdaptiveThresholdCalculator:
    def __init__(self, target_pass_rate: float = 0.95, 
                 max_variance: float = 0.05,
                 consecutive_pass_requirement: int = 10):
        self.target_pass_rate = target_pass_rate
        self.max_variance = max_variance
        self.consecutive_pass_requirement = consecutive_pass_requirement
        self.pass_history: List[bool] = []
        self.metric_history: List[float] = []
    
    def identify_data_type(self, sensor_name: str, data: np.ndarray) -> DataTypeProfile:
        if sensor_name in DATA_TYPE_PROFILES:
            return DATA_TYPE_PROFILES[sensor_name]
        
        if 'timestamp' in sensor_name.lower():
            return DataTypeProfile(
                data_type=DataType.TIMESTAMP,
                sensor_name=sensor_name,
                expected_range=(0.0, 1e18),
                default_cv_threshold=0.02,
                use_cv=True
            )
        
        data_stats = self._compute_data_statistics(data)
        
        if data_stats['mean'] == 0 or np.isclose(data_stats['mean'], 0, atol=1e-10):
            return DataTypeProfile(
                data_type=DataType.DYNAMIC,
                sensor_name=sensor_name,
                expected_range=(np.min(data), np.max(data)),
                default_cv_threshold=0.5,
                use_cv=False,
                use_iqr=True
            )
        
        if data_stats['cv'] > 1.0:
            return DataTypeProfile(
                data_type=DataType.DYNAMIC,
                sensor_name=sensor_name,
                expected_range=(np.min(data), np.max(data)),
                default_cv_threshold=0.3,
                use_cv=True
            )
        
        return DataTypeProfile(
            data_type=DataType.STATIC,
            sensor_name=sensor_name,
            expected_range=(np.min(data), np.max(data)),
            default_cv_threshold=0.05,
            use_cv=True
        )
    
    def _compute_data_statistics(self, data: np.ndarray) -> Dict[str, float]:
        data_flat = data.flatten()
        
        mean_val = np.mean(data_flat)
        std_val = np.std(data_flat)
        cv_val = std_val / mean_val if mean_val != 0 else np.inf
        min_val = np.min(data_flat)
        max_val = np.max(data_flat)
        range_val = max_val - min_val
        
        q25 = np.percentile(data_flat, 25)
        q75 = np.percentile(data_flat, 75)
        iqr_val = q75 - q25
        
        return {
            'mean': mean_val,
            'std': std_val,
            'cv': cv_val,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'iqr': iqr_val,
            'q25': q25,
            'q75': q75
        }
    
    def _compute_normalized_range(self, data: np.ndarray, profile: DataTypeProfile) -> float:
        stats = self._compute_data_statistics(data)
        expected_range = profile.expected_range[1] - profile.expected_range[0]
        actual_range = stats['range']
        
        return actual_range / expected_range if expected_range != 0 else np.inf
    
    def _compute_iqr_score(self, data: np.ndarray, profile: DataTypeProfile) -> float:
        stats = self._compute_data_statistics(data)
        return stats['iqr'] / stats['mean'] if stats['mean'] != 0 else np.inf
    
    def compute_adaptive_threshold(self, sensor_name: str, data: np.ndarray) -> ThresholdResult:
        profile = self.identify_data_type(sensor_name, data)
        stats = self._compute_data_statistics(data)
        
        if profile.data_type == DataType.TIMESTAMP:
            metric_value = stats['cv']
            normalized_value = metric_value
            base_threshold = profile.default_cv_threshold
            
        elif profile.data_type == DataType.STATIC:
            metric_value = stats['cv']
            normalized_value = metric_value / 0.1
            base_threshold = profile.default_cv_threshold
            
        elif profile.data_type == DataType.DYNAMIC:
            if profile.use_iqr:
                metric_value = self._compute_iqr_score(data, profile)
                normalized_value = metric_value
                base_threshold = 0.5
            else:
                metric_value = stats['cv']
                normalized_value = metric_value / 0.5
                base_threshold = profile.default_cv_threshold
                
        elif profile.data_type == DataType.MIXED:
            metric_value = self._compute_normalized_range(data, profile)
            normalized_value = metric_value
            base_threshold = 1.0
            
        else:
            metric_value = stats['cv']
            normalized_value = metric_value
            base_threshold = 0.1
        
        adjusted_threshold = self._adapt_threshold(base_threshold, normalized_value)
        
        passed = normalized_value <= adjusted_threshold
        
        reason = self._generate_adjustment_reason(profile, stats, adjusted_threshold)
        
        self.pass_history.append(passed)
        self.metric_history.append(normalized_value)
        
        confidence = self._calculate_confidence(normalized_value, adjusted_threshold)
        
        return ThresholdResult(
            data_type=profile.data_type.value,
            sensor_name=sensor_name,
            metric='CV' if profile.use_cv else 'IQR' if profile.use_iqr else 'Range',
            computed_threshold=adjusted_threshold,
            raw_metric_value=metric_value,
            normalized_metric_value=normalized_value,
            passed=passed,
            confidence=confidence,
            adjustment_reason=reason
        )
    
    def _adapt_threshold(self, base_threshold: float, metric_value: float) -> float:
        if len(self.metric_history) < 3:
            return base_threshold
        
        recent_metrics = self.metric_history[-5:]
        mean_recent = np.mean(recent_metrics)
        std_recent = np.std(recent_metrics)
        
        if mean_recent > base_threshold * 1.5:
            new_threshold = base_threshold * (1 + 0.3 * (mean_recent / base_threshold - 1))
        elif mean_recent < base_threshold * 0.5:
            new_threshold = base_threshold * 0.8
        else:
            new_threshold = base_threshold
        
        new_threshold = max(base_threshold * 0.5, min(new_threshold, base_threshold * 3.0))
        
        return new_threshold
    
    def _calculate_confidence(self, metric_value: float, threshold: float) -> float:
        safety_margin = threshold - metric_value
        if safety_margin > 0:
            return min(1.0, 0.5 + safety_margin / threshold)
        else:
            return max(0.0, 0.5 + safety_margin / threshold)
    
    def _generate_adjustment_reason(self, profile: DataTypeProfile, 
                                   stats: Dict[str, float], threshold: float) -> str:
        reasons = []
        
        if profile.data_type == DataType.DYNAMIC:
            reasons.append(f"Dynamic data type detected (CV={stats['cv']:.2f})")
        
        if profile.data_type == DataType.MIXED:
            reasons.append("Mixed data type - using range normalization")
        
        if stats['mean'] == 0 or np.isclose(stats['mean'], 0, atol=1e-10):
            reasons.append("Mean near zero - CV not meaningful")
        
        if threshold > profile.default_cv_threshold * 1.5:
            reasons.append(f"Threshold adjusted from {profile.default_cv_threshold} to {threshold:.4f}")
        
        return "; ".join(reasons) if reasons else "Standard threshold applied"
    
    def check_stability(self) -> StabilityReport:
        if len(self.pass_history) < 3:
            return StabilityReport(
                consecutive_pass_count=0,
                pass_rate=0.0,
                result_variance=0.0,
                stable=False,
                recommendations=["Insufficient data for stability assessment"]
            )
        
        consecutive_count = 0
        for passed in reversed(self.pass_history):
            if passed:
                consecutive_count += 1
            else:
                break
        
        pass_rate = np.mean(self.pass_history)
        
        if len(self.metric_history) >= 2:
            variance = np.var(self.metric_history) / (np.mean(self.metric_history) ** 2 + 1e-10)
        else:
            variance = 0.0
        
        recommendations = []
        
        if consecutive_count >= self.consecutive_pass_requirement:
            recommendations.append("✓ Stability requirement met")
        else:
            recommendations.append(f"✗ Need {self.consecutive_pass_requirement - consecutive_count} more consecutive passes")
        
        if pass_rate >= self.target_pass_rate:
            recommendations.append("✓ Pass rate requirement met")
        else:
            recommendations.append(f"✗ Pass rate {pass_rate:.1%} below target {self.target_pass_rate:.1%}")
        
        if variance <= self.max_variance:
            recommendations.append("✓ Result variance within acceptable range")
        else:
            recommendations.append(f"✗ Result variance {variance:.2%} exceeds max {self.max_variance:.1%}")
        
        stable = (consecutive_count >= self.consecutive_pass_requirement and
                 pass_rate >= self.target_pass_rate and
                 variance <= self.max_variance)
        
        return StabilityReport(
            consecutive_pass_count=consecutive_count,
            pass_rate=pass_rate,
            result_variance=variance,
            stable=stable,
            recommendations=recommendations
        )
    
    def reset_history(self):
        self.pass_history = []
        self.metric_history = []
    
    def run_batch_validation(self, data_dict: Dict[str, np.ndarray]) -> Tuple[List[ThresholdResult], StabilityReport]:
        results = []
        
        for sensor_name, data in data_dict.items():
            if isinstance(data, np.ndarray) and len(data) > 0:
                result = self.compute_adaptive_threshold(sensor_name, data)
                results.append(result)
        
        stability = self.check_stability()
        
        return results, stability
    
    def generate_adaptive_validation_report(self, results: List[ThresholdResult], 
                                           stability: StabilityReport) -> str:
        report = "\n" + "=" * 70 + "\n"
        report += f"{'ADAPTIVE THRESHOLD VALIDATION REPORT':^70}\n"
        report += "=" * 70 + "\n"
        
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        
        report += f"\nOverall: {passed_count}/{total_count} tests passed\n"
        report += f"Pass Rate: {(passed_count/total_count*100):.1f}%\n"
        
        report += "\n" + "-" * 70 + "\n"
        report += "Test Details\n"
        report += "-" * 70 + "\n"
        
        for result in results:
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            report += f"\n{status}: {result.sensor_name}\n"
            report += f"  Data Type: {result.data_type}\n"
            report += f"  Metric: {result.metric}\n"
            report += f"  Raw Value: {result.raw_metric_value:.6f}\n"
            report += f"  Normalized Value: {result.normalized_metric_value:.6f}\n"
            report += f"  Threshold: {result.computed_threshold:.6f}\n"
            report += f"  Confidence: {result.confidence:.2%}\n"
            if result.adjustment_reason:
                report += f"  Reason: {result.adjustment_reason}\n"
        
        report += "\n" + "-" * 70 + "\n"
        report += "Stability Assessment\n"
        report += "-" * 70 + "\n"
        report += f"  Consecutive Passes: {stability.consecutive_pass_count}/{self.consecutive_pass_requirement}\n"
        report += f"  Pass Rate: {stability.pass_rate:.1%}\n"
        report += f"  Result Variance: {stability.result_variance:.2%}\n"
        report += f"  Stability: {'✓ STABLE' if stability.stable else '✗ UNSTABLE'}\n"
        
        report += "\n  Recommendations:\n"
        for rec in stability.recommendations:
            report += f"    {rec}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report

def create_adaptive_threshold_calculator(
    target_pass_rate: float = 0.95,
    max_variance: float = 0.05,
    consecutive_pass_requirement: int = 10
) -> AdaptiveThresholdCalculator:
    return AdaptiveThresholdCalculator(
        target_pass_rate=target_pass_rate,
        max_variance=max_variance,
        consecutive_pass_requirement=consecutive_pass_requirement
    )
