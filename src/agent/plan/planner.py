import lime
import lime.lime_tabular
import numpy as np
from typing import Dict, List, Any, Tuple
from agent.memory.memory_manager import MemoryManager
from utils.types import QueryInfo, AnomalyInfo

class Planner:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
        # 先创建一些数据试试水
        self.feature_names = ['execution_time', 'cpu_usage', 'memory_usage', 'io_wait']
        self.training_data = np.array([
            [100, 50, 30, 20],  # 示例数据点
            [200, 60, 40, 25],
            [150, 55, 35, 22]
        ])
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=['normal', 'anomaly'],
            mode='classification'
        )
    
    def analyze_query(self, query_info: QueryInfo, anomaly_info: AnomalyInfo) -> Dict[str, Any]:
        """分析慢查询和异常信息，预测可能的根因"""
        
        # 构建特征向量
        # 修改
        features = self._build_features(query_info, anomaly_info, historical_data)
        
        # 使用LIME进行解释
        # 修改
        explanation = self.lime_explainer.explain_instance(
            features,
            self._predict_anomaly,  # 预测函数
            num_features=10
        )
        
        return {
            'root_causes': self._extract_root_causes(explanation),
            'confidence': explanation.score
        }
    
    def evaluate_action(self, action_result: Dict[str, Any], 
                       query_info: QueryInfo, 
                       simulation_mode: bool = False) -> Tuple[bool, List[str]]:
        """评估修复操作的结果"""
        if simulation_mode:
            return self._simulate_action_impact(action_result, query_info)
        else:
            return self._evaluate_real_action_impact(action_result, query_info)
    
    def _build_features(self, query_info: QueryInfo, 
                       anomaly_info: AnomalyInfo, 
                       historical_data: List[Dict]) -> np.ndarray:
        """构建用于LIME分析的特征向量"""
        
        # 修改  
        if not historical_data:
            # 没有历史数据的特征向量建模算法
            features = [
                float(query_info.execution_time),
                float(anomaly_info.get('metrics', {}).get('cpu_usage', 0)),
                float(anomaly_info.get('metrics', {}).get('memory_usage', 0)),
                float(anomaly_info.get('metrics', {}).get('io_wait', 0))
            ]
        else:
            # 能找到历史数据的特征向量建模算法
            features = [
                float(query_info.execution_time),
                float(anomaly_info.get('metrics', {}).get('cpu_usage', 0)),
                float(anomaly_info.get('metrics', {}).get('memory_usage', 0)),
                float(anomaly_info.get('metrics', {}).get('io_wait', 0))
            ]

        return np.array(features)
    
    def _predict_anomaly(self, features: np.ndarray) -> np.ndarray:
        """预测异常的概率"""
        # 简单的阈值判断
        is_anomaly = float(features[0]) > 150 or features[1] > 70  # 如果执行时间>150ms或CPU使用率>70%则判定为异常
        return np.array([1 - is_anomaly, is_anomaly])
    
    def _extract_root_causes(self, explanation) -> List[str]:
        """从LIME解释中提取根因"""
        root_causes = []
        for feature, importance in explanation.as_list():
            if abs(importance) > 0.1:  # 重要性阈值
                root_causes.append(f"{feature}: {importance}")
        return root_causes
    
    def _simulate_action_impact(self, action_result: Dict[str, Any], 
                              query_info: QueryInfo) -> Tuple[bool, List[str]]:
        """模拟评估修复操作的影响"""
        # TODO: 实现模拟评估逻辑
        return True, ["模拟评估通过"]
    
    def _evaluate_real_action_impact(self, action_result: Dict[str, Any], 
                                   query_info: QueryInfo) -> Tuple[bool, List[str]]:
        """评估真实执行的修复操作的影响"""
        # TODO: 实现真实评估逻辑
        return True, ["实际执行评估通过"] 