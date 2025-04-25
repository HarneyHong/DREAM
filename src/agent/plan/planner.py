import os
import sys

# Add RCRank to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
rcrank_path = os.path.join(current_dir, "RCRank")
if rcrank_path not in sys.path:
    sys.path.append(rcrank_path)

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import json
import ast
import torch
from typing import Dict, List, Any, Tuple
from agent.memory.memory_manager import MemoryManager
from utils.types import QueryInfo, AnomalyInfo
from agent.plan.online_predict import RCRankPredictor
from model.modules.QueryFormer.utils import Encoding
from RCRank.utils.plan_encoding import PlanEncoder

class Planner:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
        # Set device
        self.device = 'cpu'  # Force CPU usage to avoid CUDA warnings
        
        # Initialize RCRankPredictor
        self.predictor = RCRankPredictor(
            model_path=os.path.join(current_dir, "model_res/GateComDiffPretrainModel tpc_h confidence eta0.07/best_model.pt"),
            train_data_path=os.path.join(current_dir, "../../../data/tpc_h.csv"),
            device=self.device,
            opt_threshold=0.5
        )
        
        # Initialize encoding components
        self.encoding = Encoding(None, {'NA': 0})
    
    def encode_plan(self, query_plan: Dict) -> Dict:
        """将查询执行计划编码为模型所需的格式"""
        # 创建一个只包含当前计划的DataFrame
        df = pd.DataFrame([{
            'plan_json': json.dumps({'Plan': query_plan}) if isinstance(query_plan, dict) else query_plan
        }])
        
        # 使用PlanEncoder处理执行计划
        plan_encoder = PlanEncoder(df=df, encoding=self.encoding)
        encoded_df = plan_encoder.df
        
        # 获取编码后的计划
        return encoded_df['json_plan_tensor'].iloc[0]
    
    def analyze_query(self, query_info: QueryInfo, anomaly_info: AnomalyInfo, historical_data: List[Dict]) -> Dict[str, Any]:
        """分析慢查询和异常信息，预测可能的根因"""
        
        # 构建预测所需的输入数据
        query, plan, timeseries, log = self._build_prediction_input(query_info, anomaly_info, historical_data)
        
        # 使用RCRankPredictor进行预测
        pred_label, pred_label_binary, pred_opt, pred_rank = self.predictor.predict(
            query=query,
            plan=plan,
            timeseries=timeseries,
            log=log
        )
        
        # 解析预测结果
        return self._extract_top_root_cause(pred_label, pred_label_binary, pred_opt)
    
    def evaluate_action(self, action_result: Dict[str, Any], 
                       query_info: QueryInfo, 
                       simulation_mode: bool = False) -> Tuple[bool, List[str]]:
        """评估修复操作的结果"""
        if simulation_mode:
            return self._simulate_action_impact(action_result, query_info)
        else:
            return self._evaluate_real_action_impact(action_result, query_info)
    
    def _build_prediction_input(self, query_info: QueryInfo, 
                              anomaly_info: AnomalyInfo, 
                              historical_data: List[Dict]) -> Tuple[str, Dict, List, List]:
        """构建RCRankPredictor所需的输入数据"""
        # 获取查询语句
        query = query_info.query if hasattr(query_info, 'query') else ""
        
        # 构建执行计划数据
        plan = self.encode_plan(query_info.query_plan)
        
        # 构建时间序列数据
        # timeseries = query_info.query_kpis.apply(json.loads) if hasattr(query_info, 'query_kpis') else []
        timeseries = json.loads(query_info.query_kpis)
        
        # 构建日志数据
        # log = query_info.log_all.apply(json.loads) if hasattr(query_info, 'log_all') else []
        log = json.loads(query_info.log_all)
        
        return query, plan, timeseries, log
    
    def _extract_top_root_cause(self, pred_label, pred_label_binary, pred_opt) -> Dict[str, Any]:
        """从预测结果中提取最主要的根因"""
        # 将预测结果转换到CPU并转为numpy数组
        pred_label = pred_label.cpu().numpy()
        pred_label_binary = pred_label_binary.cpu().numpy()
        pred_opt = pred_opt.cpu().numpy()
        
        # 定义问题类型映射
        issue_types = {
            0: "outdated statistical information",
            1: "under-optimized join order",
            2: "inappropriate distribution keys",
            3: "missing indexes",
            4: "redundant indexes",
            5: "repeatedly executing subqueries",
            6: "complex table joins",
            7: "updating an entire table",
            8: "inserting large data",
            9: "unknown root cause"
        }
        
        print("pred_label_binary: ", pred_label_binary)
        print("pred_label_binary_shape: ", pred_label_binary.shape)
        print("pred_opt: ", pred_opt)
        
        if pred_label_binary.sum() == 0:
            return {"root_cause": "unknown root cause"}
            # return {"root_cause": "unknown root cause", "confidence": 0.0}
        
        # 找到优化分数最高的问题
        max_score_idx = np.argmax(pred_opt[0])
        max_score = pred_opt[0][max_score_idx]
        
        # 获取对应的问题描述
        root_cause = {"root_cause": issue_types.get(max_score_idx)}
        # root_cause = {
        #     "root_cause": issue_types.get(max_score_idx),
        #     "confidence": float(max_score)
        # }
        
        # 返回最主要的根因
        return root_cause
    
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