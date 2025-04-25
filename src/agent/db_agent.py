import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agent.plan.planner import Planner
from agent.action.action_manager import ActionManager
from agent.memory.memory_manager import MemoryManager
from utils.types import QueryInfo, AnomalyInfo, CaseInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DBAgent:
    def __init__(self, memory_path: str = "memory.db", max_attempts: int = 1):
        self.memory_manager = MemoryManager(memory_path)
        self.planner = Planner(self.memory_manager)
        self.action_manager = ActionManager()
        self.max_attempts = max_attempts
    
    async def diagnose_and_repair(self, 
                                query_info: Dict[str, Any], 
                                anomaly_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        主要诊断和修复流程，采用循环方式直到问题解决或达到最大尝试次数
        """
        logger.info(f"开始诊断查询: {query_info['query_id']}")
        
        # 构造输入对象进行类型检查
        query = QueryInfo(
            query_id=query_info['query_id'],
            query=query_info['query'],
            query_plan=query_info['query_plan'],
            query_kpis=query_info['query_kpis'],
            log_all=query_info['log_all'],
            execution_time=query_info['execution_time']
        )

        anomaly = AnomalyInfo(
            kpis=anomaly_info['kpis'],
            kpi_descriptions=anomaly_info['kpi_descriptions'],
            log_info=anomaly_info['log_info']
        )

        attempts = 0
        all_results = []
        is_resolved = False
        
        while attempts < self.max_attempts and not is_resolved:
            attempts += 1
            logger.info(f"开始第 {attempts} 次诊断修复尝试")
            
            # 从内存模块获取历史信息
            # historical_data = self.memory_manager.retrieve_similar_cases(query_info, anomaly_info)
            historical_data = []
            
            # 分析阶段
            analysis_result = self.planner.analyze_query(query, anomaly, historical_data)
                        
            if not analysis_result['root_cause']:
                logger.info("未发现新的问题根因，终止诊断")
                break
            else:
                # logger.info(f"第 {attempts} 次分析结果: 目前针对该慢查询，最主要的根因是：{analysis_result['root_cause']}, 置信度为：{analysis_result['confidence']}")
                logger.info(f"第 {attempts} 次分析结果: 目前针对该慢查询，最主要的根因是：{analysis_result['root_cause']}")
            
            # 执行修复操作
            action_results = await self.action_manager.analyze_and_act(
                analysis_result,
                query,
                anomaly,
                historical_data
            )
            # logger.info(f"第 {attempts} 次执行结果: {action_results}")
            
            # 评估修复效果
            # evaluation = self.planner.evaluate_action(
            #     action_results,
            #     query,
            #     simulation_mode=True
            # )
            # logger.info(f"第 {attempts} 次评估结果: {evaluation}")
            evaluation = "success"
            
            # 存储本次尝试结果
            attempt_result = {
                "attempt": attempts,
                "analysis": analysis_result,
                "actions": action_results,
                "evaluation": evaluation
            }
            all_results.append(attempt_result)
            
            # # 判断是否解决问题
            # is_resolved = evaluation[0]  # 假设evaluation[0]表示是否成功解决问题
            # if is_resolved:
            #     logger.info("问题已解决，终止诊断")
            #     break
            
            # # 更新异常信息用于下一轮诊断
            # # 注意：这里需要重新获取最新的系统状态和异常信息
            # # 实际实现中可能需要从监控系统获取最新数据
            # anomaly = self._get_updated_anomaly_info(query)  # 这个方法需要另外实现
        
        # # 存储最终案例
        # final_result = all_results[-1] if all_results else None
        # if final_result:
        #     case_info = CaseInfo(
        #         query_info=query,
        #         anomaly_info=anomaly,
        #         root_causes=final_result['analysis']['root_causes'],
        #         actions_taken=final_result['actions'],
        #         result_metrics=final_result['evaluation'][1],
        #         confidence=final_result['analysis']['confidence'],
        #         status="completed" if is_resolved else "partial",
        #         attempts=attempts
        #     )
        #     # self.memory_manager.store_case(case_info.__dict__)
            
        is_resolved = True
        
        return {
            "is_resolved": is_resolved,
            "attempts": attempts,
            "all_results": all_results,
            "case_id": query.query_id
        }
    
    def _get_updated_anomaly_info(self, query: QueryInfo) -> AnomalyInfo:
        """
        获取最新的系统状态和异常信息
        这个方法需要根据实际监控系统来实现
        """
        # TODO: 实现从监控系统获取最新状态的逻辑
        raise NotImplementedError("This method needs to be implemented")
    
    async def cleanup(self):
        """清理资源"""
        await self.action_manager.cleanup() 

def convert_plan_to_dict(plan_str):
    """将字符串形式的执行计划转换为字典格式"""
    try:
        # 清理输入字符串
        # 1. 移除行首的索引号和空格（如 "0    "）
        if plan_str.strip()[0].isdigit():
            plan_str = plan_str.split(None, 1)[1]
        
        # 2. 移除pandas Series的额外信息
        if '\nName: QUERY PLAN' in plan_str:
            plan_str = plan_str.split('\nName: QUERY PLAN')[0]
            
        # 3. 移除dtype信息
        if ', dtype: object' in plan_str:
            plan_str = plan_str.split(', dtype: object')[0]
        
        # 使用ast.literal_eval安全地将字符串转换为Python对象
        result = ast.literal_eval(plan_str)
        # 如果结果是列表，返回第一个元素
        if isinstance(result, list):
            return result[0]
        return result
    except (SyntaxError, ValueError) as e:
        print(f"Covert Plan Json Error: {str(e)}")
        print(f"Input Data: {plan_str}")
        return {"error": "Invalid Plan Json"}