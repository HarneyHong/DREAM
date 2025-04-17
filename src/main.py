import asyncio
import ast
import json
import logging
import time
import pandas as pd
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agent.db_agent import DBAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='慢查询诊断工具')
    parser.add_argument('--data_path', type=str, required=True, help='慢查询数据文件路径')
    return parser.parse_args()

def create_dir_if_not_exists(dir_path: str):
    """创建目录（如果不存在）"""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)

def read_slow_queries(file_path: str) -> List[Dict[str, Any]]:
    """读取慢查询数据"""
    df = pd.read_csv(file_path)
    queries = []
    for _, row in df.iterrows():
        # 修改
        # plan_dict = convert_plan_to_dict(row["plan_json"])
        plan_dict = row["plan_json"]
        query_info = {
            "query_id": row.get("query_id", f"{len(queries)}"),
            "query": row["query"],
            "query_plan": plan_dict,
            "query_kpis": row["timeseries"],
            "log_all": row["log_all"],
            "execution_time": float(row["duration"])
        }
        queries.append(query_info)
    return queries

def create_diagnostic_ticket(query: Dict[str, Any], result_dir: str) -> Tuple[Dict[str, Any], str]:
    """
    创建诊断工单并生成异常信息
    Args:
        query: 查询相关信息
        result_dir: 结果目录路径
    Returns:
        Dict[str, Any]: 异常信息
    """
    # 解析查询基本信息
    execution_time = query.get("execution_time", 0)
    query_text = query.get("query", "")
    
    # 解析时序数据
    query_kpis = json.loads(query.get("query_kpis", "[]"))
    kpi_descriptions = {}
    if query_kpis:
        # 修改
        metrics = {
            "cpu_usage": query_kpis[0],
            "memory_usage": query_kpis[1],
            "io_wait": query_kpis[2],
            "network_traffic": query_kpis[3]
        }
        for metric_name, values in metrics.items():
            avg_value = sum(values) / len(values) if values else 0
            kpi_descriptions[metric_name] = f"avg_value: {avg_value:.2f}"
    
    # 解析日志信息
    log_all = json.loads(query.get("log_all", "[]"))
    log_info = {
        "read_rows": log_all[0] if len(log_all) > 0 else 0,
        "write_rows": log_all[1] if len(log_all) > 1 else 0,
        "scan_rows": log_all[2] if len(log_all) > 2 else 0
    }
    
    # 生成工单描述
    ticket_description = {
        "query_id": query.get("query_id", "unknown"),
        "query": query_text,
        "execution_time": f"{execution_time:.2f}ms",
        "query_plan": query.get("query_plan", "unknown"),
        "kpis": metrics,
        "kpi_descriptions": kpi_descriptions,
        "log_info": log_info,
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存工单信息
    ticket_file = f"{result_dir}/ticket_slow_query_{query.get('query_id', 'unknown')}.json"
    with open(ticket_file, "w", encoding="utf8") as f:
        json.dump(ticket_description, f, ensure_ascii=False, indent=4)
    
    # 返回异常信息
    return {
        "kpis": metrics,
        "kpi_descriptions": kpi_descriptions,
        "log_info": log_info,
    }, ticket_file

async def process_slow_query(agent: DBAgent, query: Dict[str, Any], path: str):
    """处理慢查询"""

    # 为每个查询创建诊断工单和异常信息
    anomaly_info, ticket_file = create_diagnostic_ticket(query, path)
    
    # 诊断和修复
    try:
        # result = await agent.diagnose_and_repair(query, anomaly_info)
        result = "success"
    except Exception as e:
        logger.error(f"处理查询 {query['query_id']} 时出错: {str(e)}")
    
    return result, ticket_file

async def main():
    # 解析命令行参数
    args = parse_args()
    
    SLOW_QUERIES_FILE = args.data_path
    DIAGNOSTIC_RESULTS_PATH = "../diagnostic_results"

    # 创建结果目录
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = f"{DIAGNOSTIC_RESULTS_PATH}/{current_time}"
    create_dir_if_not_exists(result_dir)
    
    # 初始化代理
    agent = DBAgent()
    
    try:
        # 读取慢查询
        logger.info("读取慢查询数据...")
        queries = read_slow_queries(SLOW_QUERIES_FILE)
        logger.info(f"找到 {len(queries)} 个慢查询")
        
        # 处理查询
        logger.info("开始处理慢查询...")
        
        for i, query in enumerate(queries):
            logger.info(f"处理第 {i+1}/{len(queries)} 个查询")
            start_time = time.time()

            # 慢查询异常诊断
            result, ticket_file = await process_slow_query(agent, query, result_dir)

            end_time = time.time()
            
            # 保存结果
            output = {
                "result": result,
                "diagnostic_time": current_time,
                "processing_time": end_time - start_time
            }

            with open(ticket_file, "r", encoding="utf8") as f:
                ticket_description = json.load(f)
            
            ticket_description.update(output)

            with open(ticket_file, "w", encoding="utf8") as f:
                json.dump(ticket_description, f, ensure_ascii=False, indent=4)
            
            logger.info(f"查询处理完成，耗时: {end_time - start_time:.2f}秒")
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
