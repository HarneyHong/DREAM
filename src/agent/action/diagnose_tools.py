from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword
from sqlparse.sql import Where, Comparison
from sqlalchemy import create_engine, inspect, text
import re
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from datetime import datetime

# 初始化 FastMCP 服务器
mcp = FastMCP("db_diagnosis")

def clean_sql(sql: str) -> str:
    """清理SQL语句，去除多余的空格和换行符

    Args:
        sql: 原始SQL语句

    Returns:
        str: 清理后的SQL语句
    """
    # 去除开头和结尾的空白字符
    sql = sql.strip()
    # 将多个空白字符替换为单个空格
    sql = re.sub(r'\s+', ' ', sql)
    return sql

def extract_tables(parsed) -> list:
    """从解析后的 SQL 语句中提取表名（考虑 FROM / JOIN / 子查询 等）"""
    tables = set()

    from_seen = False
    for token in parsed.tokens:
        # 递归处理嵌套结构
        if token.is_group:
            tables.update(extract_tables(token))

        if from_seen:
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    tables.add(identifier.get_real_name())
                from_seen = False
            elif isinstance(token, Identifier):
                tables.add(token.get_real_name())
                from_seen = False
            elif token.ttype is Keyword:
                # 终止符，比如 WHERE，说明表名部分结束了
                from_seen = False
        if token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN"):
            from_seen = True

    return list(tables)

@mcp.tool()
async def update_statistics(query: str) -> Dict[str, Any]:
    """更新统计信息，通过执行ANALYZE命令更新查询中涉及的表的统计信息

    Args:
        query: 查询语句

    Returns:
        Dict[str, Any]: 包含统计信息更新结果的字典
    """
    # 清理SQL语句
    query = clean_sql(query)

    # 解析SQL语句
    parsed = sqlparse.parse(query)[0]
    
    # 提取涉及的表名
    tables = extract_tables(parsed)
    
    # 连接数据库
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/tpch')
    
    result = {
        "action": "update_statistics",
        "tables_analyzed": [],
        "executed_commands": []
    }
    
    # 对每个表执行ANALYZE
    with engine.connect() as conn:
        DEAD_TUPLE_RATIO_THRESHOLD = 0.1   # 死元组占比 >10% 建议 VACUUM
        MOD_SINCE_ANALYZE_THRESHOLD = 1000 # 自上次 ANALYZE 修改行数 >1000 建议 ANALYZE

        for table in tables:
            # 从 pg_stat_all_tables 中获取统计
            stats = conn.execute(text(f"""
                SELECT
                  n_live_tup,
                  n_dead_tup,
                  n_mod_since_analyze,
                  last_vacuum,
                  last_autoanalyze,
                  last_analyze,
                  last_autoanalyze
                FROM pg_stat_all_tables s
                JOIN pg_class c ON s.relid = c.oid
                WHERE c.relkind = 'r' AND s.schemaname = current_schema() AND s.relname = :tbl
            """), {"tbl": table}).fetchone()

            # 如果表不存在或无统计则跳过
            if not stats:
                continue

            n_live, n_dead, n_mod, last_vac, last_auto_vac, last_an, last_auto_an = stats
            dead_pct = (n_dead / (n_live + n_dead)) if (n_live + n_dead) > 0 else 0

            result["tables_checked"].append({
                "table": table,
                "n_live_tup": n_live,
                "n_dead_tup": n_dead,
                "dead_ratio": round(dead_pct, 4),
                "n_mod_since_analyze": n_mod,
                "last_vacuum": str(last_vac),
                "last_auto_vacuum": str(last_auto_vac),
                "last_analyze": str(last_an),
                "last_auto_analyze": str(last_auto_an)
            })

            # 如果表既需要 VACUUM 又需要 ANALYZE，可以合并
            if dead_pct > DEAD_TUPLE_RATIO_THRESHOLD and n_mod > MOD_SINCE_ANALYZE_THRESHOLD:
                result["vacuum_commands"].append(f"VACUUM ANALYZE {table};")

            # 判断是否需要 VACUUM
            elif dead_pct > DEAD_TUPLE_RATIO_THRESHOLD:
                result["vacuum_commands"].append(f"VACUUM {table};")

            # 判断是否需要 ANALYZE
            elif n_mod > MOD_SINCE_ANALYZE_THRESHOLD:
                result["analyze_commands"].append(f"ANALYZE {table};")

    return result

@mcp.tool()
async def join_order_optimization(query: str) -> Dict[str, Any]:
    """优化查询的连接顺序，通过测试不同的连接策略找出最优执行计划

    Args:
        query: SQL查询语句

    Returns:
        Dict[str, Any]: 包含优化结果的字典
    """
    # 清理SQL语句
    query = clean_sql(query)
    
    # 解析SQL语句
    parsed = sqlparse.parse(query)[0]
    
    # 提取涉及的表
    tables = extract_tables(parsed)
    
    # 连接数据库
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/tpch')
    
    result = {
        "action": "join_order_optimization",
        "original_query": query,
        "tables_involved": tables,
        "table_statistics": [],
        "join_conditions": [],
        "optimized_query": query
    }
    
    if len(tables) < 2:
        result["explanation"] = "查询只涉及单表，无需优化连接顺序"
        return result
    
    with engine.connect() as conn:
        # 1. 收集表统计信息
        for table in tables:
            stats = conn.execute(text(f"""
                SELECT 
                    reltuples::bigint as row_estimate,
                    relpages::bigint as page_estimate,
                    n_live_tup as live_rows,
                    n_dead_tup as dead_rows
                FROM pg_class c
                LEFT JOIN pg_stat_all_tables s ON c.oid = s.relid
                WHERE c.relname = :table
            """), {"table": table}).fetchone()
            
            if stats:
                result["table_statistics"].append({
                    "table": table,
                    "rows": stats[0],
                    "pages": stats[1],
                    "live_rows": stats[2],
                    "dead_rows": stats[3]
                })
        
        # 2. 提取连接条件
        join_conditions = []
        join_pattern = r'JOIN\s+(\w+)\s+(?:\w+\s+)?ON\s+(.+?)(?=(?:JOIN|WHERE|GROUP|ORDER|LIMIT|$))'
        matches = re.finditer(join_pattern, query, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            table = match.group(1)
            condition = match.group(2).strip()
            join_conditions.append({
                "table": table,
                "condition": condition
            })
        
        result["join_conditions"] = join_conditions
        
        # 3. 分析连接类型和选择性
        join_analysis = []
        for join in join_conditions:
            # 提取连接列
            join_cols = re.findall(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', join["condition"])
            
            for left_table, left_col, right_table, right_col in join_cols:
                # 获取连接列的选择性
                for table, col in [(left_table, left_col), (right_table, right_col)]:
                    try:
                        selectivity = conn.execute(text(f"""
                            SELECT n_distinct
                            FROM pg_stats
                            WHERE tablename = :table
                            AND attname = :column
                        """), {"table": table, "column": col}).scalar()
                        
                        join_analysis.append({
                            "table": table,
                            "column": col,
                            "selectivity": selectivity if selectivity else 0
                        })
                    except:
                        pass
        
        # 4. 优化连接顺序
        # 基于表大小和连接选择性进行排序
        sorted_tables = sorted(result["table_statistics"], 
                            key=lambda x: (x.get("rows", float('inf')), x.get("pages", float('inf'))))
        
        # 生成优化后的查询
        if sorted_tables:
            # 提取SELECT和WHERE部分
            select_pattern = r'(SELECT\s+.+?)\s+FROM'
            select_match = re.search(select_pattern, query, re.IGNORECASE | re.DOTALL)
            select_clause = select_match.group(1) if select_match else "SELECT *"
            
            where_pattern = r'WHERE\s+(.+?)(?=(GROUP|ORDER|LIMIT|$))'
            where_match = re.search(where_pattern, query, re.IGNORECASE)
            where_clause = f"WHERE {where_match.group(1)}" if where_match else ""
            
            # 构建优化后的查询
            optimized_query = f"{select_clause}\nFROM {sorted_tables[0]['table']} t1\n"
            
            for i, join in enumerate(join_conditions):
                optimized_query += f"JOIN {join['table']} t{i+2} ON {join['condition']}\n"
            
            if where_clause:
                optimized_query += where_clause
            
            # 保留原查询的GROUP BY, ORDER BY, LIMIT等子句
            remaining_clauses = re.search(r'(?:GROUP|ORDER|LIMIT).+$', query, re.IGNORECASE)
            if remaining_clauses:
                optimized_query += "\n" + remaining_clauses.group(0)
            
            result["optimized_query"] = clean_sql(optimized_query)
            
            # 估算成本降低
            original_cost = conn.execute(text(f"EXPLAIN (FORMAT JSON) {query}")).scalar()
            optimized_cost = conn.execute(text(f"EXPLAIN (FORMAT JSON) {result['optimized_query']}")).scalar()
            
            if original_cost and optimized_cost:
                cost_reduction = ((float(original_cost) - float(optimized_cost)) / float(original_cost)) * 100
                result["estimated_cost_reduction"] = f"{cost_reduction:.1f}%"
        
        # 5. 添加优化建议
        hints = []
        
        # 检查表大小差异
        if len(sorted_tables) > 1:
            size_ratio = sorted_tables[-1]["rows"] / sorted_tables[0]["rows"]
            if size_ratio > 10:
                hints.append(f"建议将小表 {sorted_tables[0]['table']} 作为驱动表")
        
        # 检查连接列的选择性
        for analysis in join_analysis:
            if analysis["selectivity"] and analysis["selectivity"] < 0.1:
                hints.append(f"表 {analysis['table']} 的连接列 {analysis['column']} 具有高选择性，适合早期过滤")
        
        result["optimization_hints"] = hints

    return result

@mcp.tool()
async def optimize_index(query: str) -> Dict[str, Any]:
    """优化数据库索引

    Args:
        query: SQL查询语句

    Returns:
        Dict[str, Any]: 包含索引优化建议的字典，包括需要创建和删除的索引
    """
    # 清理SQL语句
    query = clean_sql(query)

    # 解析SQL语句
    parsed = sqlparse.parse(query)[0]
    
    # 提取涉及的表名
    tables = extract_tables(parsed)

    # 连接数据库（这里使用示例连接字符串，实际使用时需要替换）
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/tpch')
    inspector = inspect(engine)
    
    result = {
        "action": "optimize_index",
        "tables_analyzed": [],
        "create_index_statements": [],
        "drop_index_statements": []
    }
    
    for table in tables:
        table_info = {
            "table_name": table,
            "existing_indexes": [],
            "recommended_indexes": []
        }
        
        # 获取现有索引
        existing_indexes = inspector.get_indexes(table)
        table_info["existing_indexes"] = existing_indexes
        
        # 分析查询中的WHERE子句和JOIN条件
        where_columns = []
        join_columns = []
        
        # 提取WHERE子句中的列
        where_match = re.search(r"WHERE\s+(.+?)(?=(GROUP|ORDER|LIMIT|$))", query, re.IGNORECASE)
        if where_match:
            where_conditions = where_match.group(1)
            where_columns = re.findall(r"(\w+)\s*[=<>]", where_conditions)
        
        # 提取JOIN条件中的列
        join_match = re.search(r"JOIN.+?ON\s+(.+?)(?=(WHERE|GROUP|ORDER|LIMIT|$))", query, re.IGNORECASE)
        if join_match:
            join_conditions = join_match.group(1)
            join_columns = re.findall(r"(\w+)\s*=", join_conditions)
        
        # 合并需要索引的列
        needed_columns = list(set(where_columns + join_columns))
        
        # 检查哪些列需要新建索引
        for col in needed_columns:
            if not any(col in idx.get('column_names', []) for idx in existing_indexes):
                index_name = f"idx_{table}_{col}"
                create_stmt = f"CREATE INDEX {index_name} ON {table} ({col});"
                result["create_index_statements"].append(create_stmt)
                table_info["recommended_indexes"].append({
                    "name": index_name,
                    "columns": [col]
                })
        
        # 检查是否有可能需要删除的索引
        used_columns = set(needed_columns)
        for idx in existing_indexes:
            if not any(col in used_columns for col in idx.get('column_names', [])):
                drop_stmt = f"DROP INDEX {idx['name']};"
                result["drop_index_statements"].append(drop_stmt)
        
        result["tables_analyzed"].append(table_info)
    
    return result

@mcp.tool()
async def optimize_repeatedly_subqueries(query: str) -> Dict[str, Any]:
    """优化重复执行的子查询

    Args:
        query: 包含重复子查询的SQL语句

    Returns:
        Dict[str, Any]: 包含优化后查询的字典
    """
    # 清理SQL语句
    query = clean_sql(query)

    # 解析SQL语句
    parsed = sqlparse.parse(query)[0]
    
    result = {
        "action": "optimize_subqueries",
        "original_query": query,
        "optimized_query": query,
        "explanation": ""
    }
    
    # 检查是否存在相同的子查询
    subqueries = []
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.Token) and token.ttype is None:
            # 提取所有子查询
            matches = re.finditer(r'\(SELECT[^()]*(?:\([^()]*\)[^()]*)*\)', str(token), re.IGNORECASE)
            for match in matches:
                subqueries.append(match.group())
    
    # 检查重复的子查询
    subquery_count = defaultdict(int)
    for sq in subqueries:
        subquery_count[sq] += 1
    
    repeated_subqueries = {sq: count for sq, count in subquery_count.items() if count > 1}
    
    if not repeated_subqueries:
        result["explanation"] = "未发现重复执行的子查询"
        return result
        
    # 优化重复子查询
    optimized_query = query
    for subquery, count in repeated_subqueries.items():
        # 1. 尝试使用CTE优化
        cte_name = f"cte_{len(result.get('ctes', []))}"
        cte_query = f"WITH {cte_name} AS {subquery} "
        new_query = cte_query + optimized_query.replace(subquery, cte_name)
        
        # 2. 如果子查询在WHERE子句中，尝试转换为JOIN
        if "WHERE" in optimized_query:
            try:
                # 提取子查询的选择列
                subquery_cols = re.search(r'SELECT\s+(.*?)\s+FROM', subquery, re.IGNORECASE).group(1)
                # 提取子查询的表名
                subquery_table = re.search(r'FROM\s+(.*?)(?:\s+WHERE|\s*$)', subquery, re.IGNORECASE).group(1)
                
                # 构建JOIN查询
                join_condition = re.search(rf'WHERE.*?{re.escape(subquery)}.*?(?:AND|$)', optimized_query, re.IGNORECASE)
                if join_condition:
                    join_query = optimized_query.replace(
                        subquery,
                        f"(SELECT DISTINCT {subquery_cols} FROM {subquery_table})"
                    )
                    if "EXPLAIN" in join_query.upper():
                        result["join_alternative"] = join_query
            except:
                pass
        
        optimized_query = new_query
        
    result["optimized_query"] = clean_sql(optimized_query)
    result["explanation"] = f"发现{len(repeated_subqueries)}个重复执行的子查询，已使用CTE优化。"
    
    return result

@mcp.tool()
async def tune_query_configuration(query: str) -> Dict[str, Any]:
    """调优数据库配置参数

    Args:
        query: SQL查询语句

    Returns:
        Dict[str, Any]: 包含调优参数和优化建议的字典
    """
    # 清理SQL语句
    query = clean_sql(query)
    
    # 解析SQL语句
    parsed = sqlparse.parse(query)[0]
    
    # 配置参数列表
    config_params = [ "enable_sort", "enable_memoize", "enable_hashjoin", "enable_nestloop", "enable_mergejoin", "enable_gathermerge", "enable_hashagg", "enable_material", "enable_parallel_hash", "random_page_cost", "seq_page_cost", "hash_mem_multiplier" ]
    
    # 分析查询特征
    query_features = {
        "has_sort": bool(re.search(r'ORDER\s+BY', query, re.IGNORECASE)),
        "has_group": bool(re.search(r'GROUP\s+BY', query, re.IGNORECASE)),
        "has_join": bool(re.search(r'JOIN', query, re.IGNORECASE)),
        "has_subquery": bool(re.search(r'\(SELECT', query, re.IGNORECASE)),
        "table_count": len(extract_tables(parsed)),
        "has_aggregate": bool(re.search(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', query, re.IGNORECASE))
    }
    
    # 连接数据库获取当前参数
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/tpch')
    current_params = {}

    with engine.connect() as conn:
        for param in config_params:
            result = conn.execute(text(f"SHOW {param}")).fetchone()
            if result:
                current_params[param] = result[0]
    
    # 基于查询特征调整参数
    optimized_params = current_params.copy()
    optimization_reasons = []
    
    # 1. 处理排序操作
    if query_features["has_sort"]:
        optimized_params["enable_sort"] = True
        optimization_reasons.append("启用排序优化以处理ORDER BY子句")
    
    # 2. 处理连接操作
    if query_features["has_join"]:
        if query_features["table_count"] > 3:
            # 大规模连接优先使用哈希连接
            optimized_params["enable_hashjoin"] = True
            optimized_params["enable_nestloop"] = False
            optimized_params["enable_mergejoin"] = False
            optimization_reasons.append("多表连接场景，优先使用哈希连接")
        else:
            # 小规模连接保持所有连接方式可用
            optimized_params["enable_hashjoin"] = True
            optimized_params["enable_nestloop"] = True
            optimized_params["enable_mergejoin"] = True
            optimization_reasons.append("小规模连接场景，保持所有连接方式可用")
    
    # 3. 处理聚合操作
    if query_features["has_group"] or query_features["has_aggregate"]:
        optimized_params["enable_hashagg"] = True
        optimized_params["hash_mem_multiplier"] = 2.0
        optimization_reasons.append("存在聚合操作，增加哈希内存配置")
    
    # 4. 处理子查询
    if query_features["has_subquery"]:
        optimized_params["enable_material"] = True
        optimized_params["enable_memoize"] = True
        optimization_reasons.append("存在子查询，启用结果物化和查询计划缓存")
    
    # 5. 并行处理优化
    if query_features["table_count"] > 2 or query_features["has_aggregate"]:
        optimized_params["enable_parallel_hash"] = True
        optimized_params["enable_gathermerge"] = True
        optimization_reasons.append("复杂查询场景，启用并行处理优化")
    
    # 生成优化后的查询
    optimized_query = "SET " + "; SET ".join([
        f"{param} = {str(value).lower()}" 
        for param, value in optimized_params.items() 
        if str(value) != str(current_params.get(param, ''))
    ]) + "; " + query
    
    return {
        "action": "tune_config",
        "original_query": query,
        "query_features": query_features,
        "current_parameters": current_params,
        "optimized_parameters": optimized_params,
        "optimization_reasons": optimization_reasons,
        "apply_query": clean_sql(optimized_query),
    }

@mcp.tool()
async def change_distributed_key(query: str) -> str:
    """改变分布式键

    Args:
        query: SQL查询语句
    """
    print("call tool change_distributed_key")
    
    return {
        "action": "change_distributed_key",
        "distributed_key": "new_key"
    }

@mcp.tool()
async def rewrite_query(query: str) -> Dict[str, Any]:
    """重写SQL查询以提高性能

    Args:
        query: 原始SQL查询

    Returns:
        Dict[str, Any]: 包含查询重写结果的字典
    """
    # 分析模块：解析SQL语句并提取关键信息
    # analysis_result = analyze_query_structure(query)
    
    # 深度思考模块：基于分析结果生成优化策略
    # optimization_result = generate_query_optimization(query, analysis_result)
    
    # 可信验证模块：验证优化结果
    # verification_result = verify_optimization_result(query, optimization_result["optimized_query"])
    
    return {
        "action": "rewrite_query",
        "original_query": query,
        "optimized_query": f"/* Optimized */ {query}",
    }

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='stdio') 