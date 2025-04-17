import os
import sys
import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack, asynccontextmanager
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerSse
from agents._config import set_default_openai_api
import nest_asyncio

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.types import QueryInfo, AnomalyInfo
from config.base_config import API_SETTINGS

nest_asyncio.apply()

def delete_all_loggers():
    """清理和重置所有日志记录器"""
    import logging
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(logging.CRITICAL)

# 清理所有日志记录器
delete_all_loggers()

# 从配置文件加载API设置
os.environ['OPENAI_API_KEY'] = API_SETTINGS['openai']['api_key']
os.environ['OPENAI_BASE_URL'] = API_SETTINGS['openai']['base_url']

# 设置使用 chat_completions API
set_default_openai_api("chat_completions")

class ServerConfig:
    def __init__(self, name, server_type, params):
        self.name = name
        self.server_type = server_type
        self.params = params

    @classmethod
    def from_dict(cls, config):
        server_type = MCPServerStdio if config['server_type'] == 'stdio' else MCPServerSse
        if 'cwd' in config['params'] and config['params']['cwd'] == '.':
            config['params']['cwd'] = os.path.dirname(os.path.abspath(__file__))
        return cls(
            name=config['name'],
            server_type=server_type,
            params=config['params']
        )

class MultiServerManager:
    def __init__(self, configs):
        self.configs = configs
        self.servers = {}

    @asynccontextmanager
    async def manage_servers(self):
        try:
            for config in self.configs:
                self.servers[config.name] = await config.server_type(
                    name=config.name,
                    params=config.params
                ).__aenter__()
            yield list(self.servers.values())
        finally:
            for server in self.servers.values():
                await server.__aexit__(None, None, None)

class ActionManager:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'action_server_config.yaml')
        self.agent = None
        self.server_manager = None
        self.exit_stack = AsyncExitStack()
        self._servers = []

    async def __aenter__(self):
        """异步上下文管理器的进入方法"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器的退出方法"""
        
        # 首先清理 agent
        if self.agent:
            self.agent = None

        # 清理服务器资源
        cleanup_errors = []
        if self._servers:
            for server in self._servers:
                try:
                    if hasattr(server, 'cleanup'):
                        await server.cleanup()
                except asyncio.CancelledError:
                    # 忽略取消错误，继续清理其他资源
                    pass
                except Exception as e:
                    cleanup_errors.append(str(e))
                    print(f"Warning: Error cleaning up server: {e}")

        # 最后清理 exit_stack
        try:
            await self.exit_stack.aclose()
        except asyncio.CancelledError:
            # 忽略取消错误
            pass
        except Exception as e:
            cleanup_errors.append(str(e))
            print(f"Warning: Error during exit stack cleanup: {e}")

        # 清理引用
        self._servers = []
        self.server_manager = None

        # 如果有清理错误但不是由于取消导致的，则抛出
        if cleanup_errors and exc_type is not asyncio.CancelledError:
            raise Exception(f"Cleanup errors occurred: {'; '.join(cleanup_errors)}")

    async def initialize(self):
        """初始化 MCP 服务器和 Agent"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        configs = [ServerConfig.from_dict(service_config) for service_config in config_data['services']]
        self.server_manager = MultiServerManager(configs)
            
        self._servers = await self.exit_stack.enter_async_context(self.server_manager.manage_servers())
        
        self.agent = Agent(
            name="DBDiagnosticAgent",
            model="gpt-4o-mini",
            instructions=self._get_agent_instructions(),
            mcp_servers=self._servers,
        )
        return self.agent

    def _get_agent_instructions(self) -> str:
        return """
            You are a database diagnostic expert, proficient in analyzing database performance issues and providing repair suggestions. 
        """

    async def analyze_and_act(self, root_causes: List[str], 
                            query_info: QueryInfo,
                            anomaly_info: AnomalyInfo,
                            historical_data: List[Dict]) -> Dict[str, Any]:
        """分析根因并选择合适的行动"""
        if not self.agent:
            await self.initialize()
        
        prompt = self._build_prompt(root_causes, query_info, anomaly_info, historical_data)
        result = await Runner.run(starting_agent=self.agent, input=prompt)
        print(f"result output:\n{result.final_output}")
        return {"diagnosis": result.final_output}

    def _build_prompt(self, root_causes: List[str], query_info: QueryInfo, 
                      anomaly_info: AnomalyInfo, historical_data: List[Dict]) -> str:
        """构建提示信息"""
        
        # 修改
        return f"""
        Please analyze the following database issue and provide repair suggestions:
        
        1. Query Information:
        - SQL: {query_info.query}
        - Query Plan: {query_info.query_plan}
        - Execution Time: {query_info.execution_time} seconds
        
        2. Running Environment Monitoring:
        - CPU Usage: {anomaly_info.kpis['cpu_usage']}, {anomaly_info.kpi_descriptions['cpu_usage']}
        - Memory Usage: {anomaly_info.kpis['memory_usage']}, {anomaly_info.kpi_descriptions['memory_usage']}
        - Disk I/O: {anomaly_info.kpis['io_wait']}, {anomaly_info.kpi_descriptions['io_wait']}
        - Network Traffic: {anomaly_info.kpis['network_traffic']}, {anomaly_info.kpi_descriptions['network_traffic']}
        
        
        Now the root cause of the issue is identified as follows: {root_causes}, now you need to analyze the root cause and provide repair suggestions:
        1. Analyze the root cause and provide repair suggestions
        2. Provide query rewriting suggestions
        3. Suggest configuration adjustments
        4. Suggest index optimizations
        5. Suggest hardware upgrades
        6. Suggest software upgrades
        7. Suggest other suggestions
        
        
        Some Reference Similar Historical Data that may help you:
        {historical_data}
        
        For each recommendation, please explain the expected performance improvements.
        """
    
    async def cleanup(self):
        """清理资源"""
        await self.__aexit__(None, None, None)