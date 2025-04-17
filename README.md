# DREAM - Database Recovery via Explainable Autonomous Management

DREAM 是一个智能代理系统，用于自动检测和修复数据库性能异常。该系统结合了机器学习和专家系统的优点，能够自动分析数据库查询性能问题，并提供相应的修复方案。

## 系统架构

DREAM 系统包含三个核心模块：

1. **Plan 模块**
2. **Action 模块**
3. **Memory 模块**

## 项目设置与运行

1. 项目准备

```bash
# 克隆项目
git clone https://github.com/yourusername/DREAM.git
cd DREAM

# 创建名为DREAM的Python 3.10虚拟环境
conda create -n DREAM python=3.10

# 激活虚拟环境
conda activate DREAM

# 安装依赖包
pip install -r requirements.txt
```

2. 运行项目：

```bash
cd config
cp base_config.py.example base_config.py

# 更改config/basic_config.py
OPENAI_CONFIG = {
    "api_key": "api_key",
    "base_url": "https://api.gpt.ge/v1/",
    "model": "gpt-4o-mini"
}

ANTHROPIC_CONFIG = {
    "api_key": "api_key",
    "base_url": "https://api.gpt.ge",
    "model": "claude-3-5-sonnet-20241022"
}

# 运行主程序
python main.py --data=../data/tpc_c.csv
```
