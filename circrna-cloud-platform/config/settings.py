"""
CircRNA Cloud Platform - 平台配置
"""

# 平台基本信息
PLATFORM_NAME = "CircRNA Cloud Platform"
PLATFORM_VERSION = "1.0.0"
PLATFORM_URL = "https://circrna-cloud.example.com"

# ============================================================
# 教育邮箱验证配置
# ============================================================

# 支持的教育邮箱域名（正则表达式）
EDUCATION_EMAIL_DOMAINS = [
    # 中国高校
    r'.*\.edu\.cn$',           # 所有中国高校
    r'.*\.ac\.cn$',            # 中国科学院等研究机构

    # 国际高校
    r'.*\.edu$',               # 美国高校
    r'.*\.ac\.uk$',            # 英国高校
    r'.*\.ac\.jp$',            # 日本高校
    r'.*\.ac\.kr$',            # 韩国高校
    r'.*\.ac\.de$',            # 德国高校
    r'.*\.ac\.fr$',            # 法国高校
    r'.*\.edu\.au$',           # 澳大利亚高校
    r'.*\.edu\.sg$',           # 新加坡高校
    r'.*\.edu\.hk$',           # 香港高校
    r'.*\.edu\.tw$',           # 台湾高校

    # 研究机构
    r'.*@.*\.nih\.gov$',       # NIH
    r'.*@.*\.gov\.cn$',        # 中国政府研究机构

    # 特定高校（白名单）
    r'.*@tsinghua\.edu\.cn$',
    r'.*@pku\.edu\.cn$',
    r'.*@fudan\.edu\.cn$',
    r'.*@sjtu\.edu\.cn$',
    r'.*@ustc\.edu\.cn$',
    r'.*@harvard\.edu$',
    r'.*@mit\.edu$',
    r'.*@stanford\.edu$',
    r'.*@cambridge\.ac\.uk$',
    r'.*@oxford\.ac\.uk$',
]

# ============================================================
# 定价配置
# ============================================================

# GPU每小时价格（美元）
GPU_COST_PER_HOUR = 0.50      # 学生价（商业价 $2.50）

# 每月免费额度（GPU小时）
FREE_QUOTA_MONTHLY = 5.0

# ============================================================
# 资源限制配置（防挖矿核心参数）
# ============================================================

# 单用户最大并发任务数
MAX_CONCURRENT_JOBS_PER_USER = 2     # 免费用户
MAX_CONCURRENT_JOBS_PAID = 4          # 付费用户
MAX_CONCURRENT_JOBS_TEAM = 8          # 团队账户

# 单任务最大时长（小时）
MAX_JOB_DURATION_HOURS = 0.5         # 免费用户（30分钟）
MAX_JOB_DURATION_PAID = 2.0          # 付费用户
MAX_JOB_DURATION_TEAM = 10.0         # 团队账户

# 任务优先级范围
MIN_JOB_PRIORITY = 1
MAX_JOB_PRIORITY = 100

# 请求频率限制
RATE_LIMIT_PER_MINUTE = 5            # 每分钟最多5次请求
RATE_LIMIT_PER_HOUR = 50             # 每小时最多50次请求
RATE_LIMIT_PER_DAY = 200             # 每天最多200次请求

# 单序列最大长度
MAX_SEQUENCE_LENGTH = 500            # 最大500nt

# 每任务最大序列数
MAX_SEQUENCES_PER_JOB = 100          # 免费用户
MAX_SEQUENCES_PER_JOB_PAID = 500     # 付费用户

# ============================================================
# 管理员配置
# ============================================================

ADMIN_EMAILS = [
    "admin@circrna-cloud.example.com",
    # 可添加更多管理员邮箱
]

# ============================================================
# 数据库配置
# ============================================================

DATABASE_PATH = "data/users.db"
JOB_QUEUE_PATH = "data/job_queue.json"

# ============================================================
# GPU集群配置
# ============================================================

NUM_GPUS = 8                         # DGX Spark 8 GPU
GPU_TYPE = "A100"                    # 或 H100
GPU_MEMORY_GB = 80                   # 每GPU显存

# ============================================================
# 部署配置
# ============================================================

# 监听端口
SERVER_PORT = 8501

# 是否启用HTTPS
USE_HTTPS = True

# Cookie配置
COOKIE_NAME = "circrna_cloud_auth"
COOKIE_EXPIRY_DAYS = 30
COOKIE_SECRET_KEY = "change_this_to_secure_random_key"