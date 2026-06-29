"""
容器管理模块 - Docker容器编排，用户隔离，资源限制

DGX Spark环境：
- Linux内核
- PyTorch + CUDA已预装
- 单GPU Blackwell GB10 (~1000 TOPS)
- 任务分时调度，非GPU分配

核心功能：
1. 根据API Key创建隔离容器
2. 设置资源限制（防挖矿）
3. 挂载私有数据目录
4. 自动销毁过期容器
5. 任务队列调度（串行GPU使用）
"""

import os
import json
import subprocess
import hashlib
import time
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import docker  # pip install docker

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)

# 容器配置
CONTAINER_IMAGE = "circrna-cloud-base:latest"
CONTAINER_NETWORK = "circrna-network"
MAX_CONTAINER_LIFETIME_HOURS = 2
MAX_CONTAINERS_PER_USER = 2
# 单GPU模式：同一时间只能运行一个GPU任务
GPU_AVAILABLE = True
GPU_LOCK = threading.Lock()
GPU_BUSY = False


class ContainerManager:
    """
    Docker容器管理器

    负责：
    - 创建用户隔离容器
    - 设置资源限制
    - 监控容器状态
    - 自动清理过期容器
    """

    def __init__(self):
        try:
            self.client = docker.from_env()
            self._init_network()
        except docker.errors.DockerException as e:
            print(f"Docker连接失败: {e}")
            self.client = None

    def _init_network(self):
        """初始化Docker网络"""
        try:
            self.client.networks.get(CONTAINER_NETWORK)
        except docker.errors.NotFound:
            self.client.networks.create(CONTAINER_NETWORK, driver="bridge")

    def create_user_container(
        self,
        user_id: str,
        api_key: str,
        tier: str = 'free',
        gpu_id: int = 0
    ) -> Tuple[bool, Dict, str]:
        """
        为用户创建隔离容器

        Args:
            user_id: 用户ID
            api_key: 用户API密钥
            tier: 用户等级 (free/paid/team)
            gpu_id: 分配的GPU编号

        Returns:
            (success, container_info, message)
        """
        if self.client is None:
            return False, {}, "Docker未连接"

        # 检查用户已有容器数
        existing = self._get_user_containers(user_id)
        if len(existing) >= MAX_CONTAINERS_PER_USER:
            return False, {}, f"已达最大容器数限制 ({MAX_CONTAINERS_PER_USER})"

        # 生成容器名（基于API Key哈希，避免泄露）
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        container_name = f"crc-{user_id}-{api_key_hash}"

        # 创建用户数据目录
        user_data_dir = DATA_DIR / 'users' / user_id
        user_data_dir.mkdir(parents=True, exist_ok=True)
        (user_data_dir / 'input').mkdir(exist_ok=True)
        (user_data_dir / 'output').mkdir(exist_ok=True)
        (user_data_dir / 'tmp').mkdir(exist_ok=True)

        # 配额文件
        quota_file = user_data_dir / '.quota'
        if not quota_file.exists():
            quota_file.write_text(json.dumps({
                'tier': tier,
                'monthly_limit': 5.0 if tier == 'free' else 100.0,
                'used': 0,
                'created': datetime.now().isoformat()
            }))

        # 资源限制配置（防挖矿核心）
        resources = self._get_resource_limits(tier)

        # 创建容器
        try:
            container = self.client.containers.create(
                image=CONTAINER_IMAGE,
                name=container_name,

                # 挂载卷（数据隔离）
                volumes={
                    str(user_data_dir): {
                        'bind': '/data/user',
                        'mode': 'rw'  # 用户私有读写
                    },
                    str(DATA_DIR / 'shared'): {
                        'bind': '/data/shared',
                        'mode': 'ro'  # 共享工具只读
                    },
                },

                # 资源限制
                host_config=self.client.api.create_host_config(
                    **resources,
                    binds={
                        str(user_data_dir): '/data/user',
                        str(DATA_DIR / 'shared'): '/data/shared:ro'
                    }
                ),

                # 环境变量
                environment={
                    'USER_ID': user_id,
                    'API_KEY_HASH': api_key_hash,
                    'TIER': tier,
                    'GPU_ID': str(gpu_id),
                    'MAX_RUNTIME_HOURS': str(MAX_CONTAINER_LIFETIME_HOURS),
                    'PYTHONPATH': '/app:/data/shared',
                },

                # 网络隔离
                network=CONTAINER_NETWORK,

                # 工作目录
                working_dir='/app',

                # 标签（用于自动清理）
                labels={
                    'user_id': user_id,
                    'tier': tier,
                    'created_at': datetime.now().isoformat(),
                    'auto_delete': 'true',
                    'max_lifetime': str(MAX_CONTAINER_LIFETIME_HOURS * 3600),
                },

                # 不自动启动，等待任务调度
                detach=True,
            )

            container_info = {
                'container_id': container.id[:12],
                'container_name': container_name,
                'user_id': user_id,
                'gpu_id': gpu_id,
                'tier': tier,
                'data_dir': str(user_data_dir),
                'created_at': datetime.now().isoformat(),
                'status': 'created',
            }

            return True, container_info, "容器创建成功"

        except docker.errors.ImageNotFound:
            return False, {}, f"镜像 {CONTAINER_IMAGE} 未找到，请先构建"
        except docker.errors.APIError as e:
            return False, {}, f"Docker API错误: {e}"

    def start_container(self, container_id: str) -> Tuple[bool, str]:
        """启动容器"""
        if self.client is None:
            return False, "Docker未连接"

        try:
            container = self.client.containers.get(container_id)
            container.start()
            return True, "容器已启动"
        except docker.errors.NotFound:
            return False, "容器不存在"
        except docker.errors.APIError as e:
            return False, f"启动失败: {e}"

    def stop_container(self, container_id: str) -> Tuple[bool, str]:
        """停止容器"""
        if self.client is None:
            return False, "Docker未连接"

        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=30)
            return True, "容器已停止"
        except docker.errors.NotFound:
            return False, "容器不存在"

    def remove_container(self, container_id: str, force: bool = False) -> Tuple[bool, str]:
        """
        删除容器

        Args:
            container_id: 容器ID
            force: 是否强制删除（即使运行中）
        """
        if self.client is None:
            return False, "Docker未连接"

        try:
            container = self.client.containers.get(container_id)

            # 先停止再删除
            try:
                container.stop(timeout=10)
            except:
                pass

            container.remove(force=force)

            # 清理临时数据（保留output）
            labels = container.attrs.get('Config', {}).get('Labels', {})
            user_id = labels.get('user_id')
            if user_id:
                tmp_dir = DATA_DIR / 'users' / user_id / 'tmp'
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    tmp_dir.mkdir()

            return True, "容器已删除"
        except docker.errors.NotFound:
            return False, "容器不存在"

    def get_container_status(self, container_id: str) -> Dict:
        """获取容器状态"""
        if self.client is None:
            return {'error': 'Docker未连接'}

        try:
            container = self.client.containers.get(container_id)
            attrs = container.attrs

            # 计算剩余时间
            created_str = attrs.get('Config', {}).get('Labels', {}).get('created_at', '')
            if created_str:
                created = datetime.fromisoformat(created_str)
                elapsed = datetime.now() - created
                remaining = MAX_CONTAINER_LIFETIME_HOURS * 3600 - elapsed.total_seconds()
            else:
                remaining = 0

            return {
                'container_id': container.id[:12],
                'status': container.status,
                'image': attrs.get('Config', {}).get('Image'),
                'user_id': attrs.get('Config', {}).get('Labels', {}).get('user_id'),
                'tier': attrs.get('Config', {}).get('Labels', {}).get('tier'),
                'gpu_id': attrs.get('Config', {}).get('Labels', {}).get('gpu_id', '0'),
                'created_at': created_str,
                'remaining_seconds': int(remaining),
                'remaining_minutes': int(remaining / 60),
                'memory_usage': attrs.get('MemoryUsage', 'N/A'),
            }
        except docker.errors.NotFound:
            return {'error': '容器不存在'}

    def _get_user_containers(self, user_id: str) -> List[Dict]:
        """获取用户的所有容器"""
        if self.client is None:
            return []

        containers = self.client.containers.list(
            all=True,
            filters={'label': f'user_id={user_id}'}
        )

        return [
            {
                'container_id': c.id[:12],
                'name': c.name,
                'status': c.status,
                'labels': c.attrs.get('Config', {}).get('Labels', {})
            }
            for c in containers
        ]

    def _get_resource_limits(self, tier: str) -> Dict:
        """
        根据用户等级返回资源限制配置

        这是防挖矿的核心机制：
        - CPU限制
        - 内存限制
        - GPU限制
        """
        limits = {
            'free': {
                'cpu_quota': 100000 * 2,      # 2核 (100000 = 1核)
                'cpu_period': 100000,
                'mem_limit': '8g',            # 8GB
                'memswap_limit': '8g',
            },
            'paid': {
                'cpu_quota': 100000 * 4,      # 4核
                'cpu_period': 100000,
                'mem_limit': '16g',           # 16GB
                'memswap_limit': '16g',
            },
            'team': {
                'cpu_quota': 100000 * 8,      # 8核
                'cpu_period': 100000,
                'mem_limit': '32g',           # 32GB
                'memswap_limit': '32g',
            },
            'admin': {
                # 管理员无限制
            }
        }

        return limits.get(tier, limits['free'])

    def cleanup_expired_containers(self) -> int:
        """
        清理过期容器

        Returns:
            清理的容器数量
        """
        if self.client is None:
            return 0

        cleaned = 0

        # 获取所有标记auto_delete的容器
        containers = self.client.containers.list(
            all=True,
            filters={'label': 'auto_delete=true'}
        )

        for container in containers:
            labels = container.attrs.get('Config', {}).get('Labels', {})

            # 检查创建时间
            created_str = labels.get('created_at', '')
            max_lifetime = int(labels.get('max_lifetime', 7200))

            if created_str:
                created = datetime.fromisoformat(created_str)
                elapsed = (datetime.now() - created).total_seconds()

                if elapsed > max_lifetime:
                    # 已过期，删除
                    success, _ = self.remove_container(container.id[:12], force=True)
                    if success:
                        cleaned += 1
                        print(f"已清理过期容器: {container.name}")

        return cleaned

    def get_gpu_status(self) -> Dict[int, Dict]:
        """
        获取GPU状态

        通过nvidia-smi查询
        """
        gpu_status = {}

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 5:
                        gpu_id = int(parts[0].strip())
                        gpu_status[gpu_id] = {
                            'utilization': int(parts[1].strip()),
                            'memory_used_mb': int(parts[2].strip()),
                            'memory_total_mb': int(parts[3].strip()),
                            'temperature': int(parts[4].strip()),
                            'memory_pct': int(parts[2].strip()) / int(parts[3].strip()) * 100,
                            'status': 'idle' if int(parts[1].strip()) < 10 else 'running',
                        }
        except Exception as e:
            print(f"GPU状态查询失败: {e}")
            # 返回默认状态
            for i in range(8):
                gpu_status[i] = {'status': 'unknown', 'utilization': 0}

        return gpu_status

    def find_available_gpu(self) -> Optional[int]:
        """找到空闲GPU"""
        gpu_status = self.get_gpu_status()

        for gpu_id, status in gpu_status.items():
            if status.get('status') == 'idle' and status.get('utilization', 100) < 10:
                return gpu_id

        return None

    def list_all_containers(self) -> List[Dict]:
        """列出所有容器"""
        if self.client is None:
            return []

        containers = self.client.containers.list(all=True)

        return [
            {
                'container_id': c.id[:12],
                'name': c.name,
                'status': c.status,
                'image': c.attrs.get('Config', {}).get('Image'),
                'user_id': c.attrs.get('Config', {}).get('Labels', {}).get('user_id', 'N/A'),
                'tier': c.attrs.get('Config', {}).get('Labels', {}).get('tier', 'N/A'),
                'created_at': c.attrs.get('Config', {}).get('Labels', {}).get('created_at', 'N/A'),
            }
            for c in containers
        ]


class AntiMiningGuard:
    """
    防挖矿守卫

    多层检查：
    1. 配额检查
    2. 频率限制
    3. 并发限制
    4. 行为异常检测
    """

    def __init__(
        self,
        rate_limit_per_minute: int = 5,
        rate_limit_per_hour: int = 50,
        rate_limit_per_day: int = 200,
        max_concurrent_jobs: int = 2,
        max_job_duration: float = 0.5,
        free_quota_monthly: float = 5.0
    ):
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_per_hour = rate_limit_per_hour
        self.rate_limit_per_day = rate_limit_per_day
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_job_duration = max_job_duration
        self.free_quota_monthly = free_quota_monthly

        # 用户请求记录
        self.request_history: Dict[str, List[float]] = defaultdict(list)

    def check_user(self, user_id: str) -> Dict:
        """
        检查用户是否允许创建新任务

        Returns:
            dict with quota and limit status
        """
        now = time.time()

        # 清理旧记录
        self._cleanup_history(user_id, now)

        # 请求记录
        requests = self.request_history[user_id]

        # 频率检查
        requests_last_minute = [t for t in requests if now - t < 60]
        requests_last_hour = [t for t in requests if now - t < 3600]
        requests_last_day = [t for t in requests if now - t < 86400]

        rate_limited = False
        rate_limit_wait = 0

        if len(requests_last_minute) >= self.rate_limit_per_minute:
            rate_limited = True
            rate_limit_wait = int(60 - (now - requests_last_minute[0]))
        elif len(requests_last_hour) >= self.rate_limit_per_hour:
            rate_limited = True
            rate_limit_wait = int(3600 - (now - requests_last_hour[0]))
        elif len(requests_last_day) >= self.rate_limit_per_day:
            rate_limited = True
            rate_limit_wait = int(86400 - (now - requests_last_day[0]))

        # 配额检查
        quota_file = DATA_DIR / 'users' / user_id / '.quota'
        quota_used = 0
        monthly_remaining = self.free_quota_monthly

        if quota_file.exists():
            quota_data = json.loads(quota_file.read_text())
            quota_used = quota_data.get('used', 0)
            monthly_limit = quota_data.get('monthly_limit', self.free_quota_monthly)
            monthly_remaining = monthly_limit - quota_used

        # 并发任务检查
        container_manager = ContainerManager()
        user_containers = container_manager._get_user_containers(user_id)
        running_jobs = len([c for c in user_containers if c['status'] == 'running'])
        queued_jobs = len([c for c in user_containers if c['status'] == 'created'])

        return {
            'user_id': user_id,
            'monthly_limit': self.free_quota_monthly,
            'monthly_remaining': monthly_remaining,
            'used_monthly': quota_used,
            'current_jobs': running_jobs,
            'queued_jobs': queued_jobs,
            'completed_jobs': 0,  # 需从数据库查询

            # 频率限制
            'rate_limited': rate_limited,
            'rate_limit_wait': rate_limit_wait,
            'requests_last_minute': len(requests_last_minute),
            'requests_last_hour': len(requests_last_hour),
            'requests_last_day': len(requests_last_day),

            # 允许状态
            'can_submit': (
                not rate_limited and
                monthly_remaining > 0 and
                running_jobs < self.max_concurrent_jobs
            ),
        }

    def record_request(self, user_id: str):
        """记录用户请求"""
        self.request_history[user_id].append(time.time())

    def _cleanup_history(self, user_id: str, now: float):
        """清理过期记录"""
        self.request_history[user_id] = [
            t for t in self.request_history[user_id]
            if now - t < 86400  # 只保留24小时
        ]


# 初始化模块
def init_platform():
    """初始化平台"""

    # 创建目录结构
    dirs = [
        DATA_DIR / 'users',
        DATA_DIR / 'shared' / 'tools',
        DATA_DIR / 'shared' / 'models',
        DATA_DIR / 'shared' / 'databases',
        DATA_DIR / 'queue',
        DATA_DIR / 'logs',
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("平台目录初始化完成")

    # 检查Docker
    try:
        client = docker.from_env()
        print(f"Docker连接成功: {client.info()['ContainersRunning']} 个容器运行中")
    except Exception as e:
        print(f"Docker连接失败: {e}")


if __name__ == '__main__':
    init_platform()