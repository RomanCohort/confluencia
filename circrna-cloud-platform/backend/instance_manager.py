"""
镜像管理器 - AutoDL模式
预置多种生信镜像模板，用户一键开机
"""

import os
import json
import uuid
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
MIRRORS_DIR = DATA_DIR / 'mirrors'
INSTANCES_DIR = DATA_DIR / 'instances'


class MirrorTemplate:
    """
    镜像模板

    预置配置好的环境，用户选择后直接创建实例
    """

    def __init__(self, name: str, display_name: str, description: str,
                 price_per_hour: float, image: str, ports: Dict[str, int],
                 env: Dict[str, str], max_cpu: int, max_mem: str):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.price_per_hour = price_per_hour
        self.image = image
        self.ports = ports  # 容器端口 -> 宿主机端口映射
        self.env = env
        self.max_cpu = max_cpu
        self.max_mem = max_mem


# ============================================================
# 预置镜像清单
# ============================================================

MIRROR_CATALOG = [
    MirrorTemplate(
        name="circrna-base",
        display_name="CircRNA 基础版",
        description="ViennaRNA + OpenMM + Biopython，适合circRNA二级结构预测和简单MD模拟",
        price_per_hour=0.50,
        image="circrna-cloud-base:latest",
        ports={"8501": 8501, "8000": 8000},
        env={
            "PROJECT": "circrna",
            "WORKFLOW": "basic",
        },
        max_cpu=4,
        max_mem="16g",
    ),
    MirrorTemplate(
        name="circrna-full",
        display_name="CircRNA 完整版",
        description="含RoseTTAFold2NA + 全精度OpenMM，支持完整5-stage 3D结构预测管线",
        price_per_hour=1.20,
        image="circrna-cloud-full:latest",
        ports={"8501": 8501, "8000": 8000, "8888": 8888},
        env={
            "PROJECT": "circrna",
            "WORKFLOW": "full",
            "ROSETTAFOLD2NA_HOME": "/opt/RoseTTAFold2NA",
        },
        max_cpu=8,
        max_mem="32g",
    ),
    MirrorTemplate(
        name="bio-ml",
        display_name="生物信息学-ML版",
        description="PyTorch + scikit-learn + DeepPurpose + rdkit，适合药物发现和ML训练",
        price_per_hour=0.80,
        image="circrna-cloud-ml:latest",
        ports={"8888": 8888, "6006": 6006},
        env={
            "PROJECT": "bio-ml",
        },
        max_cpu=4,
        max_mem="16g",
    ),
    MirrorTemplate(
        name="custom",
        display_name="自定义环境",
        description="Dockerfile上传，自己构建环境",
        price_per_hour=1.50,
        image="custom",
        ports={"8501": 8501, "8888": 8888, "8000": 8000},
        env={},
        max_cpu=8,
        max_mem="32g",
    ),
]

# 镜像索引
MIRROR_MAP = {m.name: m for m in MIRROR_CATALOG}


class InstanceManager:
    """
    实例管理器 - AutoDL模式

    功能：
    - 用户选择镜像 → 创建实例
    - 按时计费（按秒累计）
    - Web端启动/停止/续租
    - SSH密钥/密码访问
    - 端口映射（Jupyter/Streamlit/TensorBoard）
    """

    def __init__(self):
        INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        db_path = INSTANCES_DIR / 'instances.json'
        if not db_path.exists():
            db_path.write_text('{}')

    def _read_db(self) -> Dict:
        return json.loads((INSTANCES_DIR / 'instances.json').read_text())

    def _write_db(self, data: Dict):
        (INSTANCES_DIR / 'instances.json').write_text(json.dumps(data, indent=2, default=str))

    def create_instance(
        self,
        user_id: str,
        mirror_name: str = "circrna-base",
        instance_name: str = None,
        gpu_required: bool = True,
    ) -> Dict:
        """
        为用户创建实例（开机）

        Args:
            user_id: 用户标识
            mirror_name: 镜像模板名
            instance_name: 实例别名（用户自定义）
            gpu_required: 是否需要GPU

        Returns:
            instance_info
        """
        mirror = MIRROR_MAP.get(mirror_name)
        if not mirror:
            raise ValueError(f"镜像不存在: {mirror_name}")

        if instance_name is None:
            instance_name = f"{mirror_name}-{uuid.uuid4().hex[:6]}"

        instance_id = f"ins_{uuid.uuid4().hex[:12]}"

        # 创建实例数据目录
        instance_dir = INSTANCES_DIR / instance_id
        instance_dir.mkdir(parents=True)

        # 用户数据挂载点
        user_data_dir = instance_dir / 'data'
        user_data_dir.mkdir()

        now = datetime.now()

        # 端口分配
        ports = {}
        base_port = self._find_available_port()
        for container_port, _host_port in mirror.ports.items():
            host_port = base_port + len(ports)
            ports[str(container_port)] = host_port

        instance = {
            'instance_id': instance_id,
            'instance_name': instance_name,
            'user_id': user_id,
            'mirror': mirror_name,
            'status': 'created',          # created → running → stopped
            'gpu_required': gpu_required,

            # 计费
            'price_per_hour': mirror.price_per_hour,
            'created_at': now.isoformat(),
            'started_at': None,
            'stopped_at': None,
            'total_cost': 0.0,
            'total_seconds': 0,

            # 网络
            'ports': ports,
            'ssh_port': self._find_available_port() + 1000,

            # 目录
            'instance_dir': str(instance_dir),
            'user_data_dir': str(user_data_dir),

            # SSH访问
            'ssh_password': uuid.uuid4().hex[:8],
        }

        # 保存
        db = self._read_db()
        db[instance_id] = instance
        self._write_db(db)

        print(f"[InstanceManager] 实例已创建: {instance_id} ({instance_name})")
        return instance

    def start_instance(self, instance_id: str) -> bool:
        """启动实例（调用Docker）"""
        db = self._read_db()
        instance = db.get(instance_id)
        if not instance:
            return False

        mirror = MIRROR_MAP[instance['mirror']]

        # Docker配置
        container_config = {
            'image': mirror.image,
            'name': f"crc-{instance_id}",
            'detach': True,
            'ports': instance['ports'],
            'volumes': {
                instance['user_data_dir']: {'bind': '/data/user', 'mode': 'rw'},
            },
            'environment': {
                **mirror.env,
                'INSTANCE_ID': instance_id,
                'USER_ID': instance['user_id'],
                'SSH_PASSWORD': instance['ssh_password'],
            },
            'host_config': {
                'port_bindings': {
                    '22/tcp': instance['ssh_port'],
                    **{f'{cp}/tcp': hp for cp, hp in instance['ports'].items()}
                },
                'mem_limit': mirror.max_mem,
                'cpuset_cpus': f'0-{mirror.max_cpu - 1}',
                'runtime': 'nvidia' if instance['gpu_required'] else None,
                'device_requests': [
                    {'Driver': 'nvidia', 'Capabilities': [['gpu', 'compute', 'utility']]}
                ] if instance['gpu_required'] else [],
            }
        }

        try:
            import docker
            client = docker.from_env()
            container = client.containers.run(**container_config)

            instance['status'] = 'running'
            instance['started_at'] = datetime.now().isoformat()
            instance['container_id'] = container.id[:12]

            db[instance_id] = instance
            self._write_db(db)

            print(f"[InstanceManager] 实例已启动: {instance_id}")
            print(f"  SSH端口: {instance['ssh_port']}")
            print(f"  Web端口: {instance['ports']}")
            return True

        except Exception as e:
            print(f"[InstanceManager] 启动失败: {e}")
            return False

    def stop_instance(self, instance_id: str) -> bool:
        """停止实例"""
        db = self._read_db()
        instance = db.get(instance_id)
        if not instance or instance['status'] != 'running':
            return False

        try:
            import docker
            client = docker.from_env()
            container = client.containers.get(instance.get('container_id', ''))
            container.stop(timeout=30)
            container.remove()
        except Exception as e:
            print(f"[InstanceManager] 停止容器失败: {e}")

        # 记录费用
        if instance.get('started_at'):
            started = datetime.fromisoformat(instance['started_at'])
            elapsed = (datetime.now() - started).total_seconds()
            instance['total_seconds'] += int(elapsed)
            instance['total_cost'] += round(elapsed / 3600 * instance['price_per_hour'], 4)

        instance['status'] = 'stopped'
        instance['stopped_at'] = datetime.now().isoformat()
        instance['container_id'] = None

        db[instance_id] = instance
        self._write_db(db)

        print(f"[InstanceManager] 实例已停止: {instance_id}")
        print(f"  本次费用: ${instance['total_cost']:.2f}")
        return True

    def delete_instance(self, instance_id: str) -> bool:
        """删除实例（释放所有资源）"""
        # 先停止
        self.stop_instance(instance_id)

        db = self._read_db()
        instance = db.pop(instance_id, None)
        self._write_db(db)

        if instance:
            # 清理数据
            shutil.rmtree(instance['instance_dir'], ignore_errors=True)
            print(f"[InstanceManager] 实例已删除: {instance_id}")
            return True
        return False

    def get_instance(self, instance_id: str) -> Optional[Dict]:
        """获取实例信息"""
        db = self._read_db()
        instance = db.get(instance_id)
        if instance:
            # 计算当前费用（运行中的实时计费）
            if instance['status'] == 'running' and instance.get('started_at'):
                started = datetime.fromisoformat(instance['started_at'])
                additional = (datetime.now() - started).total_seconds()
                instance['current_cost'] = instance['total_cost'] + round(
                    additional / 3600 * instance['price_per_hour'], 4
                )
            else:
                instance['current_cost'] = instance['total_cost']
            return instance
        return None

    def get_user_instances(self, user_id: str) -> List[Dict]:
        """获取用户的所有实例"""
        db = self._read_db()
        return [
            self.get_instance(iid)
            for iid, inst in db.items()
            if inst['user_id'] == user_id
        ]

    def list_instances(self, status: str = None) -> List[Dict]:
        """列出所有实例"""
        db = self._read_db()
        instances = [self.get_instance(iid) for iid in db]
        if status:
            instances = [i for i in instances if i and i['status'] == status]
        return instances

    def _find_available_port(self, start: int = 50000) -> int:
        """找可用端口"""
        import socket
        port = start
        while port < 51000:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
            port += 1
        return port

    def cleanup_stale_instances(self, max_idle_hours: int = 4):
        """清理闲置过久的实例"""
        db = self._read_db()
        now = datetime.now()
        cleaned = []

        for iid, instance in db.items():
            if instance['status'] == 'running':
                started = datetime.fromisoformat(instance['started_at'])
                elapsed_hours = (now - started).total_seconds() / 3600
                if elapsed_hours > max_idle_hours:
                    print(f"[InstanceManager] 自动停止闲置实例: {iid} (已运行{elapsed_hours:.1f}h)")
                    self.stop_instance(iid)
                    cleaned.append(iid)

        return cleaned


def show_catalog():
    """展示镜像目录"""
    print("\n" + "=" * 80)
    print("  CircRNA Cloud - 镜像模板选择")
    print("=" * 80)

    for i, mirror in enumerate(MIRROR_CATALOG, 1):
        print(f"\n  [{i}] {mirror.display_name}")
        print(f"      {mirror.description}")
        print(f"      价格: ${mirror.price_per_hour:.2f}/小时")
        print(f"      CPU: {mirror.max_cpu}核, 内存: {mirror.max_mem}")
        print(f"      端口: {', '.join(f'{p}' for p in mirror.ports.values())}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    show_catalog()

    # 测试：创建实例
    mgr = InstanceManager()
    instance = mgr.create_instance(
        user_id="user_demo",
        mirror_name="circrna-base",
        instance_name="我的测试实例",
    )
    print(f"\n创建实例: {instance['instance_name']}")
    print(f"  ID: {instance['instance_id']}")
    print(f"  SSH密码: {instance['ssh_password']}")
    print(f"  端口: {instance['ports']}")

    # 启动
    # mgr.start_instance(instance['instance_id'])
    # mgr.stop_instance(instance['instance_id'])