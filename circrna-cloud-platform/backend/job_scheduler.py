"""
单用户模式 - 任务队列调度器
DGX Spark (单GPU Blackwell GB10)
一次只跑一个任务，其余排队

用户通过API Key提交任务，系统串行执行
每个用户的目录互相隔离
"""

import os
import json
import time
import uuid
import shutil
import hashlib
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
JOB_QUEUE_FILE = DATA_DIR / 'job_queue.json'
QUEUE_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()


class JobStatus:
    QUEUED = 'queued'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class JobScheduler:
    """
    单GPU任务调度器

    设计：
    - 一次只运行一个任务
    - 其余任务排队等待
    - 支持优先级（付费用户优先）
    - 任务超时自动终止
    - 数据完全隔离
    """

    def __init__(self):
        self.jobs_file = JOB_QUEUE_FILE
        self._init_storage()
        self._current_job = None
        self._stop_event = threading.Event()

    def _init_storage(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.jobs_file.exists():
            self._write_queue([])

    def _read_queue(self) -> List[Dict]:
        with QUEUE_LOCK:
            try:
                return json.loads(self.jobs_file.read_text())
            except:
                return []

    def _write_queue(self, queue: List[Dict]):
        with QUEUE_LOCK:
            self.jobs_file.write_text(json.dumps(queue, indent=2, default=str))

    def submit_job(
        self,
        user_id: str,
        api_key_hash: str,
        sequences: List[str],
        bsj_positions: List[Tuple[int, int]],
        mode: str = 'quality',
        priority: int = 50
    ) -> Tuple[bool, str]:
        """
        提交任务到队列

        Args:
            user_id: 用户ID
            api_key_hash: API Key哈希（用于验权）
            sequences: RNA序列列表
            bsj_positions: BSJ位置列表
            mode: quality / fast / ultra_quality
            priority: 优先级 (1-100, 越高越优先)

        Returns:
            (success, job_id)
        """
        job_id = f'job_{uuid.uuid4().hex[:12]}'

        # 创建用户隔离目录
        job_dir = DATA_DIR / 'users' / user_id / 'jobs' / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = {
            'job_id': job_id,
            'user_id': user_id,
            'api_key_hash': api_key_hash,
            'status': JobStatus.QUEUED,
            'priority': priority,
            'mode': mode,
            'num_sequences': len(sequences),
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'job_dir': str(job_dir),
            'error': None,
            'progress': 0.0,
        }

        # 保存输入数据到隔离目录
        input_file = job_dir / 'input.json'
        input_file.write_text(json.dumps({
            'sequences': sequences,
            'bsj_positions': bsj_positions,
        }, indent=2))

        # 入队
        queue = self._read_queue()
        queue.append(job)
        # 按优先级排序（高优先级在前，同优先级按创建时间）
        queue.sort(key=lambda j: (-j.get('priority', 50), j.get('created_at', '')))
        self._write_queue(queue)

        print(f"[JobScheduler] 任务已加入队列: {job_id} (用户: {user_id}, 优先级: {priority})")

        return True, job_id

    def get_next_job(self) -> Optional[Dict]:
        """获取下一个要执行的任务（带锁）"""
        with QUEUE_LOCK:
            queue = self._read_queue()
            # 找到第一个排队中的任务
            for i, job in enumerate(queue):
                if job['status'] == JobStatus.QUEUED:
                    # 标记为运行中
                    job['status'] = JobStatus.RUNNING
                    job['started_at'] = datetime.now().isoformat()
                    queue[i] = job
                    self._write_queue(queue)
                    self._current_job = job
                    return job
            return None

    def complete_job(self, job_id: str, success: bool, result_path: str = None, error: str = None):
        """完成任务"""
        with QUEUE_LOCK:
            queue = self._read_queue()
            for i, job in enumerate(queue):
                if job['job_id'] == job_id:
                    job['status'] = JobStatus.COMPLETED if success else JobStatus.FAILED
                    job['completed_at'] = datetime.now().isoformat()
                    job['error'] = error
                    job['progress'] = 1.0 if success else 0.0
                    if result_path:
                        job['result_path'] = result_path
                    queue[i] = job
                    self._write_queue(queue)
                    self._current_job = None

                    print(f"[JobScheduler] 任务 {'完成' if success else '失败'}: {job_id}")
                    return True
            return False

    def update_progress(self, job_id: str, progress: float):
        """更新任务进度"""
        with QUEUE_LOCK:
            queue = self._read_queue()
            for i, job in enumerate(queue):
                if job['job_id'] == job_id:
                    job['progress'] = progress
                    queue[i] = job
                    self._write_queue(queue)
                    return

    def cancel_job(self, job_id: str, user_id: str = None) -> Tuple[bool, str]:
        """
        取消任务

        Args:
            job_id: 任务ID
            user_id: 用户ID（非管理员只能取消自己的）
        """
        with QUEUE_LOCK:
            queue = self._read_queue()
            for i, job in enumerate(queue):
                if job['job_id'] == job_id:
                    # 权限检查
                    if user_id and job['user_id'] != user_id:
                        return False, "无权取消他人任务"
                    if job['status'] == JobStatus.RUNNING:
                        return False, "任务正在运行，无法取消"
                    job['status'] = JobStatus.CANCELLED
                    queue[i] = job
                    self._write_queue(queue)
                    return True, "任务已取消"
            return False, "任务不存在"

    def get_job_status(self, job_id: str, user_id: str = None) -> Optional[Dict]:
        """获取任务状态"""
        queue = self._read_queue()
        for job in queue:
            if job['job_id'] == job_id:
                # 权限检查
                if user_id and job['user_id'] != user_id:
                    return None
                return job
        return None

    def get_user_jobs(self, user_id: str) -> List[Dict]:
        """获取用户的所有任务"""
        queue = self._read_queue()
        return [
            {
                'job_id': j['job_id'],
                'status': j['status'],
                'mode': j['mode'],
                'num_sequences': j['num_sequences'],
                'created_at': j['created_at'],
                'started_at': j['started_at'],
                'completed_at': j['completed_at'],
                'progress': j['progress'],
                'error': j['error'],
            }
            for j in queue
            if j['user_id'] == user_id
        ]

    def get_queue_stats(self) -> Dict:
        """获取队列统计"""
        queue = self._read_queue()
        return {
            'queued': sum(1 for j in queue if j['status'] == JobStatus.QUEUED),
            'running': sum(1 for j in queue if j['status'] == JobStatus.RUNNING),
            'completed_today': sum(
                1 for j in queue
                if j['status'] == JobStatus.COMPLETED
                and j.get('completed_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))
            ),
            'failed': sum(1 for j in queue if j['status'] == JobStatus.FAILED),
            'total': len(queue),
        }

    def get_result_path(self, job_id: str, user_id: str = None) -> Optional[str]:
        """获取结果文件路径"""
        job = self.get_job_status(job_id, user_id)
        if job and job['status'] == JobStatus.COMPLETED:
            return job.get('result_path')
        return None

    def get_queue_position(self, job_id: str) -> int:
        """获取任务在队列中的位置"""
        queue = self._read_queue()
        queued = [(i, j) for i, j in enumerate(queue) if j['status'] == JobStatus.QUEUED]
        for pos, (_, j) in enumerate(queued):
            if j['job_id'] == job_id:
                return pos + 1
        return -1


class TaskRunner:
    """
    任务运行器

    从队列取任务，调用Pipeline执行
    一次只跑一个，GPU独占
    """

    def __init__(self, pipeline_path: str = None):
        self.scheduler = JobScheduler()
        self.running = False
        self.current_job_id = None

        if pipeline_path is None:
            pipeline_path = str(PROJECT_ROOT.parent / 'confluencia_3_0' / 'core' / 'circrna' / 'torusfold' / 'circrna_3d_pipeline')

        self.pipeline_path = pipeline_path

    def start_worker(self):
        """启动工作进程（后台线程）"""
        thread = threading.Thread(target=self._worker_loop, daemon=True)
        thread.start()
        print("[TaskRunner] Worker已启动，等待任务...")

    def _worker_loop(self):
        """Worker主循环"""
        while True:
            with GPU_LOCK:
                # 取下一个任务
                job = self.scheduler.get_next_job()

                if job:
                    self.running = True
                    self.current_job_id = job['job_id']
                    print(f"\n[TaskRunner] 开始执行任务: {job['job_id']}")
                    print(f"[TaskRunner] 用户: {job['user_id']}, 模式: {job['mode']}")

                    # 执行任务
                    try:
                        self._execute_job(job)
                    except Exception as e:
                        print(f"[TaskRunner] 任务执行异常: {e}")
                        self.scheduler.complete_job(
                            job['job_id'],
                            success=False,
                            error=str(e)
                        )

                    self.running = False
                    self.current_job_id = None
                else:
                    # 没有任务，等待
                    time.sleep(5)

    def _execute_job(self, job: Dict):
        """
        执行单个任务

        调用现有的Pipeline完成结构预测
        """
        job_dir = Path(job['job_dir'])
        input_file = job_dir / 'input.json'
        output_dir = job_dir / 'output'
        output_dir.mkdir(exist_ok=True)

        # 加载输入数据
        input_data = json.loads(input_file.read_text())
        sequences = input_data['sequences']
        bsj_positions = input_data['bsj_positions']

        # 写入FASTA文件（供Pipeline使用）
        fasta_path = job_dir / 'input.fasta'
        with open(fasta_path, 'w') as f:
            for i, (seq, (bsj_start, bsj_end)) in enumerate(zip(sequences, bsj_positions)):
                f.write(f">seq_{i:04d} bsj_start={bsj_start} bsj_end={bsj_end}\n{seq}\n")

        print(f"[TaskRunner] 输入: {len(sequences)} 条序列")
        print(f"[TaskRunner] 输出目录: {output_dir}")

        # 调用现有Pipeline
        # pipeline.py提供了完整的5-stage流程
        self.scheduler.update_progress(job['job_id'], 0.1)

        cmd = [
            'python', f'{self.pipeline_path}/pipeline.py',
            '--fasta', str(fasta_path),
            '--output', str(output_dir),
            '--export-torusfold',
        ]

        self.scheduler.update_progress(job['job_id'], 0.2)

        # 执行
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2h timeout

        if result.returncode != 0:
            raise RuntimeError(f"Pipeline失败: {result.stderr[:500]}")

        self.scheduler.update_progress(job['job_id'], 0.9)

        # 打包结果
        result_zip = job_dir / 'results.zip'
        shutil.make_archive(
            str(result_zip.with_suffix('')),
            'zip',
            str(output_dir)
        )

        # 完成
        self.scheduler.complete_job(
            job['job_id'],
            success=True,
            result_path=str(result_zip)
        )

        print(f"[TaskRunner] 任务完成: {job['job_id']}")
        print(f"[TaskRunner] 输出: {result_zip}")


class ApiKeyManager:
    """
    API Key管理器

    密钥分发：
    - 每位客户分配一个API Key
    - Key用于身份认证和数据隔离
    - 支持吊销和轮换
    """

    def __init__(self):
        self.keys_file = DATA_DIR / 'api_keys.json'
        self._init_storage()

    def _init_storage(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.keys_file.exists():
            self._write_keys({})

    def _read_keys(self) -> Dict:
        return json.loads(self.keys_file.read_text())

    def _write_keys(self, keys: Dict):
        self.keys_file.write_text(json.dumps(keys, indent=2))

    def create_key(self, customer_name: str, email: str, tier: str = 'standard') -> str:
        """
        创建新的API Key

        Args:
            customer_name: 客户名称/机构名
            email: 联系邮箱
            tier: standard / premium

        Returns:
            api_key (格式: crc_xxxxxxxxxxxx)
        """
        api_key = 'crc_' + uuid.uuid4().hex[:24]
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        user_id = 'usr_' + uuid.uuid4().hex[:8]

        keys = self._read_keys()
        keys[api_key_hash] = {
            'user_id': user_id,
            'customer_name': customer_name,
            'email': email,
            'tier': tier,
            'created_at': datetime.now().isoformat(),
            'is_active': True,
            'quota_used': 0,
            'monthly_limit': 100 if tier == 'premium' else 20,
        }
        self._write_keys(keys)

        # 创建用户数据目录
        user_dir = DATA_DIR / 'users' / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ApiKeyManager] 已创建Key -> 客户: {customer_name}, 等级: {tier}")
        return api_key

    def validate_key(self, api_key: str) -> Optional[Dict]:
        """
        验证API Key

        Args:
            api_key: 待验证的Key

        Returns:
            user_info if valid, None if invalid
        """
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        keys = self._read_keys()

        if api_key_hash not in keys:
            return None

        key_info = keys[api_key_hash]
        if not key_info.get('is_active', True):
            return None

        return key_info

    def revoke_key(self, api_key: str) -> bool:
        """吊销API Key"""
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        keys = self._read_keys()

        if api_key_hash in keys:
            keys[api_key_hash]['is_active'] = False
            keys[api_key_hash]['revoked_at'] = datetime.now().isoformat()
            self._write_keys(keys)
            return True
        return False

    def list_keys(self) -> List[Dict]:
        """列出所有Key"""
        keys = self._read_keys()
        return [
            {
                'user_id': info['user_id'],
                'customer_name': info['customer_name'],
                'email': info['email'],
                'tier': info['tier'],
                'created_at': info['created_at'],
                'is_active': info['is_active'],
                'quota_used': info.get('quota_used', 0),
            }
            for info in keys.values()
        ]


class DataIsolator:
    """
    数据隔离器

    每个客户的数据放在独立目录，互不可见
    """

    def __init__(self):
        self.base_dir = DATA_DIR / 'users'

    def store_input(self, user_id: str, job_id: str, filename: str, content: bytes) -> Path:
        """存储客户输入文件"""
        path = self.base_dir / user_id / 'jobs' / job_id / 'input' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def get_output(self, user_id: str, job_id: str) -> Optional[Path]:
        """获取客户输出文件"""
        result_zip = self.base_dir / user_id / 'jobs' / job_id / 'results.zip'
        if result_zip.exists():
            return result_zip
        return None

    def clean_user_data(self, user_id: str, max_age_days: int = 30):
        """清理过期用户数据"""
        user_dir = self.base_dir / user_id / 'jobs'
        if not user_dir.exists():
            return

        now = time.time()
        for job_dir in user_dir.iterdir():
            if job_dir.is_dir():
                mtime = job_dir.stat().st_mtime
                age_days = (now - mtime) / 86400
                if age_days > max_age_days:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    print(f"[DataIsolator] 已清理过期数据: {job_dir}")


if __name__ == '__main__':
    # 测试
    key_mgr = ApiKeyManager()
    scheduler = JobScheduler()

    print("=== 单用户模式任务调度器 ===")
    print(f"数据目录: {DATA_DIR}")

    # 模拟流程
    key = key_mgr.create_key("Test Lab", "test@university.edu", "premium")
    print(f"\n生成的API Key: {key}")

    key_hash = hashlib.sha256(key.encode()).hexdigest()

    # 提交任务
    success, job_id = scheduler.submit_job(
        user_id=key_mgr.validate_key(key)['user_id'],
        api_key_hash=key_hash,
        sequences=["ACGUACGUACGU"],
        bsj_positions=[(0, 12)],
        mode='fast',
    )

    print(f"任务提交: {'成功' if success else '失败'}")
    print(f"任务ID: {job_id}")
    print(f"队列状态: {scheduler.get_queue_stats()}")