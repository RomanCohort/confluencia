"""
用户管理 - AutoDL风格简化版
用户充值后创建实例，按时计费
"""

import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
USERS_FILE = DATA_DIR / 'users.json'


class UserManager:
    """
    用户管理器

    简化版：
    - 注册/登录（普通邮箱即可）
    - 余额管理
    - 充值记录
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not USERS_FILE.exists():
            USERS_FILE.write_text('{}')

    def _read_users(self) -> Dict:
        return json.loads(USERS_FILE.read_text())

    def _write_users(self, users: Dict):
        USERS_FILE.write_text(json.dumps(users, indent=2, default=str))

    def register(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """用户注册"""
        if len(password) < 6:
            return False, "密码至少6位"

        users = self._read_users()

        # 检查重复
        for u in users.values():
            if u['username'] == username:
                return False, "用户名已存在"
            if u['email'] == email:
                return False, "邮箱已注册"

        user_id = f"usr_{uuid.uuid4().hex[:8]}"
        password_hash = self._hash_password(password)

        users[user_id] = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'balance': 0.0,                    # 账户余额（美元）
            'total_spent': 0.0,               # 累计消费
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True,
            'is_admin': False,                # 管理员标记
        }

        # 第一个注册用户自动成为管理员
        if len(users) == 1:
            users[user_id]['is_admin'] = True
            print(f"[UserManager] 管理员账号: {username} ({user_id})")

        self._write_users(users)
        return True, f"注册成功！用户ID: {user_id}"

    def login(self, username: str, password: str) -> Tuple[bool, Optional[Dict], str]:
        """用户登录"""
        users = self._read_users()

        for user in users.values():
            if user['username'] == username or user['email'] == username:
                if not user['is_active']:
                    return False, None, "账户已被禁用"

                if self._verify_password(password, user['password_hash']):
                    user['last_login'] = datetime.now().isoformat()
                    self._write_users(users)

                    return True, {
                        'user_id': user['user_id'],
                        'username': user['username'],
                        'email': user['email'],
                        'balance': user['balance'],
                        'is_admin': user.get('is_admin', False),
                    }, "登录成功"

        return False, None, "用户名或密码错误"

    def get_balance(self, user_id: str) -> float:
        """获取用户余额"""
        users = self._read_users()
        return users.get(user_id, {}).get('balance', 0.0)

    def add_balance(self, user_id: str, amount: float, method: str = 'alipay') -> bool:
        """
        充值

        Args:
            user_id: 用户ID
            amount: 充值金额（美元）
            method: 支付方式
        """
        users = self._read_users()
        if user_id not in users:
            return False

        users[user_id]['balance'] += amount

        # 记录充值
        if 'recharges' not in users[user_id]:
            users[user_id]['recharges'] = []

        users[user_id]['recharges'].append({
            'amount': amount,
            'method': method,
            'time': datetime.now().isoformat(),
        })

        self._write_users(users)
        print(f"[UserManager] 充值成功: +${amount:.2f} (用户: {user_id})")
        return True

    def deduct_balance(self, user_id: str, amount: float, description: str = '') -> bool:
        """
        扣款

        Args:
            user_id: 用户ID
            amount: 扣款金额
            description: 扣款说明
        """
        users = self._read_users()
        if user_id not in users:
            return False

        if users[user_id]['balance'] < amount:
            return False  # 余额不足

        users[user_id]['balance'] -= amount
        users[user_id]['total_spent'] += amount

        # 记录消费
        if 'transactions' not in users[user_id]:
            users[user_id]['transactions'] = []

        users[user_id]['transactions'].append({
            'amount': -amount,
            'description': description,
            'time': datetime.now().isoformat(),
        })

        self._write_users(users)
        return True

    def get_user(self, user_id: str) -> Optional[Dict]:
        """获取用户信息"""
        users = self._read_users()
        user = users.get(user_id)
        if user:
            # 不返回密码哈希
            safe_user = {k: v for k, v in user.items() if k != 'password_hash'}
            return safe_user
        return None

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        return f"{salt}:{hashlib.sha256((salt + password).encode()).hexdigest()}"

    def _verify_password(self, password: str, hash_str: str) -> bool:
        salt, stored_hash = hash_str.split(':')
        return stored_hash == hashlib.sha256((salt + password).encode()).hexdigest()


class BillingEngine:
    """
    计费引擎

    按秒计费，实时扣款
    """

    def __init__(self):
        self.user_mgr = UserManager()
        self.billing_file = DATA_DIR / 'billing.json'
        if not self.billing_file.exists():
            self.billing_file.write_text('{}')

    def _read_billing(self) -> Dict:
        return json.loads(self.billing_file.read_text())

    def _write_billing(self, data: Dict):
        self.billing_file.write_text(json.dumps(data, indent=2, default=str))

    def start_billing(self, instance_id: str, user_id: str, price_per_hour: float):
        """开始计费"""
        billing = self._read_billing()
        billing[instance_id] = {
            'user_id': user_id,
            'price_per_hour': price_per_hour,
            'started_at': datetime.now().isoformat(),
            'billed_amount': 0.0,
        }
        self._write_billing(billing)

    def update_billing(self, instance_id: str) -> float:
        """
        更新计费（定时调用）

        Returns:
            当前累计费用
        """
        billing = self._read_billing()
        if instance_id not in billing:
            return 0.0

        record = billing[instance_id]
        started = datetime.fromisoformat(record['started_at'])
        elapsed = (datetime.now() - started).total_seconds()
        current_cost = round(elapsed / 3600 * record['price_per_hour'], 4)

        record['billed_amount'] = current_cost
        self._write_billing(billing)

        return current_cost

    def stop_billing(self, instance_id: str) -> Tuple[float, bool]:
        """
        停止计费并扣款

        Returns:
            (final_cost, success)
        """
        billing = self._read_billing()
        if instance_id not in billing:
            return 0.0, False

        record = billing[instance_id]
        final_cost = self.update_billing(instance_id)

        # 扣款
        success = self.user_mgr.deduct_balance(
            record['user_id'],
            final_cost,
            f"实例 {instance_id} 使用费"
        )

        # 清除计费记录
        del billing[instance_id]
        self._write_billing(billing)

        return final_cost, success

    def get_current_cost(self, instance_id: str) -> float:
        """获取当前费用"""
        return self.update_billing(instance_id)

    def check_balance_sufficient(self, user_id: str, price_per_hour: float, max_hours: float = 10) -> bool:
        """检查余额是否足够运行指定时长"""
        balance = self.user_mgr.get_balance(user_id)
        required = price_per_hour * max_hours
        return balance >= required


if __name__ == '__main__':
    # 测试
    user_mgr = UserManager()
    billing = BillingEngine()

    # 注册
    success, msg = user_mgr.register("testuser", "test@example.com", "123456")
    print(msg)

    # 登录
    success, user, msg = user_mgr.login("testuser", "123456")
    print(msg, user)

    # 充值
    user_mgr.add_balance(user['user_id'], 10.0, 'alipay')
    print(f"余额: ${user_mgr.get_balance(user['user_id']):.2f}")

    # 模拟计费
    billing.start_billing("ins_test", user['user_id'], 0.5)
    time.sleep(2)
    cost = billing.update_billing("ins_test")
    print(f"当前费用: ${cost:.4f}")

    final_cost, _ = billing.stop_billing("ins_test")
    print(f"最终费用: ${final_cost:.4f}")
    print(f"剩余余额: ${user_mgr.get_balance(user['user_id']):.2f}")