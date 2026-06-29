"""
用户管理模块 - 账户注册、认证、配额管理
"""

import os
import sqlite3
import json
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)


class UserTier(Enum):
    """用户等级"""
    FREE = 'free'           # 免费用户（教育邮箱注册）
    PAID = 'paid'           # 付费用户
    TEAM = 'team'           # 团队账户
    ADMIN = 'admin'         # 管理员


class UserManager:
    """用户管理"""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = str(DATA_DIR / 'users.db')
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email_verified INTEGER DEFAULT 0,
                tier TEXT DEFAULT 'free',
                quota_used_gpu_hours REAL DEFAULT 0,
                quota_used_this_month REAL DEFAULT 0,
                quota_reset_month TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT,
                api_key TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS email_verification (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                gpu_hours REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def register(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """
        用户注册

        Args:
            username: 用户名
            email: 教育邮箱
            password: 明文密码

        Returns:
            (success, message)
        """
        if not self._validate_password(password):
            return False, "密码长度至少6位"

        # 检查用户名或邮箱是否已注册
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('SELECT user_id FROM users WHERE username = ? OR email = ?', (username, email))
        if c.fetchone():
            conn.close()
            return False, "用户名或邮箱已注册"

        # 创建用户
        user_id = self._generate_user_id()
        password_hash = self._hash_password(password)
        now = datetime.now().isoformat()
        api_key = self._generate_api_key()
        month_str = datetime.now().strftime('%Y-%m')

        c.execute('''
            INSERT INTO users (user_id, username, email, password_hash, email_verified,
                              created_at, last_login, api_key, quota_reset_month)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
        ''', (user_id, username, email, password_hash, now, now, api_key, month_str))

        conn.commit()
        conn.close()

        return True, "注册成功！请检查邮箱完成验证。"

    def login(self, username: str, password: str) -> Tuple[bool, Optional[Dict], str]:
        """
        用户登录

        Args:
            username: 用户名或邮箱
            password: 明文密码

        Returns:
            (success, user_info, message)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 支持用户名或邮箱登录
        c.execute('''
            SELECT user_id, username, email, password_hash, tier, email_verified, is_active
            FROM users WHERE username = ? OR email = ?
        ''', (username, username))

        row = c.fetchone()
        if not row:
            conn.close()
            return False, None, "用户不存在"

        user_id, db_username, email, password_hash, tier, verified, active = row

        if not active:
            conn.close()
            return False, None, "账户已被禁用"

        if not self._verify_password(password, password_hash):
            conn.close()
            return False, None, "密码错误"

        # 更新登录时间
        c.execute('UPDATE users SET last_login = ? WHERE user_id = ?',
                 (datetime.now().isoformat(), user_id))

        # 检查月度配额重置
        current_month = datetime.now().strftime('%Y-%m')
        c.execute('SELECT quota_reset_month FROM users WHERE user_id = ?', (user_id,))
        reset_month = c.fetchone()[0]
        if reset_month != current_month:
            c.execute('UPDATE users SET quota_used_this_month = 0, quota_reset_month = ? WHERE user_id = ?',
                     (current_month, user_id))

        conn.commit()
        conn.close()

        user_info = {
            'user_id': user_id,
            'username': db_username,
            'email': email,
            'tier': UserTier(tier),
            'verified': bool(verified)
        }

        return True, user_info, "登录成功"

    def get_user_quota(self, user_id: str) -> Dict:
        """
        获取用户配额信息

        Returns:
            dict with quota details
        """
        from config.settings import FREE_QUOTA_MONTHLY

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('SELECT tier, quota_used_this_month, quota_used_gpu_hours FROM users WHERE user_id = ?', (user_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return {'error': 'User not found'}

        tier, used_this_month, total_used = row

        # 不同等级不同配额
        quotas = {
            'free': FREE_QUOTA_MONTHLY,
            'paid': 100.0,
            'team': 500.0,
            'admin': 99999.0
        }

        monthly_limit = quotas.get(tier, FREE_QUOTA_MONTHLY)
        remaining = max(0, monthly_limit - used_this_month)

        return {
            'tier': tier,
            'monthly_limit': monthly_limit,
            'used_this_month': used_this_month,
            'monthly_remaining': remaining,
            'total_used': total_used,
            'percentage': used_this_month / monthly_limit if monthly_limit > 0 else 0
        }

    def use_quota(self, user_id: str, gpu_hours: float) -> bool:
        """
        消耗用户配额

        Returns:
            bool: True if quota available and consumed
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        quota = self.get_user_quota(user_id)
        if 'error' in quota:
            conn.close()
            return False

        if quota['monthly_remaining'] < gpu_hours:
            conn.close()
            return False

        c.execute('''
            UPDATE users SET quota_used_gpu_hours = quota_used_gpu_hours + ?,
                             quota_used_this_month = quota_used_this_month + ?
            WHERE user_id = ?
        ''', (gpu_hours, gpu_hours, user_id))

        conn.commit()
        conn.close()
        return True

    def verify_email(self, email: str, code: str) -> Tuple[bool, str]:
        """验证教育邮箱"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT id FROM email_verification
            WHERE email = ? AND code = ? AND used = 0 AND expires_at > ?
        ''', (email, code, datetime.now().isoformat()))

        row = c.fetchone()
        if not row:
            conn.close()
            return False, "验证码无效或已过期"

        verification_id = row[0]
        c.execute('UPDATE email_verification SET used = 1 WHERE id = ?', (verification_id,))
        c.execute('UPDATE users SET email_verified = 1 WHERE email = ?', (email,))

        conn.commit()
        conn.close()
        return True, "邮箱验证成功！"

    def send_verification_email(self, email: str) -> Tuple[bool, str]:
        """
        发送验证邮件

        注：实际部署时需要配置SMTP
        """
        code = secrets.token_hex(32)
        now = datetime.now()
        expires = now + timedelta(hours=24)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT INTO email_verification (email, code, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (email, code, now.isoformat(), expires.isoformat()))

        conn.commit()
        conn.close()

        # 调试模式：打印验证码
        print(f"[DEV] Verification code for {email}: {code}")

        # TODO: 实际部署时替换为SMTP
        # send_email(email, f"验证您的邮箱\n验证链接: {BASE_URL}/verify?code={code}")

        return True, "验证邮件已发送"

    def generate_api_key(self, user_id: str) -> str:
        """生成新的API key"""
        api_key = self._generate_api_key()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE users SET api_key = ? WHERE user_id = ?', (api_key, user_id))
        conn.commit()
        conn.close()
        return api_key

    def _validate_password(self, password: str) -> bool:
        return len(password) >= 6

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        return f"{salt}:{hashlib.sha256((salt + password).encode()).hexdigest()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        salt, hash_value = password_hash.split(':')
        return hash_value == hashlib.sha256((salt + password).encode()).hexdigest()

    def _generate_user_id(self) -> str:
        return 'usr_' + secrets.token_hex(16)

    def _generate_api_key(self) -> str:
        return 'crc_' + secrets.token_hex(32)


class EducationEmailValidator:
    """
    教育邮箱验证器

    验证规则：
    1. 格式合法
    2. 域名在EDUCATION_EMAIL_DOMAINS中
    3. DNS MX记录查询（可选）
    """

    def __init__(self, edu_domains: List[str] = None):
        from config.settings import EDUCATION_EMAIL_DOMAINS
        self.edu_domains = edu_domains or EDUCATION_EMAIL_DOMAINS

    def validate(self, email: str) -> Tuple[bool, str]:
        """
        验证教育邮箱

        Args:
            email: 邮箱地址

        Returns:
            (is_valid, message)
        """
        # 基本格式验证
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return False, "邮箱格式无效"

        # 提取域名
        domain = email.split('@')[1].lower()

        # 检查是否匹配教育域名
        for pattern in self.edu_domains:
            if pattern.startswith(r'.*@.*'):
                # 全域名匹配
                if re.match(pattern, email.lower()):
                    return True, "教育邮箱验证通过"
            else:
                # 域名后缀匹配
                if re.match(pattern, domain):
                    return True, "教育邮箱验证通过"

        # 白名单高校
        for pattern in self.edu_domains:
            if pattern.startswith(r'.*@'):
                if re.match(pattern, email.lower()):
                    return True, "教育邮箱验证通过"

        return False, "非教育邮箱，请使用.edu/.edu.cn等教育邮箱注册"

    def check_dns_mx(self, domain: str) -> bool:
        """
        检查DNS MX记录（可选，需要dns库）
        """
        try:
            import dns.resolver
            records = dns.resolver.resolve(domain, 'MX')
            return len(records) > 0
        except ImportError:
            return True  # 未安装dns库则跳过
        except Exception:
            return False