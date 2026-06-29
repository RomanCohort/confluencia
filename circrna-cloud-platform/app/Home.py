"""
CircRNA Cloud Platform - circRNA 3D结构预测算力租赁平台
面向科研学生用户，教育邮箱验证，防挖矿机制

功能模块：
- 用户注册/登录（教育邮箱验证）
- 任务提交队列
- 资源配额管理（防挖矿）
- 任务状态追踪
- 结果下载

部署：可直接部署到DGX Spark服务器
"""

# 核心依赖
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import extra_streamlit_components as stx

# 标准库
import os
import sys
import json
import time
import hashlib
import re
import sqlite3
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# 第三方
import pandas as pd
import numpy as np
import requests

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 配置
from config.settings import (
    PLATFORM_NAME,
    EDUCATION_EMAIL_DOMAINS,
    GPU_COST_PER_HOUR,
    FREE_QUOTA_MONTHLY,
    MAX_CONCURRENT_JOBS_PER_USER,
    MAX_JOB_DURATION_HOURS,
    MIN_JOB_PRIORITY,
    MAX_JOB_PRIORITY,
    ADMIN_EMAILS,
    DATABASE_PATH,
    JOB_QUEUE_PATH,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_PER_HOUR,
)

# 后端
from backend.job_manager import JobManager, JobQueue, JobStatus
from backend.user_manager import UserManager, UserTier
from backend.resource_monitor import ResourceMonitor, AntiMiningGuard
from backend.email_validator import EducationEmailValidator

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title=f"{PLATFORM_NAME} - circRNA算力租赁",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': f"{PLATFORM_NAME} - 面向科研学生的circRNA 3D结构预测算力租赁平台",
        'Get Help': 'https://github.com/IGEM-FBH/confluencia',
    }
)

# ============================================================
# 自定义CSS
# ============================================================

st.markdown("""
<style>
    /* 深色主题 */
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

    /* 卡片样式 */
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    .pricing-card {
        background: linear-gradient(135deg, rgba(196,30,58,0.1) 0%, rgba(45,58,79,0.3) 100%);
        border: 1px solid #c41e3a;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }

    .pricing-card.featured {
        background: linear-gradient(135deg, rgba(196,30,58,0.3) 0%, rgba(45,58,79,0.5) 100%);
        border: 2px solid #c41e3a;
        transform: scale(1.05);
    }

    /* 状态指示 */
    .status-running { color: #4CAF50; font-weight: bold; }
    .status-queued { color: #FFC107; }
    .status-completed { color: #2196F3; }
    .status-failed { color: #F44336; }

    /* 进度条 */
    .progress-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }

    /* 防挖矿警告 */
    .warning-banner {
        background: rgba(244,67,54,0.2);
        border: 1px solid #F44336;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* 登录表单 */
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 30px;
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* 教育邮箱验证 */
    .edu-badge {
        background: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
    }

    /* GPU指示器 */
    .gpu-meter {
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .gpu-icon { font-size: 24px; }
    .gpu-available { color: #4CAF50; }
    .gpu-busy { color: #FFC107; }
    .gpu-full { color: #F44336; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 认证系统初始化
# ============================================================

def init_authenticator():
    """初始化Streamlit认证器"""

    # 加载用户配置
    config_path = PROJECT_ROOT / 'config' / 'users.yaml'

    if not config_path.exists():
        # 创建默认配置
        default_config = {
            'credentials': {
                'usernames': {}
            },
            'cookie': {
                'name': 'circrna_cloud',
                'key': 'random_key_change_this',
                'expiry_days': 30,
            },
            'preauthorized': {
                'emails': []
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)

    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    return authenticator, config, config_path

# ============================================================
# 教育邮箱验证
# ============================================================

def validate_education_email(email: str) -> Tuple[bool, str]:
    """
    验证教育邮箱

    支持的教育邮箱域名：
    - 中国高校：.edu.cn
    - 国际高校：.edu, .ac.uk, .ac.jp 等
    - 研究机构：部分科研域名
    """
    validator = EducationEmailValidator(EDUCATION_EMAIL_DOMAINS)
    return validator.validate(email)

# ============================================================
# 防挖矿机制
# ============================================================

def check_user_quota(user_id: str) -> Dict:
    """
    检查用户配额（核心防挖矿机制）

    配额维度：
    1. 每月免费额度
    2. 每分钟请求频率
    3. 每小时任务数
    4. 最大并发任务数
    5. 单任务最大时长
    """
    guard = AntiMiningGuard(
        rate_limit_per_minute=RATE_LIMIT_PER_MINUTE,
        rate_limit_per_hour=RATE_LIMIT_PER_HOUR,
        max_concurrent_jobs=MAX_CONCURRENT_JOBS_PER_USER,
        max_job_duration=MAX_JOB_DURATION_HOURS,
        free_quota_monthly=FREE_QUOTA_MONTHLY
    )

    return guard.check_user(user_id)

def display_quota_status(quota: Dict):
    """显示用户配额状态"""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        remaining = quota['monthly_remaining']
        total = FREE_QUOTA_MONTHLY
        pct = remaining / total * 100
        st.metric(
            "本月剩余额度",
            f"{remaining:.1f} GPU小时",
            f"{remaining - total:.1f}",
            delta_color="inverse" if remaining < total/2 else "normal"
        )
        st.progress(pct / 100)

    with col2:
        st.metric(
            "当前并发任务",
            f"{quota['current_jobs']}/{MAX_CONCURRENT_JOBS_PER_USER}"
        )

    with col3:
        st.metric(
            "本月已完成",
            f"{quota['completed_jobs']} 任务"
        )

    with col4:
        st.metric(
            "排队任务",
            f"{quota['queued_jobs']} 个"
        )

    # 配额警告
    if quota['monthly_remaining'] < FREE_QUOTA_MONTHLY * 0.2:
        st.warning(f"⚠️ 本月额度即将用尽！剩余 {quota['monthly_remaining']:.1f} GPU小时。请充值或等待下月重置。")

    if quota['rate_limited']:
        st.error(f"🚫 请求频率超限！请等待 {quota['rate_limit_wait']} 秒后再试。")

# ============================================================
# 主页
# ============================================================

def show_home():
    """显示主页"""

    st.markdown("""
    <div class="main-header">
        <h1 style="color: #ecf0f1; margin: 0;">🧬 CircRNA Cloud Platform</h1>
        <p style="color: #bdc3c7; margin-top: 10px;">
            面向科研学生的 circRNA 3D 结构预测算力租赁平台
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 功能介绍
    st.markdown("### 🎯 平台功能")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>🔬 circRNA 3D结构预测</h4>
            <p>使用 ViennaRNA + RoseTTAFold2NA + OpenMD 高质量预测</p>
            <ul>
                <li>BSJ环化约束</li>
                <li>20ns MD弛豫</li>
                <li>多重质量验证</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>⚡ GPU算力租赁</h4>
            <p>DGX Spark 高性能计算集群</p>
            <ul>
                <li>8× A100/H100 GPU</li>
                <li>并行批量处理</li>
                <li>学生优惠价格</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <h4>🛡️ 安全保障</h4>
            <p>教育邮箱验证 + 防挖矿机制</p>
            <ul>
                <li>.edu邮箱专属</li>
                <li>配额限制</li>
                <li>优先级队列</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 价格方案
    st.markdown("### 💰 价格方案（学生专属优惠）")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="pricing-card">
            <h3>免费试用</h3>
            <p style="font-size: 32px; color: #4CAF50;">$0</p>
            <p>每月 5 GPU小时</p>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <ul style="text-align: left;">
                <li>教育邮箱注册即送</li>
                <li>单任务最长30分钟</li>
                <li>最多2个并发任务</li>
            </ul>
            <p style="color: #bdc3c7; font-size: 12px;">仅限.edu邮箱用户</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pricing-card featured">
            <h3>学生标准</h3>
            <p style="font-size: 32px; color: #c41e3a;">$0.50</p>
            <p>每GPU小时</p>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <ul style="text-align: left;">
                <li>教育邮箱专属优惠</li>
                <li>单任务最长2小时</li>
                <li>最多4个并发任务</li>
                <li>优先级队列</li>
            </ul>
            <p style="color: #4CAF50; font-size: 12px;">比商业价便宜80%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="pricing-card">
            <h3>科研团队</h3>
            <p style="font-size: 32px; color: #2196F3;">$50</p>
            <p>包月100 GPU小时</p>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <ul style="text-align: left;">
                <li>团队账号共享</li>
                <li>单任务最长10小时</li>
                <li>最多8个并发任务</li>
                <li>最高优先级</li>
            </ul>
            <p style="color: #bdc3c7; font-size: 12px;">需团队邮箱认证</p>
        </div>
        """, unsafe_allow_html=True)

    # GPU状态实时显示
    st.markdown("### 🖥️ GPU集群实时状态")

    monitor = ResourceMonitor()
    gpu_status = monitor.get_gpu_status()

    cols = st.columns(8)
    gpu_icons = {
        'idle': ('✅', 'gpu-available'),
        'running': ('🔄', 'gpu-busy'),
        'full': ('⛔', 'gpu-full')
    }

    for i, col in enumerate(cols):
        gpu = gpu_status.get(i, {'status': 'idle', 'utilization': 0})
        icon, css_class = gpu_icons.get(gpu['status'], gpu_icons['idle'])
        col.markdown(f"""
        <div class="gpu-meter">
            <span class="gpu-icon {css_class}">{icon}</span>
            <span>GPU {i+1}</span>
            <span style="color: #bdc3c7;">{gpu['utilization']}%</span>
        </div>
        """, unsafe_allow_html=True)

    # 队列状态
    st.markdown("### 📊 任务队列状态")

    queue = JobQueue()
    queue_stats = queue.get_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("排队中", queue_stats['queued'])
    col2.metric("运行中", queue_stats['running'])
    col3.metric("今日完成", queue_stats['completed_today'])
    col4.metric("平均等待", f"{queue_stats['avg_wait_time']}分钟")

    # 使用说明
    st.markdown("### 📖 使用指南")
    st.markdown("""
    1. **注册账号**：使用教育邮箱（.edu/.edu.cn等）注册
    2. **验证邮箱**：收到验证邮件后点击链接激活
    3. **提交任务**：上传circRNA序列FASTA文件
    4. **等待处理**：系统自动排队处理
    5. **下载结果**：完成后在"我的任务"页面下载

    > **防挖矿机制**：平台设置了配额限制、频率限制和并发限制，确保资源公平分配给科研用户。
    """)

# ============================================================
# 认证页面
# ============================================================

def show_login_page():
    """显示登录/注册页面"""

    authenticator, config, config_path = init_authenticator()

    # 尝试登录
    try:
        name, authentication_status, username = authenticator.login('登录', 'main')
    except Exception as e:
        st.error(f"登录失败: {e}")
        authentication_status = None

    if authentication_status:
        # 登录成功
        st.session_state['authentication_status'] = True
        st.session_state['name'] = name
        st.session_state['username'] = username
        st.rerun()

    elif authentication_status == False:
        st.error('❌ 用户名或密码错误')

    elif authentication_status == None:
        # 显示注册选项
        st.markdown("---")
        st.markdown("### 📝 新用户注册")

        with st.form("register_form"):
            new_username = st.text_input("用户名", placeholder="your_name")
            new_email = st.text_input("教育邮箱", placeholder="your_name@university.edu")
            new_password = st.text_input("密码", type="password")
            new_password_confirm = st.text_input("确认密码", type="password")

            # 邮箱域名提示
            st.markdown(f"""
            <p style="color: #bdc3c7; font-size: 12px;">
                支持的教育邮箱域名：.edu, .edu.cn, .ac.uk, .ac.jp, .ac.cn 等
            </p>
            """, unsafe_allow_html=True)

            submitted = st.form_submit_button("注册账号")

            if submitted:
                # 验证教育邮箱
                is_valid, msg = validate_education_email(new_email)

                if not is_valid:
                    st.error(f"❌ {msg}")
                elif new_password != new_password_confirm:
                    st.error("❌ 两次密码不一致")
                elif len(new_password) < 6:
                    st.error("❌ 密码长度至少6位")
                elif new_username in config['credentials']['usernames']:
                    st.error("❌ 用户名已存在")
                else:
                    # 注册成功
                    hashed_password = stauth.Hasher([new_password]).generate()[0]

                    config['credentials']['usernames'][new_username] = {
                        'email': new_email,
                        'name': new_username,
                        'password': hashed_password,
                        'tier': 'free',
                        'quota_used': 0,
                        'created_at': datetime.now().isoformat(),
                        'last_login': datetime.now().isoformat()
                    }

                    # 保存配置
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)

                    st.success(f"✅ 注册成功！请使用用户名 '{new_username}' 登录")

# ============================================================
# 任务提交页面
# ============================================================

def show_submit_job():
    """显示任务提交页面"""

    st.markdown("### 🚀 提交预测任务")

    # 检查配额
    username = st.session_state.get('username', 'guest')
    quota = check_user_quota(username)

    display_quota_status(quota)

    if quota['rate_limited'] or quota['current_jobs'] >= MAX_CONCURRENT_JOBS_PER_USER:
        st.stop()

    # 任务表单
    with st.form("job_form"):
        st.markdown("#### 输入circRNA序列")

        # 文件上传
        uploaded_file = st.file_uploader(
            "上传FASTA文件",
            type=['fasta', 'fa', 'txt'],
            help="支持标准FASTA格式，每条序列标注BSJ位置"
        )

        # 或直接输入序列
        st.markdown("**或直接输入序列：**")
        sequence_input = st.text_area(
            "circRNA序列",
            placeholder="ACGUACGUACGU...",
            max_chars=500,
            help="单条序列，支持AUGC四种碱基"
        )

        bsj_start = st.number_input("BSJ起始位置", min_value=0, value=0)
        bsj_end = st.number_input("BSJ结束位置", min_value=1, value=100)

        # 任务参数
        st.markdown("#### 任务参数")

        col1, col2 = st.columns(2)
        with col1:
            quality_mode = st.selectbox(
                "质量模式",
                ["fast (5min)", "quality (20min)", "ultra_quality (60min)"],
                help="quality模式产出最佳结果"
            )
        with col2:
            num_samples = st.slider(
                "采样数量",
                min_value=1, max_value=20, value=5,
                help="每序列生成多个构象"
            )

        # 预估成本
        seq_count = 1 if sequence_input else (uploaded_file.name if uploaded_file else 0)
        estimated_hours = {
            "fast (5min)": 0.1,
            "quality (20min)": 0.3,
            "ultra_quality (60min)": 1.0
        }

        if sequence_input or uploaded_file:
            est_gpu_hours = estimated_hours[quality_mode]
            est_cost = est_gpu_hours * GPU_COST_PER_HOUR

            st.markdown(f"""
            <div class="glass-card">
                <p><strong>预估消耗：</strong> {est_gpu_hours:.1f} GPU小时</p>
                <p><strong>预估费用：</strong> ${est_cost:.2f}</p>
                <p><strong>剩余额度：</strong> {quota['monthly_remaining']:.1f} GPU小时</p>
            </div>
            """, unsafe_allow_html=True)

            if est_gpu_hours > quota['monthly_remaining']:
                st.error("❌ 额度不足！请充值或减少任务规模。")

        # 提交按钮
        submitted = st.form_submit_button("提交任务", type="primary")

        if submitted:
            # 创建任务
            job_manager = JobManager()

            sequences = []
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                # 解析FASTA
                for line in content.split('\n'):
                    if not line.startswith('>') and line.strip():
                        sequences.append(line.strip())
            elif sequence_input:
                sequences.append(sequence_input)

            if not sequences:
                st.error("❌ 请输入序列或上传文件")
            else:
                job_id = job_manager.create_job(
                    user_id=username,
                    sequences=sequences,
                    bsj_positions=[(bsj_start, bsj_end)] * len(sequences),
                    quality_mode=quality_mode.split()[0],
                    num_samples=num_samples
                )

                st.success(f"✅ 任务已提交！任务ID: {job_id}")
                st.info("⏳ 任务已加入队列，请到「我的任务」页面查看进度")

# ============================================================
# 我的任务页面
# ============================================================

def show_my_jobs():
    """显示用户任务列表"""

    st.markdown("### 📋 我的任务")

    username = st.session_state.get('username', 'guest')
    job_manager = JobManager()

    jobs = job_manager.get_user_jobs(username)

    if not jobs:
        st.info("暂无任务记录")
        return

    # 任务表格
    df = pd.DataFrame(jobs)

    # 状态样式
    def format_status(status):
        css_class = {
            'queued': 'status-queued',
            'running': 'status-running',
            'completed': 'status-completed',
            'failed': 'status-failed'
        }.get(status, '')
        return f'<span class="{css_class}">{status.upper()}</span>'

    df['status_display'] = df['status'].apply(format_status)

    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # 下载已完成任务结果
    completed_jobs = [j for j in jobs if j['status'] == 'completed']

    if completed_jobs:
        st.markdown("### 📥 下载结果")

        for job in completed_jobs:
            with st.expander(f"任务 {job['job_id']} - {job['created_at']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("结构数", job['num_structures'])
                col2.metric("平均置信度", f"{job['avg_confidence']:.2f}")
                col3.metric("GPU小时", f"{job['gpu_hours']:.1f}")

                st.download_button(
                    "下载结果包 (ZIP)",
                    job_manager.get_result_zip(job['job_id']),
                    file_name=f"circrna_results_{job['job_id']}.zip"
                )

# ============================================================
# 主入口
# ============================================================

def main():
    """主入口"""

    # 初始化session state
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None

    # 检查认证状态
    if not st.session_state['authentication_status']:
        show_login_page()
        return

    # 已登录用户
    # 侧边栏导航
    st.sidebar.markdown(f"""
    <div style="padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px;">
        <h3 style="color: #ecf0f1;">👋 {st.session_state['name']}</h3>
        <p style="color: #bdc3c7; font-size: 12px;">
            <span class="edu-badge">EDU</span> 学生账户
        </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "导航",
        ["🏠 主页", "🚀 提交任务", "📋 我的任务", "💰 账户充值", "⚙️ 设置"]
    )

    # 退出按钮
    if st.sidebar.button("退出登录"):
        st.session_state['authentication_status'] = None
        st.rerun()

    # 页面路由
    if page == "🏠 主页":
        show_home()
    elif page == "🚀 提交任务":
        show_submit_job()
    elif page == "📋 我的任务":
        show_my_jobs()
    elif page == "💰 账户充值":
        show_payment()
    elif page == "⚙️ 设置":
        show_settings()


def show_payment():
    """显示充值页面"""
    st.markdown("### 💰 账户充值")

    username = st.session_state.get('username')
    quota = check_user_quota(username)

    st.markdown(f"""
    <div class="glass-card">
        <h4>当前余额</h4>
        <p style="font-size: 24px;">{quota['monthly_remaining']:.1f} GPU小时</p>
        <p style="color: #bdc3c7;">本月已使用: {quota['used_monthly']:.1f} GPU小时</p>
    </div>
    """, unsafe_allow_html=True)

    # 充值选项
    st.markdown("#### 选择充值套餐")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.button("10 GPU小时 - $5", key="buy10")
    with col2:
        st.button("50 GPU小时 - $20", key="buy50")
    with col3:
        st.button("100 GPU小时 - $35", key="buy100")

    st.markdown("""
    > **支付方式**：支付宝 / 微信支付 / PayPal

    > **充值说明**：充值额度永久有效，不会每月重置
    """)


def show_settings():
    """显示设置页面"""
    st.markdown("### ⚙️ 账户设置")

    username = st.session_state.get('username')

    with st.form("settings_form"):
        st.text_input("用户名", value=username, disabled=True)
        st.text_input("邮箱", value="", placeholder="更新邮箱...")
        st.text_input("新密码", type="password", placeholder="修改密码...")

        st.form_submit_button("保存设置")


if __name__ == '__main__':
    main()