"""
CircRNA Cloud - AutoDL风格算力租赁平台
Web前端 (Streamlit)

功能：
1. 镜像模板选择
2. 实例创建/启动/停止
3. 实时计费显示
4. Web终端 + Jupyter访问
"""

import streamlit as st
import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.instance_manager import InstanceManager, MIRROR_CATALOG, MIRROR_MAP
from backend.user_billing import UserManager, BillingEngine
from backend.payment import PaymentGateway, PaymentMethod, PaymentStatus, admin_panel
from backend.captcha import captcha_generator

# 页面配置
st.set_page_config(
    page_title="CircRNA Cloud - 算力租赁",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background: #0f0f1a; }

    .instance-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .instance-card.running {
        border-color: #4CAF50;
        border-left: 4px solid #4CAF50;
    }
    .instance-card.stopped {
        border-color: #666;
        border-left: 4px solid #666;
    }

    .mirror-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s;
    }
    .mirror-card:hover {
        border-color: #c41e3a;
        transform: translateY(-2px);
    }

    .price-tag {
        font-size: 28px;
        font-weight: bold;
        color: #4CAF50;
    }

    .status-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
    }
    .status-badge.running {
        background: rgba(76,175,80,0.2);
        color: #4CAF50;
    }
    .status-badge.stopped {
        background: rgba(255,255,255,0.1);
        color: #999;
    }

    .console-box {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        font-family: monospace;
        font-size: 13px;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.current_page = 'instances'
    st.session_state.pending_payment = None
    st.session_state.payment_qr = None
    st.session_state.captcha_id = None
    st.session_state.captcha_img = None


# ============================================================
# 页面：登录/注册
# ============================================================
def show_auth_page():
    st.markdown("""
    <div style="text-align: center; padding: 40px;">
        <h1>🧬 CircRNA Cloud</h1>
        <p style="color: #999;">面向科研团队的circRNA算力租赁平台</p>
        <p style="color: #666; font-size: 12px;">按小时计费 · 即开即用 · 预置生信环境</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        tab1, tab2 = st.tabs(["登录", "注册"])

        user_mgr = UserManager()

        with tab1:
            # 生成验证码
            if st.session_state.captcha_id is None:
                captcha_id, captcha_img = captcha_generator.generate()
                st.session_state.captcha_id = captcha_id
                st.session_state.captcha_img = captcha_img

            with st.form("login_form"):
                username = st.text_input("用户名/邮箱", placeholder="your@email.com")
                password = st.text_input("密码", type="password")

                # 验证码
                st.markdown("**验证码：**")
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    captcha_input = st.text_input("输入验证码", max_chars=4, label_visibility="collapsed")
                with col_c2:
                    if st.button("🔄 刷新", use_container_width=True):
                        captcha_id, captcha_img = captcha_generator.generate()
                        st.session_state.captcha_id = captcha_id
                        st.session_state.captcha_img = captcha_img
                        st.rerun()

                # 显示验证码图片
                if st.session_state.captcha_img:
                    st.image(st.session_state.captcha_img, width=150)

                submitted = st.form_submit_button("登录", use_container_width=True)

                if submitted:
                    # 验证验证码
                    if not captcha_generator.verify(st.session_state.captcha_id, captcha_input):
                        st.error("验证码错误")
                        # 刷新验证码
                        captcha_id, captcha_img = captcha_generator.generate()
                        st.session_state.captcha_id = captcha_id
                        st.session_state.captcha_img = captcha_img
                    else:
                        success, user, msg = user_mgr.login(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user = user
                            st.success(f"欢迎回来, {user['username']}!")
                            st.rerun()
                        else:
                            st.error(msg)
                            # 刷新验证码
                            captcha_id, captcha_img = captcha_generator.generate()
                            st.session_state.captcha_id = captcha_id
                            st.session_state.captcha_img = captcha_img

        with tab2:
            # 注册页验证码
            if 'reg_captcha_id' not in st.session_state:
                captcha_id, captcha_img = captcha_generator.generate()
                st.session_state.reg_captcha_id = captcha_id
                st.session_state.reg_captcha_img = captcha_img

            with st.form("register_form"):
                new_username = st.text_input("用户名", placeholder="my_lab")
                new_email = st.text_input("邮箱", placeholder="me@university.edu")
                new_password = st.text_input("密码", type="password", placeholder="至少6位")
                confirm = st.text_input("确认密码", type="password")

                # 验证码
                st.markdown("**验证码：**")
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    reg_captcha_input = st.text_input("输入验证码", max_chars=4, key="reg_captcha_input", label_visibility="collapsed")
                with col_c2:
                    if st.button("🔄 刷新", key="refresh_reg_captcha", use_container_width=True):
                        captcha_id, captcha_img = captcha_generator.generate()
                        st.session_state.reg_captcha_id = captcha_id
                        st.session_state.reg_captcha_img = captcha_img
                        st.rerun()

                if st.session_state.get('reg_captcha_img'):
                    st.image(st.session_state.reg_captcha_img, width=150)

                submitted = st.form_submit_button("注册", use_container_width=True)

                if submitted:
                    # 验证验证码
                    if not captcha_generator.verify(st.session_state.reg_captcha_id, reg_captcha_input):
                        st.error("验证码错误")
                        captcha_id, captcha_img = captcha_generator.generate()
                        st.session_state.reg_captcha_id = captcha_id
                        st.session_state.reg_captcha_img = captcha_img
                    elif new_password != confirm:
                        st.error("两次密码不一致")
                    else:
                        success, msg = user_mgr.register(new_username, new_email, new_password)
                        if success:
                            st.success(msg)
                            st.info("现在去登录吧")
                        else:
                            st.error(msg)
                            captcha_id, captcha_img = captcha_generator.generate()
                            st.session_state.reg_captcha_id = captcha_id
                            st.session_state.reg_captcha_img = captcha_img

    # 底部信息
    st.markdown("""
    <div style="text-align: center; margin-top: 60px; color: #555; font-size: 12px;">
        <p>所有实例运行于 DGX Spark (Blackwell GB10 GPU)</p>
        <p>预置 ViennaRNA · OpenMM · RoseTTAFold2NA · PyTorch</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 页面：实例列表
# ============================================================
def show_instances_page():
    st.markdown("### 🖥️ 我的实例")

    user = st.session_state.user
    if not user:
        return

    ins_mgr = InstanceManager()
    instances = ins_mgr.get_user_instances(user['user_id'])

    # 余额显示
    user_mgr = UserManager()
    balance = user_mgr.get_balance(user['user_id'])
    st.sidebar.markdown(f"""
    <div style="padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px;">
        <p style="color: #999; margin: 0;">账户余额</p>
        <p style="font-size: 28px; color: #4CAF50; margin: 0; font-weight: bold;">${balance:.2f}</p>
        <p style="color: #666; font-size: 11px; margin: 0;">{user['username']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 导航
    st.sidebar.markdown("---")
    nav_options = ["📦 实例管理", "🖼️ 创建实例", "💰 充值", "⚙️ 设置"]
    if user.get('is_admin'):
        nav_options.append("👑 管理后台")
    page = st.sidebar.radio("导航", nav_options, label_visibility="collapsed")

    if page == "📦 实例管理":
        if not instances:
            st.info("暂无实例，去「创建实例」页面开一台吧")
        else:
            for inst in instances:
                status = inst['status']
                css_class = 'running' if status == 'running' else 'stopped'
                status_text = '🟢 运行中' if status == 'running' else '⏹️ 已停止'

                with st.container():
                    st.markdown(f"""
                    <div class="instance-card {css_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h3 style="margin: 0;">{inst['instance_name']}</h3>
                                <p style="color: #999; margin: 5px 0;">
                                    镜像: {inst['mirror']} · {status_text}
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <p style="font-size: 20px; margin: 0;">${inst.get('current_cost', 0):.2f}</p>
                                <p style="color: #666; font-size: 11px; margin: 0;">
                                    ${inst['price_per_hour']:.2f}/h
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

                    if status == 'running':
                        col1.button("⏹️ 停止", key=f"stop_{inst['instance_id']}",
                                    on_click=lambda iid=inst['instance_id']: ins_mgr.stop_instance(iid))
                        col2.button("🔌 连接", key=f"connect_{inst['instance_id']}")

                        # 连接信息
                        with st.expander("🔗 连接信息"):
                            st.code(f"""
SSH:  ssh -p {inst['ssh_port']} crcuser@your-server-ip
       Password: {inst['ssh_password']}

Web:  http://your-server-ip:{list(inst['ports'].values())[0]}
Jupyter:  http://your-server-ip:{list(inst['ports'].values())[1]}

GPU: NVIDIA Blackwell GB10
使用: nvidia-smi
                            """)
                    elif status == 'stopped':
                        col1.button("▶️ 启动", key=f"start_{inst['instance_id']}",
                                    on_click=lambda iid=inst['instance_id']: ins_mgr.start_instance(iid))
                        col3.button("🗑️ 删除", key=f"delete_{inst['instance_id']}",
                                    on_click=lambda iid=inst['instance_id']: ins_mgr.delete_instance(iid))

                    st.markdown("---")

    elif page == "🖼️ 创建实例":
        show_create_instance(user, ins_mgr)

    elif page == "💰 充值":
        show_recharge(user)

    elif page == "⚙️ 设置":
        show_settings(user)

    elif page == "👑 管理后台":
        show_admin_panel(user)


# ============================================================
# 创建实例页
# ============================================================
def show_create_instance(user: dict, ins_mgr: InstanceManager):
    st.markdown("### 🖼️ 选择镜像创建实例")

    # 余额不足提醒
    user_mgr = UserManager()
    balance = user_mgr.get_balance(user['user_id'])
    if balance <= 0:
        st.warning("⚠️ 余额为0，请先充值")
        if st.button("去充值"):
            st.session_state.current_page = 'recharge'
            st.rerun()
        return

    # 镜像选择
    st.markdown("#### 选择镜像模板")

    cols = st.columns(len(MIRROR_CATALOG))

    for i, (col, mirror) in enumerate(zip(cols, MIRROR_CATALOG)):
        if mirror.name == 'custom':
            continue
        with col:
            st.markdown(f"""
            <div class="mirror-card">
                <h3>{mirror.display_name}</h3>
                <p class="price-tag">${mirror.price_per_hour:.2f}</p>
                <p style="font-size: 12px; color: #999;">/ 小时</p>
                <p style="font-size: 12px; color: #bbb;">{mirror.max_cpu}核 · {mirror.max_mem}</p>
                <p style="font-size: 12px; color: #777;">{mirror.description[:50]}...</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"选择 {mirror.display_name}", key=f"select_{mirror.name}", use_container_width=True):
                st.session_state.selected_mirror = mirror.name

    # 实例配置
    selected = st.session_state.get('selected_mirror', 'circrna-base')
    mirror = MIRROR_MAP.get(selected, MIRROR_MAP['circrna-base'])

    st.markdown(f"---\n**已选择:** {mirror.display_name} (${mirror.price_per_hour:.2f}/h)")

    with st.form("create_instance_form"):
        instance_name = st.text_input("实例名称", value=f"my-{selected}-{datetime.now().strftime('%m%d')}")
        gpu = st.checkbox("启用GPU (Blackwell GB10)", value=True, help="不勾选则不使用GPU，按CPU计费")

        # 费用预估
        est_cost_10h = mirror.price_per_hour * 10 * (1 if gpu else 0.5)
        est_cost_48h = mirror.price_per_hour * 48 * (1 if gpu else 0.5)

        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; margin: 10px 0;">
            <p><strong>费用预估:</strong></p>
            <p>10小时: <strong>${est_cost_10h:.2f}</strong></p>
            <p>48小时: <strong>${est_cost_48h:.2f}</strong></p>
            <p style="color: #999; font-size: 12px;">当前余额: ${user_mgr.get_balance(user['user_id']):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        submitted = st.form_submit_button("🚀 创建实例", use_container_width=True)

        if submitted:
            if user_mgr.get_balance(user['user_id']) < mirror.price_per_hour:
                st.error("余额不足，至少需要1小时的费用")
            else:
                instance = ins_mgr.create_instance(
                    user_id=user['user_id'],
                    mirror_name=selected,
                    instance_name=instance_name,
                    gpu_required=gpu,
                )

                st.success(f"实例创建成功！是否立即启动？")

                col1, col2 = st.columns(2)
                if col1.button("▶️ 立即启动", use_container_width=True):
                    ins_mgr.start_instance(instance['instance_id'])
                    st.rerun()

                st.info(f"""
                **实例信息:**
                - 实例ID: `{instance['instance_id']}`
                - SSH密码: `{instance['ssh_password']}`
                - 请妥善保管以上信息
                """)


# ============================================================
# 充值页
# ============================================================
def show_recharge(user: dict):
    st.markdown("### 💰 账户充值")

    user_mgr = UserManager()
    payment_gw = PaymentGateway()
    balance = user_mgr.get_balance(user['user_id'])

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; text-align: center;">
        <p style="color: #999;">当前余额</p>
        <p style="font-size: 48px; color: #4CAF50; font-weight: bold;">${balance:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 如果有待支付的订单，显示二维码
    if st.session_state.get('pending_payment'):
        show_payment_qr(user, user_mgr, payment_gw)
        return

    # 充值金额选择
    st.markdown("#### 选择充值金额")

    amounts = [5, 10, 20, 50, 100, 200]

    # 支付方式选择
    pay_method = st.radio(
        "支付方式",
        ["支付宝", "微信支付"],
        horizontal=True,
    )
    method = PaymentMethod.ALIPAY if pay_method == "支付宝" else PaymentMethod.WECHAT

    cols = st.columns(3)

    for i, amount in enumerate(amounts):
        with cols[i % 3]:
            bonus = "+$2" if amount >= 20 else ""
            bonus = "+$5" if amount >= 50 else bonus
            bonus = "+$10" if amount >= 100 else bonus

            st.markdown(f"""
            <div style="border: 1px solid #333; border-radius: 10px; padding: 20px; text-align: center; margin: 10px 0;">
                <p style="font-size: 28px; font-weight: bold;">${amount}</p>
                <p style="color: #4CAF50; font-size: 12px;">{bonus}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"充值 ${amount}", key=f"pay_{amount}", use_container_width=True):
                # 计算实际到账
                actual_amount = amount
                if amount >= 100:
                    actual_amount = amount + 10
                elif amount >= 20:
                    actual_amount = amount + 2

                # 创建支付单
                payment = payment_gw.create_payment(
                    amount=amount,
                    method=method,
                    user_id=user['user_id'],
                    payer_name=user['username'],
                    payer_account=user['email'],
                )
                payment.reference = f"recharge_{user['user_id']}_{int(time.time())}"

                # 生成二维码
                qr_base64 = payment_gw.generate_qr(payment)

                # 保存到session
                st.session_state.pending_payment = {
                    'payment_id': payment.payment_id,
                    'amount': amount,
                    'actual_amount': actual_amount,
                    'method': method.value,
                    'reference': payment.reference,
                }
                st.session_state.payment_qr = qr_base64

                st.rerun()

    # 银行转账说明
    with st.expander("🏦 银行转账（企业用户）"):
        st.markdown("""
        **银行转账流程：**
        1. 转账至以下账户
        2. 发送转账凭证至 admin@circrna-cloud.example.com
        3. 管理员审核后手动到账

        **收款信息：**
        - 户名：XX科技有限公司
        - 账号：XXXX XXXX XXXX XXXX
        - 开户行：XX银行XX支行

        企业用户可开具增值税发票。
        """)

    st.markdown("""
    <p style="color: #666; font-size: 12px; text-align: center; margin-top: 30px;">
        支付遇到问题？联系客服：support@circrna-cloud.example.com
    </p>
    """, unsafe_allow_html=True)


def show_payment_qr(user: dict, user_mgr: UserManager, payment_gw: PaymentGateway):
    """显示支付二维码和状态轮询"""
    pending = st.session_state.pending_payment
    qr_base64 = st.session_state.payment_qr

    st.markdown("#### 📱 扫码支付")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # 显示二维码
        if qr_base64:
            import base64
            qr_svg = base64.b64decode(qr_base64).decode()
            st.markdown(f"""
            <div style="background: #fff; padding: 20px; border-radius: 12px; text-align: center;">
                {qr_svg}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align: center; margin-top: 15px;">
            <p style="font-size: 24px; color: #4CAF50; font-weight: bold;">${pending['amount']}</p>
            <p style="color: #999; font-size: 14px;">
                支付方式：{'支付宝' if pending['method'] == 'alipay' else '微信支付'}
            </p>
            <p style="color: #666; font-size: 12px;">
                订单号：{pending['reference']}
            </p>
            <p style="color: #ff9800; font-size: 12px;">
                ⏰ 有效期：24小时
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 模拟支付成功按钮（演示模式）
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ 我已完成支付", use_container_width=True):
            # 演示模式：直接确认支付
            success = payment_gw.verify_payment(
                type('Payment', (), {'payment_id': pending['payment_id'], 'status': PaymentStatus.PENDING})(),
                f"demo_txn_{int(time.time())}"
            )

            if success:
                # 给用户加余额
                user_mgr.add_balance(user['user_id'], pending['actual_amount'])

                # 清除pending状态
                st.session_state.pending_payment = None
                st.session_state.payment_qr = None

                st.success(f"支付成功！到账：${pending['actual_amount']:.2f}")
                time.sleep(1.5)
                st.rerun()
            else:
                st.error("支付验证失败，请联系客服")

    with col2:
        if st.button("❌ 取消支付", use_container_width=True):
            st.session_state.pending_payment = None
            st.session_state.payment_qr = None
            st.rerun()

    # 自动刷新提示
    st.markdown("""
    <p style="color: #666; font-size: 12px; text-align: center; margin-top: 20px;">
        支付完成后请点击"我已完成支付"按钮<br>
        如遇问题请联系客服
    </p>
    """, unsafe_allow_html=True)


# ============================================================
# 管理后台
# ============================================================
def show_admin_panel(user: dict):
    """管理后台 - 审核支付、管理用户"""
    st.markdown("### 👑 管理后台")

    # Tab导航
    tab1, tab2, tab3 = st.tabs(["待审核支付", "用户管理", "系统状态"])

    with tab1:
        pending_payments = admin_panel.get_pending_payments()

        if not pending_payments:
            st.info("暂无待审核支付")
        else:
            st.markdown(f"**待审核订单：{len(pending_payments)} 笔**")

            for p in pending_payments:
                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin: 10px 0;">
                        <p><strong>订单号:</strong> {p['payment_id']}</p>
                        <p><strong>金额:</strong> ${p['amount']:.2f}</p>
                        <p><strong>支付方式:</strong> {p['method']}</p>
                        <p><strong>用户:</strong> {p.get('payer_name', 'N/A')} ({p.get('user_id', 'N/A')})</p>
                        <p><strong>创建时间:</strong> {p.get('created_at', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ 批准", key=f"approve_{p['payment_id']}"):
                            if admin_panel.approve_payment(p['payment_id']):
                                st.success("已批准，余额已到账")
                                st.rerun()

                    with col2:
                        reject_reason = st.text_input("拒绝原因", key=f"reason_{p['payment_id']}")
                        if st.button(f"❌ 拒绝", key=f"reject_{p['payment_id']}"):
                            admin_panel.reject_payment(p['payment_id'], reject_reason)
                            st.warning("已拒绝该支付")
                            st.rerun()

    with tab2:
        user_mgr = UserManager()
        users = user_mgr._read_users()

        st.markdown(f"**注册用户：{len(users)} 人**")

        for uid, u in users.items():
            st.markdown(f"""
            - **{u['username']}** ({u['email']})
              余额: ${u.get('balance', 0):.2f} | 累计消费: ${u.get('total_spent', 0):.2f}
            """)

    with tab3:
        ins_mgr = InstanceManager()
        instances = ins_mgr.list_instances()

        running = [i for i in instances if i and i.get('status') == 'running']
        stopped = [i for i in instances if i and i.get('status') == 'stopped']

        col1, col2, col3 = st.columns(3)
        col1.metric("运行中实例", len(running))
        col2.metric("已停止实例", len(stopped))
        col3.metric("总实例数", len(instances))


# ============================================================
# 设置页
# ============================================================
def show_settings(user: dict):
    st.markdown("### ⚙️ 账户设置")

    user_mgr = UserManager()
    user_info = user_mgr.get_user(user['user_id'])

    if user_info:
        st.markdown(f"""
        <div class="instance-card">
            <p><strong>用户ID:</strong> {user_info['user_id']}</p>
            <p><strong>用户名:</strong> {user_info['username']}</p>
            <p><strong>邮箱:</strong> {user_info['email']}</p>
            <p><strong>注册时间:</strong> {user_info['created_at']}</p>
            <p><strong>累计消费:</strong> ${user_info.get('total_spent', 0):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # 修改密码
        with st.form("change_pwd"):
            st.markdown("#### 修改密码")
            old_pwd = st.text_input("当前密码", type="password")
            new_pwd = st.text_input("新密码", type="password")
            submitted = st.form_submit_button("修改")

        # 退出登录
        if st.button("🚪 退出登录"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()


# ============================================================
# 主入口
# ============================================================
def main():
    if not st.session_state.logged_in:
        show_auth_page()
    else:
        show_instances_page()


if __name__ == '__main__':
    main()