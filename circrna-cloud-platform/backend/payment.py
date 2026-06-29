"""
支付模块 - 支付宝/微信支付对接
支持：
- 扫码支付（支付宝/微信）
- 企业转账
- 银行转账审批
- 管理员后台审核
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
PAYMENTS_FILE = DATA_DIR / 'payments.json'
ADMIN_KEYS_FILE = DATA_DIR / 'admin_keys.txt'


class PaymentStatus(Enum):
    PENDING = "pending"      # 等待支付
    PAID = "paid"            # 已支付
    FAILED = "failed"        # 支付失败
    REFUNDED = "refunded"    # 已退款


class PaymentMethod(Enum):
    ALIPAY = "alipay"          # 支付宝扫码
    WECHAT = "wechat"          # 微信扫码
    BANK_TRANSFER = "bank"     # 银行转账
    API_CREDIT = "api"         # API充值（内部使用）


class Payment:
    """支付记录"""

    def __init__(self, amount: float, method: PaymentMethod,
                 reference: str = None, user_id: str = None,
                 payer_name: str = "", payer_account: str = ""):
        self.payment_id = f"pay_{int(time.time() * 1000)}"
        self.amount = amount
        self.method = method
        self.reference = reference or self.payment_id
        self.user_id = user_id
        self.payer_name = payer_name
        self.payer_account = payer_account
        self.status = PaymentStatus.PENDING
        self.qr_code = None       # 支付二维码URL或图片base64
        self.expires_at = datetime.now() + timedelta(hours=24)
        self.created_at = datetime.now().isoformat()
        self.paid_at = None
        self.notes = ""

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class PaymentGateway:
    """
    支付网关接口

    实际部署时需要对接真实支付渠道：
    - 支付宝开放平台 (alipay.com)
    - 微信支付商户号 (weixin.qq.com)
    - Stripe (国际用户)
    """

    def __init__(self):
        self.supported_methods = {
            PaymentMethod.ALIPAY: "https://openapi.alipay.com/gateway.do",
            PaymentMethod.WECHAT: "https://api.mch.weixin.qq.com/pay/unifiedorder",
            PaymentMethod.BANK_TRANSFER: "https://internal-api.circrna-cloud.example.com/transfer",
            PaymentMethod.API_CREDIT: "https://internal-api.circrna-cloud.example.com/credit",
        }

    def create_payment(self, amount: float, method: PaymentMethod,
                       user_id: str = None, **kwargs) -> Payment:
        """创建支付单"""
        payment = Payment(amount, method, user_id=user_id, **kwargs)
        self._save_payment(payment)
        return payment

    def generate_qr(self, payment: Payment) -> str:
        """
        生成支付二维码

        Returns:
            base64编码的图片数据（前端可直接渲染）
        """
        # ============================================================
        # 演示模式：生成模拟二维码
        # 生产环境替换为真实支付API调用
        # ============================================================

        if self._is_demo():
            return self._generate_demo_qr(payment)
        else:
            return self._call_real_payment_api(payment)

    def _is_demo(self) -> bool:
        """是否运行在演示模式"""
        return os.environ.get('CIRRNA_DEMO_MODE', 'true').lower() == 'true'

    def _generate_demo_qr(self, payment: Payment) -> str:
        """
        生成模拟二维码（用于演示）

        实际是SVG格式的二维码图片，包含支付金额和订单信息
        """
        # 构建二维码内容（模拟）
        order_info = {
            "app": "CircRNA Cloud",
            "out_trade_no": payment.reference,
            "total_amount": f"{payment.amount:.2f}",
            "currency": "USD",
            "body": "CircRNA算力租赁服务费用",
            "time": int(time.time()),
        }

        # 生成简单的QR码SVG（Base64）
        qr_svg = self._create_qr_svg(order_info)
        import base64
        return base64.b64encode(qr_svg.encode()).decode()

    def _create_qr_svg(self, data: dict) -> str:
        """生成二维码SVG"""
        payload = json.dumps(data, separators=(',', ':'))
        # 简单校验和
        checksum = hashlib.md5(payload.encode()).hexdigest()[:8]

        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">
  <rect width="300" height="300" fill="#fff"/>
  <text x="50%" y="40" text-anchor="middle" font-size="14" fill="#333">
    CircRNA Cloud
  </text>
  <text x="50%" y="90" text-anchor="middle" font-size="12" fill="#666">
    ${data["total_amount"]} USD
  </text>
  <text x="50%" y="130" text-anchor="middle" font-size="10" fill="#999">
    订单: {data["out_trade_no"]}
  </text>
  <text x="50%" y="170" text-anchor="middle" font-size="10" fill="#999">
    有效期: 24小时
  </text>
  <circle cx="150" cy="150" r="80" fill="#f0f0f0" stroke="#ccc" stroke-width="2"/>
  <g transform="translate(150,150) scale(0.4)">
    <!-- 简化的二维码图案 -->
    <rect x="0" y="0" width="20" height="20" fill="#333"/>
    <rect x="10" y="10" width="20" height="20" fill="#333"/>
    <rect x="20" y="0" width="20" height="20" fill="#333"/>
    <rect x="0" y="20" width="20" height="20" fill="#333"/>
    <rect x="10" y="30" width="20" height="20" fill="#333"/>
    <rect x="20" y="10" width="20" height="20" fill="#333"/>
    <rect x="30" y="20" width="20" height="20" fill="#333"/>
    <rect x="10" y="10" width="20" height="20" fill="#333"/>
    <rect x="20" y="30" width="20" height="20" fill="#333"/>
    <rect x="30" y="10" width="20" height="20" fill="#333"/>
    <rect x="0" y="30" width="20" height="20" fill="#333"/>
    <rect x="10" y="0" width="20" height="20" fill="#333"/>
    <rect x="20" y="20" width="20" height="20" fill="#333"/>
    <rect x="30" y="30" width="20" height="20" fill="#333"/>
    <rect x="0" y="10" width="20" height="20" fill="#333"/>
    <rect x="10" y="20" width="20" height="20" fill="#333"/>
    <rect x="20" y="0" width="20" height="20" fill="#333"/>
    <rect x="30" y="20" width="20" height="20" fill="#333"/>
    <rect x="0" y="20" width="20" height="20" fill="#333"/>
    <rect x="10" y="30" width="20" height="20" fill="#333"/>
    <rect x="20" y="10" width="20" height="20" fill="#333"/>
    <rect x="30" y="0" width="20" height="20" fill="#333"/>
    <rect x="0" y="10" width="20" height="20" fill="#333"/>
    <rect x="10" y="0" width="20" height="20" fill="#333"/>
    <rect x="20" y="30" width="20" height="20" fill="#333"/>
    <rect x="30" y="10" width="20" height="20" fill="#333"/>
    <rect x="0" y="30" width="20" height="20" fill="#333"/>
    <rect x="10" y="20" width="20" height="20" fill="#333"/>
    <rect x="20" y="10" width="20" height="20" fill="#333"/>
    <rect x="30" y="30" width="20" height="20" fill="#333"/>
  </g>
</svg>'''

    def _call_real_payment_api(self, payment: Payment) -> str:
        """
        调用真实支付API

        支付宝示例：
        POST https://openapi.alipay.com/gateway.do
        {
            "app_id": "...",
            "method": "alipay.trade.page.pay",
            "biz_content": {
                "out_trade_no": payment.reference,
                "total_amount": "0.01",
                "subject": "CircRNA算力租赁",
                "product_code": "FAST_INSTANT_TRADE_PAY"
            }
        }
        返回 qr_url 给前端展示
        """
        print("[PaymentGateway] 请配置真实的支付API密钥")
        return ""

    def verify_payment(self, payment: Payment, transaction_id: str) -> bool:
        """
        验证支付结果

        Args:
            payment: 支付单
            transaction_id: 第三方交易流水号

        Returns:
            True if verified successfully
        """
        if self._is_demo():
            # 演示模式：直接通过
            payment.status = PaymentStatus.PAID
            payment.paid_at = datetime.now()
            self._update_payment(payment)
            return True

        # 生产环境：调用支付平台查询接口
        # return self._query_payment(transaction_id)
        return False

    def refund(self, payment: Payment, reason: str = "") -> bool:
        """退款"""
        payment.status = PaymentStatus.REFUNDED
        payment.notes = f"退款原因: {reason}"
        self._update_payment(payment)
        return True

    def _save_payment(self, payment: Payment):
        payments = self._load_payments()
        payments[payment.payment_id] = {
            'payment_id': payment.payment_id,
            'amount': payment.amount,
            'method': payment.method.value,
            'reference': payment.reference,
            'user_id': payment.user_id,
            'payer_name': payment.payer_name,
            'payer_account': payment.payer_account,
            'status': payment.status.value,
            'qr_code': payment.qr_code,
            'expires_at': payment.expires_at.isoformat(),
            'created_at': payment.created_at,
            'paid_at': payment.paid_at.isoformat() if payment.paid_at else None,
            'notes': payment.notes,
        }
        Path(self._get_path()).write_text(json.dumps(payments, indent=2, default=str))

    def _load_payments(self) -> Dict:
        try:
            return json.loads(Path(self._get_path()).read_text())
        except:
            return {}

    def _update_payment(self, payment: Payment):
        payments = self._load_payments()
        key = payment.payment_id
        if key in payments:
            payments[key].update({
                'status': payment.status.value,
                'paid_at': payment.paid_at.isoformat() if payment.paid_at else None,
                'notes': payment.notes,
            })
            Path(self._get_path()).write_text(json.dumps(payments, indent=2, default=str))

    def _get_path(self) -> str:
        return str(PAYMENTS_FILE)


class AdminPanel:
    """
    管理后台 - 处理待审核支付
    """

    def __init__(self):
        self.admin_password = os.environ.get('ADMIN_PASSWORD', 'changeme')
        self._ensure_admin_keys()

    def _ensure_admin_keys(self):
        if not ADMIN_KEYS_FILE.exists():
            # 生成默认管理员密码
            token = hashlib.sha256(os.urandom(32)).hexdigest()[:32]
            ADMIN_KEYS_FILE.write_text(token + '\n')
            print(f"[AdminPanel] 管理员Token: {token}")

    def authenticate(self, password: str) -> bool:
        return password == self.admin_password

    def get_pending_payments(self) -> List[Dict]:
        """获取所有待审核支付"""
        payments = self._load_payments()
        return [p for p in payments.values() if p['status'] == PaymentStatus.PENDING]

    def approve_payment(self, payment_id: str) -> bool:
        """批准支付并给用户加余额"""
        payments = self._load_payments()
        if payment_id not in payments:
            return False

        p = payments[payment_id]

        # 更新状态
        p['status'] = PaymentStatus.PAID.value
        p['approved_at'] = datetime.now().isoformat()
        p['paid_by'] = 'admin'

        # 给用户加余额
        from backend.user_billing import UserManager
        user_mgr = UserManager()
        user_mgr.add_balance(p['user_id'], p['amount'])

        # 保存
        self._save_payments(payments)
        print(f"[AdminPanel] 已批准支付: {payment_id}, 金额: ${p['amount']}")
        return True

    def reject_payment(self, payment_id: str, reason: str):
        """拒绝支付"""
        payments = self._load_payments()
        if payment_id in payments:
            payments[payment_id]['status'] = PaymentStatus.FAILED.value
            payments[payment_id]['notes'] = f"拒绝: {reason}"
            self._save_payments(payments)

    def _load_payments(self) -> Dict:
        try:
            return json.loads(PAYMENTS_FILE.read_text())
        except:
            return {}

    def _save_payments(self, payments: Dict):
        PAYMENTS_FILE.write_text(json.dumps(payments, indent=2, default=str))


# 全局实例
payment_gateway = PaymentGateway()
admin_panel = AdminPanel()


if __name__ == '__main__':
    # 测试
    print("=== 支付系统测试 ===\n")

    # 创建支付
    payment = payment_gateway.create_payment(
        amount=10.0,
        method=PaymentMethod.ALIPAY,
        user_id='test_user',
        payer_name='Test Lab',
        payer_account='test@university.edu',
    )
    print(f"支付单: {payment.payment_id}")
    print(f"金额: ${payment.amount}")
    print(f"状态: {payment.status.value}")

    # 生成二维码
    qr = payment_gateway.generate_qr(payment)
    print(f"二维码长度: {len(qr)} chars")

    # 演示模式：模拟支付成功
    payment_gateway.verify_payment(payment, "mock_txn_12345")
    print(f"支付后状态: {payment.status.value}")