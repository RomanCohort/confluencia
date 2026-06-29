"""
验证码模块 - 图形验证码生成与验证
防止机器人批量注册/登录
"""

import random
import string
import io
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
CAPTCHA_FILE = DATA_DIR / 'captchas.json'


class CaptchaGenerator:
    """
    图形验证码生成器

    使用纯Python生成简单验证码图片
    无需额外依赖（PIL可选，无PIL时用SVG替代）
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CAPTCHA_FILE.exists():
            CAPTCHA_FILE.write_text('{}')

        # 验证码配置
        self.code_length = 4
        self.expire_minutes = 5

    def generate(self) -> Tuple[str, str]:
        """
        生成验证码

        Returns:
            (captcha_id, image_base64)
        """
        # 生成随机字符
        chars = string.ascii_uppercase + string.digits
        chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')  # 去除易混淆字符
        code = ''.join(random.choice(chars) for _ in range(self.code_length))

        # 生成captcha_id
        captcha_id = hashlib.sha256(f"{code}{time.time()}{random.random()}".encode()).hexdigest()[:16]

        # 存储验证码
        self._save_captcha(captcha_id, code)

        # 生成图片
        image_base64 = self._create_image(code)

        return captcha_id, image_base64

    def verify(self, captcha_id: str, user_input: str) -> bool:
        """
        验证用户输入

        Args:
            captcha_id: 验证码ID
            user_input: 用户输入的验证码

        Returns:
            True if valid
        """
        captchas = self._load_captchas()

        if captcha_id not in captchas:
            return False

        record = captchas[captcha_id]

        # 检查过期
        created = datetime.fromisoformat(record['created_at'])
        if datetime.now() > created + timedelta(minutes=self.expire_minutes):
            del captchas[captcha_id]
            self._write_captchas(captchas)
            return False

        # 验证码不区分大小写
        if record['code'].upper() == user_input.upper().strip():
            # 验证成功后删除，防止重复使用
            del captchas[captcha_id]
            self._write_captchas(captchas)
            return True

        return False

    def _create_image(self, code: str) -> str:
        """
        生成验证码图片

        尝试使用PIL生成PNG，失败则生成SVG
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            return self._create_pil_image(code)
        except ImportError:
            return self._create_svg_image(code)

    def _create_pil_image(self, code: str) -> str:
        """使用PIL生成PNG验证码"""
        from PIL import Image, ImageDraw, ImageFont
        import base64

        # 图片尺寸
        width, height = 150, 50

        # 创建图片
        img = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # 尝试使用系统字体
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()

        # 绘制干扰线
        for _ in range(5):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=(200, 200, 200), width=1)

        # 绘制验证码字符
        colors = [(50, 50, 50), (30, 80, 120), (120, 30, 80), (80, 120, 30)]
        x_offset = 20
        for i, char in enumerate(code):
            color = random.choice(colors)
            draw.text((x_offset + i * 30, 5), char, font=font, fill=color)

        # 绘制干扰点
        for _ in range(50):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        # 转Base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_base64}"

    def _create_svg_image(self, code: str) -> str:
        """生成SVG格式验证码"""
        import base64

        # 随机颜色
        colors = ['#333', '#1a5276', '#6c3483', '#145a32']

        svg_parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="150" height="50">
  <rect width="150" height="50" fill="#f0f0f0"/>''']

        # 干扰线
        for _ in range(5):
            x1 = random.randint(0, 150)
            y1 = random.randint(0, 50)
            x2 = random.randint(0, 150)
            y2 = random.randint(0, 50)
            svg_parts.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ccc" stroke-width="1"/>')

        # 字符
        for i, char in enumerate(code):
            x = 20 + i * 32
            y = 35 + random.randint(-3, 3)
            color = random.choice(colors)
            rotation = random.randint(-15, 15)
            svg_parts.append(f'''  <text x="{x}" y="{y}" font-size="32" font-weight="bold" fill="{color}" transform="rotate({rotation} {x} {y})">{char}</text>''')

        # 干扰点
        for _ in range(30):
            x = random.randint(0, 150)
            y = random.randint(0, 50)
            r = random.randint(1, 2)
            color = f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}"
            svg_parts.append(f'  <circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

        svg_parts.append('</svg>')
        svg = '\n'.join(svg_parts)

        return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

    def _save_captcha(self, captcha_id: str, code: str):
        captchas = self._load_captchas()
        captchas[captcha_id] = {
            'code': code,
            'created_at': datetime.now().isoformat(),
        }
        self._write_captchas(captchas)

    def _load_captchas(self) -> Dict:
        try:
            return eval(CAPTCHA_FILE.read_text())
        except:
            return {}

    def _write_captchas(self, captchas: Dict):
        CAPTCHA_FILE.write_text(str(captchas))


# 全局实例
captcha_generator = CaptchaGenerator()


if __name__ == '__main__':
    # 测试
    print("=== 验证码测试 ===\n")

    # 生成
    captcha_id, img = captcha_generator.generate()
    print(f"验证码ID: {captcha_id}")
    print(f"图片长度: {len(img)} chars")

    # 验证
    test_input = input("请输入验证码（查看console输出）: ")
    if captcha_generator.verify(captcha_id, test_input):
        print("验证成功！")
    else:
        print("验证失败")
