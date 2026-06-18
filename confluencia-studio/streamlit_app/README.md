# Confluencia Studio - Streamlit Web UI

**简易模式** - 面向湿实验用户、医学生的图形化分析界面。

## 启动方式

```bash
# 方式1: 直接启动
streamlit run Home.py

# 方式2: 指定端口
streamlit run Home.py --server.port 8501

# 方式3: 后台运行
streamlit run Home.py &
```

## 页面导航

| 页面 | 功能 | 适用场景 |
|------|------|----------|
| 🏠 首页 | 模块选择 | 入口导航 |
| 🔬 circRNA分析 | 免疫原性评估 | circRNA疫苗设计 |
| 💊 药物预测 | ADMET属性 | 药物筛选 |
| 🧬 表位筛选 | MHC结合预测 | 疫苗表位设计 |
| 🎮 TNBC仿真 | 数字孪生 | 肿瘤动态模拟 |
| 📄 报告导出 | 生成报告 | 结果整理 |

## 功能特点

1. **零编程门槛** - 点击按钮即可分析
2. **示例序列** - 内置常见示例，快速上手
3. **术语解释** - 悬停显示专业术语说明
4. **可视化报告** - 自动生成HTML报告
5. **数据导出** - JSON格式下载数据

## 技术栈

- Streamlit 1.30+
- Plotly.js 图表
- Confluencia 3.0 后端