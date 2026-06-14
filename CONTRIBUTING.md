# Contributing to Confluencia

感谢你对Confluencia的贡献兴趣！

## 如何贡献

### 报告问题

1. 在 [Issues](https://github.com/RomanCohort/confluencia/issues) 页面创建新Issue
2. 使用Issue模板选择类型（Bug/Feature/Documentation）
3. 描述清楚问题、复现步骤、预期行为

### 提交代码

1. Fork仓库
2. 创建分支：`git checkout -b feature/your-feature`
3. 编写代码并添加测试
4. 运行测试确保通过：`pytest confluencia_circrna/tests/`
5. 提交PR并填写PR模板

### 代码规范

- 使用Python 3.8+语法
- 遵循PEP8风格（使用black格式化）
- 添加docstring说明
- 新功能需添加对应测试

### 开发环境

```bash
# 克隆仓库
git clone https://github.com/RomanCohort/confluencia.git

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest confluencia_circrna/tests/ -v
```

## 许可证

贡献的代码将采用MIT许可证。