# WebPlotDigitizer 文献 PK 数据提取指南

## 目标

从 6 篇核心文献的补充材料/图表中提取 circRNA 药代动力学时间-浓度数据，填充 `real_pk_database.json`。

---

## 工具准备

1. **WebPlotDigitizer**: https://automeris.io/webplotdigitizer (免费在线工具)
2. **备用**: Engauge Digitizer (桌面版), PlotDigitizer
3. **浏览器**: Chrome/Firefox (推荐)

---

## 逐步操作流程

### Step 1: 获取论文 PDF

| 文献 | DOI | 需要提取的图 |
|------|-----|-------------|
| Wesselhoeft 2018 | 10.1038/s41467-018-05994-7 | Fig 2a (蛋白表达时间曲线), Fig 2b (修饰比较), ED Fig 2 |
| Liu 2023 | 10.1038/s41467-023-XXXXX | Fig 3 (剂量反应), Fig 4 (时间曲线), ED Fig 3 |
| Chen 2019 | 10.1038/s41586-019-1016-7 | Fig 2 (m6A稳定性), Fig 3 (免疫逃逸) |
| Gilleron 2013 | 10.1038/nbt.2688 | Fig 5 (内体逃逸时间曲线) |
| Paunovska 2018 | 10.1021/acsnano.8b05672 | Fig 2 (组织分布), Fig 3 (时间分布) |
| Hassett 2019 | 10.1016/j.ymthe.2019.08.010 | Fig 3 (血浆浓度), Fig 4 (组织分布) |

### Step 2: WebPlotDigitizer 操作

```
1. 打开 https://automeris.io/webplotdigitizer
2. 加载图片: 截取论文图表区域，粘贴或上传
3. 选择模式: "2D (X-Y) Plot"
4. 校准坐标轴:
   - 点击 X 轴上的两个已知点 (如 0h 和 24h)
   - 输入实际数值
   - 点击 Y 轴上的两个已知点 (如 0 和 100%)
   - 输入实际数值
5. 提取数据:
   - 自动模式: 选择曲线颜色，自动跟踪
   - 手动模式: 沿曲线逐点点击 (推荐用于精确提取)
6. 导出: CSV 格式
```

### Step 3: 数据格式转换

提取后的 CSV 需要转换为 `real_pk_database.json` 中的 `poppk_ready_format` 格式:

```json
{
  "id": "SUBJ_XXX",
  "study": "wesselhoeft_2018",
  "modification": "none",
  "route": "IV",
  "dose": 50.0,
  "weight_kg": 0.025,
  "sex": "M",
  "species": "mouse",
  "observations": [
    {"time": 0, "conc": 100.0, "cmdv": "plasma"},
    {"time": 1, "conc": 78.0, "cmdv": "plasma"},
    ...
  ]
}
```

### Step 4: 质量控制

每个提取的数据点需通过以下检查:

- [ ] 浓度值单调递减（血浆清除阶段）
- [ ] 半衰期计算值与文献报告值一致（±15%）
- [ ] AUC 值在合理范围内
- [ ] 时间点无重复
- [ ] 单位转换正确（ng/mL → μg/L → 归一化）

---

## 高优先级提取目标

### 1. Wesselhoeft 2018 - Fig 2a (最关键)

**描述**: circRNA vs linear mRNA 蛋白表达时间曲线

**预期数据形状**:
```
时间(h):  0   6   12  24  48  72  96  120
circRNA:  高  高  高  中  中  低  低  很低
linear:   高  中  低  很低 0   0   0   0
```

**关键参数验证**: circRNA 蛋白表达半衰期 ~116h (HeLa)

### 2. Hassett 2019 - Fig 3 (血浆 PK)

**描述**: LNP-mRNA 血浆浓度时间曲线

**预期数据形状**:
```
双指数衰减:
- 分布相 (0-1h): 快速下降
- 消除相 (1-24h): 缓慢下降
```

**关键参数验证**: 清除半衰期 ~2.8h

### 3. Liu 2023 - Fig 4 (Ψ 修饰蛋白表达)

**描述**: Ψ 修饰 circRNA 蛋白表达时间曲线

**预期数据形状**:
```
时间(h):  0   4   8   12  24  48  72  96  120  144
蛋白:     0   低  中  中  高  峰值 高  中  中   低
```

**关键参数验证**: 峰值在 48h, 表达持续 >120h

---

## 转换为 PopPK 拟合格式

```python
from real_pk_loader import RealPKLoader

# 加载已有数据
loader = RealPKLoader('data/real_pk_database.json')

# 添加新提取的数据 (修改 JSON 文件)
# 然后重新加载

dataset = loader.to_population_pk()
df = dataset.to_dataframe()

# 检查数据质量
print(dataset.summary())

# 保存为 NONMEM 兼容格式
df.to_csv('data/real_pk_for_poppk.csv', index=False)
```

---

## 下一步: PopPK 拟合

```python
from pk_model_layer import RNACTMModel, PKParameters, PopPKFitter

# 加载真实数据
df = pd.read_csv('data/real_pk_for_poppk.csv')

# 使用真实数据拟合
model = RNACTMModel(extended=False)
initial_params = PKParameters(
    tv_ka=2.82, tv_ke=0.368, tv_v=6.89, tv_f=0.001,
    omega_ka=0.85, omega_ke=0.78, omega_v=0.25, omega_f=0.5,
    sigma_prop=0.10, sigma_add=0.05,
)

fitter = PopPKFitter(model)
fit_result = fitter.fit(df, initial_params=initial_params)

# 比较拟合结果与文献值
print(f"ke = {fit_result.final_params.tv_ke:.4f} (文献: 0.368)")
print(f"V = {fit_result.final_params.tv_v:.4f} (文献: 6.89)")
print(f"R² = {fit_result.r_squared:.4f}")
```

---

## 注意事项

1. **单位一致性**: 所有时间用小时(h)，浓度用 ng/mL 或归一化值
2. **修饰标注**: 每条曲线必须标注对应的核苷酸修饰类型
3. **给药途径**: IV/IM/SC 对 PK 形状有重大影响，必须正确标注
4. **物种**: mouse/rat/human 参数差异大，必须标注
5. **数据来源**: 每条数据必须关联到原始论文 DOI 和图表编号

## 参考文献

1. Wesselhoeft RA, et al. Nat Commun. 2018;9:2629.
2. Liu CX, et al. Nat Commun. 2023.
3. Chen YG, et al. Nature. 2019;586:651-655.
4. Gilleron J, et al. Nat Biotechnol. 2013;31:638-646.
5. Paunovska K, et al. ACS Nano. 2018;12:8307-8320.
6. Hassett KJ, et al. Mol Ther. 2019;27:1885-1897.
