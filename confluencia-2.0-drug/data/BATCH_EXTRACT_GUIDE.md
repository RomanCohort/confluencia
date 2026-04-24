# 批量数据提取工具使用指南

## 工具概述

本工具包提供从已发表文献中批量提取药代动力学（PK）时间曲线数据并转换为 PopPK 拟合格式的完整工作流。

---

## 工具清单

| 文件 | 功能 |
|------|------|
| `real_pk_database.json` | 真实 PK 数据中央仓库 |
| `real_pk_loader.py` | 数据加载与格式转换 |
| `batch_extract.py` | 批量提取与验证 |
| `create_excel_template.py` | Excel 提取模板生成 |
| `extraction_template.csv` | CSV 提取模板 |
| `WEBPLOTDIGITIZER_GUIDE.md` | WebPlotDigitizer 操作指南 |

---

## 快速开始

### 方式 1: 命令行批量处理

```bash
# 1. 转换提取的 CSV 为 PopPK 格式
python core/batch_extract.py convert --input data/my_extracted.csv --output data/poppk_data.csv

# 2. 验证数据质量
python core/batch_extract.py validate --database data/real_pk_database.json

# 3. 合并到数据库
python core/batch_extract.py merge --input data/new_curves.csv --database data/real_pk_database.json

# 4. 导出 NONMEM 格式
python core/batch_extract.py export --database data/real_pk_database.json --output data/nonmem.csv

# 5. 完整流程（一步完成）
python core/batch_extract.py full --extracted data/extracted.csv --database data/real_pk_database.json
```

### 方式 2: Python API

```python
from real_pk_loader import RealPKLoader
from batch_extract import parse_extracted_csv, convert_to_poppk_format

# 加载现有数据
loader = RealPKLoader('data/real_pk_database.json')
dataset = loader.to_population_pk()
df = dataset.to_dataframe()

# 解析新提取的 CSV
curves = parse_extracted_csv('extracted_data.csv')

# 转换为 PopPK 格式
df_new = convert_to_poppk_format(curves)

# 保存
df_new.to_csv('poppk_data.csv', index=False)
```

---

## 完整工作流

### Step 1: 提取数据

1. 打开 https://automeris.io/webplotdigitizer
2. 加载论文图表截图
3. 校准坐标轴
4. 沿曲线点击数据点
5. 导出为 CSV

### Step 2: 填写模板

使用 `extraction_template.csv` 模板，或生成 Excel 模板：

```bash
pip install openpyxl
python core/create_excel_template.py
```

Excel 模板特点：
- 下拉列表选择（修饰、途径、组织）
- 数据验证
- 预填充参考值

### Step 3: 批量处理

```bash
# 完整流程
python core/batch_extract.py full --extracted data/my_data.csv --database data/real_pk_database.json
```

### Step 4: 验证

验证会检查：
- 半衰期是否与参考值一致（±30%）
- 数据点数量是否充足
- 曲线形状是否合理

```
半衰期验证:
  PASS none: extracted 6.09h vs ref 6.0h (error 1.5%)
  PASS m6A: extracted 10.43h vs ref 10.8h (error 3.4%)
  PASS psi: extracted 11.38h vs ref 15.0h (error 24.1%)
```

### Step 5: PopPK 拟合

```python
import pandas as pd
from pk_model_layer import RNACTMModel, PKParameters, PopPKFitter

# 加载数据
df = pd.read_csv('data/nonmem_export.csv')

# 拟合
model = RNACTMModel(extended=False)
initial_params = PKParameters(
    tv_ka=2.82, tv_ke=0.368, tv_v=6.89, tv_f=0.001,
    omega_ka=0.85, omega_ke=0.78, omega_v=0.25, omega_f=0.5,
    sigma_prop=0.10, sigma_add=0.05,
)

fitter = PopPKFitter(model)
result = fitter.fit(df, initial_params=initial_params)

print(f"R^2 = {result.r_squared:.4f}")
print(f"tv_ke = {result.final_params.tv_ke:.4f}")
```

---

## CSV 模板格式

```csv
curve_id,source_study,figure_reference,modification,route,dose_ug_kg,species,weight_kg,tissue,analyte,notes
time_h,concentration,cv_percent
0.0,100.0,
1.0,78.0,
2.0,62.0,
4.0,45.0,
...
,,,
curve_id,source_study,...
time_h,concentration,cv_percent
...
```

**注意**：曲线之间用空行分隔。

---

## 参考值（用于验证）

| 修饰 | 半衰期 (h) | CV% |
|------|-----------|-----|
| none | 6.0 | 25 |
| m6A | 10.8 | 22 |
| psi | 15.0 | 20 |
| 5mC | 12.5 | 22 |
| ms2m6A | 20.0 | 18 |

---

## 常见问题

### Q1: 如何处理多剂量数据？

在元数据行填写不同剂量即可：

```csv
curve_id,source_study,modification,route,dose_ug_kg,...

curve_001,liu_2023,psi,IV,50.0,...
time_h,concentration,cv_percent
0,100.0,
...

curve_002,liu_2023,psi,IV,100.0,...
time_h,concentration,cv_percent
0,100.0,
...
```

### Q2: 如何处理组织分布数据？

将 `tissue` 改为对应组织（如 liver, spleen）：

```csv
curve_id,source_study,modification,route,dose_ug_kg,tissue,...

curve_003,paunovska_2018,none,IV,1000.0,liver,...
time_h,percent_injected
1,45.0,
4,62.0,
...
```

### Q3: WebPlotDigitizer 提取的坐标不对？

检查坐标轴校准：
1. 点击 X 轴两个已知点
2. 输入实际值（如 0 和 72）
3. 点击 Y 轴两个已知点
4. 输入实际值（如 0 和 100）

### Q4: 半衰期误差超过 30%？

可能原因：
1. 数据点太少（需要 ≥5 个消除相点）
2. 早期时间点被误认为消除相
3. 浓度归一化有问题

解决：增加时间点覆盖完整的消除相。

---

## 输出格式

### NONMEM 格式 (`nonmem_export.csv`)

```csv
ID STUDY MODIFICATION ROUTE DOSE WT SEX SPECIES TISSUE ANALYTE NOTES TIME DV CV CENSORED
SUBJ_001 wesselhoeft_2018 none IV 50.0 0.025 M mouse plasma circRNA_concentration  0.0 100.0 NaN 0
SUBJ_001 wesselhoeft_2018 none IV 50.0 0.025 M mouse plasma circRNA_concentration  1.0 78.0 NaN 0
...
```

### JSON 格式 (`real_pk_database.json`)

```json
{
  "poppk_ready_format": {
    "subjects": [
      {
        "id": "SUBJ_001",
        "study": "wesselhoeft_2018",
        "modification": "none",
        "route": "IV",
        "dose": 50.0,
        "observations": [
          {"time": 0, "conc": 100.0, "cmdv": "plasma"},
          {"time": 1, "conc": 78.0, "cmdv": "plasma"}
        ]
      }
    ]
  }
}
```

---

## 下一步

1. 提取更多数据（高优先级图表见 WEBPLOTDIGITIZER_GUIDE.md）
2. 用真实数据拟合 PopPK 模型
3. 运行 VPC 验证
4. 生成 FDA/EMA 合规报告

---

## 参考文献

1. Wesselhoeft RA, et al. Nat Commun. 2018;9:2629.
2. Liu CX, et al. Nat Commun. 2023.
3. Chen YG, et al. Nature. 2019;586:651-655.
4. Gilleron J, et al. Nat Biotechnol. 2013;31:638-646.
5. Paunovska K, et al. ACS Nano. 2018;12:8307-8320.
6. Hassett KJ, et al. Mol Ther. 2019;27:1885-1897.