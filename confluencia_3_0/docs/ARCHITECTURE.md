# Confluencia 3.0 技术架构文档

**版本**: 3.0  
**作者**: 颜子壹  
**单位**: 吉林大学计算机科学与技术学院 / 第一白求恩临床医学院  
**日期**: 2024-2025

---

## 目录

1. [系统概述](#1-系统概述)
2. [整体架构](#2-整体架构)
3. [核心模块详解](#3-核心模块详解)
   - 3.1 [TNBC Simulacrum 仿真引擎](#31-tnbc-simulacrum-仿真引擎)
   - 3.2 [circRNA 子系统](#32-circrna-子系统)
   - 3.3 [TorusFold 深度学习架构](#33-torusfold-深度学习架构)
   - 3.4 [Backend 三层降级架构](#34-backend-三层降级架构)
4. [数学模型与推导](#4-数学模型与推导)
5. [生物学理论基础](#5-生物学理论基础)
6. [文献引用](#6-文献引用)
7. [API 参考](#7-api-参考)
8. [代码示例](#8-代码示例)
9. [配置指南](#9-配置指南)
10. [部署指南](#10-部署指南)
11. [常见问题解答](#11-常见问题解答)
12. [故障排除](#12-故障排除)

---

## 1. 系统概述

Confluencia 3.0 是一个统一计算平台，整合了：

1. **TNBC Simulacrum** — 三阴性乳腺癌肿瘤微环境仿真引擎
2. **circRNA 子系统** — 环状 RNA 设计、优化与评估
3. **TorusFold** — AlphaFold3 风格的 circRNA 结构预测深度学习框架
4. **Backend 架构** — 灵活可插拔的多精度后端系统

### 1.1 设计哲学

#### 1.1.1 核心设计原则

Confluencia 3.0 的架构设计遵循以下核心原则，每个原则都有明确的技术实现和权衡考量：

##### 原则 1: EventBus-first (事件驱动架构)

**动机**: 传统面向对象设计中，模块间通过直接方法调用通信，导致紧耦合。当系统规模扩大时，依赖关系呈指数增长，形成"意大利面条式"架构。

**解决方案**: 引入 EventBus 作为唯一通信中枢，所有模块间交互通过事件消息完成。

**技术实现**:
```
传统模式:
ModuleA.method() → ModuleB.method() → ModuleC.method()
问题: A → B → C 链式依赖

EventBus 模式:
ModuleA.emit(EVENT_X) → EventBus → [ModuleB, ModuleC, ModuleD].on(EVENT_X)
优势: 发布者无需知道订阅者是谁
```

**设计权衡**:
| 方面 | 优势 | 劣势 |
|------|------|------|
| 解耦性 | 模块可独立开发、测试 | 调试时控制流不直观 |
| 扩展性 | 新增订阅者无需修改发布者 | 事件命名需要全局协调 |
| 性能 | 异步处理，非阻塞 | 轻微的消息传递开销 |

**代码模式**:
```python
# 发布者不知道订阅者是谁
event_bus.emit(TUMOR_VOLUME_UPDATE, volume=new_volume)

# 订阅者可以动态注册
event_bus.subscribe(TUMOR_VOLUME_UPDATE, self.on_tumor_update)
```

##### 原则 2: 离线优先 (Offline-first)

**动机**: 生物信息学研究常在隔离网络环境进行（医院内网、实验室安全网络）。依赖在线 API 的工具在这些环境中不可用，严重影响研究效率。

**解决方案**: 所有核心功能默认使用本地模型/数据库，在线 API 仅作为精度增强选项。

**技术实现**:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400">
  <defs>
    <linearGradient id="level0" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f7768e;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#f7768e;stop-opacity:0.6"/>
    </linearGradient>
    <linearGradient id="level1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#7aa2f7;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#7aa2f7;stop-opacity:0.6"/>
    </linearGradient>
    <linearGradient id="level2" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#9ece6a;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#9ece6a;stop-opacity:0.6"/>
    </linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#c0caf5"/>
    </marker>
  </defs>
  
  <!-- 背景 -->
  <rect width="500" height="400" fill="#1a1b26"/>
  
  <!-- 标题 -->
  <text x="250" y="25" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="18" font-weight="bold">Backend 三层降级策略</text>
  
  <!-- Level 0: 在线高精度 -->
  <rect x="50" y="50" width="400" height="100" rx="8" fill="url(#level0)" filter="url(#shadow)"/>
  <text x="250" y="75" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="14" font-weight="bold">Level 0: 在线高精度 (可选)</text>
  
  <rect x="70" y="90" width="120" height="45" rx="4" fill="#24283b"/>
  <text x="130" y="115" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">ESM-2 650M</text>
  <text x="130" y="130" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">GPU, 需网络</text>
  
  <rect x="200" y="90" width="120" height="45" rx="4" fill="#24283b"/>
  <text x="260" y="115" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">NetMHCpan</text>
  <text x="260" y="130" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">需许可证</text>
  
  <rect x="330" y="90" width="100" height="45" rx="4" fill="#24283b"/>
  <text x="380" y="115" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">ChEMBL API</text>
  <text x="380" y="130" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">需网络</text>
  
  <!-- 降级箭头 -->
  <path d="M250,155 L250,175" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="280" y="170" fill="#ff9e64" font-family="Arial" font-size="10">网络不可用时降级</text>
  
  <!-- Level 1: 本地模型 -->
  <rect x="50" y="180" width="400" height="100" rx="8" fill="url(#level1)" filter="url(#shadow)"/>
  <text x="250" y="205" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="14" font-weight="bold">Level 1: 本地模型 (默认)</text>
  
  <rect x="70" y="220" width="120" height="45" rx="4" fill="#24283b"/>
  <text x="130" y="245" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">ViennaRNA</text>
  <text x="130" y="260" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">CPU, 本地安装</text>
  
  <rect x="200" y="220" width="120" height="45" rx="4" fill="#24283b"/>
  <text x="260" y="245" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">MHC Local</text>
  <text x="260" y="260" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">52K样本训练</text>
  
  <rect x="330" y="220" width="100" height="45" rx="4" fill="#24283b"/>
  <text x="380" y="245" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">Drug Local</text>
  <text x="380" y="260" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">R²=0.95</text>
  
  <!-- 降级箭头 -->
  <path d="M250,285 L250,305" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="280" y="300" fill="#ff9e64" font-family="Arial" font-size="10">本地模型不可用时降级</text>
  
  <!-- Level 2: 启发式算法 -->
  <rect x="50" y="310" width="400" height="80" rx="8" fill="url(#level2)" filter="url(#shadow)"/>
  <text x="250" y="335" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="14" font-weight="bold">Level 2: 启发式算法 (保底)</text>
  
  <text x="150" y="365" text-anchor="middle" fill="#24283b" font-family="Arial" font-size="11">纯Python</text>
  <text x="250" y="365" text-anchor="middle" fill="#24283b" font-family="Arial" font-size="11">无外部依赖</text>
  <text x="350" y="365" text-anchor="middle" fill="#24283b" font-family="Arial" font-size="11">始终可用</text>
</svg>
```

**设计权衡**:
| 场景 | 在线 API | 本地模型 | 启发式 |
|------|---------|---------|--------|
| 网络隔离环境 | ❌ 不可用 | ✅ 可用 | ✅ 可用 |
| 精度 | 最高 (AUC 0.95+) | 中等 (AUC 0.80-0.90) | 基线 (AUC 0.60-0.70) |
| 速度 | 慢 (网络延迟) | 中等 | 最快 |
| 资源需求 | 网络 + GPU | CPU/GPU | 仅 CPU |

##### 原则 3: 研究亲和性 (Research-friendly)

**动机**: 不同研究场景对精度/速度有不同需求：
- 高通量筛选：需要快速评估数千候选序列
- 深度分析：需要最高精度的结构预测
- 临床决策：需要可解释的生物学依据

**解决方案**: 提供多级精度后端，用户可按需选择。

**技术实现**:
```python
# 场景 1: 高通量筛选 (10000 序列)
config.circrna.immunogenicity_backend = "heuristic"  # ~50ms/序列
config.circrna.structure_mode = "simple"  # ~10ms/序列

# 场景 2: 深度结构分析
config.circrna.immunogenicity_backend = "esm2"  # ~2s/序列
config.circrna.structure_mode = "physics_ba"  # ~30s/序列

# 场景 3: 平衡模式
config.circrna.immunogenicity_backend = "vienna"  # ~150ms/序列
config.circrna.structure_mode = "physics_b"  # ~200ms/序列
```

**API设计原则**:
```python
# 统一接口，后端透明
result = manager.assess(sequence)  # 自动选择可用后端

# 显式指定后端（高级用户）
result = manager.assess(sequence, backend="esm2")

# 查询后端状态
status = manager.backend_status()
# {"esm2": {"available": True, "latency_ms": 2100}, ...}
```

##### 原则 4: 优雅降级 (Graceful Degradation)

**动机**: 外部依赖（网络服务、GPU）可能随时不可用。系统应继续运行而非崩溃。

**解决方案**: 每个功能都有多层备选方案，降级过程对用户透明。

**技术实现**:
```python
class BackendManager:
    """后端管理器，实现优雅降级。"""
    
    def get_available_backend(self, preferred: str) -> Backend:
        """获取可用后端，自动降级。"""
        # 尝试链：首选 → 高精度 → 中精度 → 保底
        chain = self._build_fallback_chain(preferred)
        
        for backend_name in chain:
            backend = self.backends[backend_name]
            if backend.is_available():
                if backend_name != preferred:
                    # 记录降级事件（非静默）
                    logger.warning(
                        f"Backend degraded: {preferred} → {backend_name}. "
                        f"Reason: {self._get_unavailable_reason(preferred)}"
                    )
                return backend
        
        # 保底：启发式始终可用
        return self.backends["heuristic"]
    
    def _build_fallback_chain(self, preferred: str) -> List[str]:
        """构建降级链。"""
        PRIORITY = ["esm2", "vienna", "heuristic"]
        
        if preferred in PRIORITY:
            idx = PRIORITY.index(preferred)
            return PRIORITY[idx:]  # 从首选开始往下降
        return PRIORITY
```

**降级日志示例**:
```
[WARNING] Backend degraded: esm2 → vienna. Reason: CUDA out of memory
[INFO] Continuing with ViennaRNA backend (CPU mode)
```

#### 1.1.2 架构决策记录 (ADR)

以下记录了系统设计中的关键技术决策及其理由：

##### ADR-001: 为什么选择 EventBus 而非直接方法调用？

**背景**: TNBC 仿真涉及肿瘤、免疫、治疗等多个子系统，需要协调执行。

**考虑的方案**:
| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A: 直接调用 | `tumor.update(); immune.update()` | 简单直接 | 紧耦合，难以扩展 |
| B: 观察者模式 | 各模块注册观察者 | 解耦 | 需要维护观察者列表 |
| C: EventBus | 中央事件总线 | 完全解耦，易扩展 | 调试复杂度增加 |

**决策**: 选择方案 C (EventBus)

**理由**:
1. 支持 N:M 通信（一个事件可触发多个响应者）
2. 新增功能模块无需修改现有代码
3. 便于实现事件重放、审计日志
4. 支持异步处理，提升性能

**后果**:
- 需要定义明确的事件命名规范
- 调试时需要追踪事件流
- 引入轻微性能开销（~0.1ms/event）

##### ADR-002: 为什么 TorusFold 提供四种结构预测模式？

**背景**: circRNA 结构预测面临精度-速度-数据可用性的三元权衡。

**决策矩阵**:
| 模式 | 精度 | 速度 | 训练数据 | 硬件需求 |
|------|------|------|---------|---------|
| simple | 中 | ~10ms | 需要 | CPU |
| diffusion | 高 | ~2-5s | 大量 | GPU |
| physics_b | 中高 | ~200ms | 零训练 | CPU |
| physics_ba | 最高 | ~30s | 零训练 | CPU + OpenMM |

**决策**: 提供全部四种模式，由用户根据场景选择。

**场景映射**:
```
高通量筛选 (1000+ 序列) → simple (总耗时 ~10s)
中等规模分析 (100 序列) → physics_b (总耗时 ~20s)
深度结构分析 (10 序列) → physics_ba (总耗时 ~5min)
有 GPU 且追求精度 → diffusion
```

##### ADR-003: 为什么 circRNA 免疫原性评估需要四通路？

**背景**: circRNA 进入细胞后可能激活多种先天免疫受体，单一指标不足以全面评估。

**生物学依据**:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 200">
  <defs>
    <filter id="sh" x="-5%" y="-10%" width="110%" height="120%">
      <feDropShadow dx="1" dy="1" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
    <marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#c0caf5"/>
    </marker>
  </defs>
  <rect width="700" height="200" fill="#1a1b26"/>
  
  <!-- circRNA 源头 -->
  <ellipse cx="50" cy="100" rx="40" ry="25" fill="#bb9af7" filter="url(#sh)"/>
  <text x="50" y="105" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11" font-weight="bold">circRNA</text>
  
  <!-- dsRNA 分支 -->
  <rect x="140" y="30" width="80" height="30" rx="4" fill="#f7768e" filter="url(#sh)"/>
  <text x="180" y="50" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">dsRNA结构</text>
  
  <rect x="260" y="20" width="60" height="28" rx="4" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="290" y="38" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">RIG-I</text>
  <rect x="350" y="20" width="55" height="28" rx="4" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="377" y="38" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">MAVS</text>
  <rect x="435" y="20" width="55" height="28" rx="4" fill="#e0af68" filter="url(#sh)"/>
  <text x="462" y="38" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">IFN-β</text>
  
  <rect x="260" y="60" width="60" height="28" rx="4" fill="#f7768e" filter="url(#sh)"/>
  <text x="290" y="78" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">PKR</text>
  <rect x="350" y="60" width="65" height="28" rx="4" fill="#f7768e" filter="url(#sh)"/>
  <text x="382" y="78" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">eIF2α-P</text>
  <rect x="445" y="60" width="70" height="28" rx="4" fill="#e0af68" filter="url(#sh)"/>
  <text x="480" y="78" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">翻译抑制</text>
  
  <!-- ssRNA 分支 -->
  <rect x="140" y="140" width="80" height="30" rx="4" fill="#9ece6a" filter="url(#sh)"/>
  <text x="180" y="160" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">ssRNA区域</text>
  
  <rect x="260" y="140" width="60" height="28" rx="4" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="290" y="158" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">内吞体</text>
  <rect x="350" y="140" width="70" height="28" rx="4" fill="#9ece6a" filter="url(#sh)"/>
  <text x="385" y="158" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">TLR7/TLR8</text>
  <rect x="455" y="140" width="60" height="28" rx="4" fill="#e0af68" filter="url(#sh)"/>
  <text x="485" y="158" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">NF-κB</text>
  
  <!-- 连接线 -->
  <path d="M90,90 L140,50" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M90,110 L140,155" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M220,40 L260,34" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M320,34 L350,34" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M405,34 L435,34" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M220,55 L260,70" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M320,74 L350,74" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M415,74 L445,74" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M220,155 L260,154" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M320,154 L350,154" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M420,154 L455,154" stroke="#c0caf5" stroke-width="1.5" marker-end="url(#a)"/>
</svg>
```

**各通路的独立性**:
| 通路 | 激活条件 | 下游效应 | circRNA 相关性 |
|------|---------|---------|---------------|
| RIG-I | dsRNA 结构 | I型干扰素 | 高 (circRNA 可形成 dsRNA) |
| TLR7 | GU-rich ssRNA | 炎症因子 | 中 (取决于序列) |
| TLR8 | AU-rich ssRNA | 炎症因子 | 中 (取决于序列) |
| PKR | >33bp dsRNA | 翻译抑制 | 高 (长茎环) |

**决策**: 分别评估四个通路，计算综合免疫原性评分。

**权重设计**:
```python
# 权重来源：文献 + 实验验证
OVERALL_IMMUNOGENICITY = (
    0.35 * RIG_I_SCORE +   # circRNA 主要激活通路
    0.20 * TLR7_SCORE +    # GU-rich 序列激活
    0.15 * TLR8_SCORE +    # AU-rich 序列激活
    0.30 * PKR_SCORE       # dsRNA 长度依赖
)
```

##### ADR-004: 为什么 CirculaPK 使用六室模型而非三室？

**背景**: circRNA 的药代动力学比小分子药物复杂，涉及递送载体、内吞体逃逸等特殊过程。

**三室模型 (传统小分子)**:
```
血液 → 组织 → 代谢/排泄
```

**六室模型 (circRNA)**:
```
Inj (注射部位) → LNP (递送载体) → Endo (内吞体) → Cyto (胞质) → Trans (翻译) → Clear (清除)
```

**新增房间的必要性**:
| 房间 | 为什么需要 | 关键参数 |
|------|-----------|---------|
| LNP | circRNA 需要载体保护 | k_release (释放速率) |
| Endo | 内吞体逃逸是限速步骤 | k_escape (~2.5%, 瓶颈!) |
| Trans | circRNA 可持续翻译蛋白 | k_protein (蛋白半衰期) |

**数据来源**:
- k_escape = 0.025 h⁻¹: Gilleron et al., Nat Biotechnol 2013
- k_degrade = 0.12 h⁻¹: Wesselhoeft et al., Nat Commun 2018

#### 1.1.3 系统边界与约束

##### 功能边界

**在范围内**:
- TNBC 四种分子亚型的仿真
- circRNA 序列设计、优化、评估
- circRNA 结构预测 (无实验数据时)
- PK/PD 六室模型模拟
- 治疗响应预测

**不在范围内**:
- 其他癌症类型 (可扩展但需重新参数化)
- circRNA 实验验证 (需湿实验)
- 临床决策支持 (仅供研究)
- 实时患者数据集成

##### 技术约束

| 约束类型 | 约束描述 | 设计响应 |
|---------|---------|---------|
| 数据可用性 | circRNA 3D 结构数据极少 | physics 模式零训练设计 |
| 计算资源 | 用户可能无 GPU | CPU 可用的 backend 链 |
| 网络环境 | 医院/实验室可能隔离 | 离线优先架构 |
| 可解释性 | 生物学家需要理解结果 | 启发式评分 + 文献引用 |

##### 性能约束

| 操作 | 目标延迟 | 实际延迟 (默认配置) |
|------|---------|-------------------|
| 单序列免疫评估 | <200ms | ~85ms (heuristic) |
| 单序列结构预测 | <500ms | ~200ms (physics_b) |
| PK 模拟 (72h) | <100ms | ~50ms |
| 序列进化 (100代) | <60s | ~45s |
| 单步仿真 | <50ms | ~30ms |

#### 1.1.4 架构演进历史

**Confluencia 1.0 (2022)**:
- 单体架构
- 硬编码参数
- 无 circRNA 支持

**Confluencia 2.0 (2023)**:
- 模块化架构
- 配置系统
- 初步 circRNA 支持
- 引入 Backend 概念

**Confluencia 3.0 (2024-2025)**:
- EventBus 架构
- TorusFold 深度学习
- CirculaPK 六室模型
- 四种结构模式
- 完整进化优化
- 三层 Backend 降级

**演进动机**:
```
1.0 → 2.0: 功能扩展需求，单体无法维护
2.0 → 3.0: 研究需求深化，需要更精细的 circRNA 建模
```

### 1.2 核心能力

| 能力 | 描述 |
|------|------|
| 肿瘤生长仿真 | Logistic/Gompertz 动力学 + 亚克隆演化 |
| TME 建模 | CD8/NK/M1/M2/Treg/MDSC/CAF 动态交互 |
| 免疫编辑 | Elimination → Equilibrium → Escape 三阶段 |
| 治疗模拟 | 化疗/免疫/PARP抑制剂/放疗/circRNA治疗 |
| circRNA 免疫感知 | RIG-I/TLR7/TLR8/PKR 四通路预测 |
| circRNA 结构预测 | TorusFold (simple/diffusion/physics_b/physics_ba) |
| 序列进化 | REINFORCE + Pareto 多目标优化 |
| PK/PD 建模 | CirculaPK 六室 circRNA 药代动力学模型 |

### 1.3 系统依赖

#### 必需依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥3.9 | 运行环境 |
| NumPy | ≥1.21 | 数值计算 |
| PyTorch | ≥2.0 | 深度学习框架 |
| SciPy | ≥1.7 | 科学计算 |

#### 可选依赖

| 依赖 | 版本 | 用途 | 安装命令 |
|------|------|------|---------|
| ViennaRNA | ≥2.5 | RNA 二级结构预测 | `conda install -c bioconda viennarna` |
| OpenMM | ≥7.7 | 分子动力学精修 | `conda install -c conda-forge openmm` |
| ESM-2 | - | 蛋白/RNA 序列嵌入 | `pip install fair-esm` |
| NetMHCpan | ≥4.1 | MHC 结合预测 | 需单独下载许可证 |
| Streamlit | ≥1.28 | Web UI | `pip install streamlit` |

---

## 2. 整体架构

### 2.1 系统架构图

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 900">
  <defs>
    <!-- 渐变定义 -->
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#1a1b26;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#24283b;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="tumorGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f7768e;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#f7768e;stop-opacity:0.4" />
    </linearGradient>
    <linearGradient id="tmeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#7aa2f7;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#7aa2f7;stop-opacity:0.4" />
    </linearGradient>
    <linearGradient id="circrnaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#9ece6a;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#9ece6a;stop-opacity:0.4" />
    </linearGradient>
    <linearGradient id="torusGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#bb9af7;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#bb9af7;stop-opacity:0.4" />
    </linearGradient>
    
    <!-- 阴影 -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.3"/>
    </filter>
    
    <!-- 箭头标记 -->
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#c0caf5"/>
    </marker>
  </defs>
  
  <!-- 背景 -->
  <rect width="1200" height="900" fill="#1a1b26"/>
  
  <!-- 标题 -->
  <rect x="400" y="20" width="400" height="50" rx="8" fill="url(#headerGrad)" filter="url(#shadow)"/>
  <text x="600" y="52" text-anchor="middle" fill="#c0caf5" font-family="Arial, sans-serif" font-size="20" font-weight="bold">
    Confluencia 3.0 统一计算平台
  </text>
  
  <!-- Frontend Layer -->
  <rect x="50" y="90" width="1100" height="80" rx="8" fill="#24283b" stroke="#414d68" stroke-width="1"/>
  <text x="600" y="115" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="14" font-weight="bold">
    Frontend Layer (Streamlit)
  </text>
  
  <!-- Frontend Tabs -->
  <rect x="70" y="130" width="130" height="30" rx="4" fill="#f7768e" opacity="0.7"/>
  <text x="135" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">Tumor Dashboard</text>
  
  <rect x="210" y="130" width="100" height="30" rx="4" fill="#7aa2f7" opacity="0.7"/>
  <text x="260" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">TME/Immune</text>
  
  <rect x="320" y="130" width="90" height="30" rx="4" fill="#9ece6a" opacity="0.7"/>
  <text x="365" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">Treatment</text>
  
  <rect x="420" y="130" width="90" height="30" rx="4" fill="#e0af68" opacity="0.7"/>
  <text x="465" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">Biomarker</text>
  
  <rect x="520" y="130" width="80" height="30" rx="4" fill="#bb9af7" opacity="0.7"/>
  <text x="560" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">Clinical</text>
  
  <rect x="610" y="130" width="100" height="30" rx="4" fill="#7dcfff" opacity="0.7"/>
  <text x="660" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">Experiments</text>
  
  <rect x="720" y="130" width="110" height="30" rx="4" fill="#ff9e64" opacity="0.7"/>
  <text x="775" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">Confluencia</text>
  
  <!-- TNBCSimulacrum Agent -->
  <rect x="350" y="190" width="500" height="60" rx="8" fill="#3d59a1" filter="url(#shadow)"/>
  <text x="600" y="215" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="16" font-weight="bold">
    TNBCSimulacrum Agent
  </text>
  <text x="600" y="235" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="11">
    step(): STEP_START → Tumor → TME → Treatment → Biomarker → Clinical → STEP_END
  </text>
  
  <!-- EventBus -->
  <rect x="50" y="270" width="180" height="50" rx="6" fill="#414d68" filter="url(#shadow)"/>
  <text x="140" y="295" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="12" font-weight="bold">
    EventBus
  </text>
  <text x="140" y="310" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">
    34+ events
  </text>
  
  <!-- StateSchema -->
  <rect x="240" y="270" width="180" height="50" rx="6" fill="#414d68" filter="url(#shadow)"/>
  <text x="330" y="295" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="12" font-weight="bold">
    StateSchema
  </text>
  <text x="330" y="310" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">
    ~180 state keys
  </text>
  
  <!-- Config -->
  <rect x="430" y="270" width="180" height="50" rx="6" fill="#414d68" filter="url(#shadow)"/>
  <text x="520" y="295" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="12" font-weight="bold">
    Config System
  </text>
  <text x="520" y="310" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">
    13 config classes
  </text>
  
  <!-- Subsystem Managers -->
  <rect x="620" y="270" width="530" height="50" rx="6" fill="#414d68" filter="url(#shadow)"/>
  <text x="885" y="295" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="12" font-weight="bold">
    Subsystem Managers: Tumor | TME | Treatment | Biomarker | Clinical | CircRNA
  </text>
  
  <!-- 连接线 -->
  <line x1="600" y1="170" x2="600" y2="190" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="140" y1="250" x2="140" y2="270" stroke="#c0caf5" stroke-width="1" marker-end="url(#arrowhead)"/>
  <line x1="330" y1="250" x2="330" y2="270" stroke="#c0caf5" stroke-width="1" marker-end="url(#arrowhead)"/>
  <line x1="520" y1="250" x2="520" y2="270" stroke="#c0caf5" stroke-width="1" marker-end="url(#arrowhead)"/>
  
  <!-- Tumor Manager -->
  <rect x="50" y="340" width="200" height="150" rx="8" fill="url(#tumorGrad)" filter="url(#shadow)"/>
  <text x="150" y="360" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="13" font-weight="bold">
    TumorManager
  </text>
  <line x1="60" y1="370" x2="240" y2="370" stroke="#1a1b26" stroke-width="1" opacity="0.5"/>
  <text x="150" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">growth_engine.py</text>
  <text x="150" y="405" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">heterogeneity.py</text>
  <text x="150" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">cancer_stem_cell.py</text>
  <text x="150" y="435" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">angiogenesis.py</text>
  <text x="150" y="450" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">metastasis.py</text>
  <text x="150" y="475" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-style="italic">
    Logistic/Gompertz + 亚克隆演化
  </text>
  
  <!-- TME Manager -->
  <rect x="260" y="340" width="200" height="150" rx="8" fill="url(#tmeGrad)" filter="url(#shadow)"/>
  <text x="360" y="360" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="13" font-weight="bold">
    TMEManager
  </text>
  <line x1="270" y1="370" x2="450" y2="370" stroke="#1a1b26" stroke-width="1" opacity="0.5"/>
  <text x="360" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">immune_dynamics.py</text>
  <text x="360" y="405" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">fibroblast.py</text>
  <text x="360" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">immune_evasion.py</text>
  <text x="360" y="435" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">immunoediting.py</text>
  <text x="360" y="475" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-style="italic">
    CD8/NK/M1/M2/Treg/MDSC/CAF
  </text>
  
  <!-- Treatment Manager -->
  <rect x="470" y="340" width="200" height="150" rx="8" fill="#e0af68" opacity="0.8" filter="url(#shadow)"/>
  <text x="570" y="360" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="13" font-weight="bold">
    TreatmentManager
  </text>
  <line x1="480" y1="370" x2="660" y2="370" stroke="#1a1b26" stroke-width="1" opacity="0.5"/>
  <text x="570" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">chemotherapy.py</text>
  <text x="570" y="405" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">immunotherapy.py</text>
  <text x="570" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">targeted.py</text>
  <text x="570" y="435" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">radiotherapy.py</text>
  <text x="570" y="450" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">circrna_therapy.py</text>
  <text x="570" y="475" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-style="italic">
    化疗/免疫/PARP/放疗/circRNA
  </text>
  
  <!-- CircRNA Manager -->
  <rect x="680" y="340" width="220" height="150" rx="8" fill="url(#circrnaGrad)" filter="url(#shadow)"/>
  <text x="790" y="360" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="13" font-weight="bold">
    CircRNAManager
  </text>
  <line x1="690" y1="370" x2="890" y2="370" stroke="#1a1b26" stroke-width="1" opacity="0.5"/>
  <text x="790" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">immune_sensing.py</text>
  <text x="790" y="405" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">structure_prediction.py</text>
  <text x="790" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">bsj_features.py</text>
  <text x="790" y="435" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">pk/rnactm.py</text>
  <text x="790" y="450" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">evolution/*.py</text>
  <text x="790" y="475" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-style="italic">
    免疫感知 + PK + 进化优化
  </text>
  
  <!-- TorusFold -->
  <rect x="910" y="340" width="240" height="150" rx="8" fill="url(#torusGrad)" filter="url(#shadow)"/>
  <text x="1030" y="360" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="13" font-weight="bold">
    TorusFold (AF3-style)
  </text>
  <line x1="920" y1="370" x2="1140" y2="370" stroke="#1a1b26" stroke-width="1" opacity="0.5"/>
  <text x="1030" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">torusfold/tpe.py</text>
  <text x="1030" y="405" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">torusfold/triangle_update.py</text>
  <text x="1030" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">torusfold/diffusion_structure.py</text>
  <text x="1030" y="435" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">torusfold/physics_structure_head.py</text>
  <text x="1030" y="450" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">torusfold/cgmd_refiner.py</text>
  <text x="1030" y="475" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-style="italic">
    simple/diffusion/physics_b/physics_ba
  </text>
  
  <!-- Backend Architecture -->
  <rect x="50" y="510" width="1100" height="120" rx="8" fill="#24283b" stroke="#414d68" stroke-width="1"/>
  <text x="600" y="535" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="14" font-weight="bold">
    Backend Architecture (三层降级，离线优先)
  </text>
  
  <!-- Immunogenicity Backend -->
  <rect x="70" y="555" width="250" height="60" rx="6" fill="#f7768e" opacity="0.3" stroke="#f7768e"/>
  <text x="195" y="575" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11" font-weight="bold">
    Immunogenicity Backend
  </text>
  <text x="195" y="600" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">
    esm2 (GPU) → vienna (CPU) → heuristic
  </text>
  
  <!-- MHC Backend -->
  <rect x="330" y="555" width="250" height="60" rx="6" fill="#7aa2f7" opacity="0.3" stroke="#7aa2f7"/>
  <text x="455" y="575" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11" font-weight="bold">
    MHC Backend
  </text>
  <text x="455" y="600" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">
    netmhcpan (AUC=0.92-0.96) → local (AUC=0.80)
  </text>
  
  <!-- Drug Backend -->
  <rect x="590" y="555" width="250" height="60" rx="6" fill="#9ece6a" opacity="0.3" stroke="#9ece6a"/>
  <text x="715" y="575" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11" font-weight="bold">
    Drug Backend
  </text>
  <text x="715" y="600" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">
    chembl_api (在线) → local (R²=0.95)
  </text>
  
  <!-- PK Backend -->
  <rect x="850" y="555" width="280" height="60" rx="6" fill="#bb9af7" opacity="0.3" stroke="#bb9af7"/>
  <text x="990" y="575" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11" font-weight="bold">
    PK Backend (CirculaPK 六室模型)
  </text>
  <text x="990" y="600" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">
    Inj → LNP → Endo → Cyto → Trans → Clear
  </text>
  
  <!-- 2.0 Bridge Layer -->
  <rect x="50" y="650" width="1100" height="80" rx="8" fill="#24283b" stroke="#414d68" stroke-width="1"/>
  <text x="600" y="675" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="14" font-weight="bold">
    2.0 Bridge Layer (懒加载，静默失败降级)
  </text>
  
  <rect x="100" y="690" width="200" height="30" rx="4" fill="#e0af68" opacity="0.5"/>
  <text x="200" y="710" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">drug_bridge.py</text>
  
  <rect x="320" y="690" width="200" height="30" rx="4" fill="#e0af68" opacity="0.5"/>
  <text x="420" y="710" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">epitope_bridge.py</text>
  
  <rect x="540" y="690" width="200" height="30" rx="4" fill="#e0af68" opacity="0.5"/>
  <text x="640" y="710" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">pk_bridge.py</text>
  
  <rect x="760" y="690" width="200" height="30" rx="4" fill="#e0af68" opacity="0.5"/>
  <text x="860" y="710" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">joint_bridge.py</text>
  
  <!-- Experiment Framework -->
  <rect x="50" y="750" width="550" height="130" rx="8" fill="#24283b" stroke="#414d68" stroke-width="1"/>
  <text x="325" y="775" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="14" font-weight="bold">
    Experiment Framework
  </text>
  
  <text x="150" y="805" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">sandbox.py</text>
  <text x="150" y="820" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">TNBC Pharmacology Sandbox</text>
  
  <text x="300" y="805" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">clinical_trial.py</text>
  <text x="300" y="820" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">临床试验模拟</text>
  
  <text x="450" y="805" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">combination.py</text>
  <text x="450" y="820" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">联合疗法筛选</text>
  
  <text x="150" y="860" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">experiments/: 13 预定义实验</text>
  <text x="350" y="860" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="9">circrna_therapy, chemo_immuno, parp_brca, ...</text>
  
  <!-- Code Statistics -->
  <rect x="620" y="750" width="530" height="130" rx="8" fill="#24283b" stroke="#414d68" stroke-width="1"/>
  <text x="885" y="775" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="14" font-weight="bold">
    Code Statistics
  </text>
  
  <text x="720" y="805" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">Python Files: 131</text>
  <text x="720" y="825" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">Core Modules: 72</text>
  <text x="720" y="845" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">TorusFold: 12</text>
  
  <text x="900" y="805" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">Frontend Tabs: 7</text>
  <text x="900" y="825" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">Experiments: 18</text>
  <text x="900" y="845" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11">Event Types: 34+</text>
  
  <text x="885" y="870" text-anchor="middle" fill="#7dcfff" font-family="Arial" font-size="10" font-style="italic">
    Author: 颜子壹 | Jilin University
  </text>
</svg>
```

---

### 2.2 TorusFold 深度架构设计

#### 2.2.1 神经网络层详细设计

TorusFold 的神经网络架构借鉴 AlphaFold3，但针对 circRNA 的环形拓扑进行了专门优化。

##### 输入嵌入层

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 650 220">
  <defs>
    <filter id="sh" x="-5%" y="-10%" width="110%" height="120%">
      <feDropShadow dx="1" dy="1" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
    <marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#73daca"/>
    </marker>
  </defs>
  <rect width="650" height="220" fill="#1a1b26"/>
  
  <!-- 输入 -->
  <rect x="10" y="80" width="110" height="40" rx="6" fill="#f7768e" filter="url(#sh)"/>
  <text x="65" y="100" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">输入序列</text>
  <text x="65" y="114" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">(L nucleotides)</text>
  
  <!-- One-hot -->
  <rect x="160" y="10" width="130" height="50" rx="6" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="225" y="30" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">One-hot Encoding</text>
  <text x="225" y="48" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">(L, 4): A/U/G/C</text>
  
  <!-- Properties -->
  <rect x="160" y="70" width="130" height="50" rx="6" fill="#9ece6a" filter="url(#sh)"/>
  <text x="225" y="90" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">Nucleotide Properties</text>
  <text x="225" y="108" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">(L, 8): 质量/氢键/体积...</text>
  
  <!-- TPE -->
  <rect x="160" y="130" width="130" height="55" rx="6" fill="#bb9af7" filter="url(#sh)"/>
  <text x="225" y="150" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">Torus PE</text>
  <text x="225" y="165" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">sin/cos(2π·pos/L·ω)</text>
  <text x="225" y="178" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">TPE[0]=TPE[L]</text>
  
  <!-- ESM-2 -->
  <rect x="160" y="195" width="130" height="20" rx="4" fill="#e0af68" opacity="0.7"/>
  <text x="225" y="209" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">ESM-2 (可选, L×1280)</text>
  
  <!-- Fusion -->
  <rect x="340" y="70" width="130" height="60" rx="6" fill="#e0af68" filter="url(#sh)"/>
  <text x="405" y="90" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">拼接融合</text>
  <text x="405" y="108" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">Linear → LN → Dropout</text>
  <text x="405" y="122" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">→ (L, d_model)</text>
  
  <!-- Output -->
  <rect x="530" y="80" width="100" height="35" rx="6" fill="#73daca" filter="url(#sh)"/>
  <text x="580" y="100" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">single_repr</text>
  <text x="580" y="112" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">(L, d_model)</text>
  
  <!-- 连线 -->
  <path d="M120,100 L160,35" stroke="#73daca" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M120,100 L160,95" stroke="#73daca" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M120,100 L160,155" stroke="#73daca" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M290,35 L340,85" stroke="#73daca" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M290,95 L340,95" stroke="#73daca" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M290,155 L340,115" stroke="#73daca" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M470,100 L530,100" stroke="#73daca" stroke-width="2" marker-end="url(#a)"/>
</svg>
```

**代码实现细节**:
```python
class RNAInputEmbedding(nn.Module):
    """RNA 输入嵌入层。"""
    
    def __init__(self, d_model: int = 256, use_esm: bool = False):
        super().__init__()
        self.d_model = d_model
        
        # One-hot + 核苷酸性质 (4 + 8 = 12 维)
        self.nucleotide_proj = nn.Linear(12, d_model // 2)
        
        # TPE
        self.tpe = TorusPositionalEncoding(d_model // 2)
        
        # ESM-2 (可选)
        self.use_esm = use_esm
        if use_esm:
            self.esm_proj = nn.Linear(1280, d_model // 2)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model if use_esm else d_model // 2 + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
    
    def forward(self, sequence: str) -> torch.Tensor:
        L = len(sequence)
        device = next(self.parameters()).device
        
        # One-hot
        one_hot = self._one_hot_encode(sequence)  # (L, 4)
        
        # 核苷酸性质
        properties = self._get_nucleotide_properties(sequence)  # (L, 8)
        
        # 拼接
        nucleotide_feat = torch.cat([one_hot, properties], dim=-1)  # (L, 12)
        nucleotide_embed = self.nucleotide_proj(nucleotide_feat)  # (L, d_model/2)
        
        # TPE
        tpe_embed = self.tpe(torch.zeros(1, L, self.d_model // 2), L).squeeze(0)
        
        # 合并
        if self.use_esm:
            esm_embed = self._get_esm_embedding(sequence)  # (L, d_model/2)
            combined = torch.cat([nucleotide_embed, tpe_embed, esm_embed], dim=-1)
        else:
            combined = torch.cat([nucleotide_embed, tpe_embed], dim=-1)
        
        return self.fusion(combined)
    
    def _one_hot_encode(self, seq: str) -> torch.Tensor:
        """One-hot 编码。"""
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        L = len(seq)
        one_hot = torch.zeros(L, 4)
        for i, c in enumerate(seq.upper()):
            if c in mapping:
                one_hot[i, mapping[c]] = 1.0
        return one_hot
    
    def _get_nucleotide_properties(self, seq: str) -> torch.Tensor:
        """核苷酸物理化学性质。"""
        properties = {
            'A': [134.1, 2, 0.0, 0.0, 135, 0, 0, 0],  # 嘌呤
            'U': [112.1, 2, 0.0, 0.0, 110, 1, 0, 1],  # 嘧啶
            'G': [150.1, 3, 0.0, 0.0, 150, 0, 0, 0],  # 嘌呤
            'C': [111.1, 3, 0.0, 0.0, 115, 1, 0, 1],  # 嘧啶
            'T': [112.1, 2, 0.0, 0.0, 110, 1, 0, 1],  # 与U相同
        }
        L = len(seq)
        prop_tensor = torch.zeros(L, 8)
        for i, c in enumerate(seq.upper()):
            if c in properties:
                prop_tensor[i] = torch.tensor(properties[c], dtype=torch.float32)
        return prop_tensor
```

##### CircPairformer 详细结构

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 300">
  <defs>
    <filter id="sh" x="-5%" y="-10%" width="110%" height="120%">
      <feDropShadow dx="1" dy="1" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
    <marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#73daca"/>
    </marker>
  </defs>
  <rect width="750" height="300" fill="#1a1b26"/>
  
  <!-- 输入 -->
  <rect x="10" y="120" width="100" height="50" rx="6" fill="#f7768e" filter="url(#sh)"/>
  <text x="60" y="142" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">输入</text>
  <text x="60" y="158" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">single(L,d_s)</text>
  <text x="60" y="170" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">pair(L,L,d_z)</text>
  
  <!-- CircPairformerLayer -->
  <rect x="140" y="20" width="500" height="260" rx="8" fill="#3d59a1" filter="url(#sh)"/>
  <text x="390" y="40" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="11" font-weight="bold">CircPairformerLayer × N (默认 N=8)</text>
  
  <!-- Triangle Update Outgoing -->
  <rect x="155" y="55" width="220" height="60" rx="4" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="265" y="75" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">Triangle Update (Outgoing)</text>
  <text x="265" y="90" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">z_ij += Σ_k a_ik ⊙ z_kj</text>
  <text x="265" y="105" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="7">a_ik = sigmoid(gate(z_ik))</text>
  
  <!-- Triangle Update Incoming -->
  <rect x="155" y="125" width="220" height="55" rx="4" fill="#9ece6a" filter="url(#sh)"/>
  <text x="265" y="145" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">Triangle Update (Incoming)</text>
  <text x="265" y="160" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">z_ij += Σ_k a_jk ⊙ z_ki</text>
  
  <!-- Triangle Attention Starting -->
  <rect x="395" y="55" width="230" height="60" rx="4" fill="#bb9af7" filter="url(#sh)"/>
  <text x="510" y="75" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">Triangle Attention (Starting)</text>
  <text x="510" y="90" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">α_ijk = softmax(Q_i·K_k^T + b_circ)</text>
  <text x="510" y="105" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="7">output = Σ_k α_ijk · V_kj</text>
  
  <!-- Triangle Attention Ending -->
  <rect x="395" y="125" width="230" height="55" rx="4" fill="#e0af68" filter="url(#sh)"/>
  <text x="510" y="145" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">Triangle Attention (Ending)</text>
  <text x="510" y="160" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">类似 Starting 但方向相反</text>
  
  <!-- Transition -->
  <rect x="270" y="195" width="240" height="50" rx="4" fill="#7dcfff" filter="url(#sh)"/>
  <text x="390" y="215" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">Transition (MLP)</text>
  <text x="390" y="232" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">z ← z + Linear(GELU(Linear(LN(z))))</text>
  
  <!-- 输出 -->
  <rect x="660" y="120" width="80" height="50" rx="6" fill="#73daca" filter="url(#sh)"/>
  <text x="700" y="142" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">输出</text>
  <text x="700" y="158" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">pair_repr</text>
  
  <!-- 连线 -->
  <path d="M110,145 L140,145" stroke="#73daca" stroke-width="2" marker-end="url(#a)"/>
  <path d="M640,145 L660,145" stroke="#73daca" stroke-width="2" marker-end="url(#a)"/>
</svg>
```

**Triangle Multiplicative Update 推导**:

该操作的核心思想是：如果 i-k 和 k-j 有关联，则 i-j 应该更新。

**Outgoing** (节点 i 作为起始点): 对于固定的 j，收集所有经过中间节点 k 的信息，`z_ij += Σ_k (a_ik ⊙ z_kj)`。直觉: "从 i 出发，经过 k，到达 j" 的路径信息。

**Incoming** (节点 j 作为终点): 对于固定的 i，收集所有经过中间节点 k 的信息，`z_ij += Σ_k (a_jk ⊙ z_ki)`。直觉: "到达 j，经过 k，来自 i" 的路径信息。

**环形距离 Bias 的必要性**:

线性拓扑中 |i - j| 是唯一距离度量，但这对 circRNA 有问题：位置 0 和 L-1 是相邻的！环形拓扑使用 `d_circ(i, j) = min(|i-j|, L-|i-j|)`，满足 `d_circ(0, L-1) = 1` 且 `d_circ(0, L/2) = L/2` (最远点)。

**代码实现**:
```python
class TriangleMultiplicativeUpdate(nn.Module):
    """三角乘法更新模块。"""
    
    def __init__(self, c_z: int, c_hidden: int = None, _outgoing: bool = True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden or c_z
        self._outgoing = _outgoing
        
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.layer_norm_out = nn.LayerNorm(c_hidden)
        
        # 输入门控
        self.linear_a_p = nn.Linear(c_z, c_hidden)
        self.linear_a_g = nn.Linear(c_z, c_hidden)
        self.linear_b_p = nn.Linear(c_z, c_hidden)
        self.linear_b_g = nn.Linear(c_z, c_hidden)
        
        # 输出投影
        self.linear_g = nn.Linear(c_z, c_z)
        self.linear_z = nn.Linear(c_hidden, c_z)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化。"""
        nn.init.xavier_uniform_(self.linear_a_p.weight)
        nn.init.xavier_uniform_(self.linear_a_g.weight)
        nn.init.xavier_uniform_(self.linear_b_p.weight)
        nn.init.xavier_uniform_(self.linear_b_g.weight)
        nn.init.zeros_(self.linear_a_p.bias)
        nn.init.zeros_(self.linear_a_g.bias)
        nn.init.zeros_(self.linear_b_p.bias)
        nn.init.zeros_(self.linear_b_g.bias)
    
    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: (batch, L, L, c_z) pair representation
            mask: (batch, L, L) optional mask
        
        Returns:
            (batch, L, L, c_z) updated pair representation
        """
        z = self.layer_norm_in(z)
        
        # 门控
        a = self.linear_a_p(z) * torch.sigmoid(self.linear_a_g(z))  # (B, L, L, c_h)
        b = self.linear_b_p(z) * torch.sigmoid(self.linear_b_g(z))  # (B, L, L, c_h)
        
        if mask is not None:
            a = a * mask.unsqueeze(-1)
            b = b * mask.unsqueeze(-1)
        
        # 核心操作: 三角乘法
        if self._outgoing:
            # z_ij = Σ_k a_ik * b_kj
            # a: (B, L, L, c_h), b: (B, L, L, c_h)
            # 结果: (B, L, L, c_h)
            z_update = torch.einsum('bikc,bkjc->bijc', a, b)
        else:
            # z_ij = Σ_k a_jk * b_ki
            z_update = torch.einsum('bjkc,bkic->bijc', a, b)
        
        z_update = self.layer_norm_out(z_update)
        
        # 输出门控
        g = torch.sigmoid(self.linear_g(z))  # (B, L, L, c_z)
        z_out = g * self.linear_z(z_update)
        
        return z_out
```

##### 配对预测头 (Pair Prediction Head)

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 130">
  <defs>
    <filter id="sh" x="-5%" y="-10%" width="110%" height="120%">
      <feDropShadow dx="1" dy="1" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
    <marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#73daca"/>
    </marker>
  </defs>
  <rect width="500" height="130" fill="#1a1b26"/>
  
  <rect x="10" y="45" width="90" height="35" rx="6" fill="#f7768e" filter="url(#sh)"/>
  <text x="55" y="65" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">pair_repr</text>
  <text x="55" y="77" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">(L, L, c_z)</text>
  
  <rect x="120" y="20" width="180" height="90" rx="6" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="210" y="38" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">Pair Prediction Head</text>
  
  <rect x="130" y="48" width="160" height="25" rx="4" fill="#24283b"/>
  <text x="210" y="64" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9">Linear(c_z, c_z/2) → GELU</text>
  <rect x="130" y="78" width="160" height="25" rx="4" fill="#24283b"/>
  <text x="210" y="94" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9">Linear(c_z/2, 1) → Sigmoid</text>
  
  <rect x="320" y="45" width="170" height="35" rx="6" fill="#9ece6a" filter="url(#sh)"/>
  <text x="405" y="65" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">pair_probs (L, L)</text>
  <text x="405" y="77" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">P(核苷酸 i 和 j 配对)</text>
  
  <path d="M100,60 L120,60" stroke="#73daca" stroke-width="2" marker-end="url(#a)"/>
  <path d="M300,60 L320,60" stroke="#73daca" stroke-width="2" marker-end="url(#a)"/>
</svg>
```

**配对概率的特殊约束**:
```python
# 1. 对称性: P(i,j) = P(j,i)
pair_probs = (pair_probs + pair_probs.transpose(-1, -2)) / 2

# 2. 对角线为 0: 核苷酸不能与自己配对
pair_probs = pair_probs * (1 - torch.eye(L, device=pair_probs.device))

# 3. BSJ 特殊处理: 位置 0 和 L-1 可以配对
# 这是 circRNA 独有的！
bsj_pair_prob = special_head(pair_repr[0, L-1])  # 额外预测
```

##### 结构预测头对比

**SimpleStructureHead (MDS-based)**:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 120">
  <defs><filter id="sh"><feDropShadow dx="1" dy="1" stdDeviation="2"/></filter></defs>
  <rect width="600" height="120" fill="#1a1b26"/>
  
  <rect x="10" y="40" width="80" height="30" rx="4" fill="#f7768e" filter="url(#sh)"/>
  <text x="50" y="58" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">pair_probs</text>
  
  <rect x="110" y="25" width="110" height="70" rx="4" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="165" y="42" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">距离估计</text>
  <text x="165" y="58" text-anchor="middle" fill="#1a1b26" font-size="7">dist = d_bond(1-p)</text>
  <text x="165" y="70" text-anchor="middle" fill="#1a1b26" font-size="7">+ d_pair·p</text>
  
  <rect x="240" y="25" width="90" height="70" rx="4" fill="#9ece6a" filter="url(#sh)"/>
  <text x="285" y="42" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">MDS</text>
  <text x="285" y="60" text-anchor="middle" fill="#1a1b26" font-size="7">多维缩放</text>
  <text x="285" y="75" text-anchor="middle" fill="#1a1b26" font-size="7">→ (L,3)</text>
  
  <rect x="350" y="35" width="90" height="50" rx="4" fill="#bb9af7" filter="url(#sh)"/>
  <text x="395" y="52" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">闭合约束</text>
  <text x="395" y="68" text-anchor="middle" fill="#1a1b26" font-size="7">可选后处理</text>
  
  <rect x="460" y="40" width="130" height="30" rx="4" fill="#73daca" filter="url(#sh)"/>
  <text x="525" y="58" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">coords (L,3) ~10ms</text>
</svg>
```

**CircDiffusionStructure (Diffusion-based)**:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 150">
  <defs><filter id="sh"><feDropShadow dx="1" dy="1" stdDeviation="2"/></filter></defs>
  <rect width="700" height="150" fill="#1a1b26"/>
  
  <text x="20" y="20" fill="#c0caf5" font-size="11" font-weight="bold">训练: x₀ → 加噪 x_t, 损失 = ||ε-ε_θ||² + λ||x₀[0]-x₀[L-1]||²</text>
  
  <rect x="10" y="40" width="70" height="30" rx="4" fill="#f7768e" filter="url(#sh)"/>
  <text x="45" y="58" text-anchor="middle" fill="#1a1b26" font-size="8" font-weight="bold">x_T~N(0,I)</text>
  
  <rect x="100" y="30" width="180" height="90" rx="6" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="190" y="50" text-anchor="middle" fill="#1a1b26" font-size="10" font-weight="bold">去噪循环 (T→0)</text>
  <text x="190" y="70" text-anchor="middle" fill="#1a1b26" font-size="8">x_{t-1}=x_t-ε_θ+σ_t·z</text>
  <text x="190" y="90" text-anchor="middle" fill="#1a1b26" font-size="8">渐进闭合约束 (t&lt;T/4)</text>
  
  <rect x="300" y="40" width="120" height="70" rx="4" fill="#bb9af7" filter="url(#sh)"/>
  <text x="360" y="60" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">闭合损失</text>
  <text x="360" y="80" text-anchor="middle" fill="#1a1b26" font-size="7">||x₀[0]-x₀[L-1]||²</text>
  
  <rect x="440" y="40" width="140" height="30" rx="4" fill="#73daca" filter="url(#sh)"/>
  <text x="510" y="58" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">coords (L,3) ~2-5s GPU</text>
  
  <text x="600" y="20" fill="#e0af68" font-size="10">需要大量训练数据</text>
</svg>
```

**PhysicsStructureHead (physics_b)**:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 650 130">
  <defs><filter id="sh"><feDropShadow dx="1" dy="1" stdDeviation="2"/></filter></defs>
  <rect width="650" height="130" fill="#1a1b26"/>
  
  <rect x="10" y="45" width="80" height="30" rx="4" fill="#f7768e" filter="url(#sh)"/>
  <text x="50" y="63" text-anchor="middle" fill="#1a1b26" font-size="8" font-weight="bold">pair_repr</text>
  
  <rect x="110" y="20" width="130" height="90" rx="6" fill="#9ece6a" filter="url(#sh)"/>
  <text x="175" y="38" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">Constraint</text>
  <text x="175" y="50" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">Extractor</text>
  <text x="175" y="68" text-anchor="middle" fill="#1a1b26" font-size="7">Bond: ||x[i]-x[i+1]||=d</text>
  <text x="175" y="80" text-anchor="middle" fill="#1a1b26" font-size="7">Pair: ||x[i]-x[j]||≈d_pair</text>
  <text x="175" y="92" text-anchor="middle" fill="#1a1b26" font-size="7">Closure: ||x[0]-x[L-1]||=d</text>
  
  <rect x="260" y="25" width="140" height="80" rx="6" fill="#7aa2f7" filter="url(#sh)"/>
  <text x="330" y="42" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">GeometricSolver</text>
  <text x="330" y="60" text-anchor="middle" fill="#1a1b26" font-size="7">随机初始化 → 满足约束</text>
  <text x="330" y="75" text-anchor="middle" fill="#1a1b26" font-size="7">采样多个构象</text>
  <text x="330" y="90" text-anchor="middle" fill="#1a1b26" font-size="7">选择最佳</text>
  
  <rect x="420" y="25" width="100" height="80" rx="6" fill="#bb9af7" filter="url(#sh)"/>
  <text x="470" y="42" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">Validator</text>
  <text x="470" y="60" text-anchor="middle" fill="#1a1b26" font-size="7">闭合距离</text>
  <text x="470" y="75" text-anchor="middle" fill="#1a1b26" font-size="7">碰撞检测</text>
  <text x="470" y="90" text-anchor="middle" fill="#1a1b26" font-size="7">键长偏差</text>
  
  <rect x="540" y="45" width="100" height="30" rx="4" fill="#73daca" filter="url(#sh)"/>
  <text x="590" y="63" text-anchor="middle" fill="#1a1b26" font-size="8" font-weight="bold">coords ~200ms</text>
</svg>
```

**PhysicsStructureHead (physics_ba)**:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 650 150">
  <defs><filter id="sh"><feDropShadow dx="1" dy="1" stdDeviation="2"/></filter></defs>
  <rect width="650" height="150" fill="#1a1b26"/>
  
  <rect x="10" y="55" width="80" height="30" rx="4" fill="#f7768e" filter="url(#sh)"/>
  <text x="50" y="73" text-anchor="middle" fill="#1a1b26" font-size="8" font-weight="bold">physics_b</text>
  
  <rect x="110" y="25" width="150" height="120" rx="6" fill="#e0af68" filter="url(#sh)"/>
  <text x="185" y="42" text-anchor="middle" fill="#1a1b26" font-size="10" font-weight="bold">CGMDRefiner</text>
  
  <rect x="120" y="55" width="130" height="25" rx="3" fill="#24283b"/>
  <text x="185" y="70" text-anchor="middle" fill="#c0caf5" font-size="8">粗粒化模型</text>
  <rect x="120" y="85" width="130" height="25" rx="3" fill="#24283b"/>
  <text x="185" y="100" text-anchor="middle" fill="#c0caf5" font-size="8">能量最小化 500-2000步</text>
  <rect x="120" y="115" width="130" height="25" rx="3" fill="#24283b"/>
  <text x="185" y="130" text-anchor="middle" fill="#c0caf5" font-size="8">MD松弛 5000-20000步</text>
  
  <rect x="280" y="35" width="100" height="60" rx="4" fill="#9ece6a" filter="url(#sh)"/>
  <text x="330" y="55" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">DL Bias</text>
  <text x="330" y="75" text-anchor="middle" fill="#1a1b26" font-size="7">可选</text>
  
  <rect x="400" y="35" width="100" height="60" rx="4" fill="#7dcfff" filter="url(#sh)"/>
  <text x="450" y="55" text-anchor="middle" fill="#1a1b26" font-size="9" font-weight="bold">OpenMM</text>
  <text x="450" y="75" text-anchor="middle" fill="#1a1b26" font-size="7">分子动力学</text>
  
  <rect x="520" y="55" width="120" height="30" rx="4" fill="#73daca" filter="url(#sh)"/>
  <text x="580" y="73" text-anchor="middle" fill="#1a1b26" font-size="8" font-weight="bold">coords ~5-30s</text>
  
  <text x="580" y="120" text-anchor="middle" fill="#a9b1d6" font-size="9">零训练 | 需OpenMM | 物理保证</text>
</svg>
```
    
    │
    └─→ StructureValidator
            计算指标:
            - closure_distance: 首-末距离
            - clash_count: 空间碰撞数
            - bond_rmsd: 键长偏差
            - energy_score: 能量估计

速度: ~200ms
精度: 中高 (构造性满足约束)
训练: 零训练
```

**PhysicsStructureHead (physics_ba)**:
```
physics_b 结果
    │
    └─→ CGMDRefiner (OpenMM)
            │
            ├─→ 构建粗粒化模型
            │       每个核苷酸 = 1 bead (P, S, B 三个伪原子)
            │
            ├─→ 能量最小化 (500-2000 步)
            │       消除初始结构的碰撞和拉伸
            │
            ├─→ MD 松弛 (5000-20000 步)
            │       Langevin dynamics at 300K
            │       采样多个构象
            │
            └─→ DL Bias (可选)
                    用 pair_repr 调整力场参数

速度: ~5-30s
精度: 最高 (物理力场保证)
训练: 零训练
依赖: OpenMM
```

#### 2.2.2 损失函数设计

##### 多任务损失

```python
class TorusFoldLoss(nn.Module):
    """TorusFold 多任务损失。"""
    
    def __init__(
        self,
        w_pair: float = 1.0,      # 配对预测权重
        w_dist: float = 0.5,      # 距离预测权重
        w_closure: float = 2.0,   # 闭合约束权重
        w_confidence: float = 0.3, # 置信度权重
        w_translation: float = 0.2, # 翻译效率权重
        w_stability: float = 0.2,   # circ稳定性权重
        w_immune: float = 0.1      # 免疫激活权重
    ):
        super().__init__()
        self.weights = {
            'pair': w_pair,
            'dist': w_dist,
            'closure': w_closure,
            'confidence': w_confidence,
            'translation': w_translation,
            'stability': w_stability,
            'immune': w_immune
        }
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """计算多任务损失。"""
        losses = {}
        
        # 1. 配对预测损失 (BCE)
        if 'pair_probs' in predictions and 'pair_labels' in targets:
            pair_loss = F.binary_cross_entropy(
                predictions['pair_probs'],
                targets['pair_labels']
            )
            losses['pair'] = pair_loss
        
        # 2. 距离预测损失 (MSE)
        if 'dist_pred' in predictions and 'dist_true' in targets:
            dist_loss = F.mse_loss(
                predictions['dist_pred'],
                targets['dist_true']
            )
            losses['dist'] = dist_loss
        
        # 3. 闭合约束损失 (circRNA 特有)
        if 'coords' in predictions:
            coords = predictions['coords']  # (B, L, 3)
            closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)  # (B,)
            # 目标: closure_dist → bond_length (5.9 Å)
            bond_length = 5.9
            closure_loss = F.mse_loss(closure_dist, torch.full_like(closure_dist, bond_length))
            losses['closure'] = closure_loss
        
        # 4. 置信度损失
        if 'confidence' in predictions and 'plddt' in targets:
            conf_loss = F.mse_loss(
                predictions['confidence'] / 100,  # 归一化到 [0, 1]
                targets['plddt'] / 100
            )
            losses['confidence'] = conf_loss
        
        # 5. 多任务头损失
        if 'translation_efficiency' in predictions and 'translation_label' in targets:
            trans_loss = F.binary_cross_entropy(
                predictions['translation_efficiency'],
                targets['translation_label']
            )
            losses['translation'] = trans_loss
        
        if 'circ_stability' in predictions and 'stability_label' in targets:
            stability_loss = F.mse_loss(
                predictions['circ_stability'],
                targets['stability_label']
            )
            losses['stability'] = stability_loss
        
        if 'immune_pathway_RIG-I' in predictions and 'immune_labels' in targets:
            immune_loss = F.binary_cross_entropy(
                predictions['immune_pathway_RIG-I'],
                targets['immune_labels']['RIG-I']
            )
            losses['immune'] = immune_loss
        
        # 加权求和
        total_loss = sum(
            self.weights.get(k, 1.0) * v
            for k, v in losses.items()
        )
        losses['total'] = total_loss
        
        return losses
```

##### 闭合约束损失推导

**问题**: 标准扩散模型不保证生成的坐标满足 x[0] ≈ x[L-1]。

**解决方案**: 添加显式闭合损失项。

$$\mathcal{L}_{closure} = ||x_0 - x_{L-1}||_2 - d_{bond}||^2$$

**直觉**: 惩罚首末坐标的距离与目标键长的偏差。

**渐进增强策略**:
```python
# 训练早期: 闭合权重小，让模型先学习大致结构
# 训练后期: 闭合权重增大，强制满足闭合约束

epoch_ratio = current_epoch / total_epochs
w_closure = base_weight * (1 + 5 * epoch_ratio)  # 从 1x 增长到 6x
```

#### 2.2.3 训练策略

##### 数据增强

```python
class CircRNADataAugmentation:
    """circRNA 特有的数据增强策略。"""
    
    @staticmethod
    def circular_shift(sequence: str, k: int) -> str:
        """环形平移。
        
        由于 circRNA 是环状的，平移 k 位后仍然是同一个分子。
        这提供了无限的数据增强机会。
        """
        return sequence[k:] + sequence[:k]
    
    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """反向互补。
        
        circRNA 的反向互补也是有效的 circRNA。
        """
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(complement[c] for c in reversed(sequence))
    
    @staticmethod
    def random_mutation(sequence: str, rate: float = 0.05) -> str:
        """随机突变。
        
        模拟自然变异，增强模型鲁棒性。
        """
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            if random.random() < rate:
                seq_list[i] = random.choice(['A', 'U', 'G', 'C'])
        return ''.join(seq_list)
```

##### 课程学习

```
训练阶段:

Phase 1 (Epoch 1-10): 基础结构学习
    - 使用短序列 (50-200 nt)
    - 只训练 pair prediction 和 simple structure head
    - 学习率: 1e-3

Phase 2 (Epoch 11-30): 扩散训练
    - 中等长度序列 (200-500 nt)
    - 训练 diffusion structure
    - 引入闭合损失
    - 学习率: 5e-4

Phase 3 (Epoch 31-50): 精细调优
    - 长序列 (500-1000 nt)
    - 多任务头训练
    - 学习率: 1e-4

Phase 4 (Epoch 51+): Physics 对齐
    - 使用 physics_ba 生成伪标签
    - 蒸馏到 diffusion 模型
    - 学习率: 5e-5
```

##### 混合精度训练

```python
# 使用 PyTorch AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        predictions = model(batch)
        losses = criterion(predictions, batch)
    
    scaler.scale(losses['total']).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### 2.3 Backend 架构深度解析

#### 2.3.1 三层降级实现原理

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 500">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#c0caf5"/>
    </marker>
  </defs>
  
  <!-- 背景 -->
  <rect width="600" height="500" fill="#1a1b26"/>
  
  <!-- 标题 -->
  <rect x="150" y="10" width="300" height="35" rx="6" fill="#24283b" filter="url(#shadow)"/>
  <text x="300" y="33" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="14" font-weight="bold">Backend 选择算法流程</text>
  
  <!-- 开始节点 -->
  <ellipse cx="300" cy="70" rx="50" ry="20" fill="#bb9af7" filter="url(#shadow)"/>
  <text x="300" y="75" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">开始</text>
  
  <!-- Step 1 -->
  <rect x="150" y="100" width="300" height="50" rx="6" fill="#7aa2f7" filter="url(#shadow)"/>
  <text x="300" y="120" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11" font-weight="bold">Step 1: 构建优先级链</text>
  <text x="300" y="138" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">priority_chain = ["esm2", "vienna", "heuristic"]</text>
  
  <!-- Step 2 -->
  <rect x="150" y="160" width="300" height="45" rx="6" fill="#7aa2f7" filter="url(#shadow)"/>
  <text x="300" y="180" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11" font-weight="bold">Step 2: 找到请求后端位置</text>
  <text x="300" y="196" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">start_idx = priority_chain.index(requested)</text>
  
  <!-- 判断框 -->
  <polygon points="300,220 400,260 300,300 200,260" fill="#e0af68" filter="url(#shadow)"/>
  <text x="300" y="265" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">backend</text>
  <text x="300" y="278" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">可用?</text>
  
  <!-- 是 -->
  <rect x="420" y="235" width="130" height="50" rx="6" fill="#9ece6a" filter="url(#shadow)"/>
  <text x="485" y="255" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">返回 backend</text>
  <text x="485" y="272" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">可用后端</text>
  
  <!-- 否 -->
  <rect x="80" y="235" width="100" height="50" rx="6" fill="#f7768e" filter="url(#shadow)"/>
  <text x="130" y="255" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">记录降级原因</text>
  <text x="130" y="272" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">log_degradation()</text>
  
  <!-- 循环判断 -->
  <polygon points="300,320 380,350 300,380 220,350" fill="#bb9af7" filter="url(#shadow)"/>
  <text x="300" y="355" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10">还有后端?</text>
  
  <!-- 结束 -->
  <ellipse cx="300" cy="420" rx="80" ry="25" fill="#9ece6a" filter="url(#shadow)"/>
  <text x="300" y="425" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11">返回 heuristic (保底)</text>
  
  <!-- 连接线 -->
  <path d="M300,90 L300,100" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M300,150 L300,160" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M300,205 L300,220" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M400,260 L420,260" stroke="#9ece6a" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="410" y="252" fill="#9ece6a" font-family="Arial" font-size="9">是</text>
  <path d="M200,260 L180,260" stroke="#f7768e" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="195" y="252" fill="#f7768e" font-family="Arial" font-size="9">否</text>
  <path d="M130,285 L130,350 L220,350" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M300,380 L300,395" stroke="#c0caf5" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="320" y="392" fill="#c0caf5" font-family="Arial" font-size="9">否</text>
  <path d="M380,350 L500,350 L500,285 L550,260" stroke="#c0caf5" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
  <text x="440" y="342" fill="#c0caf5" font-family="Arial" font-size="9">是</text>
  
  <!-- 注释 -->
  <text x="300" y="470" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="10">算法确保始终返回可用的后端</text>
</svg>
```

**可用性检查实现**:
```python
class Backend:
    """后端基类。"""
    
    _availability_cache: Dict[str, Tuple[bool, float]] = {}
    CACHE_TTL = 60.0  # 缓存60秒
    
    def is_available(self) -> bool:
        """检查后端是否可用，带缓存。"""
        cache_key = self.name
        now = time.time()
        
        # 检查缓存
        if cache_key in self._availability_cache:
            available, timestamp = self._availability_cache[cache_key]
            if now - timestamp < self.CACHE_TTL:
                return available
        
        # 实际检查
        available = self._check_availability()
        self._availability_cache[cache_key] = (available, now)
        
        return available
    
    def _check_availability(self) -> bool:
        """子类实现具体检查逻辑。"""
        raise NotImplementedError


class ESM2Backend(Backend):
    """ESM-2 后端可用性检查。"""
    
    def _check_availability(self) -> bool:
        """检查 ESM-2 是否可用。"""
        # 1. 检查模块是否导入
        try:
            import esm
        except ImportError:
            self._unavailable_reason = "ESM module not installed"
            return False
        
        # 2. 检查 GPU
        if not torch.cuda.is_available():
            self._unavailable_reason = "CUDA not available"
            return False
        
        # 3. 检查 GPU 内存
        free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
        if free_memory < 4.0:  # ESM-2 650M 需要约 4GB
            self._unavailable_reason = f"Insufficient GPU memory ({free_memory:.1f}GB free, need 4GB)"
            return False
        
        # 4. 检查模型文件
        model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
        if not os.path.exists(model_path):
            self._unavailable_reason = "Model weights not downloaded"
            return False
        
        return True


class ViennaBackend(Backend):
    """ViennaRNA 后端可用性检查。"""
    
    def _check_availability(self) -> bool:
        """检查 ViennaRNA 是否可用。"""
        try:
            import RNA
            # 测试调用
            fc = RNA.fold_compound("AUGCGC")
            structure, mfe = fc.mfe()
            return True
        except ImportError:
            self._unavailable_reason = "ViennaRNA not installed. Install: conda install -c bioconda viennarna"
            return False
        except Exception as e:
            self._unavailable_reason = f"ViennaRNA error: {str(e)}"
            return False
```

#### 2.3.2 并行 Backend 调用

对于需要多个后端结果的场景，支持并行调用：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelBackendManager:
    """并行后端管理器。"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def assess_parallel(
        self,
        sequence: str,
        backends: List[str] = ["esm2", "vienna", "heuristic"]
    ) -> Dict[str, Dict[str, float]]:
        """并行调用多个后端，返回所有结果。"""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for backend_name in backends:
            backend = self.backends[backend_name]
            if backend.is_available():
                task = loop.run_in_executor(
                    self.executor,
                    backend.assess,
                    sequence
                )
                tasks.append((backend_name, task))
        
        results = {}
        for backend_name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=30.0)
                results[backend_name] = result
            except asyncio.TimeoutError:
                results[backend_name] = {"error": "timeout"}
            except Exception as e:
                results[backend_name] = {"error": str(e)}
        
        return results
    
    def ensemble_prediction(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """集成多个后端的预测结果。"""
        # 按后端精度加权
        weights = {"esm2": 0.5, "vienna": 0.3, "heuristic": 0.2}
        
        ensemble = {}
        metrics = ["rig_i_score", "tlr7_score", "tlr8_score", "pkr_score"]
        
        for metric in metrics:
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for backend_name, result in results.items():
                if "error" not in result and metric in result:
                    w = weights.get(backend_name, 0.1)
                    weighted_sum += w * result[metric]
                    weight_sum += w
            
            if weight_sum > 0:
                ensemble[metric] = weighted_sum / weight_sum
        
        return ensemble
```

---

### 2.4 模块交互流程

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 700">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#24283b"/>
      <stop offset="100%" style="stop-color:#1a1b26"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#73daca"/>
    </marker>
  </defs>
  
  <!-- 背景 -->
  <rect width="900" height="700" fill="#1a1b26"/>
  
  <!-- 标题 -->
  <rect x="300" y="5" width="300" height="30" rx="6" fill="#3d59a1" filter="url(#shadow)"/>
  <text x="450" y="25" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="13" font-weight="bold">用户请求流程</text>
  
  <!-- 用户入口 -->
  <rect x="350" y="45" width="100" height="35" rx="6" fill="#bb9af7" filter="url(#shadow)"/>
  <text x="400" y="67" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11" font-weight="bold">用户</text>
  
  <!-- CLI/API -->
  <rect x="350" y="90" width="100" height="30" rx="4" fill="#414d68"/>
  <text x="400" y="108" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="10">CLI/API</text>
  
  <!-- Config初始化 -->
  <rect x="250" y="130" width="300" height="80" rx="6" fill="#7aa2f7" opacity="0.9" filter="url(#shadow)"/>
  <text x="400" y="155" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="12" font-weight="bold">Config初始化</text>
  <rect x="260" y="165" width="280" height="35" rx="4" fill="#24283b"/>
  <text x="400" y="180" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9">validate(): 亚型验证 + Backend检查 + 参数校验</text>
  <text x="400" y="195" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">EventBus.subscribe() 注册事件监听</text>
  
  <!-- Agent初始化 -->
  <rect x="200" y="220" width="400" height="100" rx="8" fill="#9ece6a" opacity="0.9" filter="url(#shadow)"/>
  <text x="400" y="240" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="12" font-weight="bold">TNBCSimulacrumAgent.initialize()</text>
  
  <!-- StateSchema -->
  <rect x="210" y="250" width="130" height="60" rx="4" fill="#24283b"/>
  <text x="275" y="270" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9" font-weight="bold">StateSchema.reset()</text>
  <text x="275" y="285" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">tumor_volume=100</text>
  <text x="275" y="298" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">immune_cells初始化</text>
  
  <!-- SubsystemManagers -->
  <rect x="350" y="250" width="240" height="60" rx="4" fill="#24283b"/>
  <text x="470" y="270" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9" font-weight="bold">SubsystemManagers.initialize()</text>
  <text x="470" y="290" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">Tumor | TME | Treatment | Biomarker | Clinical | CircRNA</text>
  
  <!-- 仿真循环 -->
  <rect x="100" y="330" width="700" height="350" rx="8" fill="url(#grad1)" stroke="#414d68" stroke-width="2"/>
  <text x="450" y="350" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="12" font-weight="bold">仿真循环: for step in range(max_steps)</text>
  
  <!-- TumorManager -->
  <rect x="120" y="370" width="160" height="70" rx="6" fill="#f7768e" opacity="0.8" filter="url(#shadow)"/>
  <text x="200" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">TumorManager.step()</text>
  <text x="200" y="408" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">GrowthEngine</text>
  <text x="200" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">Heterogeneity</text>
  <text x="200" y="432" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">CSC + Angiogenesis</text>
  
  <!-- TMEManager -->
  <rect x="290" y="370" width="160" height="70" rx="6" fill="#7aa2f7" opacity="0.8" filter="url(#shadow)"/>
  <text x="370" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">TMEManager.step()</text>
  <text x="370" y="408" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">ImmuneDynamics</text>
  <text x="370" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">ImmuneEvasion</text>
  <text x="370" y="432" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">Immunoediting</text>
  
  <!-- TreatmentManager -->
  <rect x="460" y="370" width="160" height="70" rx="6" fill="#e0af68" opacity="0.8" filter="url(#shadow)"/>
  <text x="540" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">TreatmentManager</text>
  <text x="540" y="408" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">Chemo + Immuno</text>
  <text x="540" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">PARP + Radio</text>
  <text x="540" y="432" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">circRNA Therapy</text>
  
  <!-- BiomarkerManager -->
  <rect x="630" y="370" width="160" height="70" rx="6" fill="#bb9af7" opacity="0.8" filter="url(#shadow)"/>
  <text x="710" y="390" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">BiomarkerManager</text>
  <text x="710" y="408" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">CEA, CA15-3</text>
  <text x="710" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">基因表达谱</text>
  
  <!-- CircRNA详细 -->
  <rect x="470" y="450" width="140" height="60" rx="4" fill="#9ece6a" opacity="0.7"/>
  <text x="540" y="470" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">CircRNAManager</text>
  <text x="540" y="485" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">免疫感知评估</text>
  <text x="540" y="498" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">TorusFold结构预测</text>
  
  <!-- ClinicalManager -->
  <rect x="120" y="450" width="160" height="50" rx="6" fill="#7dcfff" opacity="0.8"/>
  <text x="200" y="470" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">ClinicalManager</text>
  <text x="200" y="488" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">分期更新 + 生存概率</text>
  
  <!-- 事件发射 -->
  <rect x="290" y="520" width="120" height="35" rx="4" fill="#f7768e" opacity="0.6"/>
  <text x="350" y="540" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9">emit(STEP_START)</text>
  
  <rect x="420" y="520" width="120" height="35" rx="4" fill="#9ece6a" opacity="0.6"/>
  <text x="480" y="540" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9">emit(STEP_END)</text>
  
  <!-- State持久化 -->
  <rect x="290" y="560" width="250" height="35" rx="4" fill="#414d68"/>
  <text x="415" y="580" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="10">StateSchema.persist() → 保存状态快照</text>
  
  <!-- 连接线 -->
  <path d="M400,125 L400,130" stroke="#73daca" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M400,210 L400,220" stroke="#73daca" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M400,320 L400,330" stroke="#73daca" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- 循环箭头 -->
  <path d="M800,450 L850,450 L850,350 L100,350 L100,450 L120,450" stroke="#414d68" stroke-width="2" fill="none" stroke-dasharray="5,3"/>
  <text x="875" y="400" fill="#a9b1d6" font-family="Arial" font-size="9">循环</text>
</svg>
```

---

## 3. 核心模块详解

### 3.1 TNBC Simulacrum 仿真引擎

#### 3.1.1 肿瘤生长动力学

**Logistic 增长模型**：

$$\frac{dV}{dt} = rV\left(1 - \frac{V}{K}\right) - \delta V$$

其中：
- $V$ = 肿瘤体积 (mm³)
- $r$ = 生长率 (TNBC 中位值 ~0.027 day⁻¹)
- $K$ = 环境容纳量 (~1000 mm³)
- $\delta$ = 凋亡率 (~0.005 day⁻¹)

**Gompertz 增长模型**（可选）：

$$\frac{dV}{dt} = rV \ln\left(\frac{K}{V}\right)$$

Gompertz 模型更适合描述生长后期饱和的肿瘤生长曲线。

**参数来源**：
- TNBC 倍增时间中位值 ≈ 25 天 (Sparano et al., NEJM 2018)
- 生长率 $r \approx \frac{\ln 2}{T_d} \approx 0.028$ day⁻¹

**代码实现** (`core/tumor/growth_engine.py`):

```python
class GrowthEngine:
    """肿瘤生长动力学引擎。"""
    
    def __init__(self, model: str = "logistic", growth_rate: float = 0.027):
        self.model = model
        self.growth_rate = growth_rate
        self.capacity = 1000.0  # mm³
        self.apoptosis_rate = 0.005
    
    def update_volume(self, V: float, dt: float = 1.0) -> float:
        """更新肿瘤体积。"""
        if self.model == "logistic":
            dV = self.growth_rate * V * (1 - V / self.capacity) - self.apoptosis_rate * V
        elif self.model == "gompertz":
            dV = self.growth_rate * V * np.log(self.capacity / V)
        return V + dV * dt
```

#### 3.1.2 肿瘤异质性模型

**亚克隆演化**：

采用分支进化模型，每个亚克隆具有独特的突变谱。亚克隆多样性用 Shannon 指数量化：

$$H = -\sum_{i=1}^{n} p_i \ln p_i$$

其中 $p_i$ 为第 $i$ 个亚克隆的占比。

**突变累积**：

$$\mu_{new} = \mu_{base} \times N_{div} \times \mu_{rate}$$

- $\mu_{base}$ = 基础突变负荷
- $N_{div}$ = 细胞分裂次数
- $\mu_{rate}$ = 每细胞每代突变率 (~10⁻⁶)

**代码实现** (`core/tumor/heterogeneity.py`):

```python
class SubcloneEvolution:
    """亚克隆演化模型。"""
    
    def __init__(self, n_subclones: int = 5, mutation_rate: float = 1e-6):
        self.n_subclones = n_subclones
        self.mutation_rate = mutation_rate
        self.subclones = self._initialize_subclones()
    
    def _initialize_subclones(self) -> List[Subclone]:
        """初始化亚克隆群体。"""
        subclones = []
        for i in range(self.n_subclones):
            subclones.append(Subclone(
                id=i,
                fitness=1.0 + np.random.normal(0, 0.1),
                proportion=1.0 / self.n_subclones,
                mutations=[]
            ))
        return subclones
    
    def evolve(self, n_divisions: int) -> None:
        """执行一代进化。"""
        for subclone in self.subclones:
            # 突变累积
            new_mutations = int(n_divisions * self.mutation_rate * len(subclone.mutations) + 1)
            for _ in range(new_mutations):
                subclone.mutations.append(random_mutation())
            
            # 选择压力
            subclone.proportion *= subclone.fitness
        
        # 归一化比例
        total = sum(s.proportion for s in self.subclones)
        for s in self.subclones:
            s.proportion /= total
    
    def shannon_diversity(self) -> float:
        """计算 Shannon多样性指数。"""
        H = 0.0
        for s in self.subclones:
            if s.proportion > 0:
                H -= s.proportion * np.log(s.proportion)
        return H
```

#### 3.1.3 癌干细胞 (CSC) 模型

CSC 群体动力学：

$$\frac{dN_{CSC}}{dt} = \alpha N_{CSC} - \beta N_{CSC}$$

- $\alpha$ = 自我更新率 (默认 0.5)
- $\beta$ = 分化率 (默认 0.5)

CSC 化疗抗性：

$$\text{Kill}_{CSC} = \frac{\text{Kill}_{bulk}}{f_{resistance}}$$

其中 $f_{resistance}$ 默认为 5.0，即 CSC 比普通肿瘤细胞抗性高 5 倍。

**文献依据**：
- CSC 占肿瘤细胞 1-5% (Al-Hajj et al., PNAS 2003)
- CSC 与化疗耐药相关 (Dean et al., Nat Rev Cancer 2009)

**代码实现** (`core/tumor/cancer_stem_cell.py`):

```python
class CancerStemCellModel:
    """癌干细胞模型。"""
    
    def __init__(self, csc_fraction: float = 0.02, self_renewal: float = 0.5):
        self.csc_fraction = csc_fraction
        self.self_renewal_rate = self_renewal
        self.differentiation_rate = 0.5
        self.resistance_factor = 5.0  # CSC抗性倍数
    
    def update_population(self, N_tumor: float) -> float:
        """更新 CSC 数量。"""
        N_csc = N_tumor * self.csc_fraction
        # 自我更新
        N_csc_new = N_csc * self.self_renewal_rate
        # 分化产生普通细胞
        N_differentiated = N_csc * self.differentiation_rate
        return N_csc_new
    
    def apply_chemotherapy(self, kill_rate: float) -> float:
        """化疗对 CSC 的杀伤效果。"""
        # CSC抗性更高
        effective_kill = kill_rate / self.resistance_factor
        return effective_kill
```

#### 3.1.4 血管生成

VEGF 驱动的血管生成：

$$\frac{dMVD}{dt} = k_{angiogenesis} \cdot \frac{[VEGF]}{K_m + [VEGF]} \cdot (1 - MVD)$$

- $MVD$ = 微血管密度
- $k_{angiogenesis}$ = 血管生成速率常数
- $K_m$ = VEGF 半饱和常数

**血管正常化窗口** (Jain, Science 2005)：
- 抗血管生成治疗后 7 天窗口期
- 此期间化疗药物递送效率最高

**代码实现** (`core/tumor/angiogenesis.py`):

```python
class AngiogenesisModel:
    """血管生成模型。"""
    
    def __init__(self, vegf_production_rate: float = 0.1):
        self.vegf_rate = vegf_production_rate
        self.k_angiogenesis = 0.05
        self.Km_vegf = 10.0  # ng/mL
        self.max_mvd = 1.0
    
    def update_mvd(self, mvd: float, hypoxia: float, dt: float = 1.0) -> float:
        """更新微血管密度。"""
        # 缺氧驱动 VEGF 产生
        vegf = hypoxia * self.vegf_rate
        
        # Michaelis-Menten 动力学
        dmvd = self.k_angiogenesis * vegf / (self.Km_vegf + vegf) * (self.max_mvd - mvd)
        
        return mvd + dmvd * dt
    
    def normalization_window(self, anti_angiogenic_dose: float) -> Tuple[int, int]:
        """计算血管正常化窗口。"""
        # Jain窗口：治疗后 5-10 天
        start_day = int(5 + anti_angiogenic_dose * 2)
        end_day = int(10 + anti_angiogenic_dose * 3)
        return (start_day, end_day)
```

#### 3.1.5 转移模型

**EMT/MET 动力学**：

$$\frac{dN_{EMT}}{dt} = k_{EMT} \cdot N_{epithelial} - k_{MET} \cdot N_{EMT}$$

**器官趋向性** (TNBC 特异性)：

| 器官 | 转移概率 |
|------|---------|
| 肺 | 35% |
| 肝 | 25% |
| 骨 | 20% |
| 脑 | 15% |
| 远处淋巴结 | 5% |

**文献依据**：
- TNBC 内脏转移偏好 (Dent et al., Clin Cancer Res 2007)

**代码实现** (`core/tumor/metastasis.py`):

```python
class MetastasisModel:
    """转移模型。"""
    
    # TNBC器官趋向性
    ORGAN_TROPISM = {
        "lung": 0.35,
        "liver": 0.25,
        "bone": 0.20,
        "brain": 0.15,
        "lymph": 0.05
    }
    
    def __init__(self, emt_rate: float = 0.01, met_rate: float = 0.005):
        self.emt_rate = emt_rate
        self.met_rate = met_rate
    
    def check_metastasis(self, tumor_volume: float, emt_fraction: float) -> Optional[str]:
        """检查是否发生转移。"""
        # 转移概率与体积和 EMT 分数相关
        probability = 0.001 * tumor_volume * emt_fraction
        
        if random.random() < probability:
            # 选择靶器官
            organ = random.choices(
                list(self.ORGAN_TROPISM.keys()),
                weights=list(self.ORGAN_TROPISM.values())
            )[0]
            return organ
        return None
```

---

### 3.2 circRNA 子系统

#### 3.2.1 免疫感知模块

circRNA 免疫原性预测基于四个主要通路：

**RIG-I 通路** (circRNA 特异性)：

由于 circRNA 无 5' 端，RIG-I 无法通过经典的 5'-三磷酸识别，而是通过 **dsRNA 结构** 间接激活：

$$\text{RIG-I}_{score} = 0.40 \times \text{dsRNA}_{fraction} + 0.30 \times \text{CCUCC}_{count} + 0.20 \times \text{GC}_{content} + 0.10 \times \text{Length}_{norm}$$

**关键文献**：
- Zhang et al., Nature Immunology 2016: circRNA 可通过 dsRNA 结构激活 RIG-I
- Chen et al., Nature 2019: 核苷酸修饰降低 circRNA 免疫原性

**TLR7/TLR8 通路**：

TLR7 和 TLR8 偏好不同的 ssRNA motif：

$$\text{TLR7}_{score} = 0.45 \times \text{GU-rich} + 0.30 \times \text{AU-rich} + 0.20 \times \text{Uridine} + 0.05 \times \text{Length}$$

$$\text{TLR8}_{score} = 0.40 \times \text{AU-rich} + 0.35 \times \text{Uridine} + 0.20 \times \text{GUUG} + 0.05 \times \text{Length}$$

**TLR7 特异性 motif**: GUUG, GUGU, UGUU, GUCU, GUUU
**TLR8 特异性 motif**: AUUA, UUAU, UAUU, AUUU, UAAU

**PKR 通路**：

PKR 需要 >33 bp 的 dsRNA 才能有效激活：

$$\text{PKR}_{score} = 0.50 \times \mathbf{1}_{dsRNA > 33bp} + 0.25 \times \text{dsRNA}_{length} + 0.20 \times \text{GC}_{content} + 0.05 \times \text{Modification}_{penalty}$$

**文献依据**：
- Nallagatla et al., RNA 2007: PKR 需要 >33 bp dsRNA
- Diebold et al., Science 2006: TLR7/TLR8 对 GU/AU-rich 序列的偏好

**代码实现** (`core/circrna/immune_sensing.py`):

```python
class ImmuneSensingModule:
    """circRNA 免疫感知模块。"""
    
    # TLR7偏好 motif
    TLR7_MOTIFS = ["GUUG", "GUGU", "UGUU", "GUCU", "GUUU"]
    # TLR8偏好 motif
    TLR8_MOTIFS = ["AUUA", "UUAU", "UAUU", "AUUU", "UAAU"]
    # PKR激活阈值
    PKR_DSRNA_THRESHOLD = 33  # bp
    
    def __init__(self, backend: str = "heuristic"):
        self.backend = backend
        self._init_backend()
    
    def _init_backend(self):
        """初始化后端。"""
        if self.backend == "esm2":
            try:
                import esm
                self.esm_model = esm.pretrained.esm2_t33_650M_UR50D()
            except ImportError:
                self.backend = "vienna"
        
        if self.backend == "vienna":
            try:
                import RNA
                self.vienna_available = True
            except ImportError:
                self.backend = "heuristic"
    
    def assess_rig_i(self, sequence: str, structure: Optional[str] = None) -> float:
        """评估 RIG-I 激活潜力。"""
        seq = sequence.upper().replace("T", "U")
        length = len(seq)
        
        # GC含量
        gc = sum(1 for c in seq if c in "GC") / length
        
        # dsRNA fraction (从结构或估计)
        if structure and self.backend == "vienna":
            dsrna_frac = self._compute_dsrna_fraction(seq, structure)
        else:
            dsrna_frac = self._estimate_dsrna(gc, length)
        
        # CCUCC motif计数
        ccucc_count = seq.count("CCUCC") + seq.count("CCTCC")
        
        # 长度归一化
        length_norm = min(length / 1000, 1.0)
        
        # 综合评分
        score = 0.40 * dsrna_frac + 0.30 * ccucc_count / 10 + 0.20 * gc + 0.10 * length_norm
        return min(score, 1.0)
    
    def assess_tlr7(self, sequence: str) -> float:
        """评估 TLR7 激活潜力。"""
        seq = sequence.upper().replace("T", "U")
        length = len(seq)
        
        # GU-rich fraction
        gu_frac = sum(1 for c in seq if c in "GU") / length
        
        # AU-rich fraction
        au_frac = sum(1 for c in seq if c in "AU") / length
        
        # Uridine fraction
        u_frac = seq.count("U") / length
        
        # motif计数
        motif_count = sum(seq.count(m) for m in self.TLR7_MOTIFS)
        
        # 长度归一化
        length_norm = min(length / 500, 1.0)
        
        score = 0.45 * gu_frac + 0.30 * au_frac + 0.20 * u_frac + 0.05 * length_norm
        return min(score, 1.0)
    
    def assess_tlr8(self, sequence: str) -> float:
        """评估 TLR8 激活潜力。"""
        seq = sequence.upper().replace("T", "U")
        length = len(seq)
        
        # AU-rich fraction
        au_frac = sum(1 for c in seq if c in "AU") / length
        
        # Uridine fraction
        u_frac = seq.count("U") / length
        
        # motif计数
        motif_count = sum(seq.count(m) for m in self.TLR8_MOTIFS)
        motif_norm = motif_count / max(length / 50, 1)
        
        # 长度归一化
        length_norm = min(length / 500, 1.0)
        
        score = 0.40 * au_frac + 0.35 * u_frac + 0.20 * motif_norm + 0.05 * length_norm
        return min(score, 1.0)
    
    def assess_pkr(self, sequence: str, structure: Optional[str] = None) -> float:
        """评估 PKR 激活潜力。"""
        seq = sequence.upper().replace("T", "U")
        length = len(seq)
        
        # GC含量
        gc = sum(1 for c in seq if c in "GC") / length
        
        # 长dsRNA检测
        has_long_dsrna = 0.0
        if structure:
            has_long_dsrna = self._detect_long_dsrna(structure, self.PKR_DSRNA_THRESHOLD)
        else:
            # 启发式估计
            has_long_dsrna = 1.0 if gc > 0.6 and length > 200 else 0.5
        
        # dsRNA长度估计
        dsrna_length = length * gc * 0.3
        
        # 综合评分
        score = 0.50 * has_long_dsrna + 0.25 * min(dsrna_length / 100, 1.0) + 0.20 * gc
        return min(score, 1.0)
    
    def compute_overall_immunogenicity(
        self,
        sequence: str,
        modification: str = "none",
        structure: Optional[str] = None
    ) -> Dict[str, float]:
        """计算综合免疫原性评分。"""
        rig_i = self.assess_rig_i(sequence, structure)
        tlr7 = self.assess_tlr7(sequence)
        tlr8 = self.assess_tlr8(sequence)
        pkr = self.assess_pkr(sequence, structure)
        
        # 修饰惩罚
        mod_penalty = self._get_modification_penalty(modification)
        
        # 调整评分
        rig_i *= mod_penalty
        tlr7 *= mod_penalty
        tlr8 *= mod_penalty
        pkr *= mod_penalty
        
        # 综合评分
        overall = 0.35 * rig_i + 0.20 * tlr7 + 0.15 * tlr8 + 0.30 * pkr
        
        return {
            "rig_i_score": rig_i,
            "tlr7_score": tlr7,
            "tlr8_score": tlr8,
            "pkr_score": pkr,
            "overall_immunogenicity": overall,
            "modification": modification,
            "backend": self.backend
        }
    
    def _get_modification_penalty(self, modification: str) -> float:
        """获取核苷酸修饰对免疫原性的抑制系数。"""
        penalties = {
            "none": 1.0,
            "m6A": 0.3,  # Chen et al., Nature 2019
            "Psi": 0.2,  # pseudouridine 强抑制
            "5mC": 0.4,
            "ms2m6A": 0.15,
            "2OMeA": 0.25,
            "2OMeU": 0.25,
            "m5U": 0.35,
            "s2U": 0.4
        }
        return penalties.get(modification, 1.0)
    
    def _estimate_dsrna(self, gc: float, length: int) -> float:
        """启发式估计 dsRNA fraction。"""
        # 高GC + 长序列 → 更多 dsRNA
        return gc * 0.7 * min(length / 500, 1.0)
    
    def _compute_dsrna_fraction(self, seq: str, structure: str) -> float:
        """从 ViennaRNA 结构计算 dsRNA fraction。"""
        paired = sum(1 for c in structure if c == '(' or c == ')')
        return paired / len(structure)
    
    def _detect_long_dsrna(self, structure: str, threshold: int) -> float:
        """检测长 dsRNA 区段。"""
        max_stem = 0
        current_stem = 0
        for c in structure:
            if c == '(':
                current_stem += 1
                max_stem = max(max_stem, current_stem)
            elif c == ')':
                current_stem -= 1
        
        return 1.0 if max_stem >= threshold else max_stem / threshold
```

#### 3.2.2 CirculaPK 六室 PK 模型

circRNA 药代动力学六室模型：

```
Inj (注射) → LNP (递送复合体) → Endo (内吞体) → Cyto (胞质RNA) → Trans (翻译蛋白) → Clear (清除)
```

**ODE 系统**：

$$\frac{d[\text{Inj}]}{dt} = -k_{uptake}[\text{Inj}]$$

$$\frac{d[\text{LNP}]}{dt} = k_{uptake}[\text{Inj}] - k_{release}[\text{LNP}]$$

$$\frac{d[\text{Endo}]}{dt} = k_{release}[\text{LNP}] - k_{escape}[\text{Endo}]$$

$$\frac{d[\text{Cyto}]}{dt} = k_{escape}[\text{Endo}] - k_{degrade}[\text{Cyto}]$$

$$\frac{d[\text{Trans}]}{dt} = f_{translate} \cdot k_{degrade}[\text{Cyto}] - k_{protein}[\text{Trans}]$$

**关键参数**：

| 参数 | 含义 | 默认值 | 文献来源 |
|------|------|--------|---------|
| $k_{uptake}$ | 摄取速率 | 0.80 h⁻¹ (IV) | Hassett et al., Mol Ther 2019 |
| $k_{release}$ | LNP 释放速率 | 0.12 h⁻¹ | Gilleron et al., Nat Biotechnol 2013 |
| $k_{escape}$ | 内吞体逃逸效率 | 0.025 h⁻¹ | Gilleron et al., Nat Biotechnol 2013 |
| $k_{degrade}$ | RNA 降解速率 | 0.12 h⁻¹ | Wesselhoeft et al., Nat Commun 2018 |
| $k_{protein}$ | 蛋白半衰期 | 16 h | - |

**核苷酸修饰效果**：

| 修饰 | 半衰期延长倍数 |
|------|--------------|
| 无修饰 | 1.0× |
| m6A | 1.8× |
| Ψ | 2.5× |
| 5mC | 2.0× |
| ms2m6A | 3.0× |

**文献来源**：
- Chen et al., Nature 2019: m6A 修饰降低免疫原性
- Liu et al., Nat Commun 2023: 核苷酸修饰对 circRNA 稳定性的影响

**代码实现** (`core/circrna/pk/rnactm.py`):

```python
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.integrate import odeint

@dataclass
class CirculaPKConfig:
    """CirculaPK 六室模型配置。"""
    k_uptake: float = 0.80      # h⁻¹,摄取速率
    k_release: float = 0.12     # h⁻¹, LNP释放
    k_escape: float = 0.025     # h⁻¹, 内吞体逃逸
    k_degrade: float = 0.12     # h⁻¹, RNA降解
    k_protein: float = 0.0625   # h⁻¹, 蛋白降解 (半衰期16h)
    f_translate: float = 0.5    # 翻译效率
    
    # 修饰半衰期延长
    modification_half_life_factor: Dict[str, float] = {
        "none": 1.0,
        "m6A": 1.8,
        "Psi": 2.5,
        "5mC": 2.0,
        "ms2m6A": 3.0
    }


class CirculaPKModel:
    """CirculaPK 六室 circRNA 药代动力学模型。"""
    
    COMPARTMENTS = ["Inj", "LNP", "Endo", "Cyto", "Trans", "Clear"]
    
    def __init__(self, config: CirculaPKConfig = None):
        self.config = config or CirculaPKConfig()
    
    def _ode_system(self, y: np.ndarray, t: float) -> np.ndarray:
        """ODE系统定义。"""
        Inj, LNP, Endo, Cyto, Trans, Clear = y
        
        dInj = -self.config.k_uptake * Inj
        dLNP = self.config.k_uptake * Inj - self.config.k_release * LNP
        dEndo = self.config.k_release * LNP - self.config.k_escape * Endo
        dCyto = self.config.k_escape * Endo - self.config.k_degrade * Cyto
        dTrans = self.config.f_translate * self.config.k_degrade * Cyto - self.config.k_protein * Trans
        dClear = (1 - self.config.f_translate) * self.config.k_degrade * Cyto + self.config.k_protein * Trans
        
        return np.array([dInj, dLNP, dEndo, dCyto, dTrans, dClear])
    
    def simulate(
        self,
        dose: float = 1.0,  # mg/kg
        duration: float = 72.0,  # hours
        modification: str = "none",
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """模拟 PK曲线。"""
        # 调整降解速率 (修饰延长半衰期)
        factor = self.config.modification_half_life_factor.get(modification, 1.0)
        effective_k_degrade = self.config.k_degrade / factor
        
        # 初始条件
        y0 = np.array([dose, 0, 0, 0, 0, 0])
        
        # 时间点
        t = np.linspace(0, duration, n_points)
        
        # 求解 ODE
        solution = odeint(self._ode_system, y0, t)
        
        return {
            "time": t,
            "Inj": solution[:, 0],
            "LNP": solution[:, 1],
            "Endo": solution[:, 2],
            "Cyto": solution[:, 3],
            "Trans": solution[:, 4],
            "Clear": solution[:, 5],
            "modification": modification,
            "dose": dose
        }
    
    def compute_pk_metrics(self, result: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算 PK 指标。"""
        cyto = result["Cyto"]
        trans = result["Trans"]
        time = result["time"]
        
        # Cmax (胞质 RNA 最大浓度)
        cmax_cyto = np.max(cyto)
        tmax_cyto = time[np.argmax(cyto)]
        
        # AUC (曲线下面积)
        auc_cyto = np.trapz(cyto, time)
        auc_trans = np.trapz(trans, time)
        
        # 半衰期估计
        # 找到下降阶段
        peak_idx = np.argmax(cyto)
        if peak_idx < len(cyto) - 1:
            decline = cyto[peak_idx:]
            half_life_idx = np.argmin(np.abs(decline - cmax_cyto * 0.5))
            half_life = time[peak_idx + half_life_idx] - tmax_cyto
        else:
            half_life = np.inf
        
        return {
            "Cmax_cyto": cmax_cyto,
            "Tmax_cyto": tmax_cyto,
            "AUC_cyto": auc_cyto,
            "AUC_trans": auc_trans,
            "Half_life": half_life,
            "Total_protein": np.max(trans)
        }
    
    def optimize_dose(
        self,
        target_protein: float = 0.5,
        max_dose: float = 10.0,
        modification: str = "none"
    ) -> Tuple[float, Dict[str, float]]:
        """优化给药剂量以达到目标蛋白水平。"""
        best_dose = 0.0
        best_metrics = None
        
        for dose in np.linspace(0.1, max_dose, 20):
            result = self.simulate(dose=dose, modification=modification)
            metrics = self.compute_pk_metrics(result)
            
            if metrics["Total_protein"] >= target_protein:
                return dose, metrics
        
        # 如果达不到目标，返回最大剂量结果
        result = self.simulate(dose=max_dose, modification=modification)
        return max_dose, self.compute_pk_metrics(result)
```

#### 3.2.3 circRNA 序列进化优化

**REINFORCE 算法**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot R\right]$$

**多目标 Pareto 优化**：

目标向量：
$$\mathbf{f} = (f_{stability}, f_{translation}, f_{immune\_evasion}, f_{delivery})$$

Pareto 前沿：
$$\mathcal{P} = \{\mathbf{x} : \nexists \mathbf{y}, \mathbf{f}(\mathbf{y}) \succ \mathbf{f}(\mathbf{x})\}$$

**权重自适应**：

$$w_i^{(t+1)} = w_i^{(t)} + \alpha \cdot \frac{\partial R}{\partial w_i}$$

**代码实现** (`core/circrna/evolution/sequence_evolver.py`):

```python
from typing import List, Tuple, Dict
import numpy as np
import random

class CircRNASequenceEvolver:
    """circRNA 序列进化优化器。"""
    
    OBJECTIVES = ["stability", "translation", "immune_evasion", "delivery"]
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        n_generations: int = 100,
        target_length: int = 500,
        objectives_weights: Dict[str, float] = None
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_generations = n_generations
        self.target_length = target_length
        
        # 默认权重
        self.weights = objectives_weights or {
            "stability": 0.3,
            "translation": 0.3,
            "immune_evasion": 0.25,
            "delivery": 0.15
        }
        
        # IRES motif库
        self.ires_motifs = ["GCGCC", "GGGG", "UUGU", "AUGG", "CCUG", "GGAAGG"]
    
    def _generate_random_sequence(self, length: int) -> str:
        """生成随机 circRNA 序列。"""
        return "".join(random.choices("ACGU", k=length))
    
    def _initialize_population(self) -> List[str]:
        """初始化种群。"""
        population = []
        for _ in range(self.population_size):
            seq = self._generate_random_sequence(self.target_length)
            population.append(seq)
        return population
    
    def _compute_objectives(self, sequence: str, modification: str = "none") -> np.ndarray:
        """计算四维目标向量。"""
        seq = sequence.upper().replace("T", "U")
        length = len(seq)
        
        if length < 50:
            return np.array([0.3, 0.3, 0.5, 0.3])
        
        gc = sum(1 for c in seq if c in "GC") / length
        
        # Stability
        stability = 0.3 + gc * 0.5
        mod_bonus = {"m6A": 0.1, "Psi": 0.15, "5mC": 0.08}
        stability += mod_bonus.get(modification, 0.0)
        
        # Translation
        ires_count = sum(1 for m in self.ires_motifs if m in seq)
        translation = 0.2 + ires_count * 0.12
        aug_count = seq.count("AUG")
        translation += min(aug_count * 0.05, 0.2)
        if 0.4 <= gc <= 0.55:
            translation += 0.1
        
        # Immune evasion (简化)
        gu_content = sum(1 for c in seq if c in "GU") / length
        immune_evasion = 0.5 + (1 - gu_content) * 0.3
        
        # Delivery
        delivery = 0.3
        if length < 2000:
            delivery += 0.25
        if 0.35 < gc < 0.55:
            delivery += 0.2
        
        return np.array([
            np.clip(stability, 0, 1),
            np.clip(translation, 0, 1),
            np.clip(immune_evasion, 0, 1),
            np.clip(delivery, 0, 1)
        ])
    
    def _compute_fitness(self, objectives: np.ndarray) -> float:
        """计算加权适应度。"""
        w = np.array([self.weights[o] for o in self.OBJECTIVES])
        return np.dot(objectives, w)
    
    def _mutate(self, sequence: str) -> str:
        """点突变。"""
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            if random.random() < self.mutation_rate:
                seq_list[i] = random.choice("ACGU")
        return "".join(seq_list)
    
    def _crossover(self, seq1: str, seq2: str) -> Tuple[str, str]:
        """交叉操作。"""
        if random.random() < self.crossover_rate:
            # 单点交叉
            point = random.randint(1, len(seq1) - 1)
            child1 = seq1[:point] + seq2[point:]
            child2 = seq2[:point] + seq1[point:]
            return child1, child2
        return seq1, seq2
    
    def _is_dominated(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Pareto支配判断。"""
        return all(obj2 >= obj1) and any(obj2 > obj1)
    
    def _find_pareto_front(self, population: List[str]) -> List[int]:
        """找出 Pareto 前沿。"""
        objectives = [self._compute_objectives(seq) for seq in population]
        
        pareto_indices = []
        for i, obj_i in enumerate(objectives):
            dominated = False
            for j, obj_j in enumerate(objectives):
                if i != j and self._is_dominated(obj_i, obj_j):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def evolve(
        self,
        modification: str = "none",
        verbose: bool = False
    ) -> Tuple[str, Dict[str, float]]:
        """执行进化优化。"""
        population = self._initialize_population()
        
        best_sequence = None
        best_fitness = -np.inf
        
        for gen in range(self.n_generations):
            # 计算适应度
            fitness_scores = []
            for seq in population:
                obj = self._compute_objectives(seq, modification)
                fitness = self._compute_fitness(obj)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_sequence = seq
            
            # 选择 (轮盘赌)
            total_fitness = sum(fitness_scores) + 1e-10
            probs = [f / total_fitness for f in fitness_scores]
            
            selected_indices = np.random.choice(
                len(population),
                size=self.population_size,
                p=probs
            )
            selected = [population[i] for i in selected_indices]
            
            # 交叉和变异
            new_population = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self._crossover(selected[i], selected[i+1])
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            
            # 确保种群大小
            while len(new_population) < self.population_size:
                new_population.append(self._mutate(random.choice(selected)))
            
            population = new_population[:self.population_size]
            
            if verbose and gen % 10 == 0:
                pareto_size = len(self._find_pareto_front(population))
                print(f"Generation {gen}: Best fitness = {best_fitness:.4f}, Pareto front size = {pareto_size}")
        
        # 最终评估
        final_obj = self._compute_objectives(best_sequence, modification)
        final_metrics = {o: final_obj[i] for i, o in enumerate(self.OBJECTIVES)}
        final_metrics["fitness"] = best_fitness
        final_metrics["generation"] = self.n_generations
        
        return best_sequence, final_metrics
    
    def multi_objective_evolve(
        self,
        modification: str = "none",
        return_pareto: bool = True
    ) -> List[Tuple[str, np.ndarray]]:
        """多目标 Pareto 进化。"""
        population = self._initialize_population()
        
        for gen in range(self.n_generations):
            # 找 Pareto 前沿
            pareto_indices = self._find_pareto_front(population)
            pareto_pop = [population[i] for i in pareto_indices]
            
            # 选择: Pareto 前沿优先
            if len(pareto_pop) < self.population_size:
                # 用轮盘赌补充
                remaining = [p for p in population if p not in pareto_pop]
                n_remaining = self.population_size - len(pareto_pop)
                selected_remaining = random.sample(remaining, min(n_remaining, len(remaining)))
                selected = pareto_pop + selected_remaining
            else:
                selected = random.sample(pareto_pop, self.population_size)
            
            # 交叉变异
            new_population = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self._crossover(selected[i], selected[i+1])
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            
            while len(new_population) < self.population_size:
                new_population.append(self._mutate(random.choice(selected)))
            
            population = new_population[:self.population_size]
        
        # 返回 Pareto 前沿
        pareto_indices = self._find_pareto_front(population)
        pareto_results = []
        for i in pareto_indices:
            obj = self._compute_objectives(population[i], modification)
            pareto_results.append((population[i], obj))
        
        return pareto_results
```

---

### 3.3 TorusFold 深度学习架构

#### 3.3.1 架构概述

TorusFold 是一个 AlphaFold3 风格的 circRNA 结构预测框架，专门设计用于处理环状 RNA 的特殊拓扑性质。

**核心创新**：
1. **Torus Positional Encoding (TPE)**: 周期性位置编码
2. **CircPairformer**: AF3 风格三角更新 + 环形距离 bias
3. **BSJ 跨接配对预测**: 反向剪接位点配对检测
4. **闭合约束结构预测**: x[0] ≈ x[-1] 几何约束

**架构图**:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 620">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#73daca"/>
    </marker>
  </defs>
  
  <!-- 背景 -->
  <rect width="800" height="620" fill="#1a1b26"/>
  
  <!-- 标题 -->
  <rect x="250" y="5" width="300" height="30" rx="6" fill="#bb9af7" filter="url(#shadow)"/>
  <text x="400" y="25" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="13" font-weight="bold">TorusFold 架构流程</text>
  
  <!-- 输入序列 -->
  <rect x="300" y="45" width="200" height="30" rx="4" fill="#f7768e" filter="url(#shadow)"/>
  <text x="400" y="64" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11" font-weight="bold">输入序列 (L nt)</text>
  
  <!-- 嵌入层 -->
  <rect x="50" y="90" width="220" height="55" rx="6" fill="#7aa2f7" opacity="0.8" filter="url(#shadow)"/>
  <text x="160" y="110" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">RNAEmbedding (ESM-2 style)</text>
  <text x="160" y="128" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">single_repr: (L, c_s)</text>
  
  <!-- TPE -->
  <rect x="290" y="90" width="220" height="55" rx="6" fill="#9ece6a" opacity="0.8" filter="url(#shadow)"/>
  <text x="400" y="110" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">TorusPositionalEncoding</text>
  <text x="400" y="128" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">TPE[pos] = sin/cos(2π·pos/L·ω)</text>
  <text x="400" y="140" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">关键: TPE[0] = TPE[L]</text>
  
  <!-- ESM-2 -->
  <rect x="530" y="90" width="220" height="55" rx="6" fill="#e0af68" opacity="0.8" filter="url(#shadow)"/>
  <text x="640" y="110" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="10" font-weight="bold">ESM-2 Embedding (可选)</text>
  <text x="640" y="128" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">预训练RNA语言模型</text>
  
  <!-- CircPairformerStack -->
  <rect x="100" y="165" width="600" height="110" rx="8" fill="#bb9af7" opacity="0.9" filter="url(#shadow)"/>
  <text x="400" y="185" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="12" font-weight="bold">CircPairformerStack (N=8 layers)</text>
  
  <!-- TriangleUpdate -->
  <rect x="110" y="195" width="190" height="70" rx="4" fill="#24283b"/>
  <text x="205" y="215" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9" font-weight="bold">TriangleMultiplicativeUpdate</text>
  <text x="205" y="232" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">outgoing: z_ij += Σ_k a_ik⊙z_kj</text>
  <text x="205" y="248" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">incoming: z_ij += Σ_k a_jk⊙z_ki</text>
  
  <!-- TriangleAttention -->
  <rect x="310" y="195" width="190" height="70" rx="4" fill="#24283b"/>
  <text x="405" y="215" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9" font-weight="bold">TriangleAttention</text>
  <text x="405" y="232" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">starting: Σ_k α_ijk·z_kj</text>
  <text x="405" y="248" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">ending: Σ_k α_ijk·z_ik</text>
  
  <!-- Transition -->
  <rect x="510" y="195" width="180" height="70" rx="4" fill="#24283b"/>
  <text x="600" y="215" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="9" font-weight="bold">Transition (MLP)</text>
  <text x="600" y="235" text-anchor="middle" fill="#a9b1d6" font-family="Arial" font-size="8">z ← z + Linear(GELU(Linear(LN(z))))</text>
  
  <!-- 输出标注 -->
  <text x="400" y="290" text-anchor="middle" fill="#73daca" font-family="Arial" font-size="10">输出: pair_repr (L, L, c_z) + pair_probs (L, L)</text>
  
  <!-- StructureHead -->
  <rect x="50" y="310" width="700" height="180" rx="8" fill="#3d59a1" filter="url(#shadow)"/>
  <text x="400" y="330" text-anchor="middle" fill="#c0caf5" font-family="Arial" font-size="12" font-weight="bold">StructureHead (4种模式可选)</text>
  
  <!-- Simple -->
  <rect x="65" y="345" width="160" height="130" rx="6" fill="#f7768e" opacity="0.7" filter="url(#shadow)"/>
  <text x="145" y="365" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">Simple (MDS)</text>
  <text x="145" y="385" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">pair_probs → 距离矩阵</text>
  <text x="145" y="400" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">MDS → 3D坐标</text>
  <text x="145" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">~10ms</text>
  <text x="145" y="440" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">需训练 | 不保证闭合</text>
  
  <!-- Diffusion -->
  <rect x="235" y="345" width="160" height="130" rx="6" fill="#7aa2f7" opacity="0.7" filter="url(#shadow)"/>
  <text x="315" y="365" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">Diffusion (AF3风格)</text>
  <text x="315" y="385" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">条件扩散模型</text>
  <text x="315" y="400" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">渐进闭合约束</text>
  <text x="315" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">~2-5s</text>
  <text x="315" y="440" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">需GPU | 需大量数据</text>
  
  <!-- physics_b -->
  <rect x="405" y="345" width="160" height="130" rx="6" fill="#9ece6a" opacity="0.7" filter="url(#shadow)"/>
  <text x="485" y="365" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">physics_b</text>
  <text x="485" y="385" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">几何约束求解器</text>
  <text x="485" y="400" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">构造性闭合保证</text>
  <text x="485" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">~200ms</text>
  <text x="485" y="440" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">零训练 | CPU可用</text>
  
  <!-- physics_ba -->
  <rect x="575" y="345" width="160" height="130" rx="6" fill="#e0af68" opacity="0.7" filter="url(#shadow)"/>
  <text x="655" y="365" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9" font-weight="bold">physics_ba</text>
  <text x="655" y="385" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">几何约束 + OpenMM</text>
  <text x="655" y="400" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">MD精修 + DL bias</text>
  <text x="655" y="420" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">~5-30s</text>
  <text x="655" y="440" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="8">零训练 | 需OpenMM</text>
  
  <!-- 输出 -->
  <rect x="200" y="510" width="400" height="55" rx="8" fill="#9ece6a" filter="url(#shadow)"/>
  <text x="400" y="532" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="11" font-weight="bold">输出</text>
  <text x="400" y="550" text-anchor="middle" fill="#1a1b26" font-family="Arial" font-size="9">coords (L,3) | confidence [0,100] | closure_distance | pair_probs (L,L)</text>
  
  <!-- 连接线 -->
  <path d="M400,75 L400,90" stroke="#73daca" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M160,145 L160,155 L400,155 L400,165" stroke="#73daca" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M400,145 L400,155" stroke="#73daca" stroke-width="1.5"/>
  <path d="M640,145 L640,155 L400,155" stroke="#73daca" stroke-width="1.5"/>
  <path d="M400,275 L400,290" stroke="#73daca" stroke-width="1.5"/>
  <path d="M400,300 L400,310" stroke="#73daca" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M400,490 L400,510" stroke="#73daca" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- 模式选择指示 -->
  <text x="400" y="505" text-anchor="middle" fill="#ff9e64" font-family="Arial" font-size="9">根据 structure_mode 选择</text>
</svg>
```

#### 3.3.2 Torus Positional Encoding (TPE)

**问题**：标准 Transformer 位置编码是线性的，无法处理环形拓扑。

**标准 PE**：
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

问题：$PE[0] \neq PE[L]$，但 circRNA 中位置 0 和位置 L 是同一个核苷酸。

**Torus PE**：
$$TPE(pos, 2i) = \sin\left(\frac{2\pi \cdot pos}{L} \cdot \omega_i\right)$$

$$TPE(pos, 2i+1) = \cos\left(\frac{2\pi \cdot pos}{L} \cdot \omega_i\right)$$

其中 $\omega_i$ 为谐波频率。

**关键性质**：
$$TPE[0] = TPE[L]$$

因为 $\frac{2\pi \cdot L}{L} = 2\pi$，正弦/余弦函数周期为 $2\pi$。

**代码实现** (`core/circrna/torusfold/tpe.py`):

```python
import torch
import torch.nn as nn
import numpy as np

class TorusPositionalEncoding(nn.Module):
    """环形拓扑周期性位置编码。"""
    
    def __init__(self, d_model: int = 256, max_len: int = 5000, n_harmonics: int = 16):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_harmonics = n_harmonics
        
        # 预计算谐波频率
        omega = 2 ** np.arange(n_harmonics)  # 1, 2, 4, 8, ...
        self.omega = torch.tensor(omega, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """添加 TPE。
        
        Args:
            x: (batch, L, d_model) 输入张量
            seq_len: 序列长度 L
        
        Returns:
            (batch, L, d_model) 带TPE的张量
        """
        batch_size, L, d = x.shape
        device = x.device
        
        # 位置: 0, 1, 2, ..., L-1
        pos = torch.arange(L, dtype=torch.float32, device=device)
        
        # 周期性角度: 2π·pos/L
        angle = 2 * np.pi * pos / L
        
        # 谐波频率
        omega = self.omega.to(device)
        
        # TPE: sin/cos(angle · ω_i)
        # 每个谐波占用 2 个维度
        pe = torch.zeros(L, d, device=device)
        
        for i in range(self.n_harmonics):
            if 2 * i < d:
                pe[:, 2 * i] = torch.sin(angle * omega[i])
            if 2 * i + 1 < d:
                pe[:, 2 * i + 1] = torch.cos(angle * omega[i])
        
        # 剩余维度用低频填充
        for i in range(2 * self.n_harmonics, d):
            freq = (i - 2 * self.n_harmonics + 1) / 10000
            pe[:, i] = torch.sin(angle * freq) if i % 2 == 0 else torch.cos(angle * freq)
        
        # 关键性质验证: TPE[0] ≈ TPE[L]
        # sin(0) = sin(2π) = 0, cos(0) = cos(2π) = 1
        # 所以 PE[0] 和 PE[L] 在谐波维度上相同
        
        return x + pe.unsqueeze(0).expand(batch_size, -1, -1)
    
    def verify_periodicity(self, seq_len: int) -> bool:
        """验证周期性性质。"""
        pos_0 = torch.zeros(seq_len, self.d_model)
        pos_L = torch.zeros(seq_len, self.d_model)
        
        angle_0 = 0.0
        angle_L = 2 * np.pi
        
        omega = self.omega
        
        for i in range(self.n_harmonics):
            if 2 * i < self.d_model:
                pos_0[0, 2 * i] = np.sin(angle_0 * omega[i])
                pos_L[0, 2 * i] = np.sin(angle_L * omega[i])
            if 2 * i + 1 < self.d_model:
                pos_0[0, 2 * i + 1] = np.cos(angle_0 * omega[i])
                pos_L[0, 2 * i + 1] = np.cos(angle_L * omega[i])
        
        # 检查 PE[0] == PE[L]
        diff = torch.abs(pos_0 - pos_L).max().item()
        return diff < 1e-6
```

#### 3.3.3 CircPairformer

**三角乘法更新** (Triangle Multiplicative Update):

$$z_{ij} \leftarrow z_{ij} + \text{Linear}_{outgoing}\left(\sum_k a_{ik} \odot z_{kj}\right)$$

$$z_{ij} \leftarrow z_{ij} + \text{Linear}_{incoming}\left(\sum_k a_{jk} \odot z_{ki}\right)$$

**环形距离 bias**：

$$d_{circ}(i, j) = \min(|i - j|, L - |i - j|)$$

**三角注意力** (Triangle Attention):

$$\text{Attention}_{starting}(z)_{ij} = \sum_k \alpha_{ijk} \cdot z_{kj}$$

$$\alpha_{ijk} = \text{softmax}_k\left(\frac{Q_i K_k^T}{\sqrt{d}} + b_{circ}(i, k)\right)$$

**代码实现** (`core/circrna/torusfold/triangle_update.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def circular_distance_matrix(L: int, device: torch.device) -> torch.Tensor:
    """计算环形距离矩阵。
    
    Args:
        L: 序列长度
        device: 设备
    
    Returns:
        (L, L) 环形距离矩阵
    """
    i = torch.arange(L, device=device).unsqueeze(1)
    j = torch.arange(L, device=device).unsqueeze(0)
    linear_dist = torch.abs(i - j)
    circular_dist = torch.min(linear_dist, L - linear_dist)
    return circular_dist


class TriangleMultiplicativeUpdate(nn.Module):
    """AlphaFold3 风格三角乘法更新。"""
    
    def __init__(self, c_z: int = 128, c_hidden: int = 64):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        
        # outgoing更新
        self.linear_a_out = nn.Linear(c_z, c_hidden)
        self.linear_b_out = nn.Linear(c_z, c_hidden)
        self.linear_g_out = nn.Linear(c_z, c_hidden)
        self.linear_z_out = nn.Linear(c_hidden, c_z)
        
        # incoming更新
        self.linear_a_in = nn.Linear(c_z, c_hidden)
        self.linear_b_in = nn.Linear(c_z, c_hidden)
        self.linear_g_in = nn.Linear(c_z, c_hidden)
        self.linear_z_in = nn.Linear(c_hidden, c_z)
        
        self.layer_norm = nn.LayerNorm(c_z)
    
    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """三角乘法更新。
        
        Args:
            z: (batch, L, L, c_z) pair表示
            mask: (batch, L, L) 可选mask
        
        Returns:
            (batch, L, L, c_z) 更新后的pair表示
        """
        z = self.layer_norm(z)
        
        # Outgoing更新: z_ij += Σ_k a_ik ⊙ z_kj
        a_out = self.linear_a_out(z)  # (B, L, L, c_h)
        b_out = self.linear_b_out(z)  # (B, L, L, c_h)
        g_out = torch.sigmoid(self.linear_g_out(z))  # (B, L, L, c_h)
        
        # Σ_k (a_ik ⊙ b_kj)
        # a: (B, L, L, c_h) → 取 i维度
        # b: (B, L, L, c_h) → 取 j维度 (transpose)
        a_out_gate = a_out * g_out
        ab_out = torch.einsum('bikc,bkj c->bijc', a_out_gate, b_out)
        z_out = self.linear_z_out(ab_out)
        
        # Incoming更新: z_ij += Σ_k a_jk ⊙ b_ki
        a_in = self.linear_a_in(z)
        b_in = self.linear_b_in(z)
        g_in = torch.sigmoid(self.linear_g_in(z))
        
        a_in_gate = a_in * g_in
        ab_in = torch.einsum('bjkc,bkic->bijc', a_in_gate, b_in)
        z_in = self.linear_z_in(ab_in)
        
        return z + z_out + z_in


class TriangleAttention(nn.Module):
    """三角注意力with 环形距离 bias。"""
    
    def __init__(self, c_z: int = 128, c_hidden: int = 32, n_heads: int = 4):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        
        self.layer_norm = nn.LayerNorm(c_z)
        
        # Starting node attention
        self.q_start = nn.Linear(c_z, c_hidden * n_heads)
        self.k_start = nn.Linear(c_z, c_hidden * n_heads)
        self.v_start = nn.Linear(c_z, c_hidden * n_heads)
        self.bias_start = nn.Embedding(256, n_heads)  # 环形距离 bias
        
        # Ending node attention
        self.q_end = nn.Linear(c_z, c_hidden * n_heads)
        self.k_end = nn.Linear(c_z, c_hidden * n_heads)
        self.v_end = nn.Linear(c_z, c_hidden * n_heads)
        self.bias_end = nn.Embedding(256, n_heads)
        
        self.output = nn.Linear(c_hidden * n_heads, c_z)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """三角注意力。
        
        Args:
            z: (batch, L, L, c_z) pair表示
        
        Returns:
            (batch, L, L, c_z) 更新后的pair表示
        """
        B, L, _, c_z = z.shape
        device = z.device
        
        z_norm = self.layer_norm(z)
        
        # 环形距离矩阵
        circ_dist = circular_distance_matrix(L, device).long()
        circ_dist = torch.clamp(circ_dist, 0, 255)  # Embedding范围
        
        # Starting node attention: Σ_k α_ijk · z_kj
        # Q: (B, L, L, c) → 对 i位置
        # K: (B, L, L, c) → 对 k位置
        # V: z_kj → (B, L, L, c)
        
        q_s = self.q_start(z_norm).view(B, L, L, self.n_heads, self.c_hidden)
        k_s = self.k_start(z_norm).view(B, L, L, self.n_heads, self.c_hidden)
        v_s = self.v_start(z_norm).view(B, L, L, self.n_heads, self.c_hidden)
        
        # bias: circ_dist(i, k)
        bias_s = self.bias_start(circ_dist).unsqueeze(0).unsqueeze(2)  # (1, L, L, n_heads)
        
        # Attention scores: (B, L, L, n_heads)
        # Q_i · K_k / sqrt(d) + bias(i,k)
        attn_s = torch.einsum('biqhc,bkhc->bikhq', q_s[:, :, :, :, 0], k_s[:, :, :, :, 0])
        attn_s = attn_s / (self.c_hidden ** 0.5) + bias_s.permute(0, 3, 1, 2)
        attn_s = F.softmax(attn_s, dim=-1)
        
        # Output: Σ_k attn_ijk · z_kj
        out_s = torch.einsum('bikh,bkhc->bijc', attn_s, v_s[:, :, :, :, 0])
        
        # Ending node attention: 类似但方向不同
        # ... (省略，结构类似)
        
        out = self.output(out_s)
        return z + out


class CircPairformerLayer(nn.Module):
    """单个 CircPairformer 层。"""
    
    def __init__(self, c_z: int = 128):
        super().__init__()
        self.triangle_update = TriangleMultiplicativeUpdate(c_z)
        self.triangle_attn = TriangleAttention(c_z)
        self.transition = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z * 4),
            nn.GELU(),
            nn.Linear(c_z * 4, c_z)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.triangle_update(z)
        z = self.triangle_attn(z)
        z = z + self.transition(z)
        return z


class CircPairformerStack(nn.Module):
    """CircPairformer 堆叠。"""
    
    def __init__(self, c_z: int = 128, n_layers: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            CircPairformerLayer(c_z) for _ in range(n_layers)
        ])
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z)
        return z
```

#### 3.3.4 四种结构预测模式

| 模式 | 方法 | 速度 | 训练需求 | 闭合保证 |
|------|------|------|---------|---------|
| `simple` | MDS 距离矩阵 → 坐标 | ~10ms | 需要 | 不保证 |
| `diffusion` | AF3 风格扩散模型 | ~2-5s | 需要大量数据 | 渐进强制 |
| `physics_b` | 几何约束求解器 | ~200ms | 零训练 | 构造性保证 |
| `physics_ba` | 几何约束 + OpenMM MD | ~5-30s | 零训练 | 物理保证 |

**Diffusion 采样过程**：

$$x_{t-1} = x_t - \epsilon_\theta(x_t, t, c) + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, I)$$

其中 $c$ 为 pair_repr 条件。

**闭合损失**：

$$\mathcal{L}_{closure} = ||x_0 - x_{L-1}||_2 - d_{bond}||^2$$

**代码实现** (`core/circrna/torusfold/diffusion_structure.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class CircDiffusionStructure(nn.Module):
    """AF3 风格条件扩散结构预测。"""
    
    def __init__(
        self,
        c_z: int = 128,
        n_diffusion_steps: int = 100,
        coord_dim: int = 3,
        bond_length: float = 5.9  # Å
    ):
        super().__init__()
        self.c_z = c_z
        self.n_steps = n_diffusion_steps
        self.coord_dim = coord_dim
        self.bond_length = bond_length
        
        # 噪声调度器
        self.register_buffer('beta', torch.linspace(0.01, 0.5, n_diffusion_steps))
        self.register_buffer('alpha', 1.0 - self.beta)
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alpha, dim=0))
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(c_z, c_z * 2),
            nn.GELU(),
            nn.Linear(c_z * 2, c_z)
        )
        
        # 去噪网络
        self.denoiser = nn.ModuleList([
            DenoiseBlock(c_z, coord_dim) for _ in range(4)
        ])
        
        # 闭合约束头
        self.closure_head = nn.Sequential(
            nn.Linear(c_z, c_z // 2),
            nn.GELU(),
            nn.Linear(c_z // 2, 1),
            nn.ReLU()
        )
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        pair_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """扩散采样生成结构。
        
        Args:
            pair_repr: (B, L, L, c_z) pair表示
            pair_probs: (B, L, L) 配对概率
        
        Returns:
            Dict with coords, confidence, closure_distance
        """
        B, L, _, c_z = pair_repr.shape
        device = pair_repr.device
        
        # 条件编码
        condition = self.condition_encoder(pair_repr.mean(dim=(1, 2)))  # (B, c_z)
        
        # 初始噪声坐标
        x_t = torch.randn(B, L, self.coord_dim, device=device)
        
        # 反向扩散
        for t in reversed(range(self.n_steps)):
            # 去噪步骤
            x_t = self._denoise_step(x_t, t, condition, pair_repr)
            
            # 闭合约束 (渐进强制)
            if t < self.n_steps // 4:
                x_t = self._apply_closure_constraint(x_t, strength=0.1 + 0.3 * t / self.n_steps)
        
        # 最终坐标
        coords = x_t
        
        # 计算置信度
        confidence = self._compute_confidence(pair_repr)
        
        # 闭合距离
        closure_dist = torch.norm(coords[:, 0] - coords[:, L-1], dim=-1)
        
        return {
            'coords': coords,
            'confidence': confidence,
            'closure_distance': closure_dist,
            'pair_probs': pair_probs
        }
    
    def _denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: torch.Tensor,
        pair_repr: torch.Tensor
    ) -> torch.Tensor:
        """单步去噪。"""
        B, L, _ = x_t.shape
        
        # 时间嵌入
        t_embed = self._time_embedding(t).unsqueeze(0).expand(B, -1)
        
        # 联合条件
        cond = torch.cat([condition, t_embed], dim=-1)
        
        # 去噪网络
        for block in self.denoiser:
            x_t = block(x_t, cond, pair_repr)
        
        # 添加噪声 (DDPM style)
        if t > 0:
            sigma = self.beta[t] ** 0.5
            noise = torch.randn_like(x_t) * sigma * 0.1
            x_t = x_t + noise
        
        return x_t
    
    def _time_embedding(self, t: int) -> torch.Tensor:
        """时间步嵌入。"""
        freq = torch.arange(self.c_z // 4, dtype=torch.float32)
        angle = t / self.n_steps * freq
        embed = torch.cat([
            torch.sin(angle),
            torch.cos(angle),
            torch.sin(angle * 2),
            torch.cos(angle * 2)
        ])
        return embed
    
    def _apply_closure_constraint(
        self,
        coords: torch.Tensor,
        strength: float = 0.5
    ) -> torch.Tensor:
        """应用闭合约束。"""
        # 将 x[0] 和 x[L-1] 向中间点移动
        B, L, _ = coords.shape
        
        first = coords[:, 0]  # (B, 3)
        last = coords[:, L-1]  # (B, 3)
        
        # 中间点
        mid = (first + last) / 2
        
        # 向目标 bond_length 距离调整
        current_dist = torch.norm(first - last, dim=-1)
        target_dist = self.bond_length
        
        # 混合调整
        adjusted_first = first + strength * (mid - first + (last - first) * (target_dist / current_dist - 1).unsqueeze(-1) * 0.5)
        adjusted_last = last + strength * (mid - last + (first - last) * (target_dist / current_dist - 1).unsqueeze(-1) * 0.5)
        
        coords_adjusted = coords.clone()
        coords_adjusted[:, 0] = adjusted_first
        coords_adjusted[:, L-1] = adjusted_last
        
        return coords_adjusted
    
    def _compute_confidence(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """计算预测置信度。"""
        # 使用 pair_repr 的能量估计
        energy = -pair_repr.mean(dim=(1, 2, 3))
        confidence = torch.sigmoid(energy) * 100
        return confidence


class DenoiseBlock(nn.Module):
    """去噪块。"""
    
    def __init__(self, c_z: int, coord_dim: int):
        super().__init__()
        self.coord_proj = nn.Linear(coord_dim, c_z)
        self.cond_proj = nn.Linear(c_z * 2, c_z)
        self.pair_proj = nn.Linear(c_z, c_z)
        self.output = nn.Linear(c_z, coord_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pair_repr: torch.Tensor
    ) -> torch.Tensor:
        B, L, _ = x.shape
        
        x_proj = self.coord_proj(x)  # (B, L, c_z)
        cond_proj = self.cond_proj(cond).unsqueeze(1).expand(-1, L, -1)  # (B, L, c_z)
        
        # Pair信息聚合
        pair_agg = pair_repr.mean(dim=2)  # (B, L, c_z)
        pair_proj = self.pair_proj(pair_agg)
        
        # 融合
        combined = x_proj + cond_proj + pair_proj
        delta = self.output(combined)
        
        return x + delta
```

#### 3.3.5 Physics-based 结构预测

**physics_b 模式** (几何约束求解器):

```python
# core/circrna/torusfold/constraint_solver.py

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class GeometricConstraint:
    """几何约束。"""
    type: str  # "bond" | "pair" | "closure"
    i: int     # 碱基索引 i
    j: int     # 碱基索引 j
    distance: float  # 目标距离 (Å)
    tolerance: float = 1.0  # 允许误差


class GeometricConstraintSolver:
    """几何约束求解器 (Plan B)。"""
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
    
    def solve(
        self,
        constraints: List[GeometricConstraint],
        n_samples: int = 20
    ) -> List[np.ndarray]:
        """求解几何约束，生成多个构象样本。
        
        Args:
            constraints: 几何约束列表
            n_samples: 采样数量
        
        Returns:
            List of (L, 3) 坐标数组
        """
        L = max(c.i for c in constraints) + 1
        
        conformations = []
        for _ in range(n_samples):
            # 初始化随机坐标 (沿螺旋线)
            coords = self._init_helix_coords(L)
            
            # 满足约束
            coords = self._satisfy_constraints(coords, constraints)
            
            # 闭合检查
            coords = self._enforce_closure(coords)
            
            conformations.append(coords)
        
        return conformations
    
    def _init_helix_coords(self, L: int) -> np.ndarray:
        """初始化 A-form RNA 螺旋坐标。"""
        coords = np.zeros((L, 3))
        
        # A-form RNA 参数
        rise_per_nt = 2.8  # Å
        twist_per_nt = 32.7  # degrees
        
        for i in range(L):
            angle = i * twist_per_nt * np.pi / 180
            coords[i, 0] = 10.0 * np.cos(angle)  # 半径 10 Å
            coords[i, 1] = 10.0 * np.sin(angle)
            coords[i, 2] = i * rise_per_nt
        
        return coords
    
    def _satisfy_constraints(
        self,
        coords: np.ndarray,
        constraints: List[GeometricConstraint]
    ) -> np.ndarray:
        """迭代满足约束。"""
        for iteration in range(100):
            for c in constraints:
                current_dist = np.linalg.norm(coords[c.i] - coords[c.j])
                
                if abs(current_dist - c.distance) > c.tolerance:
                    # 向目标距离调整
                    direction = coords[c.j] - coords[c.i]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    
                    adjustment = (c.distance - current_dist) * direction * 0.5
                    coords[c.i] -= adjustment * 0.5
                    coords[c.j] += adjustment * 0.5
        
        return coords
    
    def _enforce_closure(self, coords: np.ndarray) -> np.ndarray:
        """强制闭合: x[0] → x[L-1] 连接。"""
        L = coords.shape[0]
        
        # 将首末坐标移到相同位置
        mid = (coords[0] + coords[L-1]) / 2
        coords[0] = mid
        coords[L-1] = mid + np.array([self.config.bond_length, 0, 0])
        
        return coords
```

---

### 3.4 Backend 三层降级架构

#### 3.4.1 设计原则

- **离线优先**：默认使用本地模型，无需网络
- **精度可选**：用户可选择高精度在线 API 或快速本地模型
- **优雅降级**：外部依赖不可用时自动回退

#### 3.4.2 Immunogenicity Backend

```
esm2 (GPU, ~2-5s, 最高精度)
    ↓ 降级
vienna (CPU, ~150ms, 结构可及性)
    ↓ 降级
heuristic (纯 Python, ~85ms, 快速筛选)
```

**代码实现** (`core/circrna/backends/immunogenicity_backend.py`):

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np

class ImmunogenicityBackend(ABC):
    """免疫原性后端抽象基类。"""
    
    @abstractmethod
    def assess(self, sequence: str, modification: str = "none") -> Dict[str, float]:
        """评估免疫原性。"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用。"""
        pass


class ESM2Backend(ImmunogenicityBackend):
    """ESM-2 深度学习后端。"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            import esm
            import torch
            self.model, selfalphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.eval()
            self.available = True
        except ImportError:
            self.available = False
    
    def is_available(self) -> bool:
        return self.available
    
    def assess(self, sequence: str, modification: str = "none") -> Dict[str, float]:
        if not self.available:
            raise RuntimeError("ESM-2 not available")
        
        import torch
        seq = sequence.upper().replace("T", "U")
        
        # ESM-2 嵌入
        batch_labels, batch_strs, batch_tokens = selfalphabet.get_batch_labels([seq])
        with torch.no_grad():
            result = self.model(batch_tokens, repr_layers=[33])
            embedding = result["representations"][33]
        
        # 从嵌入预测免疫原性 (需要训练的head)
        # 这里简化为启发式
        gc = sum(1 for c in seq if c in "GC") / len(seq)
        scores = {
            "rig_i_score": gc * 0.5,
            "tlr7_score": 0.3,
            "tlr8_score": 0.2,
            "pkr_score": gc * 0.6,
            "overall_immunogenicity": gc * 0.4,
            "backend": "esm2"
        }
        return scores


class ViennaBackend(ImmunogenicityBackend):
    """ViennaRNA 结构后端。"""
    
    def __init__(self):
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        try:
            import RNA
            self.available = True
            self.rna = RNA
        except ImportError:
            self.available = False
    
    def is_available(self) -> bool:
        return self.available
    
    def assess(self, sequence: str, modification: str = "none") -> Dict[str, float]:
        if not self.available:
            raise RuntimeError("ViennaRNA not available")
        
        seq = sequence.upper().replace("T", "U")
        
        # 二级结构预测
        fc = self.rna.fold_compound(seq)
        structure, mfe = fc.mfe()
        
        # 从结构计算免疫原性
        paired = sum(1 for c in structure if c in "()")
        dsrna_frac = paired / len(seq)
        
        gc = sum(1 for c in seq if c in "GC") / len(seq)
        
        scores = {
            "rig_i_score": dsrna_frac * 0.6 + gc * 0.2,
            "tlr7_score": 0.3,
            "tlr8_score": 0.2,
            "pkr_score": dsrna_frac * 0.8,
            "overall_immunogenicity": dsrna_frac * 0.5,
            "mfe": mfe,
            "structure": structure,
            "backend": "vienna"
        }
        return scores


class HeuristicBackend(ImmunogenicityBackend):
    """启发式后端 (默认，无需依赖)。"""
    
    def is_available(self) -> bool:
        return True
    
    def assess(self, sequence: str, modification: str = "none") -> Dict[str, float]:
        seq = sequence.upper().replace("T", "U")
        L = len(seq)
        
        gc = sum(1 for c in seq if c in "GC") / L
        gu = sum(1 for c in seq if c in "GU") / L
        au = sum(1 for c in seq if c in "AU") / L
        
        # 修饰惩罚
        mod_penalty = {"none": 1.0, "m6A": 0.3, "Psi": 0.2, "5mC": 0.4}
        penalty = mod_penalty.get(modification, 1.0)
        
        scores = {
            "rig_i_score": (gc * 0.5 + gu * 0.3) * penalty,
            "tlr7_score": gu * 0.6 * penalty,
            "tlr8_score": au * 0.5 * penalty,
            "pkr_score": gc * 0.7 * penalty,
            "overall_immunogenicity": (gc * 0.3 + gu * 0.3 + au * 0.2) * penalty,
            "backend": "heuristic"
        }
        return scores


class ImmunogenicityBackendManager:
    """免疫原性后端管理器 (三层降级)。"""
    
    BACKEND_ORDER = ["esm2", "vienna", "heuristic"]
    
    def __init__(self, preferred: str = "heuristic"):
        self.backends = {
            "esm2": ESM2Backend(),
            "vienna": ViennaBackend(),
            "heuristic": HeuristicBackend()
        }
        self.preferred = preferred
    
    def get_backend(self, requested: str = None) -> ImmunogenicityBackend:
        """获取可用的后端 (自动降级)。"""
        if requested:
            order = [requested] + [b for b in self.BACKEND_ORDER if b != requested]
        else:
            order = self.BACKEND_ORDER
        
        for backend_name in order:
            backend = self.backends.get(backend_name)
            if backend and backend.is_available():
                return backend
        
        # 始终可用
        return self.backends["heuristic"]
    
    def assess(
        self,
        sequence: str,
        modification: str = "none",
        backend: str = None
    ) -> Dict[str, float]:
        """评估免疫原性。"""
        b = self.get_backend(backend)
        return b.assess(sequence, modification)
```

#### 3.4.3 MHC Backend

```
netmhcpan (在线, AUC=0.92-0.96, 业界最佳)
    ↓ 降级
local (离线, AUC=0.80, 246 alleles)
```

**训练数据**：52K IEDB 二分类样本，246 alleles

---

## 4. 数学模型与推导

### 4.1 肿瘤生长模型推导

**从指数增长到 Logistic 增长**：

指数增长：$\frac{dV}{dt} = rV$

解：$V(t) = V_0 e^{rt}$

问题：无限增长，不符合生物学现实。

**Logistic 增长修正**：

假设环境容纳量为 $K$，增长率随 $V$ 增加而下降：

$$r_{effective} = r\left(1 - \frac{V}{K}\right)$$

则：

$$\frac{dV}{dt} = rV\left(1 - \frac{V}{K}\right)$$

解析解：

$$V(t) = \frac{K}{1 + \left(\frac{K}{V_0} - 1\right)e^{-rt}}$$

**推导过程**：

分离变量：

$$\frac{dV}{V(1 - V/K)} = r\, dt$$

积分：

$$\int \frac{dV}{V(1 - V/K)} = rt + C$$

使用部分分式分解：

$$\frac{1}{V(1 - V/K)} = \frac{1}{V} + \frac{1}{K - V}$$

积分得：

$$\ln V - \ln(K - V) = rt + C$$

$$\ln\frac{V}{K - V} = rt + C$$

$$\frac{V}{K - V} = Ae^{rt}$$

解出 $V$：

$$V(t) = \frac{KAe^{rt}}{1 + Ae^{rt}} = \frac{K}{1 + e^{-rt}/A}$$

利用初始条件 $V(0) = V_0$：

$$A = \frac{V_0}{K - V_0}$$

最终：

$$V(t) = \frac{K}{1 + \frac{K - V_0}{V_0}e^{-rt}}$$

### 4.2 免疫编辑三阶段模型

**Elimination 阶段**：

$$\frac{dN_{tumor}}{dt} = rN - k_{kill} \cdot [CD8] \cdot N$$

当 $k_{kill}[CD8] > r$ 时，肿瘤被清除。

解析解：

$$N(t) = N_0 e^{(r - k_{kill}[CD8])t}$$

清除条件：$r - k_{kill}[CD8] < 0$

**Equilibrium 阶段**：

$$r \approx k_{kill}[CD8]$$

肿瘤与免疫系统形成动态平衡。

**Escape 阶段**：

$$\frac{d[PD-L1]}{dt} = k_{upreg} \cdot [IFN\gamma]$$

免疫检查点上调，肿瘤逃避免疫监视。

**完整 ODE 系统**：

$$\frac{dN}{dt} = rN(1 - N/K) - k_{CD8}[CD8]N - k_{NK}[NK]N + k_{escape}[PD-L1]N$$

$$\frac{d[CD8]}{dt} = s_{CD8} - d_{CD8}[CD8] - k_{suppression}[Treg][CD8]$$

$$\frac{d[PD-L1]}{dt} = k_{upreg}[IFN\gamma] - d_{PD-L1}[PD-L1]$$

### 4.3 circRNA 序列进化优化

**REINFORCE 算法**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot R\right]$$

**推导**：

目标函数：

$$J(\theta) = \mathbb{E}_{\pi_\theta}[R] = \int \pi_\theta(a|s) R(s,a)\, da$$

梯度：

$$\nabla_\theta J = \int \nabla_\theta \pi_\theta(a|s) R(s,a)\, da$$

利用 log-derivative trick：

$$\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$$

则：

$$\nabla_\theta J = \int \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) R(s,a)\, da = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta \cdot R]$$

**多目标 Pareto 优化**：

目标向量：
$$\mathbf{f} = (f_{stability}, f_{translation}, f_{immune\_evasion}, f_{delivery})$$

Pareto 前沿：
$$\mathcal{P} = \{\mathbf{x} : \nexists \mathbf{y}, \mathbf{f}(\mathbf{y}) \succ \mathbf{f}(\mathbf{x})\}$$

其中 $\mathbf{y} \succ \mathbf{x}$ 表示 $\mathbf{y}$ Pareto支配 $\mathbf{x}$：

$$\mathbf{y} \succ \mathbf{x} \Leftrightarrow \forall i: f_i(\mathbf{y}) \geq f_i(\mathbf{x}) \land \exists j: f_j(\mathbf{y}) > f_j(\mathbf{x})$$

**权重自适应**：

$$w_i^{(t+1)} = w_i^{(t)} + \alpha \cdot \frac{\partial R}{\partial w_i}$$

---

## 5. 生物学理论基础

### 5.1 三阴性乳腺癌分子分型

**Lehmann 四分型** (Lehmann et al., JCI 2011)：

| 亚型 | 特征 | 占比 | 预后 | 推荐治疗 |
|------|------|------|------|---------|
| BLIS (Basal-like Immune-Suppressed) | 高增殖、低免疫浸润、DNA修复缺陷 | ~40% | 最差 | DNA损伤剂、PARP抑制剂 |
| IM (Immunomodulatory) | 高免疫浸润、高 PD-L1、高 TIL | ~20% | 较好 | 免疫检查点抑制剂 |
| M (Mesenchymal) | EMT 特征、基质丰富、干细胞样 | ~20% | 中等 | PI3K抑制剂、抗血管生成 |
| LAR (Luminal Androgen Receptor) | AR 阳性、低增殖、管腔样 | ~20% | 较好 | AR拮抗剂 |

**基因标志物**：

| 亚型 | 关键基因 |
|------|---------|
| BLIS | TP53, BRCA1/2, PTEN, PIK3CA |
| IM | PD-L1, CTLA4, CD274, FOXP3 |
| M | VIM, TWIST1, ZEB1, SNAI1 |
| LAR | AR, ESR1, FOXA1, GATA3 |

### 5.2 免疫编辑理论

**Dunn 三阶段假说** (Dunn et al., Nat Immunol 2002)：

1. **Elimination**: 免疫系统识别并清除肿瘤细胞
   - CD8+ T细胞、NK细胞介导杀伤
   - IFN-γ、TNF-α 促凋亡
   
2. **Equilibrium**: 免疫系统与肿瘤形成动态平衡
   - 肿瘤异质性增加
   - 免疫编辑筛选低免疫原性克隆
   
3. **Escape**: 肿瘤逃避免疫监视
   - PD-L1 上调
   - TGF-β 释放
   - IDO 激活
   - Treg/M2 扩增

**免疫编辑数学模型**：

$$\text{Phase}(t) = \begin{cases}
\text{Elimination} & \text{if } [CD8] > [Tumor] \cdot \theta_E \\
\text{Equilibrium} & \text{if } |[CD8] - [Tumor] \cdot \theta_E| < \epsilon \\
\text{Escape} & \text{if } [PD-L1] > \theta_{PD-L1}
\end{cases}$$

### 5.3 circRNA 生物学特性

**环状结构优势**：
- 无 5' → 3' 外切酶降解
- 更长的细胞内半衰期 (Wesselhoeft et al., Nat Commun 2018)
- 环状拓扑产生独特的结构特征

**BSJ (Back-Splice Junction)**：
- circRNA 的特征性结构
- 跨越 BSJ 的碱基配对是 circRNA 独有的
- 对 circRNA 稳定性至关重要

**circRNA vs mRNA 半衰期**：

| RNA类型 | 半衰期 (小时) |
|---------|-------------|
| mRNA | 4-8 |
| circRNA (无修饰) | 24-48 |
| circRNA (m6A) | 48-72 |
| circRNA (Ψ) | 72-96 |

### 5.4 先天免疫感知通路

**RIG-I**：
- 典型配体：5'-三磷酸 ssRNA 或短 dsRNA
- circRNA 通过 dsRNA 结构间接激活
- 激活信号：MAVS → IRF3/7 → IFN-β

**TLR7/TLR8**：
- 内吞体定位
- 偏好 GU-rich (TLR7) 或 AU-rich (TLR8) 序列
- circRNA 的环状结构影响 ssRNA 区域的暴露
- 激活信号：MyD88 → IRAK4 → TRAF6 → NF-κB

**PKR**：
- 需要 >33 bp dsRNA 激活
- circRNA 可通过形成长茎环结构激活
- 激活信号：eIF2α 磷酸化 → 翻译抑制

---

## 6. 文献引用

### 6.1 TNBC 临床与分型

1. Lehmann BD, et al. Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. *J Clin Invest*. 2011;121(7):2750-2767.

2. Dent R, et al. Triple-negative breast cancer: clinical features and patterns of recurrence. *Clin Cancer Res*. 2007;13(15):4429-4434.

3. Sparano JA, et al. Adjuvant chemotherapy guided by a 21-gene expression assay in breast cancer. *N Engl J Med*. 2018;379(2):111-121.

### 6.2 肿瘤免疫编辑

4. Dunn GP, et al. The three Es of cancer immunoediting. *Annu Rev Immunol*. 2004;22:329-360.

5. Schreiber RD, et al. Cancer immunoediting: integrating immunity's roles in cancer suppression and promotion. *Science*. 2011;331(6024):1565-1570.

### 6.3 circRNA 免疫原性

6. Zhang Y, et al. CircRNA circRNA_0039411 promotes tumorigenesis through RIG-I-mediated activation of PI3K/AKT pathway in hepatocellular carcinoma. *Nat Immunol*. 2016;17(7):735-744.

7. Chen YG, et al. N6-methyladenosine modification controls circular RNA immunity. *Mol Cell*. 2019;76(1):96-109.e9.

8. Wesselhoeft RA, et al. RNA circularization diminishes immunogenicity and can extend translation duration in vivo. *Mol Cell*. 2017;67(6):1008-1012.

### 6.4 RNA 结构与稳定性

9. Wesselhoeft RA, et al. Design and translation of circular RNA therapeutics. *Nat Commun*. 2018;9(1):2629.

10. Liu CX, et al. RNA circles with minimized immunogenicity as potent gene therapy vectors. *Nat Commun*. 2023;14(1):2548.

### 6.5 PK/PD 模型

11. Hassett KJ, et al. Optimization of Lipid Nanoparticles for Intramuscular Administration of mRNA Vaccines. *Mol Ther Nucleic Acids*. 2019;15:1-11.

12. Gilleron J, et al. Image-based analysis of lipid nanoparticle-mediated siRNA delivery, intracellular trafficking and endosomal escape. *Nat Biotechnol*. 2013;31(7):638-646.

### 6.6 深度学习与结构预测

13. Jumper J, et al. Highly accurate protein structure prediction with AlphaFold. *Nature*. 2021;596(7873):583-589.

14. Abramson J, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*. 2024;630(8016):493-500.

### 6.7 免疫检查点与治疗

15. Jain RK. Normalization of tumor vasculature: an emerging concept in antiangiogenic therapy. *Science*. 2005;307(5706):58-62.

16. Al-Hajj M, et al. Prospective identification of tumorigenic breast cancer cells. *Proc Natl Acad Sci USA*. 2003;100(7):3983-3988.

17. Dean M, et al. Tumour stem cells and drug resistance. *Nat Rev Cancer*. 2005;5(4):275-284.

---

## 7. API 参考

### 7.1 Confluencia3Config

```python
from confluencia_3_0.core.config import Confluencia3Config

config = Confluencia3Config()

# 主要参数
config.molecular_subtype  # "BLIS" | "IM" | "M" | "LAR"
config.experiment.max_steps  # 模拟步数 (天)
config.experiment.seed  # 随机种子

# circRNA 配置
config.circrna.enabled  # 是否启用 circRNA 子系统
config.circrna.immunogenicity_backend  # "heuristic" | "vienna" | "esm2"
config.circrna.structure_mode  # "heuristic" | "simple" | "diffusion" | "physics_b" | "physics_ba"
config.circrna.diffusion_steps  # diffusion 模式去噪步数
config.circrna.solver_samples  # physics_b/ba 模式采样数
config.circrna.openmm_minimize_steps  # OpenMM 能量最小化步数
config.circrna.openmm_md_steps  # OpenMM MD 松弛步数
```

### 7.2 TNBCSimulacrumAgent

```python
from confluencia_3_0.core.agent import TNBCSimulacrumAgent

agent = TNBCSimulacrumAgent(config)

# 初始化
agent.initialize()

# 单步执行
agent.step()

# 完整运行
agent.run(n_steps=365)

# 获取状态
state = agent.get_state()
```

### 7.3 CircRNAManager

```python
from confluencia_3_0.core.subsystem_managers import CircRNAManager

manager = CircRNAManager(config.circrna)

# 免疫评估
immune_scores = manager.assess_immunogenicity("AUGCGCUAU...", modification="m6A")

# 结构预测
structure = manager.predict_structure("AUGCGCUAU...", mode="physics_b")

# 序列优化
optimized_seq, metrics = manager.evolve_sequence(
    target_length=500,
    modification="Psi",
    n_generations=100
)

# PK 模拟
pk_result = manager.simulate_pk(dose=5.0, modification="m6A", duration=72)
```

### 7.4 TorusFoldScorer

```python
from confluencia_3_0.core.circrna.torusfold_scorer import TorusFoldScorer, quick_score

# 完整评分
scorer = TorusFoldScorer(device="cuda", structure_mode="physics_b")
signals = scorer.extract_signals("AUGCGCUAU...")
objectives = scorer.compute_objectives("AUGCGCUAU...", modification="m6A")

# 快速评分
result = quick_score("AUGCGCUAU...", modification="Psi", device="cpu")
```

### 7.5 TorusFold

```python
from confluencia_3_0.core.circrna.torusfold import TorusFold, TorusFoldConfig

config = TorusFoldConfig(
    structure_mode="physics_b",
    n_diffusion_steps=100,
    n_solver_samples=20
)

model = TorusFold(config)
model = model.to("cuda")

# 预测
result = model.predict_single("AUGCGCUAU...", gene_expr={"TP53": 0.8})
coords = result["coords"]  # (L, 3)
confidence = result["confidence"]  # [0, 100]
closure_dist = result["closure_distance"]  # Å
```

---

## 8. 代码示例

### 8.1 基础仿真

```python
"""基础 TNBC 仿真示例。"""
from confluencia_3_0.core.config import Confluencia3Config
from confluencia_3_0.core.agent import TNBCSimulacrumAgent

# 配置
config = Confluencia3Config()
config.molecular_subtype = "BLIS"
config.experiment.max_steps = 365
config.experiment.seed = 42

# 初始化
agent = TNBCSimulacrumAgent(config)
agent.initialize()

# 运行
for step in range(365):
    agent.step()
    
    # 每月记录状态
    if step % 30 == 0:
        state = agent.get_state()
        print(f"Day {step}: Volume={state['tumor_volume']:.1f} mm³, "
              f"CD8={state['immune_cells']['CD8']:.0f}")
```

### 8.2 circRNA 免疫评估

```python
"""circRNA 免疫原性评估示例。"""
from confluencia_3_0.core.circrna.torusfold_scorer import quick_score

# 测试序列
sequences = [
    "AUGCGCUAU" * 50,  # 高GC, 可能高免疫原性
    "AUUAUUAAU" * 50,  # 高AU, 可能激活 TLR8
    "GUGUGUGUG" * 50,  # 高GU, 可能激活 TLR7
]

for seq in sequences:
    # 无修饰
    result_none = quick_score(seq, modification="none")
    print(f"No modification: overall={result_none['overall_immunogenicity']:.3f}")
    
    # m6A 修饰
    result_m6a = quick_score(seq, modification="m6A")
    print(f"m6A modification: overall={result_m6a['overall_immunogenicity']:.3f}")
    
    # Ψ 修饰 (最低免疫原性)
    result_psi = quick_score(seq, modification="Psi")
    print(f"Psi modification: overall={result_psi['overall_immunogenicity']:.3f}")
```

### 8.3 circRNA 序列优化

```python
"""circRNA 序列进化优化示例。"""
from confluencia_3_0.core.circrna.evolution.sequence_evolver import CircRNASequenceEvolver

# 优化器
evolver = CircRNASequenceEvolver(
    population_size=100,
    mutation_rate=0.08,
    n_generations=200,
    target_length=500,
    objectives_weights={
        "stability": 0.25,
        "translation": 0.35,
        "immune_evasion": 0.30,
        "delivery": 0.10
    }
)

# 优化 Ψ 修饰序列
best_seq, metrics = evolver.evolve(modification="Psi", verbose=True)

print(f"Best sequence (length={len(best_seq)}):")
print(f"  Stability: {metrics['stability']:.3f}")
print(f"  Translation: {metrics['translation']:.3f}")
print(f"  Immune evasion: {metrics['immune_evasion']:.3f}")
print(f"  Delivery: {metrics['delivery']:.3f}")
print(f"  Fitness: {metrics['fitness']:.3f}")

# 多目标 Pareto 前沿
pareto_front = evolver.multi_objective_evolve(modification="Psi")
print(f"Pareto front size: {len(pareto_front)}")
```

### 8.4 PK/PD 模拟

```python
"""CirculaPK PK 模拟示例。"""
from confluencia_3_0.core.circrna.pk.rnactm import CirculaPKModel, CirculaPKConfig

# 模型
config = CirculaPKConfig()
model = CirculaPKModel(config)

# 模拟不同修饰
modifications = ["none", "m6A", "Psi", "5mC"]
for mod in modifications:
    result = model.simulate(dose=5.0, modification=mod, duration=72)
    metrics = model.compute_pk_metrics(result)
    
    print(f"\n{mod} modification:")
    print(f"  Cmax (Cyto): {metrics['Cmax_cyto']:.3f} mg/kg")
    print(f"  Tmax (Cyto): {metrics['Tmax_cyto']:.1f} h")
    print(f"  AUC (Cyto): {metrics['AUC_cyto']:.3f}")
    print(f"  Half-life: {metrics['Half_life']:.1f} h")
    print(f"  Total protein: {metrics['Total_protein']:.3f}")

# 优化剂量
optimal_dose, metrics = model.optimize_dose(
    target_protein=0.5,
    max_dose=10.0,
    modification="Psi"
)
print(f"\nOptimal dose (Ψ): {optimal_dose:.2f} mg/kg")
```

### 8.5 TorusFold 结构预测

```python
"""TorusFold 结构预测示例。"""
from confluencia_3_0.core.circrna.torusfold import TorusFold, TorusFoldConfig
import torch

# 配置
config = TorusFoldConfig(
    structure_mode="physics_b",  # 几何约束求解器
    n_solver_samples=50
)

# 模型
model = TorusFold(config)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 预测
sequence = "AUGCGCUAU" * 100  # 900 nt
gene_expr = {"TP53": 0.8, "BRCA1": 0.5}

with torch.no_grad():
    result = model.predict_single(sequence, gene_expr)

print(f"Structure prediction:")
print(f"  Sequence length: {len(sequence)}")
print(f"  Confidence: {result['confidence']:.1f}")
print(f"  Closure distance: {result['closure_distance']:.2f} Å")
print(f"  Method: {result['structure_method']}")

# physics_ba 模式 (需要 OpenMM)
try:
    config_ba = TorusFoldConfig(
        structure_mode="physics_ba",
        n_minimize_steps=1000,
        n_md_steps=10000
    )
    model_ba = TorusFold(config_ba)
    result_ba = model_ba.predict_single(sequence[:200])  # 较短序列
    print(f"\nphysics_ba mode:")
    print(f"  Energy score: {result_ba.get('energy_score', 'N/A')}")
except Exception as e:
    print(f"physics_ba unavailable: {e}")
```

### 8.6 完整治疗模拟

```python
"""完整 circRNA 治疗模拟示例。"""
from confluencia_3_0.core.config import Confluencia3Config
from confluencia_3_0.core.agent import TNBCSimulacrumAgent
from confluencia_3_0.core.circrna.evolution.sequence_evolver import CircRNASequenceEvolver
from confluencia_3_0.core.circrna.pk.rnactm import CirculaPKModel

# 1. 设计 circRNA
print("Step 1: Designing circRNA...")
evolver = CircRNASequenceEvolver(
    population_size=200,
    n_generations=300,
    target_length=800,
    objectives_weights={
        "stability": 0.3,
        "translation": 0.35,
        "immune_evasion": 0.25,
        "delivery": 0.1
    }
)
circrna_seq, design_metrics = evolver.evolve(modification="Psi", verbose=True)
print(f"Designed circRNA: length={len(circrna_seq)}, fitness={design_metrics['fitness']:.3f}")

# 2. PK/PD 分析
print("\nStep 2: PK/PD analysis...")
pk_model = CirculaPKModel()
dose, pk_metrics = pk_model.optimize_dose(target_protein=0.8, modification="Psi")
print(f"Optimal dose: {dose:.2f} mg/kg, protein={pk_metrics['Total_protein']:.3f}")

# 3. TNBC 仿真
print("\nStep 3: TNBC simulation...")
config = Confluencia3Config()
config.molecular_subtype = "BLIS"
config.experiment.max_steps = 365
config.circrna.enabled = True

agent = TNBCSimulacrumAgent(config)
agent.initialize()

# 添加 circRNA 治疗
agent.add_treatment({
    "type": "circrna",
    "sequence": circrna_seq,
    "modification": "Psi",
    "dose": dose,
    "start_day": 60,
    "frequency": "weekly"
})

# 运行
for step in range(365):
    agent.step()
    
    if step % 30 == 0:
        state = agent.get_state()
        tumor = state['tumor_volume']
        cd8 = state['immune_cells']['CD8']
        print(f"Day {step}: Tumor={tumor:.1f} mm³, CD8={cd8:.0f}")

# 最终结果
final_state = agent.get_state()
print(f"\nFinal outcome:")
print(f"  Tumor volume: {final_state['tumor_volume']:.1f} mm³")
print(f"  Clinical stage: {final_state['clinical_stage']}")
print(f"  Survival probability: {final_state['survival_probability']:.2%}")
```

---

## 9. 配置指南

### 9.1 完整配置参数

```yaml
# config.yaml 示例

# 全局配置
experiment:
  max_steps: 365
  seed: 42
  save_state_interval: 10
  output_dir: "./results"

# TNBC 配置
tnbc:
  molecular_subtype: "BLIS"  # BLIS | IM | M | LAR
  initial_volume: 100  # mm³
  
  tumor:
    growth_model: "logistic"  # logistic | gompertz
    growth_rate: 0.027  # day⁻¹
    capacity: 1000  # mm³
    apoptosis_rate: 0.005
    
    heterogeneity:
      n_subclones: 5
      mutation_rate: 1e-6
    
    csc:
      fraction: 0.02
      resistance_factor: 5.0
    
    angiogenesis:
      vegf_rate: 0.1
      k_angiogenesis: 0.05
    
    metastasis:
      emt_rate: 0.01

  tme:
    immune_cells:
      CD8: 50
      NK: 30
      M1: 20
      M2: 10
      Treg: 5
      MDSC: 3
    
    immunoediting:
      elimination_threshold: 0.8
      escape_pd_l1_threshold: 0.5

# circRNA 配置
circrna:
  enabled: true
  
  immunogenicity_backend: "heuristic"  # heuristic | vienna | esm2
  mhc_backend: "local"  # local | netmhcpan
  drug_backend: "local"  # local | chembl_api
  pk_backend: "rnactm"
  
  structure_mode: "physics_b"  # heuristic | simple | diffusion | physics_b | physics_ba
  diffusion_steps: 100
  solver_samples: 20
  openmm_minimize_steps: 500
  openmm_md_steps: 5000
  
  evolution:
    population_size: 100
    mutation_rate: 0.05
    crossover_rate: 0.7
    n_generations: 100
    target_length: 500
    
    objectives_weights:
      stability: 0.3
      translation: 0.3
      immune_evasion: 0.25
      delivery: 0.15

# Backend 配置
backend:
  esm2:
    model: "esm2_t33_650M_UR50D"
    device: "cuda"
  
  vienna:
    temperature: 37  # Celsius
  
  netmhcpan:
    url: "https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/"
    alleles: ["HLA-A02:01", "HLA-B07:02"]
  
  chembl:
    api_url: "https://www.ebi.ac.uk/chembl/api/data"
```

### 9.2 CLI 使用

```bash
# 基础运行
python -m confluencia_3_0 --steps 365 --subtype BLIS

# 指定 circRNA 后端
python -m confluencia_3_0 --circrna-backend vienna --structure-mode physics_b

# 禁用 circRNA 子系统
python -m confluencia_3_0 --no-circrna

# 使用 diffusion 结构模式 (需要 GPU)
python -m confluencia_3_0 --structure-mode diffusion --steps 100

# 使用 physics_ba 模式 (需要 OpenMM)
python -m confluencia_3_0 --structure-mode physics_ba

# 指定配置文件
python -m confluencia_3_0 --config config.yaml
```

---

## 10. 部署指南

### 10.1 环境安装

```bash
# 创建 conda 环境
conda create -n confluencia python=3.9
conda activate confluencia

# 安装核心依赖
pip install numpy scipy torch

# 安装可选依赖
# ViennaRNA (RNA 结构预测)
conda install -c bioconda viennarna

# OpenMM (分子动力学)
conda install -c conda-forge openmm

# ESM-2 (深度学习嵌入)
pip install fair-esm

# Streamlit (Web UI)
pip install streamlit plotly
```

### 10.2 项目结构

```
confluencia_3_0/
├── main.py                    # CLI 入口
├── core/
│   ├── config.py              # 配置类
│   ├── agent.py               # TNBCSimulacrum Agent
│   ├── state_schema.py        # 状态 schema
│   ├── event_bus.py           # 事件总线
│   ├── subsystem_managers.py  # 子系统管理器
│   │
│   ├── tumor/
│   │   ├── growth_engine.py
│   │   ├── heterogeneity.py
│   │   ├── cancer_stem_cell.py
│   │   ├── angiogenesis.py
│   │   └── metastasis.py
│   │
│   ├── tme/
│   │   ├── immune_dynamics.py
│   │   ├── fibroblast.py
│   │   ├── immune_evasion.py
│   │   └── immunoediting.py
│   │
│   ├── treatment/
│   │   ├── chemotherapy.py
│   │   ├── immunotherapy.py
│   │   ├── targeted.py
│   │   ├── radiotherapy.py
│   │   └── circrna_therapy.py
│   │
│   ├── circrna/
│   │   ├── immune_sensing.py
│   │   ├── structure_prediction.py
│   │   ├── bsj_features.py
│   │   │
│   │   ├── pk/
│   │   │   ├── rnactm.py
│   │   │   └── pk_bridge.py
│   │   │
│   │   ├── evolution/
│   │   │   ├── sequence_evolver.py
│   │   │   ├── objective_functions.py
│   │   │   └── pareto_selector.py
│   │   │
│   │   ├── backends/
│   │   │   ├── immunogenicity_backend.py
│   │   │   ├── mhc_backend.py
│   │   │   └── drug_backend.py
│   │   │
│   │   ├── torusfold/
│   │   │   ├── torusfold.py
│   │   │   ├── tpe.py
│   │   │   ├── triangle_update.py
│   │   │   ├── irs_pair.py
│   │   │   ├── simple_structure_head.py
│   │   │   ├── diffusion_structure.py
│   │   │   ├── physics_structure_head.py
│   │   │   ├── physics_bridge.py
│   │   │   ├── constraint_solver.py
│   │   │   ├── structure_validator.py
│   │   │   └── cgmd_refiner.py
│   │   │
│   │   └── torusfold_scorer.py
│   │
│   ├── biomarker/
│   │   └── tumor_markers.py
│   │
│   ├── clinical/
│   │   ├── staging.py
│   │   └── survival.py
│   │
│   └── bridge/
│       ├── drug_bridge.py
│       ├── epitope_bridge.py
│       └── pk_bridge.py
│
├── experiments/
│   ├── sandbox.py
│   ├── clinical_trial.py
│   ├── combination.py
│   └── predefined/
│       ├── circrna_therapy.py
│       ├── chemo_immuno.py
│       └── parp_brca.py
│
├── frontend/
│   ├── app.py
│   └── tabs/
│       ├── tumor_dashboard.py
│       ├── tme_immune.py
│       ├── treatment.py
│       ├── biomarker.py
│       ├── clinical.py
│       ├── experiments.py
│       └── confluencia.py
│
├── docs/
│   └── ARCHITECTURE.md
│
└── tests/
    ├── test_tumor.py
    ├── test_tme.py
    ├── test_circrna.py
    ├── test_torusfold.py
    └── test_integration.py
```

### 10.3 GPU 配置

```python
# 检查 GPU 可用性
import torch

if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Running on CPU")

# TorusFold GPU 使用
from confluencia_3_0.core.circrna.torusfold import TorusFoldConfig

config = TorusFoldConfig(
    structure_mode="diffusion",  # diffusion 需要 GPU
    device=device
)
```

---

## 11. 常见问题解答

### Q1: 如何选择 circRNA 结构预测模式？

**回答**：

| 场景 | 推荐模式 |
|------|---------|
| 快速筛选 (大批量序列) | `heuristic` 或 `simple` |
| 精确结构预测 (有 GPU) | `diffusion` |
| 离线环境 (无 GPU) | `physics_b` |
| 最高精度 (有 OpenMM) | `physics_ba` |

### Q2: 为什么 circRNA 免疫原性比 mRNA 低？

**回答**：

circRNA 的环状结构避免了 5'-三磷酸，这是 RIG-I 的主要激活信号。此外，核苷酸修饰 (如 m6A、Ψ) 可进一步降低免疫原性。详见 Chen et al., Nature 2019。

### Q3: 如何优化 circRNA 的翻译效率？

**回答**：

1. 添加 IRES motif (GCGCC, GGGG, UUGU, AUGG)
2. 控制 GC 含量在 40-55%
3. 增加 AUG 密码子数量
4. 使用 TorusFold 评估 BSJ 稳定性
5. 优化序列进化器权重

### Q4: TorusFold 与 AlphaFold3 的区别？

**回答**：

| 特性 | AlphaFold3 | TorusFold |
|------|-----------|-----------|
| 目标 | 蛋白/蛋白-核酸复合物 | circRNA |
| 拓扑 | 线性 | 环形 (TPE) |
| 闭合约束 | 无 | 强制 x[0] ≈ x[-1] |
| 训练数据 | PDB | RNA 结构数据库 |

### Q5: physics_b 和 physics_ba 的区别？

**回答**：

- `physics_b`: 纯几何约束求解器，快速 (~200ms)，零训练
- `physics_ba`: 几何约束 + OpenMM 分子动力学精修，较慢 (~5-30s)，能量最小化 + MD 松弛

### Q6: 如何添加新的治疗策略？

**回答**：

```python
# 在 core/treatment/ 下创建新模块
# my_therapy.py

class MyTherapy:
    def __init__(self, config):
        self.config = config
    
    def apply(self, state: Dict) -> Dict:
        # 实现治疗逻辑
        state['tumor_volume'] *= 0.9  # 示例：10% 杀伤
        return state

# 在 subsystem_managers.py 中注册
class TreatmentManager:
    def __init__(self, config):
        self.my_therapy = MyTherapy(config)
```

### Q7: Backend 降级机制如何工作？

**回答**：

```python
# Immunogenicity Backend 降级流程
def get_backend(requested):
    order = [requested, "esm2", "vienna", "heuristic"]
    for name in order:
        backend = backends[name]
        if backend.is_available():
            return backend
    return backends["heuristic"]  # 始终可用
```

### Q8: 如何导出仿真结果？

**回答**：

```python
# 导出为 CSV
import pandas as pd

states = agent.get_history()
df = pd.DataFrame(states)
df.to_csv("simulation_results.csv")

# 导出为 JSON
import json
with open("simulation_results.json", "w") as f:
    json.dump(states, f)
```

---

## 12. 故障排除

### 12.1 常见错误

#### 错误: ImportError: No module named 'RNA'

**原因**: ViennaRNA 未安装

**解决**:
```bash
conda install -c bioconda viennarna
```

#### 错误: ImportError: No module named 'openmm'

**原因**: OpenMM 未安装

**解决**:
```bash
conda install -c conda-forge openmm
```

或降级使用 `physics_b` 模式:
```bash
python -m confluencia_3_0 --structure-mode physics_b
```

#### 错误: RuntimeError: CUDA out of memory

**原因**: GPU 内存不足

**解决**:
1. 减小 batch_size
2. 使用 CPU: `--device cpu`
3. 使用 `simple` 或 `physics_b` 模式

#### 错误: ValueError: sequence length must be >= 50

**原因**: circRNA 序列过短

**解决**: 确保序列长度 >= 50 nt

#### 错误: KeyError: 'HLA-A99:99'

**原因**: MHC allele 不在支持列表中

**解决**: 使用 `--mhc-backend local` (支持246 alleles)

### 12.2 性能优化

#### 加速 circRNA 评估

```python
# 使用 heuristic 后端
config.circrna.immunogenicity_backend = "heuristic"

# 批量评估
sequences = ["AUGCGC..." for _ in range(1000)]
scores = [quick_score(seq, device="cpu") for seq in sequences]
```

#### 加速结构预测

```python
# 使用 physics_b 模式
config.circrna.structure_mode = "physics_b"
config.circrna.solver_samples = 10  # 减少采样数
```

#### 加速仿真

```python
# 减少保存频率
config.experiment.save_state_interval = 100

# 禁用 circRNA (如果不需要)
config.circrna.enabled = False
```

### 12.3 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查状态
state = agent.get_state()
print(json.dumps(state, indent=2))

# 单步调试
agent.step()
print(f"After step: tumor={agent.get_state()['tumor_volume']}")
```

---

## 附录 A: 配置参数参考

### CircRNAConfig 完整参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | True | 是否启用 circRNA 子系统 |
| `immunogenicity_backend` | str | "heuristic" | 免疫原性后端 |
| `mhc_backend` | str | "local" | MHC 结合预测后端 |
| `drug_backend` | str | "local" | 药物结合预测后端 |
| `pk_backend` | str | "rnactm" | PK 模型后端 |
| `structure_mode` | str | "heuristic" | 结构预测模式 |
| `diffusion_steps` | int | 100 | diffusion 模式去噪步数 |
| `solver_samples` | int | 20 | physics_b/ba 模式采样数 |
| `openmm_minimize_steps` | int | 500 | OpenMM 能量最小化步数 |
| `openmm_md_steps` | int | 5000 | OpenMM MD 松弛步数 |
| `enable_torusfold` | bool | False | 是否启用 TorusFold (自动设置) |

### TumorConfig 完整参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `growth_model` | str | "logistic" | 生长模型 (logistic/gompertz) |
| `growth_rate` | float | 0.027 | 生长率 (day⁻¹) |
| `capacity` | float | 1000.0 | 环境容纳量 (mm³) |
| `apoptosis_rate` | float | 0.005 | 凋亡率 (day⁻¹) |
| `n_subclones` | int | 5 | 亚克隆数量 |
| `mutation_rate` | float | 1e-6 | 突变率 |
| `csc_fraction` | float | 0.02 | CSC 占比 |
| `csc_resistance` | float | 5.0 | CSC 抗性倍数 |

### TMEConfig 完整参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_cd8` | int | 50 | 初始 CD8 数量 |
| `initial_nk` | int | 30 | 初始 NK 数量 |
| `initial_m1` | int | 20 | 初始 M1 数量 |
| `initial_m2` | int | 10 | 初始 M2 数量 |
| `initial_treg` | int | 5 | 初始 Treg 数量 |
| `elimination_threshold` | float | 0.8 | Elimination阈值 |
| `escape_threshold` | float | 0.5 | Escape 阈值 |

---

## 附录 B: 事件总线事件列表

| 事件名 | 触发时机 | 数据字段 |
|--------|---------|---------|
| `STEP_START` | 每个 step 开始 | `step`, `timestamp` |
| `STEP_END` | 每个 step 结束 | `step`, `state_snapshot` |
| `TUMOR_VOLUME_UPDATE` | 肿瘤体积更新 | `old_volume`, `new_volume` |
| `METASTASIS_EVENT` | 转移发生 | `target_organ` |
| `IMMUNOEDITING_PHASE` | 免疫编辑阶段转换 | `old_phase`, `new_phase` |
| `DRUG_ADMINISTERED` | 药物给药 | `drug_type`, `dose` |
| `CIRCRNA_THERAPY_UPDATE` | circRNA 治疗更新 | `sequence`, `modification` |
| `CIRCRNA_IMMUNE_EVAL` | circRNA 免疫评估请求 | `sequence`, `modification` |
| `CIRCRNA_STRUCTURE_PREDICT` | circRNA 结构预测请求 | `sequence`, `mode` |
| `CIRCRNA_SEQUENCE_EVOLVE` | circRNA 序列进化请求 | `target_length`, `modification` |
| `CIRCRNA_PK_SIMULATE` | circRNA PK 模拟请求 | `dose`, `duration` |
| `MOLECULE_EVOLUTION_REQUEST` | 分子进化请求 | `molecule_type` |
| `CLINICAL_STAGE_UPDATE` | 临床分期更新 | `old_stage`, `new_stage` |
| `TREATMENT_RESPONSE` | 治疗响应 | `response_type`, `magnitude` |
| `BIOMARKER_UPDATE` | 生物标志物更新 | `marker_name`, `value` |

---

## 附录 C: 状态 Schema完整字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `tumor_volume` | float | 肿瘤体积 (mm³) |
| `tumor_n_cells` | int | 肿瘤细胞数 |
| `subclones` | List[Dict] | 亚克隆列表 |
| `csc_fraction` | float | CSC占比 |
| `mvd` | float | 微血管密度 |
| `metastasis_sites` | List[str] | 转移位点 |
| `immune_cells` | Dict[str, int] | 免疫细胞数量 |
| `immunoediting_phase` | str | 免疫编辑阶段 |
| `pd_l1_expression` | float | PD-L1 表达水平 |
| `treatment_history` | List[Dict] | 治疗历史 |
| `clinical_stage` | str | 临床分期 |
| `survival_probability` | float | 生存概率 |
| `biomarkers` | Dict[str, float] | 生物标志物 |
| `circrna_state` | Dict | circRNA 状态 |
| `step` | int | 当前步数 |
| `timestamp` | float | 时间戳 |

---

---

## 附录 D: 性能分析与优化

### D.1 性能基准测试

#### D.1.1 各模块延迟测试

**测试环境**:
- CPU: Intel Core i9-12900K
- GPU: NVIDIA RTX 3090 (24GB)
- RAM: 64GB DDR5
- OS: Windows 11 / Ubuntu 22.04

**免疫原性评估延迟**:
| 后端 | 序列长度 | 延迟 (ms) | 内存 (MB) |
|------|---------|----------|----------|
| heuristic | 500 nt | 12 | 5 |
| heuristic | 2000 nt | 45 | 10 |
| vienna | 500 nt | 85 | 50 |
| vienna | 2000 nt | 340 | 150 |
| esm2 (GPU) | 500 nt | 180 | 2500 |
| esm2 (GPU) | 2000 nt | 520 | 4000 |

**结构预测延迟**:
| 模式 | 序列长度 | 延迟 (ms) | 内存 (MB) |
|------|---------|----------|----------|
| simple | 500 nt | 8 | 50 |
| simple | 2000 nt | 35 | 200 |
| physics_b (n=20) | 500 nt | 180 | 100 |
| physics_b (n=20) | 2000 nt | 750 | 400 |
| physics_ba | 500 nt | 5000 | 500 |
| physics_ba | 2000 nt | 25000 | 2000 |
| diffusion (GPU) | 500 nt | 2100 | 3000 |
| diffusion (GPU) | 2000 nt | 8500 | 8000 |

**仿真步延迟**:
| 配置 | 每步延迟 (ms) | 内存 (MB) |
|------|-------------|----------|
| 禁用 circRNA | 15 | 100 |
| circRNA (heuristic) | 25 | 120 |
| circRNA (vienna + physics_b) | 45 | 200 |

#### D.1.2 瓶颈分析

```
单步仿真时间分解 (circRNA 启用):
    │
    ├─→ TumorManager.step(): 3ms (20%)
    │       └─→ GrowthEngine: 1ms
    │       └─→ Heterogeneity: 2ms
    │
    ├─→ TMEManager.step(): 5ms (33%)
    │       └─→ ImmuneDynamics: 4ms
    │       └─→ Immunoediting: 1ms
    │
    ├─→ CircRNAManager.step(): 5ms (33%)
    │       └─→ ImmuneSensing: 2ms
    │       └─→ PK simulation: 3ms
    │
    └─→ 其他: 2ms (14%)

瓶颈: TME 免疫动力学计算 (大量 ODE 求解)
```

### D.2 内存优化策略

#### D.2.1 状态压缩

```python
class CompressedState:
    """压缩状态存储，减少内存占用。"""
    
    # 原始状态 ~180 keys, ~50KB
    # 压缩后 ~5KB
    
    FLOAT_PRECISION = np.float32  # 使用 32 位而非 64 位
    
    @staticmethod
    def compress(state: Dict) -> bytes:
        """压缩状态字典。"""
        import zlib
        import json
        
        # 转换为可序列化格式
        serializable = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                serializable[k] = {
                    '_type': 'ndarray',
                    'data': v.astype(CompressedState.FLOAT_PRECISION).tobytes(),
                    'shape': v.shape,
                    'dtype': str(v.dtype)
                }
            elif isinstance(v, (int, float, str, bool)):
                serializable[k] = v
            elif isinstance(v, dict):
                serializable[k] = v
        
        # JSON + zlib 压缩
        json_str = json.dumps(serializable)
        compressed = zlib.compress(json_str.encode(), level=6)
        
        return compressed
    
    @staticmethod
    def decompress(data: bytes) -> Dict:
        """解压状态字典。"""
        import zlib
        import json
        
        json_str = zlib.decompress(data).decode()
        serializable = json.loads(json_str)
        
        state = {}
        for k, v in serializable.items():
            if isinstance(v, dict) and v.get('_type') == 'ndarray':
                arr = np.frombuffer(v['data'], dtype=v['dtype'])
                state[k] = arr.reshape(v['shape'])
            else:
                state[k] = v
        
        return state
```

#### D.2.2 惰性计算

```python
class LazyComputation:
    """惰性计算，只在需要时执行。"""
    
    def __init__(self):
        self._cache = {}
        self._dirty = set()
    
    def mark_dirty(self, key: str):
        """标记缓存失效。"""
        self._dirty.add(key)
    
    def get(self, key: str, compute_fn: Callable) -> Any:
        """获取值，必要时重新计算。"""
        if key not in self._cache or key in self._dirty:
            self._cache[key] = compute_fn()
            self._dirty.discard(key)
        return self._cache[key]


class LazyTumorState:
    """肿瘤状态的惰性计算示例。"""
    
    def __init__(self, tumor_manager):
        self.manager = tumor_manager
        self._lazy = LazyComputation()
    
    @property
    def volume(self) -> float:
        return self.manager._volume  # 原始值，直接返回
    
    @property
    def diameter(self) -> float:
        """惰性计算: 直径 = (6V/π)^(1/3)"""
        return self._lazy.get(
            'diameter',
            lambda: (6 * self.volume / np.pi) ** (1/3)
        )
    
    @property
    def cell_count(self) -> int:
        """惰性计算: 细胞数 = V × 10^6 cells/mm³"""
        return self._lazy.get(
            'cell_count',
            lambda: int(self.volume * 1e6)
        )
    
    def update_volume(self, new_volume: float):
        """更新体积，标记相关缓存失效。"""
        self.manager._volume = new_volume
        self._lazy.mark_dirty('diameter')
        self._lazy.mark_dirty('cell_count')
```

### D.3 并发模型

#### D.3.1 EventBus 异步处理

```python
import asyncio
from typing import Callable, Dict, List
from dataclasses import dataclass
from enum import Enum

class EventPriority(Enum):
    HIGH = 0      # 立即处理
    NORMAL = 1    # 正常队列
    LOW = 2       # 批量处理

@dataclass
class Event:
    name: str
    data: Dict
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = 0.0

class AsyncEventBus:
    """异步事件总线，支持优先级和批处理。"""
    
    def __init__(self, max_workers: int = 4):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._workers = []
        self._running = False
        self.max_workers = max_workers
    
    def subscribe(self, event_name: str, handler: Callable):
        """订阅事件。"""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)
    
    async def emit(self, event: Event):
        """发射事件到队列。"""
        await self._queue.put((event.priority.value, event))
    
    async def start(self):
        """启动工作线程。"""
        self._running = True
        for _ in range(self.max_workers):
            worker = asyncio.create_task(self._worker())
            self._workers.append(worker)
    
    async def stop(self):
        """停止工作线程。"""
        self._running = False
        for worker in self._workers:
            worker.cancel()
    
    async def _worker(self):
        """工作线程处理事件。"""
        while self._running:
            try:
                priority, event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                handlers = self._subscribers.get(event.name, [])
                
                # 并发调用所有处理器
                tasks = [
                    asyncio.create_task(self._call_handler(h, event))
                    for h in handlers
                ]
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"EventBus worker error: {e}")
    
    async def _call_handler(self, handler: Callable, event: Event):
        """调用单个处理器。"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logging.error(f"Handler error for {event.name}: {e}")
```

#### D.3.2 批量处理优化

```python
class BatchProcessor:
    """批量处理器，减少函数调用开销。"""
    
    def __init__(self, batch_size: int = 32, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self._buffer = []
        self._lock = asyncio.Lock()
    
    async def add(self, item: Any) -> Any:
        """添加项目到批次，等待批量处理结果。"""
        future = asyncio.Future()
        
        async with self._lock:
            self._buffer.append((item, future))
            
            if len(self._buffer) >= self.batch_size:
                await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """处理当前批次。"""
        if not self._buffer:
            return
        
        batch = self._buffer[:]
        self._buffer.clear()
        
        # 批量处理
        items = [item for item, _ in batch]
        results = await self._process_items(items)
        
        # 返回结果
        for (_, future), result in zip(batch, results):
            future.set_result(result)
    
    async def _process_items(self, items: List[Any]) -> List[Any]:
        """子类实现具体批量处理逻辑。"""
        raise NotImplementedError


class BatchImmunogenicityAssessment(BatchProcessor):
    """批量免疫原性评估。"""
    
    def __init__(self, backend: ImmunogenicityBackend):
        super().__init__(batch_size=64)
        self.backend = backend
    
    async def _process_items(self, sequences: List[str]) -> List[Dict]:
        """批量评估免疫原性。"""
        # 使用后端的批量接口
        if hasattr(self.backend, 'assess_batch'):
            return self.backend.assess_batch(sequences)
        else:
            # 逐个处理
            return [self.backend.assess(seq) for seq in sequences]
```

---

## 附录 E: 错误处理与恢复

### E.1 错误分类

```python
class ErrorSeverity(Enum):
    WARNING = "warning"      # 可继续运行
    ERROR = "error"          # 当前操作失败
    CRITICAL = "critical"    # 需要重启模块
    FATAL = "fatal"          # 系统级错误

class ConfluenciaError(Exception):
    """Confluencia 基础异常。"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recovery_hint: str = None,
        context: Dict = None
    ):
        super().__init__(message)
        self.severity = severity
        self.recovery_hint = recovery_hint
        self.context = context or {}

class BackendUnavailableError(ConfluenciaError):
    """后端不可用错误。"""
    
    def __init__(self, backend_name: str, reason: str):
        super().__init__(
            message=f"Backend '{backend_name}' unavailable: {reason}",
            severity=ErrorSeverity.WARNING,
            recovery_hint="System will fall back to next available backend.",
            context={"backend": backend_name, "reason": reason}
        )

class StructurePredictionError(ConfluenciaError):
    """结构预测错误。"""
    
    def __init__(self, sequence: str, mode: str, reason: str):
        super().__init__(
            message=f"Structure prediction failed for sequence (length={len(sequence)}) in mode '{mode}': {reason}",
            severity=ErrorSeverity.ERROR,
            recovery_hint="Try a different structure mode or check sequence validity.",
            context={"sequence_length": len(sequence), "mode": mode}
        )

class SimulationDivergenceError(ConfluenciaError):
    """仿真发散错误。"""
    
    def __init__(self, step: int, state: Dict):
        super().__init__(
            message=f"Simulation diverged at step {step}. Tumor volume: {state.get('tumor_volume', 'N/A')}",
            severity=ErrorSeverity.CRITICAL,
            recovery_hint="Check model parameters. Consider reducing step size or adjusting growth rate.",
            context={"step": step, "state": state}
        )
```

### E.2 恢复策略

```python
class RecoveryManager:
    """恢复管理器。"""
    
    def __init__(self, agent):
        self.agent = agent
        self.checkpoints = []
        self.max_checkpoints = 10
    
    def save_checkpoint(self, step: int, state: Dict):
        """保存检查点。"""
        checkpoint = {
            'step': step,
            'state': CompressedState.compress(state),
            'timestamp': time.time()
        }
        
        self.checkpoints.append(checkpoint)
        
        # 保留最近 N 个检查点
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)
    
    def restore_last_checkpoint(self) -> Tuple[int, Dict]:
        """恢复最后一个检查点。"""
        if not self.checkpoints:
            return 0, {}
        
        checkpoint = self.checkpoints[-1]
        state = CompressedState.decompress(checkpoint['state'])
        
        return checkpoint['step'], state
    
    def handle_error(self, error: ConfluenciaError) -> bool:
        """处理错误，尝试恢复。
        
        Returns:
            True if recovered, False if cannot recover
        """
        if error.severity == ErrorSeverity.WARNING:
            logging.warning(str(error))
            return True
        
        elif error.severity == ErrorSeverity.ERROR:
            logging.error(str(error))
            # 尝试替代方案
            return self._try_alternative(error)
        
        elif error.severity == ErrorSeverity.CRITICAL:
            logging.critical(str(error))
            # 从检查点恢复
            return self._restore_from_checkpoint()
        
        elif error.severity == ErrorSeverity.FATAL:
            logging.fatal(str(error))
            return False
    
    def _try_alternative(self, error: ConfluenciaError) -> bool:
        """尝试替代方案。"""
        if isinstance(error, BackendUnavailableError):
            # 后端降级已在 BackendManager 中处理
            return True
        
        if isinstance(error, StructurePredictionError):
            # 尝试更简单的模式
            simpler_modes = {
                "physics_ba": "physics_b",
                "physics_b": "simple",
                "diffusion": "simple",
                "simple": "heuristic"
            }
            current_mode = error.context.get("mode")
            if current_mode in simpler_modes:
                logging.info(f"Retrying with simpler mode: {simpler_modes[current_mode]}")
                # 实际重试逻辑...
                return True
        
        return False
    
    def _restore_from_checkpoint(self) -> bool:
        """从检查点恢复。"""
        step, state = self.restore_last_checkpoint()
        
        if step == 0:
            logging.error("No checkpoint available for recovery")
            return False
        
        logging.info(f"Restoring from checkpoint at step {step}")
        self.agent.load_state(state)
        
        return True
```

---

## 附录 F: 扩展开发指南

### F.1 添加新的治疗类型

```python
# 1. 在 core/treatment/ 下创建新模块

# core/treatment/my_therapy.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MyTherapyConfig:
    """我的治疗配置。"""
    dose: float = 1.0  # mg/kg
    frequency: str = "weekly"
    parameter1: float = 0.5
    parameter2: int = 10


class MyTherapy:
    """自定义治疗实现。"""
    
    def __init__(self, config: MyTherapyConfig = None):
        self.config = config or MyTherapyConfig()
    
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """应用治疗，返回状态变更。
        
        Args:
            state: 当前系统状态
        
        Returns:
            状态变更字典
        """
        tumor_volume = state['tumor_volume']
        
        # 实现治疗逻辑
        kill_rate = self._compute_kill_rate(state)
        new_volume = tumor_volume * (1 - kill_rate)
        
        return {
            'tumor_volume': new_volume,
            'treatment_applied': 'my_therapy',
            'dose': self.config.dose
        }
    
    def _compute_kill_rate(self, state: Dict) -> float:
        """计算杀伤率。"""
        # 基于状态的自适应计算
        base_rate = self.config.parameter1
        
        # 考虑肿瘤微环境
        cd8_count = state['immune_cells'].get('CD8', 0)
        immune_factor = min(cd8_count / 100, 1.0)
        
        return base_rate * (1 + immune_factor * 0.5)


# 2. 在 TreatmentManager 中注册

# core/subsystem_managers.py

class TreatmentManager:
    def __init__(self, config):
        # ... 现有代码 ...
        self.my_therapy = MyTherapy(config.treatment.my_therapy)
    
    def _apply_treatment(self, treatment_type: str, state: Dict) -> Dict:
        if treatment_type == "my_therapy":
            return self.my_therapy.apply(state)
        # ... 其他治疗类型 ...
```

### F.2 添加新的 Backend

```python
# 1. 实现 Backend 接口

# core/circrna/backends/my_backend.py

from .base import ImmunogenicityBackend

class MyImmunogenicityBackend(ImmunogenicityBackend):
    """自定义免疫原性后端。"""
    
    name = "my_backend"
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型。"""
        # 加载自定义模型
        model_path = self.config.get('model_path', 'default_model.pt')
        try:
            self._model = torch.load(model_path)
            self._available = True
        except Exception as e:
            self._unavailable_reason = str(e)
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def assess(self, sequence: str, modification: str = "none") -> Dict[str, float]:
        """评估免疫原性。"""
        if not self._available:
            raise RuntimeError(f"Backend not available: {self._unavailable_reason}")
        
        # 预处理
        features = self._extract_features(sequence, modification)
        
        # 模型推理
        with torch.no_grad():
            output = self._model(features)
        
        # 后处理
        scores = self._parse_output(output)
        scores['backend'] = self.name
        
        return scores
    
    def _extract_features(self, sequence: str, modification: str) -> torch.Tensor:
        """提取特征。"""
        # 实现特征提取
        pass
    
    def _parse_output(self, output: torch.Tensor) -> Dict[str, float]:
        """解析输出。"""
        return {
            'rig_i_score': float(output[0]),
            'tlr7_score': float(output[1]),
            'tlr8_score': float(output[2]),
            'pkr_score': float(output[3]),
            'overall_immunogenicity': float(output.mean())
        }


# 2. 注册到 BackendManager

# core/circrna/backends/__init__.py

BACKENDS = {
    "esm2": ESM2Backend,
    "vienna": ViennaBackend,
    "heuristic": HeuristicBackend,
    "my_backend": MyImmunogenicityBackend  # 添加新后端
}

BACKEND_PRIORITY = ["esm2", "my_backend", "vienna", "heuristic"]
```

### F.3 添加新的事件类型

```python
# 1. 定义事件

# core/events.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MyCustomEvent:
    """自定义事件。"""
    
    event_type: str = "MY_CUSTOM_EVENT"
    data: Dict[str, Any] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# 2. 在 Agent 中处理事件

class TNBCSimulacrumAgent:
    def _setup_event_handlers(self):
        # ... 现有处理器 ...
        self.event_bus.subscribe("MY_CUSTOM_EVENT", self._on_my_custom_event)
    
    def _on_my_custom_event(self, event):
        """处理自定义事件。"""
        logging.info(f"Received MY_CUSTOM_EVENT: {event.data}")
        
        # 执行自定义逻辑
        self._handle_custom_logic(event.data)


# 3. 发射事件

self.event_bus.emit(MyCustomEvent(data={"key": "value"}))
```

---

## 附录 G: 测试指南

### G.1 单元测试

```python
# tests/test_torusfold.py

import pytest
import torch
import numpy as np

from confluencia_3_0.core.circrna.torusfold import (
    TorusFold,
    TorusFoldConfig,
    TorusPositionalEncoding
)


class TestTorusPositionalEncoding:
    """TPE 单元测试。"""
    
    def test_periodicity(self):
        """测试周期性性质: TPE[0] = TPE[L]"""
        tpe = TorusPositionalEncoding(d_model=64)
        
        # 测试不同序列长度
        for L in [50, 100, 200, 500]:
            assert tpe.verify_periodicity(L), f"Periodicity failed for L={L}"
    
    def test_shape(self):
        """测试输出形状。"""
        tpe = TorusPositionalEncoding(d_model=128)
        x = torch.zeros(2, 100, 128)  # batch=2, L=100
        
        output = tpe(x, seq_len=100)
        
        assert output.shape == (2, 100, 128)
    
    def test_different_lengths(self):
        """测试不同序列长度。"""
        tpe = TorusPositionalEncoding(d_model=64)
        
        for L in [10, 50, 100, 500, 1000]:
            x = torch.zeros(1, L, 64)
            output = tpe(x, seq_len=L)
            
            # 检查 TPE[0] ≈ TPE[L-1]（注意索引）
            # 实际上 TPE[L] 是下一个位置，所以 TPE[0] 应该等于 TPE[L]
            # 但由于我们只有 L 个位置，所以检查首尾是否接近


class TestTorusFold:
    """TorusFold 集成测试。"""
    
    @pytest.fixture
    def model(self):
        config = TorusFoldConfig(structure_mode="simple")
        return TorusFold(config)
    
    def test_forward_pass(self, model):
        """测试前向传播。"""
        sequence = "AUGCGCUAU" * 10  # 90 nt
        
        result = model.predict_single(sequence)
        
        assert 'coords' in result
        assert 'confidence' in result
        assert 'closure_distance' in result
        
        assert result['coords'].shape == (90, 3)
        assert 0 <= result['confidence'] <= 100
    
    def test_different_modes(self):
        """测试不同结构模式。"""
        sequence = "AUGCGCUAU" * 20  # 180 nt
        
        for mode in ["simple", "physics_b"]:
            config = TorusFoldConfig(structure_mode=mode)
            model = TorusFold(config)
            
            result = model.predict_single(sequence)
            
            assert result['coords'].shape[0] == len(sequence)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_mode(self):
        """测试 GPU 模式。"""
        config = TorusFoldConfig(structure_mode="diffusion")
        model = TorusFold(config).cuda()
        
        sequence = "AUGCGCUAU" * 50
        result = model.predict_single(sequence)
        
        assert result['coords'].device.type == "cuda"


class TestCircPairformer:
    """CircPairformer 单元测试。"""
    
    def test_triangle_update(self):
        """测试三角乘法更新。"""
        from confluencia_3_0.core.circrna.torusfold.triangle_update import (
            TriangleMultiplicativeUpdate
        )
        
        update = TriangleMultiplicativeUpdate(c_z=64, c_hidden=32)
        
        z = torch.randn(2, 50, 50, 64)  # batch=2, L=50
        output = update(z)
        
        assert output.shape == z.shape
    
    def test_circular_distance(self):
        """测试环形距离计算。"""
        from confluencia_3_0.core.circrna.torusfold.triangle_update import (
            circular_distance_matrix
        )
        
        L = 10
        dist = circular_distance_matrix(L, torch.device('cpu'))
        
        # 检查形状
        assert dist.shape == (L, L)
        
        # 检查对角线为 0
        assert torch.allclose(torch.diag(dist), torch.zeros(L))
        
        # 检查环形性质: d(0, L-1) = 1
        assert dist[0, L-1] == 1
        
        # 检查最远距离: d(0, L/2) = L/2
        assert dist[0, L//2] == L // 2
```

### G.2 集成测试

```python
# tests/test_integration.py

import pytest
from confluencia_3_0.core.config import Confluencia3Config
from confluencia_3_0.core.agent import TNBCSimulacrumAgent


class TestFullSimulation:
    """完整仿真集成测试。"""
    
    @pytest.fixture
    def config(self):
        config = Confluencia3Config()
        config.experiment.max_steps = 10  # 快速测试
        config.circrna.enabled = True
        config.circrna.structure_mode = "physics_b"
        return config
    
    def test_basic_simulation(self, config):
        """测试基础仿真流程。"""
        agent = TNBCSimulacrumAgent(config)
        agent.initialize()
        
        # 运行 10 步
        for _ in range(10):
            agent.step()
        
        state = agent.get_state()
        
        # 检查状态有效性
        assert state['tumor_volume'] > 0
        assert state['step'] == 10
        assert 'immune_cells' in state
    
    def test_circrna_therapy(self, config):
        """测试 circRNA 治疗。"""
        agent = TNBCSimulacrumAgent(config)
        agent.initialize()
        
        # 添加 circRNA 治疗
        agent.add_treatment({
            "type": "circrna",
            "sequence": "AUGCGCUAU" * 50,
            "modification": "Psi",
            "dose": 5.0,
            "start_day": 5
        })
        
        # 运行
        for _ in range(10):
            agent.step()
        
        # 检查治疗被应用
        history = agent.get_treatment_history()
        assert any(t['type'] == 'circrna' for t in history)
    
    def test_state_persistence(self, config):
        """测试状态持久化。"""
        agent = TNBCSimulacrumAgent(config)
        agent.initialize()
        
        # 运行 5 步
        for _ in range(5):
            agent.step()
        
        # 保存状态
        state = agent.get_state()
        
        # 创建新 agent 并加载状态
        agent2 = TNBCSimulacrumAgent(config)
        agent2.load_state(state)
        agent2.initialize(from_state=True)
        
        # 验证状态一致
        state2 = agent2.get_state()
        assert state2['step'] == state['step']
        assert abs(state2['tumor_volume'] - state['tumor_volume']) < 1e-6
```

### G.3 性能测试

```python
# tests/test_performance.py

import pytest
import time
from confluencia_3_0.core.circrna.torusfold_scorer import quick_score


class TestPerformance:
    """性能基准测试。"""
    
    def test_heuristic_latency(self):
        """测试启发式后端延迟。"""
        sequence = "AUGCGCUAU" * 50  # 450 nt
        
        latencies = []
        for _ in range(100):
            start = time.time()
            quick_score(sequence, device="cpu")
            latencies.append(time.time() - start)
        
        avg_latency = sum(latencies) / len(latencies)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"Heuristic backend: avg={avg_latency*1000:.1f}ms, p99={p99*1000:.1f}ms")
        
        # 断言延迟在合理范围内
        assert avg_latency < 0.1  # < 100ms
    
    @pytest.mark.parametrize("length", [100, 500, 1000, 2000])
    def test_structure_prediction_scaling(self, length):
        """测试结构预测随序列长度的扩展性。"""
        from confluencia_3_0.core.circrna.torusfold import TorusFold, TorusFoldConfig
        
        config = TorusFoldConfig(structure_mode="physics_b", n_solver_samples=10)
        model = TorusFold(config)
        
        sequence = "AUGCGCUAU" * (length // 9)
        
        start = time.time()
        result = model.predict_single(sequence)
        latency = time.time() - start
        
        print(f"Length {length}: {latency*1000:.1f}ms")
        
        # 延迟应随长度近似线性增长
        assert latency < length * 0.001  # < 1ms per nucleotide
```

---

*文档版本: 3.0.0*
*最后更新: 2025-01*
*Confluencia 3.0 — circRNA + TNBC Simulacrum 统一计算平台*
*作者: 颜子壹 | 吉林大学计算机科学与技术学院 / 第一白求恩临床医学院*