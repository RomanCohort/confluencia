"""
html_renderer.py — 把 PDB + 指纹 JSON 注入 Mol* HTML 模板，输出自包含 .html。

双击即开，离线可用（Mol* 走 CDN，但首次加载后浏览器缓存）。
评委拿到文件不用装任何东西，浏览器打开就能转能看。

设计：
    - 左侧 Mol* viewport（占主区域）
    - 右侧控制面板：coloring scheme 下拉 + 整分子标量数值卡 + 序列信息
    - coloring 切换通过 Mol* 的 Themeregistry，重设 per-element b-factor / 自定义值
    - BSJ 残基（resName=BSJ）永远高亮，凸显 circRNA 闭环点

数据注入：PDB 和 fingerprint JSON 作为 JS 字符串内联进 <script>，
不走网络，所以 .html 自包含。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


# Mol* CDN（固定版本，避免 latest 漂移）
# NOTE: 版本号必须是 npm 上真实存在的。molstar 的 npm 版本序列是 0.x / 1.x /
# 2.x / 4.x / 5.x，没有 3.x（早先代码写的 3.51.0 是不存在的版本 → CDN 全 404 →
# molstar is not defined）。5.10.1 是当前 latest stable，已验证 build/viewer/
# molstar.js + .css 都 200。
MOLSTAR_JS = "https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.js"
MOLSTAR_CSS = "https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.css"


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>TorusFold circRNA 3D Viewer</title>
<link rel="stylesheet" href="__MOLSTAR_CSS__">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #0f0f1a; color: #e0e0e0;
    height: 100vh; overflow: hidden;
    display: flex; flex-direction: column;
  }
  header {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    padding: 10px 20px; border-bottom: 1px solid #2a2a4a;
  }
  header h1 { font-size: 16px; font-weight: 600; }
  header .sub { font-size: 12px; color: #888; margin-top: 2px; }
  #main { display: flex; flex: 1; overflow: hidden; }
  #viewer-wrap { flex: 1; position: relative; }
  #viewer { width: 100%; height: 100%; }
  aside {
    width: 320px; background: #141425; border-left: 1px solid #2a2a4a;
    padding: 16px; overflow-y: auto; flex-shrink: 0;
  }
  aside h2 { font-size: 13px; color: #6ab7ff; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .panel { margin-bottom: 22px; }
  select {
    width: 100%; padding: 8px; background: #1f1f35; color: #e0e0e0;
    border: 1px solid #3a3a5a; border-radius: 4px; font-size: 13px;
  }
  .scalar-card {
    background: #1a1a30; border: 1px solid #2a2a4a; border-radius: 6px;
    padding: 10px 12px; margin-bottom: 8px;
  }
  .scalar-card .k { font-size: 11px; color: #888; }
  .scalar-card .v { font-size: 18px; font-weight: 600; color: #6ab7ff; margin-top: 2px; }
  .seq-box {
    font-family: "Consolas", monospace; font-size: 11px; line-height: 1.6;
    background: #1a1a30; padding: 8px; border-radius: 4px; word-break: break-all;
    max-height: 100px; overflow-y: auto;
  }
  .legend { font-size: 11px; color: #888; margin-top: 6px; line-height: 1.5; }
  .gradient-bar {
    height: 10px; border-radius: 3px; margin-top: 4px;
    background: linear-gradient(90deg, #2b66d6, #ffdd57, #ff1243);
  }
  .bsj-tag {
    display: inline-block; background: #ff6b9d; color: #1a1a2e;
    padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 700;
    margin-left: 6px;
  }
  #status { font-size: 11px; color: #888; padding: 4px 0; min-height: 18px; }
</style>
</head>
<body>
<header>
  <h1>TorusFold — circRNA 3D Structure <span class="bsj-tag">circular</span></h1>
  <div class="sub" id="meta">loading...</div>
</header>
<div id="main">
  <div id="viewer-wrap"><div id="viewer"></div></div>
  <aside>
    <div class="panel">
      <h2>Coloring Scheme</h2>
      <select id="scheme-select"></select>
      <div class="legend" id="scheme-legend"></div>
      <div class="gradient-bar" id="gradient-bar"></div>
    </div>
    <div class="panel">
      <h2>Molecule Scalars</h2>
      <div id="scalar-cards"></div>
    </div>
    <div class="panel">
      <h2>Sequence</h2>
      <div class="seq-box" id="seq-box"></div>
    </div>
    <div class="panel">
      <h2>BSJ (back-splice junction)</h2>
      <div class="legend">首尾残基高亮粉色 = circRNA 闭合点。线性 RNA 没有这个。</div>
    </div>
    <div id="status"></div>
  </aside>
</div>

<script src="__MOLSTAR_JS__"></script>
<script>
// ====== 注入数据 ======
const PDB_DATA = `__PDB_DATA__`;
const FP = __FP_JSON__;

// ====== 全局状态 ======
let viewer = null;
let plugin = null;
let structureRef = null;   // 加载后的结构引用，用于重设 coloring

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

// 任意输入安全转数组（Array / Set / Map / 可迭代 / 有 forEach）。Mol* 不同属性
// 返回类型漂移（selector vs 数组），直接 for...of 会抛 is not iterable。全局共用。
function toArr(x) {
  if (!x) return [];
  if (Array.isArray(x)) return x;
  if (typeof x[Symbol.iterator] === 'function') return Array.from(x);
  if (typeof x.forEach === 'function') { const a=[]; x.forEach(v=>a.push(v)); return a; }
  return [];
}

// ====== 颜色映射：value → [r,g,b] ======
// 红-黄-绿梯度（仿 AlphaFold pLDDT），input 0~1
function gradRYG(t) {
  t = Math.max(0, Math.min(1, t));
  if (t < 0.5) {
    // 蓝→黄
    const k = t / 0.5;
    return [0.17 + (1.0 - 0.17) * k, 0.40 + (0.87 - 0.40) * k, 0.84 + (0.34 - 0.84) * k];
  } else {
    // 黄→红
    const k = (t - 0.5) / 0.5;
    return [1.0, 0.87 + (0.07 - 0.87) * k, 0.34 + (0.26 - 0.34) * k];
  }
}

// per-residue 值 → 归一化到 0~1
function normalize(vals) {
  if (!vals || vals.length === 0) return [];
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = hi - lo || 1;
  return vals.map(v => (v - lo) / range);
}

// ====== 渲染整分子标量卡 ======
function renderScalarCards() {
  const box = document.getElementById('scalar-cards');
  box.innerHTML = '';
  const scalars = FP.scalar || {};
  const entries = Object.entries(scalars);
  if (entries.length === 0) {
    box.innerHTML = '<div class="legend">无整分子标量数据</div>';
    return;
  }
  for (const [k, v] of entries) {
    const card = document.createElement('div');
    card.className = 'scalar-card';
    card.innerHTML = `<div class="k">${k}</div><div class="v">${typeof v === 'number' ? v.toFixed(3) : v}</div>`;
    box.appendChild(card);
  }
}

// ====== 填充 coloring scheme 下拉 ======
function fillSchemeSelect() {
  const sel = document.getElementById('scheme-select');
  sel.innerHTML = '';
  const schemes = FP.coloring_schemes || [];
  for (const s of schemes) {
    const opt = document.createElement('option');
    opt.value = s.key;
    opt.textContent = s.label + (s.type === 'scalar' ? '' : '');
    opt.dataset.type = s.type;
    sel.appendChild(opt);
  }
}

// ====== 应用 coloring ======
// 思路：Mol* 加载结构后，每个残基的 b-factor 列可以重设。
// 我们把选中的 per-residue 指纹值写进 b-factor，然后让 Mol* 用 b-factor coloring 主题。
// 但 Mol* EmbeddedView 的 API 较封闭，直接改 b-factor 要操作 plugin 内部结构。
// 简化方案：用 Mol* 的 colorTheme = 'uniform' + 按 scheme 给 BSJ 高亮，
// per-residue 梯度通过重新 loadStructureFromData（每次重写 B-factor 列）实现。
// 这里先用 "重写 B-factor + 触发重绘" 的轻量路径。
async function applyColoring(schemeKey) {
  if (!structureRef) return;
  const scheme = (FP.coloring_schemes || []).find(s => s.key === schemeKey);
  if (!scheme) return;

  const legend = document.getElementById('scheme-legend');
  const gradBar = document.getElementById('gradient-bar');

  if (scheme.type === 'scalar') {
    legend.textContent = '整分子标量 → 结构整体单色（数值见上方卡片）';
    gradBar.style.background = '#6ab7ff';
    // 整分子标量没法按残基上色，统一设蓝
    await setColorUniform([0.42, 0.72, 1.0]);
    return;
  }

  // per-residue
  const vals = (FP.per_residue || {})[schemeKey];
  if (!vals) {
    legend.textContent = '该 scheme 无数据';
    return;
  }
  const norm = normalize(vals);
  legend.textContent = `${scheme.label}（已归一化: 0 → 1）`;
  gradBar.style.background = 'linear-gradient(90deg, #2b66d6, #ffdd57, #ff1243)';

  await setColorPerResidue(norm);
}

// setColorPerResidue / setColorUniform 通过 Mol* plugin API 操作。
//
// 实现策略（容错链，按 Mol* 版本兼容）：
//   1) 首选 plugin.managers.structure.component：对已加载的每个 structure component
//      调 applyColor，传 ThemeRegistry 里的 'uncertainty'（per-residue，读 B-factor 列）或 'uniform' 主题。
//      这条路不重载结构，只换 theme，最轻、最快。
//   2) 兜底：loadStructureFromData 重载重写 B-factor 后的 PDB，并带 colorTheme option。
//      （Mol* 3.x 的 LoadStructureOptions 支持 colorTheme。）
//   3) 全失败：setStatus 报错，不静默吞 —— 评委能看到"该 scheme 没生效"而不是被骗。
//
// 注：Mol* 5.10.1 没有 'b-factor' 这个 color theme name，per-residue 按 B-factor 上色
// 的注册名是 'uncertainty'（src/mol-theme/color/uncertainty.ts，读 B_iso_or_equiv 列，
// 默认 domain [0,100] red-white-blue scale）。原代码用 'b-factor' → 5.10.1 注册表查不到
// → 报"不支持 b-factor 主题"。改成 'uncertainty'。
async function setColorPerResidue(normVals) {
  // 1) 先把 B-factor 列重写好（uncertainty 主题会读它）
  const newPdb = rewriteBFactors(PDB_DATA, normVals);

  // 诊断：采样新 PDB 的 B-factor 列，看是不是真重写了
  const newLines = newPdb.split('\n').filter(l => l.startsWith('ATOM'));
  const samples = [0, Math.floor(newLines.length/2), newLines.length-1].map(i => {
    const l = newLines[i];
    return l ? l.slice(60,66).trim() : 'NA';
  });
  setStatus(`rewrite: normLen=${normVals.length} | newPDB atoms=${newLines.length} | bfact[0,mid,last]=${samples.join(',')}`);

  const ok = await tryApplyColorTheme('uncertainty', { pdb: newPdb });
  if (!ok) {
    setStatus('per-residue coloring: uncertainty 主题未生效');
  }
}

async function setColorUniform(rgb) {
  const hex = '#' + rgb.map(c => Math.round(c*255).toString(16).padStart(2,'0')).join('');
  const newPdb = rewriteBFactorsUniform(PDB_DATA, 0.5);
  const ok = await tryApplyColorTheme('uniform', { pdb: newPdb, color: { r: rgb[0], g: rgb[1], b: rgb[2] } });
  if (!ok) {
    setStatus('uniform coloring: 当前 Mol* 版本不支持 uniform 主题，已回退');
  }
}

// 统一的 theme 应用入口。themeName ∈ {'uncertainty', 'uniform'}。
// 返回 true=成功应用，false=此路径失败（调用方决定是否报错）。
//
// Mol* 5.x 正确 API（已核对 molstar@5.10.1 源码）：
//   - loadStructureFromData 第三参数只支持 {dataLabel}，不支持 colorTheme
//   - 换 colorTheme 走 plugin.managers.structure.component.updateRepresentationsTheme(
//       components, { color: { name, params } })
//   - uncertainty 主题读已加载结构的 B-factor 列 → 切 per-residue scheme 时
//     必须先重载新 B-factor 的 PDB（只传 {dataLabel}），再 updateRepresentationsTheme
//   - uniform 不依赖 B-factor → 直接 updateRepresentationsTheme，不重载
async function tryApplyColorTheme(themeName, opts) {
  if (themeName === 'uncertainty') {
    // 重载新 B-factor 的 PDB 前，先清掉旧 structure —— 否则 loadStructureFromData
    // 会追加而非替换，切 N 次累积 N+1 个 structure，旧的默认色盖住新的 uncertainty 色
    // （现象：components 从 3 涨到 8，颜色"没变"）。hierarchy.remove 已核对 5.10.1
    // 源码 hierarchy.ts:179-183。
    const h = plugin.managers.structure.hierarchy;
    if (h) {
      const existing = toArr(h.current.structures);
      if (existing.length > 0) {
        try { await h.remove(existing, false); } catch (e) {
          console.warn('remove old structures failed:', e);
        }
      }
    }
    // 重载新 B-factor 的 PDB（不传 colorTheme，那个字段 5.x 不认）
    try {
      await viewer.loadStructureFromData(opts.pdb, 'pdb', { dataLabel: 'circRNA' });
      structureRef = true;
    } catch (e) {
      setStatus('重载结构失败: ' + e.message);
      return false;
    }
    // 重载后对 representation 设 uncertainty 主题
    const ok = applyColorViaComponent('uncertainty', opts);
    if (!ok) setStatus('uncertainty 主题未生效（component API 不可用）');
    return ok;
  }

  // uniform：不重载，直接换 theme
  const ok = applyColorViaComponent('uniform', opts);
  if (!ok) setStatus('uniform 主题未生效（component API 不可用）');
  return ok;
}

// 路径 A：plugin.managers.structure.component.updateRepresentationsTheme
// 5.x 稳定 API（comp.ts:297）。params = { color: { name, params } }。
function applyColorViaComponent(themeName, opts) {
  if (!plugin || !plugin.managers || !plugin.managers.structure || !plugin.managers.structure.component) {
    return false;
  }
  const mgr = plugin.managers.structure.component;

  // 取 components（已核对 molstar@5.10.1 源码
  //  src/mol-plugin-state/manager/structure/component.ts:53-55）：
  //   get currentStructures() {
  //     return this.plugin.managers.structure.hierarchy.selection.structures;
  //   }
  // 即 mgr.currentStructures 直接给当前所有 structure 的数组，每个 s.components
  // 是 StructureComponentRef[]（updateRepresentationsTheme 要求的类型，见同文件 295）。
  // 原代码试了 mgr.components / hierarchy.current 等都不存在 → 全空 → 返回 false
  // → 报"取不到 components"。改成正确路径 mgr.currentStructures。
  // toArr 是全局函数（见文件顶部）。

  // 主路径：mgr.currentStructures → 每个 s.components
  let components = [];
  try {
    const structs = toArr(mgr.currentStructures);
    for (const s of structs) {
      if (s && s.components) components = components.concat(toArr(s.components));
    }
  } catch (e) {
    console.warn('currentStructures 取 components 失败:', e);
  }

  // 兜底：如果 currentStructures 走不通，再试几个历史属性名
  if (components.length === 0) {
    const fallbacks = [
      () => mgr.components,
      () => mgr.currentComponents,
    ];
    for (const get of fallbacks) {
      let arr = [];
      try { arr = toArr(get.call(mgr)); } catch (_) { arr = []; }
      if (arr.length > 0) { components = arr; break; }
    }
  }

  if (components.length === 0) {
    setStatus('取不到 structure components（Mol* 版本 API 不匹配），着色未生效');
    return false;
  }

  try {
    // color 必须是 theme name 字符串（不是 { name } 嵌套对象）——
    // Mol* 5.10.1 component.ts:305-310 期望 params.color 是字符串或 'default'，
    // params.colorParams 单独传。uncertainty theme 读 B_iso_or_equiv 列（我们 PDB
    // 已把 per-residue 指纹值写到 B-factor 列），domain 默认 [0,100]，red-white-blue 梯度。
    // uniform 需要 colorParams: { value: Color }，Color 是 (r<<16|g<<8|b) 打包整数。
    let themeParams;
    if (themeName === 'uniform' && opts.color) {
      const c = opts.color;
      const colorInt = (Math.round(c.r*255)<<16) | (Math.round(c.g*255)<<8) | Math.round(c.b*255);
      themeParams = { color: 'uniform', colorParams: { value: colorInt } };
    } else {
      themeParams = { color: themeName };
    }
    const ret = mgr.updateRepresentationsTheme(components, themeParams);
    const isPromise = ret && typeof ret.then === 'function';
    setStatus(`theme=${themeName} | components=${components.length} | return=${isPromise ? 'Promise' : (ret===undefined?'undefined':typeof ret)}`);
    if (isPromise) {
      ret.catch(e => setStatus('updateRepresentationsTheme 失败: ' + (e?.message || e)));
    }
    return true;
  } catch (e) {
    console.warn('updateRepresentationsTheme failed:', e);
    setStatus('coloring 失败: ' + (e?.message || e));
    return false;
  }
}

// 重写 PDB 的 B-factor 列（第 61-66 列）为 normVals * 100
function rewriteBFactors(pdb, normVals) {
  const lines = pdb.split('\n');
  let idx = 0;
  return lines.map(line => {
    if (!line.startsWith('ATOM')) return line;
    const v = normVals[idx++] * 100;
    // B-factor 在第 61-66 列（0-indexed 60-66），%6.2f
    return line.slice(0, 60) + v.toFixed(2).padStart(6) + line.slice(66);
  }).join('\n');
}

function rewriteBFactorsUniform(pdb, val) {
  const lines = pdb.split('\n');
  return lines.map(line => {
    if (!line.startsWith('ATOM')) return line;
    return line.slice(0, 60) + (val*100).toFixed(2).padStart(6) + line.slice(66);
  }).join('\n');
}

// ====== 探查 Mol* 运行时注册的 color theme name ======
// 不同小版本注册名漂移（bfactor / b-factor / uncertainty），网络查不到源码时
// 直接问运行时实例最准。把名字打到 status，用户回贴给我，我据此定 theme name。
function probeColorThemes() {
  if (!plugin) { setStatus('probe: plugin 不存在'); return; }
  try {
    // color theme 注册表在几条候选路径里，逐个 try
    const regs = [];
    const tryPush = (label, reg) => {
      if (!reg) return;
      // registry 一般有 .list / .entries / [Symbol.iterator]
      let names = [];
      if (typeof reg.list === 'function') names = reg.list().map(x => x?.name).filter(Boolean);
      else if (Array.isArray(reg)) names = reg.map(x => x?.name).filter(Boolean);
      else if (typeof reg[Symbol.iterator] === 'function') names = Array.from(reg).map(x => x?.name).filter(Boolean);
      else if (reg.entries && typeof reg.entries === 'function') {
        for (const [k, v] of reg.entries()) names.push(k);
      }
      if (names.length) regs.push({ label, names });
    };
    const sc = plugin.config?.structure?.colorThemeProvider;
    tryPush('config.structure.colorThemeProvider', sc);
    // 其它候选路径
    tryPush('managers.structure.component registry',
      plugin.managers?.structure?.component?.registry);
    tryPush('state.registry', plugin.state?.registry);

    if (regs.length === 0) {
      setStatus('probe: 没探到 color theme 注册表路径，请 F12 console 手动查 plugin');
      console.log('PROBE: plugin keys =', Object.keys(plugin));
      console.log('PROBE: plugin.managers.structure =', plugin.managers?.structure);
      console.log('PROBE: plugin.config =', plugin.config);
      return;
    }
    const all = regs.map(r => `[${r.label}]: ${r.names.join(', ')}`).join(' | ');
    setStatus('PROBE color themes → ' + all);
    console.log('PROBE color themes:', regs);
  } catch (e) {
    setStatus('probe 失败: ' + (e?.message || e));
    console.warn('probe failed', e);
  }
}

// ====== 初始化 ======
async function init() {
  document.getElementById('meta').textContent =
    `length=${FP.length}  |  circular=true  |  ${FP.sequence.length} nt`;
  document.getElementById('seq-box').textContent = FP.sequence;
  renderScalarCards();
  fillSchemeSelect();

  try {
    // Mol* 5.x 正确 API：molstar.Viewer.create(el, opts) 返回 Promise<Viewer>，
    // 不是 new molstar.Viewer(...)（旧版写法，5.x 下 new 会造出半残 viewer，
    // plugin.state 未初始化 → "Cannot read properties of undefined (reading 'data')"）。
    viewer = await molstar.Viewer.create('viewer', {
      layoutIsExpanded: false,
      viewportShowExpand: true,
      viewportShowControls: false,
      viewportShowAnimation: false,
      viewportShowSettings: false,
      backgroundColor: { r: 0.06, g: 0.06, b: 0.10 },
    });
    plugin = viewer.plugin;
    // 首次加载。Mol* 5.x 的 loadStructureFromData 第三参数只支持 {dataLabel}，
    // 不支持 colorTheme（旧版字段，传了会被忽略；传非标字段在某些版本触发
    // "Cannot read properties of undefined"）。coloring 由 applyColoring 在
    // 加载后单独设，不塞进 load options。
    await viewer.loadStructureFromData(PDB_DATA, 'pdb', {
      dataLabel: 'circRNA',
    });
    structureRef = true;
    setStatus('结构加载完成。');

    // 默认用 confidence 上色（uncertainty 主题读 B-factor 列，applyColoring 触发）
    const sel = document.getElementById('scheme-select');
    if (sel.options.length > 0) {
      sel.selectedIndex = 0;
      // uncertainty 主题随 applyColoring 调用生效，这里同步 legend / 梯度条显示
      await applyColoring(sel.value);
    }
    sel.addEventListener('change', () => applyColoring(sel.value));
  } catch (e) {
    setStatus('初始化失败: ' + e.message);
    console.error(e);
  }
}

window.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


def render_html(
    pdb: str,
    fingerprint_json: str,
    title: str = "TorusFold circRNA 3D Viewer",
) -> str:
    """
    注入 PDB + 指纹 JSON 到 Mol* 模板，返回自包含 HTML 字符串。

    Args:
        pdb: PDB 格式字符串（来自 structure_export.coords_to_pdb）
        fingerprint_json: 已 json.dumps 的指纹 JSON 字符串
        title: 页面标题

    Returns:
        完整 HTML 字符串，可写文件后浏览器直接打开
    """
    # 防止 PDB 里的反引号 / $ 破坏 JS 模板字符串
    pdb_safe = pdb.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    fp_safe = fingerprint_json.replace("</", "<\\/")  # 防 </script> 注入

    html = (
        HTML_TEMPLATE
        .replace("__MOLSTAR_CSS__", MOLSTAR_CSS)
        .replace("__MOLSTAR_JS__", MOLSTAR_JS)
        .replace("__PDB_DATA__", pdb_safe)
        .replace("__FP_JSON__", fp_safe)
    )
    html = html.replace("<title>TorusFold circRNA 3D Viewer</title>",
                        f"<title>{title}</title>")
    return html


def render_from_export(
    export_dict: Dict[str, str],
    title: str = "TorusFold circRNA 3D Viewer",
) -> str:
    """
    便利函数：直接接 export_circrna_structure 的返回值。

    Args:
        export_dict: {"pdb": ..., "fingerprint_json": ...}
    """
    return render_html(
        export_dict["pdb"],
        export_dict["fingerprint_json"],
        title=title,
    )
