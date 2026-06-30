"""literature_tracker.py — circRNA 免疫学文献追踪系统

目的：
  1. 集中管理所有免疫感知相关的文献依据
  2. 支持版本追踪（哪些文献被推翻、哪些是新增）
  3. 自动生成 Methods 部分的文献引用
  4. 提醒用户代码与最新文献的同步状态

使用方式：
  from literature_tracker import LiteratureTracker, get_all_active_refs

  tracker = LiteratureTracker()
  tracker.list_active()
  tracker.check_superseded()
  tracker.generate_methods_section()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json


@dataclass
class LiteratureRef:
    """单条文献依据。"""
    key: str                    # 短键名（唯一标识）
    claim: str                  # 该文献支持的科学主张
    citation: str               # 完整引用（作者, 年份, 期刊, 卷:页）
    year: int                   # 发表年份
    pmid: Optional[str] = None  # PubMed ID（可选）
    doi: Optional[str] = None   # DOI（可选）

    # 适用性
    applies_to_circrna: bool = False  # 是否直接适用于 circRNA 语境

    # 状态
    status: str = "active"      # active / superseded / contested
    supersedes: List[str] = field(default_factory=list)  # 推翻了哪些旧依据
    superseded_by: List[str] = field(default_factory=list)  # 被哪些新依据推翻

    # 代码引用
    code_refs: List[str] = field(default_factory=list)  # 引用该文献的代码模块

    # 元数据
    added_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    notes: str = ""


class LiteratureTracker:
    """文献追踪器。"""

    def __init__(self):
        self._registry: Dict[str, LiteratureRef] = {}

    def register(self, ref: LiteratureRef) -> None:
        """注册一条文献。"""
        self._registry[ref.key] = ref

    def get(self, key: str) -> Optional[LiteratureRef]:
        """查询文献。"""
        return self._registry.get(key)

    def list_active(self) -> List[LiteratureRef]:
        """列出所有 active 状态的文献。"""
        return [r for r in self._registry.values() if r.status == "active"]

    def list_superseded(self) -> List[LiteratureRef]:
        """列出被推翻的旧文献。"""
        return [r for r in self._registry.values() if r.status == "superseded"]

    def list_contested(self) -> List[LiteratureRef]:
        """列出有争议的文献。"""
        return [r for r in self._registry.values() if r.status == "contested"]

    def list_by_year(self, min_year: int = 2020) -> List[LiteratureRef]:
        """列出指定年份之后的文献。"""
        return [r for r in self._registry.values() if r.year >= min_year]

    def list_circrna_specific(self) -> List[LiteratureRef]:
        """列出 circRNA 专用的文献。"""
        return [r for r in self._registry.values() if r.applies_to_circrna]

    def check_superseded(self) -> Dict[str, List[str]]:
        """检查被推翻的文献是否有残留代码引用。"""
        warnings = {}
        for ref in self.list_superseded():
            if ref.code_refs:
                warnings[ref.key] = ref.code_refs
        return warnings

    def generate_methods_section(self, section: str = "immunogenicity") -> str:
        """生成 Methods 部分的文献引用文本。

        Args:
            section: "immunogenicity" / "pk" / "structure" 等

        Returns:
            LaTeX 格式的 Methods 文本
        """
        refs = self.list_active()
        if section == "immunogenicity":
            refs = [r for r in refs if any(k in r.key for k in ["rig_i", "mda5", "tlr", "pkr", "circrig", "ds_crna"])]
        elif section == "pk":
            refs = [r for r in refs if "pk" in r.key.lower() or "half" in r.key.lower()]
        elif section == "structure":
            refs = [r for r in refs if "topology" in r.key.lower() or "bsj" in r.key.lower()]

        lines = []
        for ref in refs:
            lines.append(f"\\textit{{{ref.claim}}} \\cite{{{ref.key}}}")

        return "\n\n".join(lines)

    def to_json(self) -> str:
        """导出为 JSON。"""
        data = {}
        for key, ref in self._registry.items():
            data[key] = {
                "claim": ref.claim,
                "citation": ref.citation,
                "year": ref.year,
                "pmid": ref.pmid,
                "doi": ref.doi,
                "applies_to_circrna": ref.applies_to_circrna,
                "status": ref.status,
                "supersedes": ref.supersedes,
                "superseded_by": ref.superseded_by,
                "code_refs": ref.code_refs,
                "added_date": ref.added_date,
                "notes": ref.notes,
            }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def from_json(self, json_str: str) -> None:
        """从 JSON 导入。"""
        data = json.loads(json_str)
        for key, d in data.items():
            self.register(LiteratureRef(
                key=key,
                claim=d["claim"],
                citation=d["citation"],
                year=d["year"],
                pmid=d.get("pmid"),
                doi=d.get("doi"),
                applies_to_circrna=d.get("applies_to_circrna", False),
                status=d.get("status", "active"),
                supersedes=d.get("supersedes", []),
                superseded_by=d.get("superseded_by", []),
                code_refs=d.get("code_refs", []),
                added_date=d.get("added_date", ""),
                notes=d.get("notes", ""),
            ))


# ═══════════════════════════════════════════════════════════════
# 预注册的 circRNA 免疫学文献库
# ═══════════════════════════════════════════════════════════════

_global_tracker = LiteratureTracker()


def get_global_tracker() -> LiteratureTracker:
    """获取全局文献追踪器。"""
    return _global_tracker


def register_default_literature() -> None:
    """注册默认的 circRNA 免疫学文献库。"""
    tracker = get_global_tracker()

    # === RIG-I 识别 ===
    tracker.register(LiteratureRef(
        key="rig_i_dsrna_19bp",
        claim="RIG-I 识别 ≥19bp dsRNA，5'-三磷酸增强激活",
        citation="Schlee M, et al. (2009) Immunity 31:257-270; Baum A, et al. (2010) J Virol 84:3705-3709",
        year=2009,
        pmid="19631270",
        applies_to_circrna=False,  # 病毒 RNA 语境
        status="active",
        code_refs=["immune_sensing_v2._score_rig_i_v2", "immune_sensing_v3._score_rig_i_v3"],
        notes="RIG-I 在 circRNA 语境为次识别者（见 MDA5）",
    ))

    # === MDA5（circRNA 主识别者）===
    tracker.register(LiteratureRef(
        key="mda5_circrna_primary",
        claim="circRNA 共价闭环产生长 dsRNA（>500bp 连续），主要由 MDA5 识别而非 RIG-I",
        citation="Peisley A, Hur S. (2013) Nature 500:352-356; Wu Y, et al. (2022) Nat Commun circRIG-I",
        year=2022,
        pmid="35750088",
        applies_to_circrna=True,
        status="active",
        supersedes=["rig_i_dsrna_19bp"],  # 在 circRNA 语境部分推翻 RIG-I 主路径假设
        code_refs=["immune_sensing_v3._score_mda5_v3"],
        notes="V3 新增：MDA5 是 circRNA 长 dsRNA 主识别者",
    ))

    # === circRIG-I 反馈 ===
    tracker.register(LiteratureRef(
        key="circrig_i_feedback",
        claim="RIG-I 自身存在 circ 亚型（circRIG-I），通路在 circRNA 语境非单向'被识别'",
        citation="吕丹组 (2022) Nat Commun; 尹玉新组 (2023) Cell Rep PTENα/PTIR1",
        year=2022,
        pmid="35750088",
        applies_to_circrna=True,
        status="active",
        code_refs=["immune_sensing_v3._score_mda5_v3"],
        notes="V3 新增：circRIG-I 反馈调控因子",
    ))

    # === PKR ===
    tracker.register(LiteratureRef(
        key="pkr_dsrna_30bp",
        claim="PKR 识别 ≥30bp 连续 dsRNA，二聚化后抑制翻译",
        citation="Lemaire PA, et al. (2008) J Mol Biol 384:550-564",
        year=2008,
        pmid="18938177",
        applies_to_circrna=True,
        status="active",
        code_refs=["immune_sensing_v2._score_pkr_v2", "immune_sensing_v3._score_pkr_v3"],
    ))

    # === ds-cRNA 悖论 ===
    tracker.register(LiteratureRef(
        key="ds_crna_pkr_inhibition",
        claim="含双链区的短环状 RNA（ds-cRNA）对 PKR 免疫原性极低，反可抑制 PKR 过度激活",
        citation="陈玲玲组 (2024) Nat Biotechnol ds-cRNA 工作",
        year=2024,
        applies_to_circrna=True,
        status="active",
        code_refs=["immune_sensing_v3.compute_adaptive_weights_v3"],
        notes="V3 核心：推翻'dsRNA 高=免疫毒'假设，需看连续段长度+上下文",
    ))

    # === Ψ 修饰（circRNA 禁用）===
    tracker.register(LiteratureRef(
        key="psi_circrna_ires_disruption",
        claim="体外制备 circRNA 若用 Ψ 替代 U，会破坏 IRES 二级结构，导致环化失败、翻译失败",
        citation="复旦璩良团队（原魏文胜组 Cell 2022 circRNA 疫苗一作）(2026) Nature 子刊",
        year=2026,
        applies_to_circrna=True,
        status="active",
        supersedes=["kariko_2005_psi"],
        code_refs=["immune_sensing_v3.is_modification_safe_for_circrna"],
        notes="V3 核心：Ψ 在 linear mRNA 降免疫，在 circRNA 破坏环化",
    ))

    # === 被推翻：Ψ 在 linear mRNA ===
    tracker.register(LiteratureRef(
        key="kariko_2005_psi",
        claim="Ψ 修饰可降低 RIG-I 识别（linear mRNA 语境）",
        citation="Karikó K, et al. (2005) Immunity 23:165-175",
        year=2005,
        pmid="16111635",
        applies_to_circrna=False,
        status="superseded",
        superseded_by=["psi_circrna_ires_disruption"],
        code_refs=[],  # V3 已移除
        notes="在 circRNA 语境被复旦璩良 2026 推翻",
    ))

    # === m6A 修饰 ===
    tracker.register(LiteratureRef(
        key="anderson_2011_m6a",
        claim="m6A 修饰降低免疫激活，circRNA 中有效",
        citation="Anderson BR, et al. (2011) N Engl J Med; Chen YG, et al. (2019) Nature 586:651-655",
        year=2011,
        pmid="21798045",
        applies_to_circrna=True,
        status="active",
        code_refs=["immune_sensing_v3.get_modification_half_life_factor_circrna"],
    ))

    # === 拓扑约束警告 ===
    tracker.register(LiteratureRef(
        key="chen_lingling_2026_topology",
        claim="circRNA 拓扑约束直接左右翻译效率和免疫原性，但现有算法无法可靠建模",
        citation="陈玲玲组 (2026) Nature Chemical Biology 专访",
        year=2026,
        applies_to_circrna=True,
        status="active",
        code_refs=["immune_sensing_v3.analyze_dsrna_segments"],
        notes="BSJ ±30nt 区域高误差，需 confidence_interval",
    ))

    # === LNP 递送 ===
    tracker.register(LiteratureRef(
        key="lnp_uptake_hassett_2019",
        claim="LNP 递送系统摄取速率与给药途径相关",
        citation="Hassett KJ, et al. (2019) Mol Ther 27:1885-1897",
        year=2019,
        pmid="31474412",
        applies_to_circrna=True,
        status="active",
        code_refs=["pk.rnactm.infer_rna_ctm_params"],
    ))

    # === circRNA 半衰期 ===
    tracker.register(LiteratureRef(
        key="wesselhoeft_2018_circ_half_life",
        claim="circRNA 半衰期显著长于线性 mRNA，未修饰约 6-24h",
        citation="Wesselhoeft RA, et al. (2018) Nat Commun 9:2629",
        year=2018,
        pmid="29955060",
        applies_to_circrna=True,
        status="active",
        code_refs=["pk.rnactm.infer_rna_ctm_params"],
    ))


# 在模块加载时注册默认文献
register_default_literature()


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def get_all_active_refs() -> List[LiteratureRef]:
    """获取所有 active 状态的文献。"""
    return get_global_tracker().list_active()


def get_ref(key: str) -> Optional[LiteratureRef]:
    """查询单条文献。"""
    return get_global_tracker().get(key)


def check_superseded_refs() -> Dict[str, List[str]]:
    """检查被推翻的文献是否有残留代码引用。"""
    return get_global_tracker().check_superseded()


def generate_methods_section(section: str = "immunogenicity") -> str:
    """生成 Methods 文献引用。"""
    return get_global_tracker().generate_methods_section(section)


def export_literature_db() -> str:
    """导出文献库为 JSON。"""
    return get_global_tracker().to_json()


def print_literature_summary() -> None:
    """打印文献库摘要。"""
    tracker = get_global_tracker()
    print("=== circRNA 免疫学文献库摘要 ===")
    print(f"总计: {len(tracker._registry)} 条")
    print(f"Active: {len(tracker.list_active())} 条")
    print(f"Superseded: {len(tracker.list_superseded())} 条")
    print(f"Contested: {len(tracker.list_contested())} 条")
    print(f"circRNA 专用: {len(tracker.list_circrna_specific())} 条")
    print(f"2022+ 新文献: {len(tracker.list_by_year(2022))} 条")
    print()

    warnings = check_superseded_refs()
    if warnings:
        print("⚠️ 警告：被推翻的文献仍有代码引用：")
        for key, code_refs in warnings.items():
            print(f"  {key}: {code_refs}")
    else:
        print("✅ 无残留代码引用")


if __name__ == "__main__":
    print_literature_summary()
