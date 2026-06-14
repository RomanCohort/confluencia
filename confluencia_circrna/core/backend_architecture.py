"""
Confluencia Backend Architecture Design
=========================================

核心设计：灵活的orchestrator + 可插拔的backend

作者：颜子壹
吉林大学计算机科学与技术学院
吉林大学第一白求恩临床医学院
"""

# ============================================================
# 1. 架构设计
# ============================================================

class ConfluenciaEvaluator:
    """
    Confluencia主评估器 - 灵活的orchestrator

    Design Philosophy:
    - 研究亲和性：给用户选择权
    - 灵活性：可插拔backend
    - 离线优先：默认离线可用，可选在线API
    """

    def __init__(self,
                 immunogenicity_backend="heuristic",  # heuristic/vienna/esm2
                 mhc_backend="local",                 # local/netmhcpan
                 drug_backend="local",                # local/external
                 pk_backend="rnactm"):                # rnactm/external
        """
        初始化评估器，配置backend

        Parameters:
        -----------
        immunogenicity_backend : str
            "heuristic" - 快速heuristic模型（默认，离线）
            "vienna" - ViennaRNA结构预测（高精度，离线）
            "esm2" - ESM-2蛋白语言模型（最高精度，需要GPU）

        mhc_backend : str
            "local" - 本地训练模型（AUC=0.80，离线，快）
            "netmhcpan" - NetMHCpan API（AUC=0.92-0.96，在线，高精度）

        drug_backend : str
            "local" - 本地模型（离线）
            "external" - 外部API（在线）
        """
        self.immunogenicity_backend = immunogenicity_backend
        self.mhc_backend = mhc_backend
        self.drug_backend = drug_backend

        # 初始化backend
        self._init_backends()

    def _init_backends(self):
        """初始化各个backend"""
        # 免疫原性backend
        if self.immunogenicity_backend == "heuristic":
            self._imm_engine = HeuristicImmunogenicity()
        elif self.immunogenicity_backend == "vienna":
            self._imm_engine = ViennaImmunogenicity()
        elif self.immunogenicity_backend == "esm2":
            self._imm_engine = ESM2Immunogenicity()

        # MHC backend
        if self.mhc_backend == "local":
            self._mhc_engine = LocalMHC()
        elif self.mhc_backend == "netmhcpan":
            self._mhc_engine = NetMHCpanAPI()

        # Drug backend
        if self.drug_backend == "local":
            self._drug_engine = LocalDrugBinding()
        elif self.drug_backend == "external":
            self._drug_engine = ExternalDrugAPI()


# ============================================================
# 2. 免疫原性Backend设计
# ============================================================

class ImmunogenicityBackendBase:
    """免疫原性评分基类"""

    def score(self, sequence, modification=None):
        """
        计算免疫原性评分

        Returns:
        --------
        dict with keys:
            - overall: 总评分
            - rig_i: RIG-I通路评分
            - tlr7: TLR7评分
            - tlr8: TLR8评分
            - pkr: PKR评分
            - backend: 使用的backend名称
            - metadata: backend特定元数据
        """
        raise NotImplementedError


class HeuristicImmunogenicity(ImmunogenicityBackendBase):
    """
    Heuristic免疫原性评分（当前模型）

    特点：
    - 快速（~85ms）
    - 离线可用
    - 精度中等
    """

    def score(self, sequence, modification=None):
        # 当前的评分逻辑
        rig_i_score = self._score_rig_i(sequence)
        tlr7_score = self._score_tlr7(sequence)
        tlr8_score = self._score_tlr8(sequence)
        pkr_score = self._score_pkr(sequence)

        overall = 0.35 * rig_i_score + \
                  0.20 * tlr7_score + \
                  0.15 * tlr8_score + \
                  0.30 * pkr_score

        return {
            "overall": overall,
            "rig_i": rig_i_score,
            "tlr7": tlr7_score,
            "tlr8": tlr8_score,
            "pkr": pkr_score,
            "backend": "heuristic",
            "metadata": {
                "weights": "heuristic (RIG-I=0.35, TLR7=0.20, TLR8=0.15, PKR=0.30)",
                "note": "Author-informed heuristics, not empirically calibrated"
            }
        }


class ViennaImmunogenicity(ImmunogenicityBackendBase):
    """
    ViennaRNA增强免疫原性评分

    新功能：
    - TLR结构可及性（ViennaRNA unpaired probability）
    - 更精确的dsRNA backbone检测

    特点：
    - 中等速度（~150ms）
    - 离线可用
    - 精度较高
    """

    def score(self, sequence, modification=None):
        import ViennaRNA

        # 获取RNA结构
        fc = ViennaRNA.fold_compound(sequence)
        mfe_structure, mfe = fc.mfe()

        # RIG-I评分（与heuristic相同，但用ViennaRNA精确结构）
        rig_i_score = self._score_rig_i_vienna(mfe_structure, mfe)

        # TLR评分（新增：结构可及性）
        tlr7_score = self._score_tlr7_vienna(sequence, mfe_structure)
        tlr8_score = self._score_tlr8_vienna(sequence, mfe_structure)

        # PKR评分
        pkr_score = self._score_pkr_vienna(mfe_structure)

        overall = 0.35 * rig_i_score + \
                  0.20 * tlr7_score + \
                  0.15 * tlr8_score + \
                  0.30 * pkr_score

        return {
            "overall": overall,
            "rig_i": rig_i_score,
            "tlr7": tlr7_score,
            "tlr8": tlr8_score,
            "pkr": pkr_score,
            "backend": "vienna",
            "metadata": {
                "mfe": mfe,
                "structure": mfe_structure,
                "unpaired_prob": self._get_unpaired_probs(fc),
                "note": "ViennaRNA-enhanced with structural accessibility"
            }
        }

    def _score_tlr7_vienna(self, sequence, structure):
        """TLR7评分 + 结构可及性"""
        # 基础motif计数
        base_score = self._count_gu_motifs(sequence)

        # 新增：结构可及性权重
        accessibility = self._get_motif_accessibility(sequence, structure)

        # 可及性加权评分
        return base_score * accessibility

    def _get_motif_accessibility(self, sequence, structure):
        """计算motif在unpaired region的比例"""
        # 找到motif位置
        motif_positions = self._find_gu_motifs(sequence)

        # 计算每个motif是否在unpaired region
        unpaired_count = 0
        for pos in motif_positions:
            if structure[pos] == '.':
                unpaired_count += 1

        return unpaired_count / len(motif_positions) if motif_positions else 0.5


class ESM2Immunogenicity(ImmunogenicityBackendBase):
    """
    ESM-2蛋白语言模型免疫原性评分

    特点：
    - 慢（~2-5s）
    - 需要GPU
    - 最高精度（待验证）
    """

    def score(self, sequence, modification=None):
        import torch
        from esm import pretrained

        # 加载ESM-2模型
        model, alphabet = pretrained.esm2_t33_650M_UR50D()

        # 编码序列
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq', sequence)])

        # 掐取embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33]

        # 用embeddings预测免疫原性（需要训练的下游模型）
        # TODO: 需要训练或使用预训练的免疫原性预测头

        return {
            "overall": self._predict_from_embeddings(embeddings),
            "backend": "esm2",
            "metadata": {
                "model": "esm2_t33_650M_UR50D",
                "embedding_dim": 1280,
                "note": "Experimental - requires downstream model training"
            }
        }


# ============================================================
# 3. MHC Backend设计
# ============================================================

class MHCBackendBase:
    """MHC结合预测基类"""

    def predict(self, sequence, alleles=None):
        """
        预测MHC结合

        Parameters:
        -----------
        sequence : str
            蛋白序列或circRNA编码的蛋白
        alleles : list
            要预测的MHC alleles列表

        Returns:
        --------
        dict with keys:
            - predictions: {allele: score} dict
            - backend: backend名称
            - auc_estimate: 该backend的预期AUC
            - metadata: 其他信息
        """
        raise NotImplementedError


class LocalMHC(MHCBackendBase):
    """
    本地MHC预测模型

    特点：
    - AUC = 0.80
    - 快速（~50ms）
    - 离线可用
    - 支持所有246 alleles
    """

    def predict(self, sequence, alleles=None):
        # 当前的本地预测逻辑
        predictions = {}
        for allele in (alleles or self._default_alleles):
            predictions[allele] = self._predict_single(sequence, allele)

        return {
            "predictions": predictions,
            "backend": "local",
            "auc_estimate": 0.80,
            "metadata": {
                "note": "For high-accuracy prediction, use backend='netmhcpan'",
                "training_data": "52K IEDB binary, 246 alleles"
            }
        }


class NetMHCpanAPI(MHCBackendBase):
    """
    NetMHCpan API集成

    特点：
    - AUC = 0.92-0.96 (业界最佳)
    - 中等速度（~200ms）
    - 需要网络
    - 需要NetMHCpan服务
    """

    def __init__(self, api_url=None, api_key=None):
        self.api_url = api_url or "https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/"
        self.api_key = api_key

    def predict(self, sequence, alleles=None):
        import requests

        # 调用NetMHCpan API
        response = requests.post(
            self.api_url,
            data={
                "sequence": sequence,
                "alleles": alleles or self._default_alleles,
                "api_key": self.api_key
            }
        )

        predictions = self._parse_api_response(response)

        return {
            "predictions": predictions,
            "backend": "netmhcpan",
            "auc_estimate": 0.92,
            "metadata": {
                "note": "Industry-best MHC binding predictor",
                "source": "NetMHCpan 4.1",
                "url": self.api_url
            }
        }


# ============================================================
# 4. 用户接口
# ============================================================

def evaluate(sequence,
             immunogenicity_backend="heuristic",
             mhc_backend="local",
             drug_backend="local"):
    """
    快速评估函数

    Example Usage:
    --------------

    # 默认快速评估
    result = evaluate(sequence)

    # 高精度评估（使用ViennaRNA + NetMHCpan）
    result = evaluate(sequence,
                      immunogenicity_backend="vienna",
                      mhc_backend="netmhcpan")

    # 最高精度评估（ESM-2 + NetMHCpan）
    result = evaluate(sequence,
                      immunogenicity_backend="esm2",
                      mhc_backend="netmhcpan")
    """
    evaluator = ConfluenciaEvaluator(
        immunogenicity_backend=immunogenicity_backend,
        mhc_backend=mhc_backend,
        drug_backend=drug_backend
    )

    return evaluator.evaluate(sequence)


# ============================================================
# 5. Backend性能对比表
# ============================================================

BACKEND_COMPARISON = {
    "immunogenicity": {
        "heuristic": {
            "speed": "85ms",
            "accuracy": "medium",
            "offline": True,
            "description": "Default fast heuristic model"
        },
        "vienna": {
            "speed": "150ms",
            "accuracy": "high",
            "offline": True,
            "description": "ViennaRNA-enhanced with structural accessibility"
        },
        "esm2": {
            "speed": "2-5s",
            "accuracy": "highest (experimental)",
            "offline": True,
            "gpu_required": True,
            "description": "ESM-2 embeddings-based prediction"
        }
    },
    "mhc": {
        "local": {
            "speed": "50ms",
            "auc": 0.80,
            "offline": True,
            "description": "Local trained model"
        },
        "netmhcpan": {
            "speed": "200ms",
            "auc": "0.92-0.96",
            "offline": False,
            "description": "Industry-best NetMHCpan API"
        }
    }
}


# ============================================================
# 6. 论文中如何描述
# ============================================================

METHODS_DESCRIPTION = """
\textbf{Flexible Backend Architecture:}
Confluencia supports multiple backend options for each prediction module:

\textbf{Immunogenicity:}
- `heuristic` (default): Fast heuristic model with pathway weights
- `vienna`: ViennaRNA-enhanced with TLR structural accessibility
- `esm2`: ESM-2 embeddings-based prediction (experimental)

\textbf{MHC Binding:}
- `local` (default): Local trained model (AUC=0.80)
- `netmhcpan`: NetMHCpan API integration (AUC=0.92-0.96, industry-best)

This design allows users to choose accuracy-speed trade-offs based on
their research requirements. Default backends prioritize offline
availability and speed; switching to external backends provides higher
accuracy when network access is available.
"""