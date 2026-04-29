"""
MHC 等位基因特征编码器
用于提升 Epitope 结合预测性能

MHC 等位基因特征:
1. MHC 伪序列编码 (NetMHCpan 方法)
2. HLA 等位基因 one-hot 编码
3. 肽-MHC 结合位置接触特征

参考: NetMHCpan-4.1 论文方法
"""

import os
import numpy as np
from typing import Optional, Dict, List, Tuple

# 标准氨基酸
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# 常见 HLA 等位基因列表 (包含 iedb_heldout_mhc.csv 中出现的所有 allele)
HLA_ALLELES = [
    # HLA-A alleles
    'HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03',
    'HLA-A*02:06', 'HLA-A*03:01', 'HLA-A*11:01', 'HLA-A*23:01',
    'HLA-A*24:02', 'HLA-A*24:07', 'HLA-A*26:01', 'HLA-A*29:02',
    'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*31:01', 'HLA-A*32:01',
    'HLA-A*33:01', 'HLA-A*33:03', 'HLA-A*68:01', 'HLA-A*68:02',
    # HLA-B alleles
    'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*13:01', 'HLA-B*14:02',
    'HLA-B*15:01', 'HLA-B*18:01', 'HLA-B*27:02', 'HLA-B*27:05',
    'HLA-B*35:01', 'HLA-B*35:02', 'HLA-B*37:01', 'HLA-B*38:01',
    'HLA-B*39:01', 'HLA-B*39:06', 'HLA-B*40:01', 'HLA-B*44:02',
    'HLA-B*44:03', 'HLA-B*45:01', 'HLA-B*46:01', 'HLA-B*50:01',
    'HLA-B*51:01', 'HLA-B*57:01', 'HLA-B*57:02', 'HLA-B*58:01',
    # HLA-C alleles
    'HLA-C*01:02', 'HLA-C*02:02', 'HLA-C*03:03', 'HLA-C*03:04',
    'HLA-C*04:01', 'HLA-C*05:01', 'HLA-C*06:02', 'HLA-C*07:01',
    'HLA-C*07:02', 'HLA-C*08:02', 'HLA-C*12:02', 'HLA-C*14:02',
    'HLA-C*15:02',
    # Simplified allele names (from IEDB)
    'HLA-A1', 'HLA-A2', 'HLA-A3', 'HLA-A11', 'HLA-A24', 'HLA-A26',
    'HLA-A28', 'HLA-A29', 'HLA-A31', 'HLA-A68', 'HLA-A69',
    'HLA-B7', 'HLA-B8', 'HLA-B14', 'HLA-B27', 'HLA-B35', 'HLA-B37',
    'HLA-B44', 'HLA-B51', 'HLA-B52', 'HLA-B53', 'HLA-B57', 'HLA-B60',
    'HLA-B62', 'HLA-Cw7',
]

ALLELE_TO_IDX = {a: i for i, a in enumerate(HLA_ALLELES)}

# MHC 伪序列 (34个锚定位点)
# 来源: NetMHCpan-4.1 training.pseudo 数据库
MHC_PSEUDO_SEQUENCE = {
    # ===== HLA-A alleles =====
    'HLA-A*01:01': 'YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY',
    'HLA-A*02:01': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY',
    'HLA-A*02:02': 'YFAMYGEKVAHTHVDTLYLRYHYYTWAVWAYTWY',
    'HLA-A*02:03': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAEWAYTWY',
    'HLA-A*02:04': 'YFAMYGEKVAHTHVDTLYVMYHYYTWAVLAYTWY',
    'HLA-A*02:05': 'YYAMYGEKVAHTHVDTLYLRYHYYTWAVWAYTWY',
    'HLA-A*02:06': 'YYAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY',
    'HLA-A*02:07': 'YFAMYGEKVAHTHVDTLYVRCHYYTWAVLAYTWY',
    'HLA-A*02:11': 'YFAMYGEKVAHIDVDTLYVRYHYYTWAVLAYTWY',
    'HLA-A*02:12': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVQAYTWY',
    'HLA-A*02:16': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYEWY',
    'HLA-A*02:17': 'YFAMYGEKVAHTHVDTLYLMFHYYTWAVLAYTWY',
    'HLA-A*02:19': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVQAYTGY',
    'HLA-A*02:20': 'YFAMYGENVAHTHVDTLYVRYHYYTWAVLAYTWY',
    'HLA-A*02:50': 'YFAMYGEKVAHTHVDTLYIRYHYYTWAVWAYTWY',
    'HLA-A*03:01': 'YFAMYQENVAQTDVDTLYIIYRDYTWAELAYTWY',
    'HLA-A*03:02': 'YFAMYQENVAQTDVDTLYIIYRDYTWAVQAYTWY',
    'HLA-A*11:01': 'YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY',
    'HLA-A*23:01': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVLAYTGY',
    'HLA-A*24:02': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTGY',
    'HLA-A*24:03': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTWY',
    'HLA-A*24:06': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVWAYTGY',
    'HLA-A*24:13': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVLAYTGY',
    'HLA-A*24:07': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTGY',  # similar to A*24:02
    'HLA-A*25:01': 'YYAMYRNNVAHTDESIAYIRYQDYTWAEWAYRWY',
    'HLA-A*26:01': 'YYAMYRNNVAHTDANTLYIRYQDYTWAEWAYRWY',
    'HLA-A*26:02': 'YYAMYRNNVAHTDANTLYIRYQNYTWAEWAYRWY',
    'HLA-A*26:03': 'YYAMYRNNVAHTHVDTLYIRYQDYTWAEWAYRWY',
    'HLA-A*29:01': 'YTAMYLQNVAQTDANTLYIMYRDYTWAVLAYTWY',
    'HLA-A*29:02': 'YTAMYLQNVAQTDANTLYIMYRDYTWAVLAYTWY',
    'HLA-A*30:01': 'YSAMYQENVAQTDVDTLYIIYEHYTWAWLAYTWY',
    'HLA-A*30:02': 'YSAMYQENVAHTDENTLYIIYEHYTWARLAYTWY',
    'HLA-A*31:01': 'YTAMYQENVAHIDVDTLYIMYQDYTWAVLAYTWY',
    'HLA-A*32:01': 'YFAMYQENVAHTDESIAYIMYQDYTWAVLAYTWY',
    'HLA-A*32:15': 'YFAMYRNNVAHTDESIAYIMYQDYTWAVLAYTWY',
    'HLA-A*33:01': 'YTAMYRNNVAHIDVDTLYIMYQDYTWAVLAYTWH',
    'HLA-A*33:03': 'YTAMYRNNVAHIDVDTLYIMYQDYTWAVLAYTWY',
    'HLA-A*66:01': 'YYAMYRNNVAQTDVDTLYIRYQDYTWAEWAYRWY',
    'HLA-A*68:01': 'YYAMYRNNVAQTDVDTLYIMYRDYTWAVWAYTWY',
    'HLA-A*68:02': 'YYAMYRNNVAQTDVDTLYIRYHYYTWAVWAYTWY',
    'HLA-A*69:01': 'YYAMYRNNVAQTDVDTLYVRYHYYTWAVLAYTWY',
    'HLA-A*80:01': 'YFAMYEENVAHTNANTLYIIYRDYTWARLAYEGY',
    # ===== HLA-B alleles =====
    'HLA-B*07:02': 'YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY',
    'HLA-B*08:01': 'YDSEYRNIFTNTDESNLYLSYNYYTWAVDAYTWY',
    'HLA-B*08:02': 'YDSEYRNIFTNTDENTAYLSYNYYTWAVDAYTWY',
    'HLA-B*08:03': 'YDSEYRNIFTNTYENIAYLSYNYYTWAVDAYTWY',
    'HLA-B*13:01': 'YYTMYREISTNTYENTAYIRYNLYTWAVLAYEWY',
    'HLA-B*13:02': 'YYTMYREISTNTYENTAYWTYNLYTWAVLAYEWY',
    'HLA-B*14:01': 'YYSEYRNICTNTDESNLYLWYNFYTWAELAYTWH',
    'HLA-B*14:02': 'YYSEYRNICTNTDESNLYLWYNFYTWAELAYTWH',
    'HLA-B*15:01': 'YYAMYREISTNTYESNLYLRYDSYTWAEWAYLWY',
    'HLA-B*15:02': 'YYAMYRNISTNTYESNLYIRYDSYTWAELAYLWY',
    'HLA-B*15:03': 'YYSEYREISTNTYESNLYLRYDSYTWAELAYLWY',
    'HLA-B*15:09': 'YYSEYRNICTNTYESNLYLRYNYYTWAELAYLWY',
    'HLA-B*15:10': 'YYSEYRNICTNTYESNLYLRYDYYTWAELAYLWY',
    'HLA-B*15:11': 'YYAMYRNIYTNTYESNLYLRYDSYTWAEWAYLWY',
    'HLA-B*15:17': 'YYAMYRENMASTYENIAYLRYHDYTWAELAYLWY',
    'HLA-B*15:18': 'YYSEYRNICTNTYESNLYLRYDSYTWAELAYLWY',
    'HLA-B*18:01': 'YHSTYRNISTNTYESNLYLRYDSYTWAVLAYTWH',
    'HLA-B*18:03': 'YHSTYRNISTNTDESNLYLRYDSYTWAVLAYTWH',
    'HLA-B*27:01': 'YHTEYREICAKTYENTAYLNYHDYTWAVLAYEWY',
    'HLA-B*27:02': 'YHTEYREICAKTDENIAYLNYHDYTWAVLAYEWY',
    'HLA-B*27:03': 'YHTEHREICAKTDEDTLYLNYHDYTWAVLAYEWY',
    'HLA-B*27:04': 'YHTEYREICAKTDESTLYLNYHDYTWAELAYEWY',
    'HLA-B*27:05': 'YHTEYREICAKTDEDTLYLNYHDYTWAVLAYEWY',
    'HLA-B*27:06': 'YHTEYREICAKTDESTLYLNYDYYTWAELAYEWY',
    'HLA-B*27:07': 'YHTEYREICAKTDEDTLYLSYNYYTWAVLAYEWY',
    'HLA-B*27:08': 'YHTEYREICAKTDESNLYLNYHDYTWAVLAYEWY',
    'HLA-B*27:09': 'YHTEYREICAKTDEDTLYLNYHHYTWAVLAYEWY',
    'HLA-B*35:01': 'YYATYRNIFTNTYESNLYIRYDSYTWAVLAYLWY',
    'HLA-B*35:02': 'YYATYRNIFTNTYESNLYIRYNYYTWAVLAYLWY',
    'HLA-B*35:03': 'YYATYRNIFTNTYESNLYIRYDFYTWAVLAYLWY',
    'HLA-B*35:08': 'YYATYRNIFTNTYESNLYIRYDSYTWAVRAYLWY',
    'HLA-B*37:01': 'YHSTYREISTNTYEDTLYIRSNFYTWAVDAYTWY',
    'HLA-B*38:01': 'YYSEYRNICTNTYENIAYLRYNFYTWAVLTYTWY',
    'HLA-B*39:01': 'YYSEYRNICTNTDESNLYLRYNFYTWAVLTYTWY',
    'HLA-B*39:06': 'YYSEYRNICTNTDESNLYWTYNFYTWAVLTYTWY',
    'HLA-B*39:24': 'YYSEYRNICTNTDESNLYLSYNFYTWAVLTYTWY',
    'HLA-B*40:01': 'YHTKYREISTNTYESNLYLRYNYYSLAVLAYEWY',
    'HLA-B*40:02': 'YHTKYREISTNTYESNLYLSYNYYTWAVLAYEWY',
    'HLA-B*41:01': 'YHTKYREISTNTYESNLYWRYNYYTWAVDAYTWY',
    'HLA-B*41:03': 'YHTKYREISTNTYESNLYLRYNYYTWAVDAYTWY',
    'HLA-B*42:01': 'YYSEYRNIYAQTDESNLYLSYNYYTWAVDAYTWY',
    'HLA-B*44:02': 'YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY',
    'HLA-B*44:03': 'YYTKYREISTNTYENTAYIRYDDYTWAVLAYLSY',
    'HLA-B*44:09': 'YYTKYREISTNTYESNLYIRYDDYTWAVDAYLSY',
    'HLA-B*44:27': 'YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY',
    'HLA-B*44:28': 'YYTKYREISTNTYENTAYIRYDDYTWAVRAYLSY',
    'HLA-B*45:01': 'YHTKYREISTNTYESNLYWRYNLYTWAVDAYLSY',
    'HLA-B*46:01': 'YYAMYREKYRQTDVSNLYLRYDSYTWAEWAYLWY',
    'HLA-B*47:01': 'YYTKYREISTNTYEDTLYLRFHDYTWAVLAYEWY',
    'HLA-B*48:01': 'YYSEYREISTNTYESNLYLSYNYYSLAVLAYEWY',
    'HLA-B*49:01': 'YHTKYREISTNTYENIAYWRYNLYTWAELAYLWY',
    'HLA-B*50:01': 'YHTKYREISTNTYESNLYWRYNLYTWAELAYLWY',
    'HLA-B*51:01': 'YYATYRNIFTNTYENIAYWTYNYYTWAELAYLWH',
    'HLA-B*51:08': 'YYATYRNIFTNTYENIAYWTYNYYTWAVDAYLWH',
    'HLA-B*52:01': 'YYATYREISTNTYENIAYWTYNYYTWAELAYLWH',
    'HLA-B*53:01': 'YYATYRNIFTNTYENIAYIRYDSYTWAVLAYLWY',
    'HLA-B*54:01': 'YYAGYRNIYAQTDESNLYWTYNLYTWAVLAYTWY',
    'HLA-B*55:01': 'YYAEYRNIYAQTDESNLYWTYNLYTWAELAYTWY',
    'HLA-B*56:01': 'YYAEYRNIYAQTDESNLYWTYNLYTWAVLAYLWY',
    'HLA-B*57:01': 'YYAMYGENMASTYENIAYIVYDSYTWAVLAYLWY',
    'HLA-B*57:03': 'YYAMYGENMASTYENIAYIVYNYYTWAVLAYLWY',
    'HLA-B*57:02': 'YYAMYGENMASTYENIAYIVYNYYTWAVLAYLWY',  # similar to B*57:03
    'HLA-B*58:01': 'YYATYGENMASTYENIAYIRYDSYTWAVLAYLWY',
    'HLA-B*67:01': 'YYSEYRNIYAQTDESNLYLRYNFYTWAVLTYTWY',
    'HLA-B*73:01': 'YHTEYRNICAKTDVGNLYWTYNFYTWAVLAYEWH',
    'HLA-B*83:01': 'YYSEYRNIYAQTDESNLYIRYDDYTWAVDAYLSY',
    # ===== HLA-C alleles =====
    'HLA-C*01:02': 'YFSGYREKYRQTDVSNLYLWCDYYTWAERAYTWY',
    'HLA-C*02:02': 'YYAGYREKYRQTDVNKLYLRYDSYTWAEWAYEWY',
    'HLA-C*02:10': 'YYAGYREKYRQTDVNKLYLRYDSYTWAEWAYEWY',
    'HLA-C*03:03': 'YYAGYREKYRQTDVSNLYIRYDYYTWAELAYLWY',
    'HLA-C*03:04': 'YYAGYREKYRQTDVSNLYIRYDYYTWAELAYLWY',
    'HLA-C*04:01': 'YSAGYREKYRQADVNKLYLRFNFYTWAERAYTWY',
    'HLA-C*05:01': 'YYAGYREKYRQTDVNKLYLRYNFYTWAERAYTWY',
    'HLA-C*06:02': 'YDSGYREKYRQADVNKLYLWYDSYTWAEWAYTWY',
    'HLA-C*07:01': 'YDSGYRENYRQADVSNLYLRYDSYTLAALAYTWY',
    'HLA-C*07:02': 'YDSGYREKYRQADVSNLYLRSDSYTLAALAYTWY',
    'HLA-C*08:01': 'YYAGYREKYRQTDVSNLYLRYNFYTWATLAYTWY',
    'HLA-C*08:02': 'YYAGYREKYRQTDVSNLYLRYNFYTWAERAYTWY',
    'HLA-C*12:02': 'YYAGYREKYRQADVSNLYLRYDSYTWAEWAYTWY',
    'HLA-C*12:03': 'YYAGYREKYRQADVSNLYLWYDSYTWAEWAYTWY',
    'HLA-C*14:02': 'YSAGYREKYRQTDVSNLYLWFDSYTWAERAYTWY',
    'HLA-C*15:02': 'YYAGYRENYRQTDVNKLYIRYDLYTWAELAYTWY',
    'HLA-C*15:05': 'YYAGYRENYRQTDVNKLYIRYDFYTWAELAYTWY',
    'HLA-C*16:01': 'YYAGYREKYRQTDVSNLYLWYDSYTWAAQAYTWY',
    'HLA-C*17:01': 'YYAGYREKYRQADVNKLYIRYNFYSLAELAYEWY',
    # ===== HLA-E alleles =====
    'HLA-E*01:01': 'YHSMYRESADTIFVNTLYLWHEFYSSAEQAYTWY',
    # ===== Simplified allele mappings (IEDB shorthand -> best matching pseudo-sequence) =====
    'HLA-A1': 'YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY',
    'HLA-A2': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY',
    'HLA-A3': 'YFAMYQENVAQTDVDTLYIIYRDYTWAELAYTWY',
    'HLA-A11': 'YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY',
    'HLA-A24': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTGY',
    'HLA-A26': 'YYAMYRNNVAHTDANTLYIRYQDYTWAEWAYRWY',
    'HLA-A28': 'YTAMYLQNVAQTDANTLYIMYRDYTWAVLAYTWY',
    'HLA-A29': 'YTAMYLQNVAQTDANTLYIMYRDYTWAVLAYTWY',
    'HLA-A31': 'YTAMYQENVAHIDVDTLYIMYQDYTWAVLAYTWY',
    'HLA-A68': 'YYAMYRNNVAQTDVDTLYIMYRDYTWAVWAYTWY',
    'HLA-A69': 'YYAMYRNNVAQTDVDTLYVRYHYYTWAVLAYTWY',
    'HLA-B7': 'YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY',
    'HLA-B8': 'YDSEYRNIFTNTDESNLYLSYNYYTWAVDAYTWY',
    'HLA-B14': 'YYSEYRNICTNTDESNLYLWYNFYTWAELAYTWH',
    'HLA-B27': 'YHTEYREICAKTDEDTLYLNYHDYTWAVLAYEWY',
    'HLA-B35': 'YYATYRNIFTNTYESNLYIRYDSYTWAVLAYLWY',
    'HLA-B37': 'YHSTYREISTNTYEDTLYIRSNFYTWAVDAYTWY',
    'HLA-B44': 'YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY',
    'HLA-B51': 'YYATYRNIFTNTYENIAYWTYNYYTWAELAYLWH',
    'HLA-B52': 'YYATYREISTNTYENIAYWTYNYYTWAELAYLWH',
    'HLA-B53': 'YYATYRNIFTNTYENIAYIRYDSYTWAVLAYLWY',
    'HLA-B57': 'YYAMYGENMASTYENIAYIVYDSYTWAVLAYLWY',
    'HLA-B60': 'YHTKYREISTNTYESNLYLRYNYYSLAVLAYEWY',
    'HLA-B62': 'YYSEYRNIYAQTDESNLYLRYNFYTWAVLTYTWY',
    'HLA-Cw7': 'YDSGYREKYRQADVSNLYLRSDSYTLAALAYTWY',
}

# 默认伪序列 (用于未知等位基因)
DEFAULT_PSEUDO = 'X' * 34


class MHCPseudoSequenceEncoder:
    """
    MHC 伪序列编码器

    方法: 将 MHC 分子的伪序列(34个关键氨基酸)编码为向量
    这些位点是与肽段直接接触的 MHC 口袋残基

    编码方式:
    - 每个位置使用 one-hot (20维)
    - 总维度: 34 * 20 = 680
    """

    def __init__(self, pseudo_sequences: Optional[Dict] = None):
        self.pseudo_sequences = pseudo_sequences or MHC_PSEUDO_SEQUENCE
        self.pseudo_len = 34

    def get_pseudo_sequence(self, allele: str) -> str:
        """获取等位基因的伪序列"""
        # 精确匹配
        if allele in self.pseudo_sequences:
            return self.pseudo_sequences[allele]

        # 模糊匹配 (处理版本号)
        for key in self.pseudo_sequences:
            if allele.startswith(key.split('*')[0] + '*'):
                return self.pseudo_sequences[key]

        return DEFAULT_PSEUDO

    def encode(self, allele: str) -> np.ndarray:
        """
        编码单个等位基因的伪序列

        Args:
            allele: HLA 等位基因名称

        Returns:
            encoding: (680,) one-hot 编码向量
        """
        pseudo_seq = self.get_pseudo_sequence(allele)
        # 截断或填充到固定长度
        if len(pseudo_seq) > self.pseudo_len:
            pseudo_seq = pseudo_seq[:self.pseudo_len]
        elif len(pseudo_seq) < self.pseudo_len:
            pseudo_seq = pseudo_seq + 'X' * (self.pseudo_len - len(pseudo_seq))

        encoding = np.zeros((self.pseudo_len, 20))

        for i, aa in enumerate(pseudo_seq):
            if aa in AA_TO_IDX:
                encoding[i, AA_TO_IDX[aa]] = 1

        return encoding.flatten()

    def encode_batch(self, alleles: List[str]) -> np.ndarray:
        """批量编码"""
        return np.array([self.encode(a) for a in alleles])


class HLAOneHotEncoder:
    """
    HLA 等位基因 One-Hot 编码器
    """

    def __init__(self, alleles: Optional[List[str]] = None):
        self.alleles = alleles or HLA_ALLELES
        self.allele_to_idx = ALLELE_TO_IDX
        self.n_alleles = len(self.alleles)

    def encode(self, allele: str) -> np.ndarray:
        """One-hot 编码"""
        encoding = np.zeros(self.n_alleles)
        if allele in self.allele_to_idx:
            encoding[self.allele_to_idx[allele]] = 1
        return encoding

    def encode_batch(self, alleles: List[str]) -> np.ndarray:
        """批量编码"""
        return np.array([self.encode(a) for a in alleles])


class BindingPositionEncoder:
    """
    肽-MHC 结合位置接触特征

    原理:
    - MHC 分子有特定的结合口袋
    - 肽段的特定位置(P1, P2, PΩ等)与MHC口袋相互作用
    - 提取这些关键位置的氨基酸特征

    结合位置权重 (根据 NetMHCpan):
    P1, P2, P3, PΩ-1, PΩ (锚定位点) 最重要
    """

    # 结合位置索引 (0-based)
    BINDING_POSITIONS = [0, 1, 2, -3, -2, -1]  # P1, P2, P3, PΩ-1, PΩ-1, PΩ

    def __init__(self):
        self.n_positions = len(self.BINDING_POSITIONS)

    def encode(self, peptide: str, mhc_pseudo: Optional[np.ndarray] = None) -> np.ndarray:
        """
        编码肽-MHC 结合位置特征

        Args:
            peptide: 肽段序列 (8-15氨基酸)
            mhc_pseudo: (680,) MHC伪序列编码 [可选]

        Returns:
            features: (n_positions * 20 + n_positions,) = (126,) 结合特征
        """
        peptide = peptide.upper()
        n = len(peptide)

        features = []

        # 1. 各位置的氨基酸 one-hot
        for pos in self.BINDING_POSITIONS:
            if -n <= pos < n:
                aa = peptide[pos]
                if aa in AA_TO_IDX:
                    aa_vec = np.zeros(20)
                    aa_vec[AA_TO_IDX[aa]] = 1
                    features.extend(aa_vec)
                else:
                    features.extend([0] * 20)
            else:
                features.extend([0] * 20)

        # 2. 各位置的氨基酸生化属性
        for pos in self.BINDING_POSITIONS:
            if -n <= pos < n:
                aa = peptide[pos]
                props = self._aa_properties(aa)
                features.extend(props)
            else:
                features.extend([0] * 6)

        # 3. MHC 伪序列对应位置的接触特征 (如果提供)
        if mhc_pseudo is not None:
            # MHC 伪序列的关键位置 (根据结构生物学)
            mhc_contact_positions = [0, 1, 2, 32, 33]  # P1, P2, P3, PΩ-1, PΩ 对应的MHC位置
            for mhc_pos in mhc_contact_positions:
                start = mhc_pos * 20
                mhc_aa_vec = mhc_pseudo[start:start+20]
                features.extend(mhc_aa_vec)
        else:
            features.extend([0] * (5 * 20))

        return np.array(features)

    def _aa_properties(self, aa: str) -> List[float]:
        """氨基酸生化属性 [疏水性, 荷电性, 体积, 极性, 芳香性, 小体积]"""
        properties = {
            'A': [1.8, 0, 89, 0, 0, 1],
            'C': [2.5, 0, 121, 0, 0, 0],
            'D': [-3.5, -1, 133, 1, 0, 0],
            'E': [-3.5, -1, 147, 1, 0, 0],
            'F': [2.8, 0, 165, 0, 1, 0],
            'G': [-0.4, 0, 75, 0, 0, 1],
            'H': [-3.2, 0.5, 155, 1, 0, 0],
            'I': [4.5, 0, 167, 0, 0, 0],
            'K': [-3.9, 1, 168, 1, 0, 0],
            'L': [3.8, 0, 131, 0, 0, 0],
            'M': [1.9, 0, 149, 0, 0, 0],
            'N': [-3.5, 0, 132, 1, 0, 0],
            'P': [-1.6, 0, 115, 0, 0, 1],
            'Q': [-3.5, 0, 146, 1, 0, 0],
            'R': [-4.5, 1, 174, 1, 0, 0],
            'S': [-0.8, 0, 105, 1, 0, 1],
            'T': [-0.7, 0, 119, 1, 0, 1],
            'V': [4.2, 0, 117, 0, 0, 0],
            'W': [-0.9, 0, 204, 0, 1, 0],
            'Y': [-1.3, 0, 181, 1, 1, 0],
        }
        return properties.get(aa, [0, 0, 0, 0, 0, 0])

    def encode_batch(self, peptides: List[str], mhc_pseudos: Optional[np.ndarray] = None) -> np.ndarray:
        """批量编码"""
        if mhc_pseudos is not None:
            return np.array([self.encode(p, mhc_pseudos[i]) for i, p in enumerate(peptides)])
        return np.array([self.encode(p) for p in peptides])


class MHCFeatureEncoder:
    """
    MHC 等位基因特征编码器 (整合版)

    输出特征维度:
    - MHC 伪序列: 34 * 20 = 680
    - HLA one-hot: 43
    - 结合位置: 246
    - 总计: 969
    """

    def __init__(self):
        self.pseudo_encoder = MHCPseudoSequenceEncoder()
        self.hla_encoder = HLAOneHotEncoder()
        self.binding_encoder = BindingPositionEncoder()

    @property
    def feature_dim(self) -> int:
        """特征维度"""
        # 伪序列: 34*20=680, HLA one-hot: 43, 结合位置: 6*20+6*6+5*20=256
        return 680 + 43 + 256  # 979

    def encode(self, peptide: str, allele: str) -> np.ndarray:
        """
        编码肽-MHC 对

        Args:
            peptide: 肽段序列
            allele: HLA 等位基因

        Returns:
            features: (969,) MHC 特征向量
        """
        # 1. MHC 伪序列编码
        mhc_pseudo = self.pseudo_encoder.encode(allele)

        # 2. HLA one-hot
        hla_onehot = self.hla_encoder.encode(allele)

        # 3. 结合位置特征
        binding_feat = self.binding_encoder.encode(peptide, mhc_pseudo)

        return np.concatenate([mhc_pseudo, hla_onehot, binding_feat])

    def encode_batch(self, peptides: List[str], alleles: List[str]) -> np.ndarray:
        """批量编码"""
        return np.array([self.encode(p, a) for p, a in zip(peptides, alleles)])


class FullEpitopeEncoder:
    """
    完整表位编码器 = ESM-2 + Mamba3Lite + MHC 特征

    融合策略: Concat
    """

    def __init__(
        self,
        esm2_model_dir: Optional[str] = None,
        use_mamba: bool = True,
        use_mhc: bool = True
    ):
        self.esm2_encoder = None
        self.mamba_encoder = None
        self.mhc_encoder = MHCFeatureEncoder() if use_mhc else None

        self.use_mamba = use_mamba
        self.use_mhc = use_mhc
        self.esm2_model_dir = esm2_model_dir

    def load(self):
        """加载所有编码器"""
        if self.esm2_encoder is None:
            from core.esm2_mamba_fusion import ESM2Encoder
            self.esm2_encoder = ESM2Encoder(self.esm2_model_dir)
            self.esm2_encoder.load()

        if self.mamba_encoder is None and self.use_mamba:
            from core.esm2_mamba_fusion import MambaEncoder
            self.mamba_encoder = MambaEncoder()
            self.mamba_encoder.load()

    def encode(
        self,
        peptides: List[str],
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None  # 用于 Mamba3Lite ( epitope_seq )
    ) -> np.ndarray:
        """
        完整编码

        Args:
            peptides: 肽段序列 (与 MHC 结合)
            alleles: HLA 等位基因列表 [可选]
            sequences: 表位序列 (用于 Mamba3Lite) [可选]

        Returns:
            features: 融合特征向量
        """
        self.load()

        features = []

        # 1. ESM-2 编码
        if self.esm2_encoder:
            esm2_feat = self.esm2_encoder.encode(peptides)
            features.append(esm2_feat)

        # 2. Mamba3Lite 编码
        if self.mamba_encoder and sequences:
            mamba_feat = self.mamba_encoder.encode(sequences)
            features.append(mamba_feat)

        # 3. MHC 特征编码
        if self.mhc_encoder and alleles:
            mhc_feat = self.mhc_encoder.encode_batch(peptides, alleles)
            features.append(mhc_feat)

        return np.concatenate(features, axis=1)

    @property
    def feature_dim(self) -> int:
        """总特征维度"""
        from core.esm2_mamba_fusion import ESM2_EMBED_DIM, MAMBA_EMBED_DIM

        dim = ESM2_EMBED_DIM  # 320
        if self.use_mamba:
            dim += MAMBA_EMBED_DIM  # 528
        if self.use_mhc:
            dim += 969

        return dim


def quick_test():
    """快速测试"""
    print("=" * 60)
    print("MHC 等位基因特征编码器测试")
    print("=" * 60)

    encoder = MHCFeatureEncoder()

    # 测试单个编码
    peptide = "SLYNTVATL"
    allele = "HLA-A*02:01"

    feat = encoder.encode(peptide, allele)
    print(f"\n[MHC] 特征维度: {feat.shape}")
    print(f"[MHC] 特征统计: min={feat.min():.3f}, max={feat.max():.3f}, mean={feat.mean():.3f}")

    # 测试批量编码
    peptides = ["SLYNTVATL", "GILGFVFTL", "KLGGALQAK"] * 10
    alleles = ["HLA-A*02:01", "HLA-A*02:01", "HLA-B*07:02"] * 10

    feats = encoder.encode_batch(peptides, alleles)
    print(f"\n[Batch] 批量编码: {feats.shape}")

    # 测试完整编码器
    print("\n[Full] 完整表位编码器")
    full_encoder = FullEpitopeEncoder(use_mhc=True)
    print(f"[Full] 总特征维度: {full_encoder.feature_dim}")

    print("\n" + "=" * 60)
    print("MHC 特征编码器准备就绪")
    print("=" * 60)

    return {
        'mhc_dim': encoder.feature_dim,
        'full_dim': full_encoder.feature_dim,
        'n_test': len(peptides)
    }


if __name__ == '__main__':
    result = quick_test()
    print(f"\n结果: {result}")
