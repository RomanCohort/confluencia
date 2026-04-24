#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
real_pk_loader.py - 真实 PK 数据加载器

从文献提取的真实 circRNA PK 数据加载到 PopPK 格式。

用法:
    from real_pk_loader import RealPKLoader

    loader = RealPKLoader('data/real_pk_database.json')
    pop_data = loader.to_population_pk()
    df = pop_data.to_dataframe()
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class DeliveryRoute(Enum):
    IV = "IV"
    IM = "IM"
    SC = "SC"
    ORAL = "ORAL"


class NucleotideModification(Enum):
    NONE = "none"
    M6A = "m6A"
    PSI = "psi"
    M5C = "5mC"
    MS2M6A = "ms2m6A"


@dataclass
class PKObservation:
    """单个时间点观测"""
    subject_id: str
    time_h: float
    concentration: float
    tissue: str = "plasma"
    analyte: str = "circRNA_concentration"
    cv_percent: Optional[float] = None
    lloq: Optional[float] = None  # lower limit of quantification
    censored: bool = False


@dataclass
class PKSubject:
    """单个受试者的 PK 数据"""
    subject_id: str
    study_id: str
    modification: str
    route: str
    dose: float  # μg/kg
    weight_kg: float = 0.025  # 默认小鼠体重
    sex: str = "M"
    species: str = "mouse"
    observations: List[PKObservation] = field(default_factory=list)

    def add_observation(self, obs: PKObservation):
        self.observations.append(obs)


@dataclass
class RealPKDataset:
    """真实 PK 数据集"""
    dataset_id: str
    subjects: List[PKSubject] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为长格式 DataFrame（用于 PopPK 拟合）"""
        rows = []
        for subj in self.subjects:
            for obs in subj.observations:
                rows.append({
                    'ID': subj.subject_id,
                    'STUDY': subj.study_id,
                    'MODIFICATION': subj.modification,
                    'ROUTE': subj.route,
                    'DOSE': subj.dose,
                    'WT': subj.weight_kg,
                    'SEX': subj.sex,
                    'SPECIES': subj.species,
                    'TIME': obs.time_h,
                    'DV': obs.concentration,
                    'TISSUE': obs.tissue,
                    'ANALYTE': obs.analyte,
                    'CV': obs.cv_percent if obs.cv_percent else np.nan,
                    'CENSORED': 1 if obs.censored else 0,
                })
        return pd.DataFrame(rows)

    def summary(self) -> Dict:
        """数据集摘要统计"""
        df = self.to_dataframe()
        return {
            'n_subjects': len(self.subjects),
            'n_observations': len(df),
            'studies': df['STUDY'].unique().tolist(),
            'modifications': df['MODIFICATION'].unique().tolist(),
            'routes': df['ROUTE'].unique().tolist(),
            'time_range_h': (df['TIME'].min(), df['TIME'].max()),
            'dose_range_ug_kg': (df['DOSE'].min(), df['DOSE'].max()),
        }


class RealPKLoader:
    """从 JSON 加载真实 PK 数据"""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.data = self._load_json()

    def _load_json(self) -> Dict:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_studies(self) -> List[str]:
        """获取所有研究 ID"""
        return list(self.data.get('studies', {}).keys())

    def get_study_info(self, study_id: str) -> Dict:
        """获取研究信息"""
        return self.data['studies'].get(study_id, {})

    def get_compiled_curves(self) -> List[Dict]:
        """获取编译的 PK 曲线"""
        return self.data.get('compiled_pk_curves', {}).get('curves', [])

    def to_population_pk(self) -> RealPKDataset:
        """转换为 PopPK 格式数据集"""
        dataset = RealPKDataset(
            dataset_id='real_pk_compiled',
            metadata={
                'source': self.json_path.name,
                'compiled_date': self.data.get('database_info', {}).get('compiled_date'),
                'data_sources': self.data.get('database_info', {}).get('data_sources', []),
            }
        )

        # 从 poppk_ready_format 加载
        poppk_data = self.data.get('poppk_ready_format', {})

        for subj_data in poppk_data.get('subjects', []):
            subject = PKSubject(
                subject_id=subj_data['id'],
                study_id=subj_data['study'],
                modification=subj_data['modification'],
                route=subj_data['route'],
                dose=subj_data['dose'],
                weight_kg=subj_data.get('weight_kg', 0.025),
                sex=subj_data.get('sex', 'M'),
                species=subj_data.get('species', 'mouse'),
            )

            for obs_data in subj_data.get('observations', []):
                obs = PKObservation(
                    subject_id=subject.subject_id,
                    time_h=obs_data['time'],
                    concentration=obs_data['conc'],
                    tissue=obs_data.get('cmdv', 'plasma'),
                    analyte=obs_data.get('analyte', 'circRNA_concentration'),
                )
                subject.add_observation(obs)

            dataset.subjects.append(subject)

        return dataset

    def get_half_life_by_modification(self) -> pd.DataFrame:
        """获取按修饰分组的半衰期数据"""
        rows = []
        for study_id, study_data in self.data.get('studies', {}).items():
            pk_data = study_data.get('pk_data', {})
            half_life_data = pk_data.get('half_life_by_modification', {})

            for mod, params in half_life_data.items():
                rows.append({
                    'study': study_id,
                    'modification': mod,
                    'half_life_h': params.get('value_h', params.get('mean_h')),
                    'cv_percent': params.get('cv_percent'),
                    'method': params.get('method', 'unknown'),
                    'sample_size': params.get('sample_size'),
                })

        return pd.DataFrame(rows)

    def get_tissue_distribution(self) -> pd.DataFrame:
        """获取组织分布数据"""
        rows = []
        for study_id, study_data in self.data.get('studies', {}).items():
            dist_data = study_data.get('tissue_distribution', {}).get('distribution', {})

            for tissue, value in dist_data.items():
                if isinstance(value, dict):
                    percent = value.get('percent', value.get('value', 0))
                else:
                    percent = value

                rows.append({
                    'study': study_id,
                    'tissue': tissue.replace('_percent', ''),
                    'percent': percent,
                })

        return pd.DataFrame(rows)


def main():
    """测试加载器"""
    import sys

    # 默认路径
    json_path = Path(__file__).parent.parent / 'data' / 'real_pk_database.json'

    if not json_path.exists():
        print(f"错误: 文件不存在 {json_path}")
        sys.exit(1)

    loader = RealPKLoader(str(json_path))

    print("=" * 60)
    print("真实 PK 数据库摘要")
    print("=" * 60)

    # 研究列表
    print(f"\n研究数量: {len(loader.get_studies())}")
    for study_id in loader.get_studies():
        study = loader.get_study_info(study_id)
        print(f"  - {study_id}: {study.get('journal', 'Unknown')} ({study.get('year', 'N/A')})")

    # 加载为 PopPK 格式
    dataset = loader.to_population_pk()

    print(f"\n数据集摘要:")
    summary = dataset.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 半衰期数据
    print(f"\n半衰期数据 (按修饰):")
    hl_df = loader.get_half_life_by_modification()
    print(hl_df.to_string(index=False))

    # 组织分布
    print(f"\n组织分布数据:")
    td_df = loader.get_tissue_distribution()
    print(td_df.to_string(index=False))

    # 导出为 DataFrame
    df = dataset.to_dataframe()
    print(f"\nPopPK 格式 DataFrame (前 10 行):")
    print(df.head(10).to_string())

    # 保存为 CSV
    output_path = json_path.parent / 'real_pk_for_poppk.csv'
    df.to_csv(output_path, index=False)
    print(f"\n已保存为: {output_path}")


if __name__ == '__main__':
    main()
