#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_extract.py - 批量提取 WebPlotDigitizer 数据并转换

用法:
    # 1. 手动提取后，使用模板 CSV
    python batch_extract.py convert --input extracted_data.csv --output poppk_data.csv

    # 2. 添加到现有数据库
    python batch_extract.py merge --input new_curves.csv --database real_pk_database.json

    # 3. 运行质量检查
    python batch_extract.py validate --database real_pk_database.json

    # 4. 导出 NONMEM 格式
    python batch_extract.py export --database real_pk_database.json --format NONMEM

    # 5. 完整流程
    python batch_extract.py full --extracted extracted_data.csv --database real_pk_database.json
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


# =============================================================================
# 常量定义
# =============================================================================

SUPPORTED_MODIFICATIONS = ['none', 'm6A', 'psi', '5mC', 'ms2m6A']
SUPPORTED_ROUTES = ['IV', 'IM', 'SC', 'oral']
SUPPORTED_TISSUES = ['plasma', 'muscle', 'liver', 'spleen', 'lung', 'kidney']
SUPPORTED_SPECIES = ['mouse', 'rat', 'human', 'dog']

# 半衰期参考值（用于验证提取数据）
REFERENCE_HALF_LIVES = {
    'none': {'mean': 6.0, 'cv': 25},
    'm6A': {'mean': 10.8, 'cv': 22},
    'psi': {'mean': 15.0, 'cv': 20},
    '5mC': {'mean': 12.5, 'cv': 22},
    'ms2m6A': {'mean': 20.0, 'cv': 18},
}


# =============================================================================
# 数据转换
# =============================================================================

def parse_extracted_csv(csv_path: str) -> List[Dict]:
    """
    解析 WebPlotDigitizer 提取的 CSV 格式

    预期格式:
    curve_id,source_study,figure_reference,modification,route,dose_ug_kg,species,tissue,notes
    time_h,concentration,cv_percent
    0.0,100.0,
    1.0,78.0,
    ...
    ,,,
    curve_id,...
    """
    curves = []
    current_header = None
    current_data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                # 空行分隔符
                if current_header and current_data:
                    curves.append({
                        'header': current_header,
                        'data': current_data
                    })
                    current_header = None
                    current_data = []
                continue

            if current_header is None:
                # 跳过 time_h,concentration 行
                if row[0].strip().startswith('time'):
                    continue
                # 解析头部行
                current_header = {
                    'curve_id': row[0] if len(row) > 0 else '',
                    'source_study': row[1] if len(row) > 1 else '',
                    'figure_reference': row[2] if len(row) > 2 else '',
                    'modification': row[3] if len(row) > 3 else 'none',
                    'route': row[4] if len(row) > 4 else 'IV',
                    'dose_ug_kg': float(row[5]) if len(row) > 5 and row[5] and not row[5].startswith('dose') else 50.0,
                    'species': row[6] if len(row) > 6 else 'mouse',
                    'weight_kg': float(row[7]) if len(row) > 7 and row[7] and not row[7].startswith('weight') else 0.025,
                    'tissue': row[8] if len(row) > 8 else 'plasma',
                    'analyte': row[9] if len(row) > 9 else 'circRNA_concentration',
                    'notes': row[10] if len(row) > 10 else '',
                }
            elif row[0].replace('.', '').replace('-', '').isdigit() or (row[0] and row[0][0].isdigit()):
                # 解析数据行
                try:
                    time_h = float(row[0])
                    conc = float(row[1]) if len(row) > 1 else 0.0
                    cv = float(row[2]) if len(row) > 2 and row[2] else None
                    current_data.append({
                        'time_h': time_h,
                        'concentration': conc,
                        'cv_percent': cv
                    })
                except ValueError:
                    pass

        # 最后一个曲线
        if current_header and current_data:
            curves.append({
                'header': current_header,
                'data': current_data
            })

    return curves


def convert_to_poppk_format(curves: List[Dict]) -> pd.DataFrame:
    """
    将提取的曲线转换为 PopPK 拟合格式
    """
    rows = []

    for i, curve in enumerate(curves):
        header = curve['header']
        data = curve['data']

        # 生成受试者 ID
        curve_id = header['curve_id'].strip() if header['curve_id'] else ''
        subject_id = f"EXTRACTED_{curve_id}" if curve_id else f"EXTRACTED_{i+1:03d}"
        study_id = header['source_study']
        modification = header['modification']
        route = header['route']

        for obs in data:
            rows.append({
                'ID': subject_id,
                'STUDY': study_id,
                'FIG_REF': header['figure_reference'],
                'MODIFICATION': modification,
                'ROUTE': route,
                'DOSE': header['dose_ug_kg'],
                'WT': header['weight_kg'],
                'SPECIES': header['species'],
                'TISSUE': header['tissue'],
                'ANALYTE': header['analyte'],
                'NOTES': header['notes'],
                'TIME': obs['time_h'],
                'DV': obs['concentration'],
                'CV': obs['cv_percent'] if obs['cv_percent'] else np.nan,
            })

    return pd.DataFrame(rows)


def compute_half_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    从时间-浓度数据计算半衰期（用于验证）
    """
    results = []

    for (subject_id, modification), group in df.groupby(['ID', 'MODIFICATION']):
        group = group.sort_values('TIME')

        # 线性拟合 (log-scale)
        times = group['TIME'].values
        concs = group['DV'].values

        # 只使用消除相数据 (time > 2h, conc > 10%)
        elim_mask = (times > 2) & (concs > concs.max() * 0.1)
        if elim_mask.sum() < 3:
            continue

        times_elim = times[elim_mask]
        concs_elim = concs[elim_mask]

        try:
            # log-linear 回归
            log_concs = np.log(concs_elim)
            slope, intercept = np.polyfit(times_elim, log_concs, 1)
            ke = -slope  # 消除速率常数
            half_life = np.log(2) / ke if ke > 0 else np.nan

            results.append({
                'ID': subject_id,
                'MODIFICATION': modification,
                'ke': ke,
                'half_life_h': half_life,
                'n_points': elim_mask.sum(),
            })
        except Exception:
            pass

    return pd.DataFrame(results)


def validate_half_life(extracted_df: pd.DataFrame) -> Dict:
    """
    验证提取的半衰期是否与参考值一致
    """
    computed = compute_half_life(extracted_df)

    if computed.empty:
        return {'status': 'warning', 'message': '无法计算半衰期（数据点不足）'}

    results = []
    for _, row in computed.iterrows():
        mod = row['MODIFICATION']
        if mod not in REFERENCE_HALF_LIVES:
            continue

        ref = REFERENCE_HALF_LIVES[mod]
        computed_hl = row['half_life_h']
        reference_hl = ref['mean']
        error_pct = abs(computed_hl - reference_hl) / reference_hl * 100

        results.append({
            'modification': mod,
            'computed_hl_h': round(computed_hl, 2),
            'reference_hl_h': reference_hl,
            'error_pct': round(error_pct, 1),
            'pass': error_pct < 30,  # 30% 容差
        })

    return {
        'status': 'ok' if all(r['pass'] for r in results) else 'warning',
        'results': results,
    }


# =============================================================================
# 数据库操作
# =============================================================================

def load_database(db_path: str) -> Dict:
    """加载 JSON 数据库"""
    with open(db_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_database(db_path: str, data: Dict):
    """保存 JSON 数据库"""
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_to_database(curves: List[Dict], db_path: str) -> Dict:
    """
    将提取的曲线合并到现有数据库
    """
    db = load_database(db_path)

    # 获取现有受试者数量
    existing_subjects = len(db.get('poppk_ready_format', {}).get('subjects', []))
    new_subjects = []

    for i, curve in enumerate(curves):
        header = curve['header']
        data = curve['data']

        subject_id = f"SUBJ_{existing_subjects + i + 1:03d}"

        subject = {
            'id': subject_id,
            'study': header['source_study'],
            'modification': header['modification'],
            'route': header['route'],
            'dose': header['dose_ug_kg'],
            'weight_kg': header['weight_kg'],
            'sex': 'M',
            'species': header['species'],
            'observations': [
                {
                    'time': obs['time_h'],
                    'conc': obs['concentration'],
                    'cmdv': header['tissue'],
                    'analyte': header['analyte'],
                }
                for obs in data
            ]
        }
        new_subjects.append(subject)

    # 添加到数据库
    if 'poppk_ready_format' not in db:
        db['poppk_ready_format'] = {'subjects': []}

    db['poppk_ready_format']['subjects'].extend(new_subjects)

    # 更新元数据
    if 'metadata' not in db:
        db['metadata'] = {}

    db['metadata']['last_updated'] = datetime.now().isoformat()
    db['metadata']['n_extracted_subjects'] = len(new_subjects)

    save_database(db_path, db)

    return {
        'status': 'success',
        'n_new_subjects': len(new_subjects),
        'total_subjects': len(db['poppk_ready_format']['subjects']),
    }


def export_nonmem_format(db_path: str, output_path: str):
    """
    导出为 NONMEM 兼容格式
    """
    from real_pk_loader import RealPKLoader

    loader = RealPKLoader(db_path)
    dataset = loader.to_population_pk()
    df = dataset.to_dataframe()

    # 按 ID 和 TIME 排序
    df = df.sort_values(['ID', 'TIME'])

    # 保存为 CSV
    df.to_csv(output_path, index=False, sep=' ')

    print(f"NONMEM 格式数据已保存: {output_path}")
    print(f"总记录数: {len(df)}")
    print(f"受试者数: {df['ID'].nunique()}")

    return df


# =============================================================================
# 可视化
# =============================================================================

def plot_extracted_curves(df: pd.DataFrame, output_path: str = None):
    """
    绘制提取的 PK 曲线（用于质量检查）
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 所有曲线（线性坐标）
        ax1 = axes[0, 0]
        for (subject_id, mod), group in df.groupby(['ID', 'MODIFICATION']):
            group = group.sort_values('TIME')
            ax1.plot(group['TIME'], group['DV'], 'o-', label=f"{mod} ({subject_id})", markersize=4)
        ax1.set_xlabel('时间 (h)')
        ax1.set_ylabel('浓度')
        ax1.set_title('提取的 PK 曲线 (线性坐标)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 所有曲线（半对数坐标）
        ax2 = axes[0, 1]
        for (subject_id, mod), group in df.groupby(['ID', 'MODIFICATION']):
            group = group.sort_values('TIME')
            valid = group['DV'] > 0
            ax2.semilogy(group.loc[valid, 'TIME'], group.loc[valid, 'DV'], 'o-',
                        label=f"{mod}", markersize=4)
        ax2.set_xlabel('时间 (h)')
        ax2.set_ylabel('浓度 (log)')
        ax2.set_title('提取的 PK 曲线 (半对数坐标)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 半衰期验证
        ax3 = axes[1, 0]
        computed = compute_half_life(df)
        if not computed.empty:
            mods = computed['MODIFICATION'].values
            hls = computed['half_life_h'].values
            ref_hls = [REFERENCE_HALF_LIVES.get(m, {}).get('mean', 0) for m in mods]

            x = np.arange(len(mods))
            width = 0.35
            ax3.bar(x - width/2, hls, width, label='提取值', color='steelblue')
            ax3.bar(x + width/2, ref_hls, width, label='参考值', color='coral')
            ax3.set_xticks(x)
            ax3.set_xticklabels(mods, rotation=45)
            ax3.set_ylabel('半衰期 (h)')
            ax3.set_title('半衰期验证')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. 数据点分布
        ax4 = axes[1, 1]
        for mod, group in df.groupby('MODIFICATION'):
            ax4.scatter(group['TIME'], group['DV'], alpha=0.5, label=mod, s=20)
        ax4.set_xlabel('时间 (h)')
        ax4.set_ylabel('浓度')
        ax4.set_title('数据点分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {output_path}")

        plt.show()

    except ImportError:
        print("警告: matplotlib 未安装，跳过绘图")


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='批量提取 WebPlotDigitizer 数据并转换为 PopPK 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换提取的 CSV 为 PopPK DataFrame
  python batch_extract.py convert --input data/extracted.csv --output poppk_data.csv

  # 合并到数据库
  python batch_extract.py merge --input data/new_curves.csv --database data/real_pk_database.json

  # 验证数据质量
  python batch_extract.py validate --database data/real_pk_database.json

  # 导出 NONMEM 格式
  python batch_extract.py export --database data/real_pk_database.json --output nonmem_data.csv

  # 完整流程
  python batch_extract.py full --extracted data/extracted.csv --database data/real_pk_database.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # convert 命令
    convert_parser = subparsers.add_parser('convert', help='转换 CSV 为 PopPK 格式')
    convert_parser.add_argument('--input', '-i', required=True, help='提取的 CSV 文件')
    convert_parser.add_argument('--output', '-o', default='poppk_data.csv', help='输出文件')

    # merge 命令
    merge_parser = subparsers.add_parser('merge', help='合并到数据库')
    merge_parser.add_argument('--input', '-i', required=True, help='提取的 CSV 文件')
    merge_parser.add_argument('--database', '-d', required=True, help='JSON 数据库文件')

    # validate 命令
    validate_parser = subparsers.add_parser('validate', help='验证数据质量')
    validate_parser.add_argument('--database', '-d', required=True, help='JSON 数据库文件')
    validate_parser.add_argument('--plot', '-p', help='输出图表路径')

    # export 命令
    export_parser = subparsers.add_parser('export', help='导出 NONMEM 格式')
    export_parser.add_argument('--database', '-d', required=True, help='JSON 数据库文件')
    export_parser.add_argument('--output', '-o', default='nonmem_data.csv', help='输出文件')

    # full 命令
    full_parser = subparsers.add_parser('full', help='完整流程')
    full_parser.add_argument('--extracted', '-e', required=True, help='提取的 CSV 文件')
    full_parser.add_argument('--database', '-d', required=True, help='JSON 数据库文件')
    full_parser.add_argument('--plot', '-p', help='输出图表路径')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'convert':
        print(f"读取: {args.input}")
        curves = parse_extracted_csv(args.input)
        print(f"解析了 {len(curves)} 条曲线")

        df = convert_to_poppk_format(curves)
        df.to_csv(args.output, index=False)
        print(f"保存为: {args.output}")
        print(f"总记录数: {len(df)}")

    elif args.command == 'merge':
        print(f"读取: {args.input}")
        curves = parse_extracted_csv(args.input)
        print(f"解析了 {len(curves)} 条曲线")

        result = merge_to_database(curves, args.database)
        print(f"合并结果: {result}")

    elif args.command == 'validate':
        from real_pk_loader import RealPKLoader

        loader = RealPKLoader(args.database)
        dataset = loader.to_population_pk()
        df = dataset.to_dataframe()

        print("=" * 60)
        print("数据质量验证")
        print("=" * 60)

        # 半衰期验证
        validation = validate_half_life(df)

        print("\n半衰期验证:")
        for r in validation.get('results', []):
            status = "PASS" if r['pass'] else "FAIL"
            print(f"  {status} {r['modification']}: extracted {r['computed_hl_h']}h vs ref {r['reference_hl_h']}h (error {r['error_pct']}%)")

        # 绘图
        if args.plot:
            plot_extracted_curves(df, args.plot)

    elif args.command == 'export':
        export_nonmem_format(args.database, args.output)

    elif args.command == 'full':
        # 1. 转换
        print(f"步骤 1: 转换 {args.extracted}")
        curves = parse_extracted_csv(args.extracted)
        print(f"  - 解析了 {len(curves)} 条曲线")

        df = convert_to_poppk_format(curves)

        # 2. 验证
        print("\n步骤 2: 验证数据质量")
        validation = validate_half_life(df)
        for r in validation.get('results', []):
            status = "PASS" if r['pass'] else "FAIL"
            print(f"  {status} {r['modification']}: {r['computed_hl_h']}h vs {r['reference_hl_h']}h (error {r['error_pct']}%)")

        # 3. 合并到数据库
        print(f"\n步骤 3: 合并到 {args.database}")
        result = merge_to_database(curves, args.database)
        print(f"  - 新增 {result['n_new_subjects']} 个受试者")
        print(f"  - 数据库总计 {result['total_subjects']} 个受试者")

        # 4. 导出
        output_path = args.database.replace('.json', '_nonmem.csv')
        print(f"\n步骤 4: 导出 NONMEM 格式")
        export_nonmem_format(args.database, output_path)

        print("\n" + "=" * 60)
        print("完成!")
        print("=" * 60)


if __name__ == '__main__':
    main()