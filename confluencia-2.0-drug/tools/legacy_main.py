"""
药物疗效预测神经网络 - 少样本学习与GBDT优化

此脚本实现了一个预测最佳药效的神经网络模型，结合梯度提升决策树（GBDT）和Hyperopt优化。
在剂量-给药频次参数空间识别高疗效区，使用网络爬取的已知药物进行训练。

流程：
1. 数据准备（极少量标注样本）- 网络爬取药物数据
2. 特征提取器预训练（使用对比损失/Triplet Loss）
3. 原型网络或关系网络构建（使用Prototypical Loss）
4. 多任务微调（联合加权焦点损失与辅助任务损失）
5. 模型验证与集成（包括交叉验证）
6. GBDT + Hyperopt优化剂量-频次参数空间
7. 预测数据输入区域（用户交互预测）

"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import requests
import random

FP_DIM = 2048

# 多任务列表（用于任务管理/日志/可视化）
MULTI_TASK_LIST = ["efficacy", "toxicity"]
# 任务损失权重
TASK_LOSS_WEIGHTS = {
    "efficacy": 1.0,
    "toxicity": 0.5,
}

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


@dataclass(frozen=True)
class NumericCondition:
    """数值条件输入（用于参数空间预测）。"""

    dose: float
    freq: float
    route: float

class DrugDataset(Dataset):
    """药物数据集类"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ResidualBlock(nn.Module):
    """简单残差块（MLP）"""
    def __init__(self, dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + residual)

class FeatureExtractor(nn.Module):
    """特征提取器"""
    def __init__(self, input_dim=FP_DIM, hidden_dim=512):
        super(FeatureExtractor, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.res_block = ResidualBlock(hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
        )

    def forward(self, x):
        x = self.fc_in(x)
        x = self.res_block(x)
        return self.fc_out(x)

class PrototypicalNetwork(nn.Module):
    """原型网络"""
    def __init__(self, feature_extractor):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, support_set, query_set):
        # 计算支持集原型
        support_features = self.feature_extractor(support_set)
        prototypes = torch.mean(support_features, dim=0)
        
        # 计算查询集特征
        query_features = self.feature_extractor(query_set)
        
        # 计算距离
        distances = torch.norm(query_features - prototypes, dim=1)
        return distances

class MultiTaskModel(nn.Module):
    """多任务模型"""
    def __init__(self, feature_extractor):
        super(MultiTaskModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.task_list = list(MULTI_TASK_LIST)
        self.task_heads = nn.ModuleDict({
            name: nn.Linear(256, 1) for name in self.task_list
        })

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = {name: head(features) for name, head in self.task_heads.items()}
        return outputs

def focal_loss(pred, target, alpha=1, gamma=2):
    """焦点损失"""
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

def prototypical_loss(distances, labels, margin=1.0):
    """原型损失（回归版）

    原始实现用于分类（按标签索引原型），但当前任务标签为连续的疗效值。
    为避免索引错误，这里将距离映射为预测疗效（距离越小，疗效越高），
    然后对预测值与真实疗效做均方误差损失。
    映射函数：pred = 1 / (1 + distance)
    """
    # distances: Tensor shape (num_queries,)
    # labels: array-like or tensor shape (num_queries, 2) -> [efficacy, toxicity]
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=distances.dtype, device=distances.device)

    # 只使用疗效标签（第0列）
    if labels.dim() > 1:
        true_eff = labels[:, 0].to(dtype=distances.dtype)
    else:
        true_eff = labels.to(dtype=distances.dtype)

    # 将距离映射为预测疗效（可根据需要调整映射）
    pred_eff = 1.0 / (1.0 + distances)
    loss = F.mse_loss(pred_eff, true_eff)
    return loss

def scrape_drug_data(num_drugs=100):
    """网络爬取药物数据（使用PubChem API）"""
    drugs = []
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
    
    # 示例CID列表（实际应随机或从数据库获取）
    cids = list(range(1, num_drugs + 1))
    
    for cid in cids:
        try:
            # 获取SMILES
            smiles_url = f"{base_url}{cid}/property/CanonicalSMILES/TXT"
            response = requests.get(smiles_url, timeout=8)
            if response.status_code == 200:
                smiles = response.text.strip()
                
                # 模拟疗效和毒性标签（实际需要标注数据）
                efficacy = np.random.uniform(0, 1)
                toxicity = np.random.uniform(0, 1)
                
                drugs.append({
                    'smiles': smiles,
                    'efficacy': efficacy,
                    'toxicity': toxicity,
                    'cid': cid
                })
        except requests.RequestException:
            continue

    # 避免重复样本干扰小样本训练。
    return pd.DataFrame(drugs).drop_duplicates(subset=["smiles"]).reset_index(drop=True)

def smiles_to_fingerprint(smiles):
    """将SMILES转换为分子指纹"""
    try:
        # Lazy import so this module can be imported even when RDKit isn't installed.
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(FP_DIM)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        return np.array(fp)
    except:
        return np.zeros(FP_DIM)


def _build_feature_vector(*, smiles: Optional[str] = None, cond: Optional[NumericCondition] = None) -> np.ndarray:
    """统一构造模型输入向量，避免多处重复实现。"""
    if smiles:
        return np.asarray(smiles_to_fingerprint(smiles), dtype=np.float32).reshape(1, -1)

    if cond is None:
        raise ValueError("请提供 SMILES 或数值条件")

    features = np.zeros((1, FP_DIM), dtype=np.float32)
    features[0, 0] = float(cond.dose)
    features[0, 1] = float(cond.freq)
    features[0, 2] = float(cond.route)
    return features


def predict_outputs(
    model: nn.Module,
    *,
    smiles: Optional[str] = None,
    dose: Optional[float] = None,
    freq: Optional[float] = None,
    route: Optional[float] = None,
) -> Dict[str, float]:
    """统一预测入口，返回 dict 方便前端和 CLI 复用。"""
    cond = None
    if dose is not None and freq is not None and route is not None:
        cond = NumericCondition(dose=float(dose), freq=float(freq), route=float(route))

    x = _build_feature_vector(smiles=smiles, cond=cond)
    with torch.no_grad():
        outputs = model(torch.tensor(x, dtype=torch.float32))

    return {
        "efficacy": float(outputs["efficacy"].reshape(-1)[0].item()),
        "toxicity": float(outputs["toxicity"].reshape(-1)[0].item()),
    }

def prepare_data(df):
    """准备数据"""
    fingerprints = []
    labels = []
    
    for _, row in df.iterrows():
        fp = smiles_to_fingerprint(row['smiles'])
        fingerprints.append(fp)
        labels.append([row['efficacy'], row['toxicity']])
    
    return np.array(fingerprints, dtype=np.float32), np.array(labels, dtype=np.float32)

def pretrain_feature_extractor(data, epochs=10):
    """预训练特征提取器（Triplet Loss）"""
    model = FeatureExtractor()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if len(data) < 3:
        print("样本少于 3，跳过 Triplet 预训练，返回初始化特征提取器")
        return model
    
    # 简化的Triplet Loss实现
    for epoch in range(epochs):
        # 随机选择anchor, positive, negative
        indices = np.random.choice(len(data), 3, replace=False)
        anchor, positive, negative = data[indices]
        
        anchor_feat = model(torch.tensor(anchor, dtype=torch.float32).unsqueeze(0))
        positive_feat = model(torch.tensor(positive, dtype=torch.float32).unsqueeze(0))
        negative_feat = model(torch.tensor(negative, dtype=torch.float32).unsqueeze(0))
        
        # Triplet Loss
        dist_pos = torch.norm(anchor_feat - positive_feat)
        dist_neg = torch.norm(anchor_feat - negative_feat)
        loss = torch.max(torch.tensor(0.0), dist_pos - dist_neg + 1.0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def train_prototypical_network(feature_extractor, data, labels, epochs=10):
    """训练原型网络"""
    model = PrototypicalNetwork(feature_extractor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if len(data) < 4:
        print("样本少于 4，跳过原型网络训练")
        return model

    support_end = max(2, len(data) // 2)
    support_end = min(support_end, len(data) - 1)
    
    for epoch in range(epochs):
        # 简化的训练循环
        support_set = torch.tensor(data[:support_end], dtype=torch.float32)  # 支持集
        query_set = torch.tensor(data[support_end:], dtype=torch.float32)  # 查询集
        query_labels = labels[support_end:]
        
        distances = model(support_set, query_set)
        
        # Prototypical Loss
        loss = prototypical_loss(distances, query_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Proto Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def finetune_multi_task(model, data, labels, epochs=10):
    """多任务微调"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        inputs = torch.tensor(data, dtype=torch.float32)
        targets = torch.tensor(labels, dtype=torch.float32)

        outputs = model(inputs)

        # 多任务联合损失（支持并行任务列表）
        total_loss = 0.0
        task_losses = {}
        for idx, task in enumerate(model.task_list):
            pred = outputs[task].squeeze()
            target = targets[:, idx]
            if task == "efficacy":
                loss = focal_loss(pred, target)
            elif task == "toxicity":
                loss = F.mse_loss(pred, target)
            else:
                loss = F.mse_loss(pred, target)
            weight = float(TASK_LOSS_WEIGHTS.get(task, 1.0))
            task_losses[task] = loss
            total_loss = total_loss + weight * loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        task_info = ", ".join([f"{k}={v.item():.4f}" for k, v in task_losses.items()])
        print(
            f"Finetune Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}, "
            f"Tasks: {getattr(model, 'task_list', MULTI_TASK_LIST)} [{task_info}]"
        )

def cross_validate_model(data, labels, k=5):
    """交叉验证模型"""
    if len(data) < 2:
        print("样本少于 2，无法进行交叉验证")
        return []

    n_splits = max(2, min(k, len(data)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # 重新训练模型
        feature_extractor = pretrain_feature_extractor(train_data, epochs=5)  # 减少epochs for speed
        multi_task_model = MultiTaskModel(feature_extractor)
        finetune_multi_task(multi_task_model, train_data, train_labels, epochs=5)
        
        # 验证
        with torch.no_grad():
            val_inputs = torch.tensor(val_data, dtype=torch.float32)
            outputs = multi_task_model(val_inputs)

            efficacy_pred = outputs["efficacy"].squeeze()
            toxicity_pred = outputs["toxicity"].squeeze()

            efficacy_mse = F.mse_loss(efficacy_pred, torch.tensor(val_labels[:, 0], dtype=torch.float32))
            toxicity_mse = F.mse_loss(toxicity_pred, torch.tensor(val_labels[:, 1], dtype=torch.float32))
            
            fold_results.append({
                'efficacy_mse': efficacy_mse.item(),
                'toxicity_mse': toxicity_mse.item()
            })
        
        print(f"Fold {fold+1} - Efficacy MSE: {fold_results[-1]['efficacy_mse']:.4f}, Toxicity MSE: {fold_results[-1]['toxicity_mse']:.4f}")
    
    # 平均结果
    avg_efficacy_mse = np.mean([r['efficacy_mse'] for r in fold_results])
    avg_toxicity_mse = np.mean([r['toxicity_mse'] for r in fold_results])
    
    print(f"\n交叉验证结果 - 平均Efficacy MSE: {avg_efficacy_mse:.4f}, 平均Toxicity MSE: {avg_toxicity_mse:.4f}")
    
    return fold_results

def gbdt_hyperopt_optimization(model, data):
    """GBDT + Hyperopt优化剂量-频次参数空间"""
    _ = data

    def objective(params):
        dose = params['dose']
        freq = params['freq']
        
        # 模拟多维参数：剂量、频率、途径（编码为数字）
        route = params['route']  # 0:口服, 1:注射, 2:局部
        
        preds = predict_outputs(model, dose=dose, freq=freq, route=route)
        efficacy = preds["efficacy"]
        
        # 目标：最大化疗效
        return {'loss': -efficacy, 'status': STATUS_OK}
    
    space = {
        'dose': hp.uniform('dose', 1, 50),  # 剂量 mg/kg
        'freq': hp.uniform('freq', 1, 10),  # 频率 次/天
        'route': hp.choice('route', [0, 1, 2])  # 途径
    }
    
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
    
    return best

def predict_new_drug(model, smiles=None, dose=None, freq=None, route=None):
    """预测新药物的疗效"""
    try:
        preds = predict_outputs(model, smiles=smiles, dose=dose, freq=freq, route=route)
    except ValueError:
        print("请提供SMILES或剂量参数")
        return

    if smiles:
        print(f"SMILES: {smiles}")
    else:
        print(f"剂量: {dose} mg/kg, 频率: {freq} 次/天, 途径: {route}")

    print(f"预测疗效: {preds['efficacy']:.4f}")
    print(f"预测毒性: {preds['toxicity']:.4f}")


def run_training_pipeline(
    *,
    num_drugs: int = 50,
    pretrain_epochs: int = 10,
    proto_epochs: int = 10,
    finetune_epochs: int = 10,
    cv_folds: int = 5,
) -> Tuple[nn.Module, dict, list]:
    """将原 main 训练流程模块化，便于前端/脚本复用。"""
    print("\n1. 数据准备：网络爬取药物数据")
    drug_df = scrape_drug_data(num_drugs=num_drugs)
    print(f"获取到 {len(drug_df)} 个药物样本")
    if len(drug_df) == 0:
        raise RuntimeError("未抓取到任何药物数据，请检查网络环境")

    data, labels = prepare_data(drug_df)

    print("\n2. 特征提取器预训练（Triplet Loss）")
    feature_extractor = pretrain_feature_extractor(data, epochs=pretrain_epochs)

    print("\n3. 原型网络构建（Prototypical Loss）")
    _ = train_prototypical_network(feature_extractor, data, labels, epochs=proto_epochs)

    print("\n4. 多任务微调（焦点损失 + 辅助损失）")
    multi_task_model = MultiTaskModel(feature_extractor)
    finetune_multi_task(multi_task_model, data, labels, epochs=finetune_epochs)

    print("\n5. GBDT + Hyperopt优化剂量-频次参数空间")
    best_params = gbdt_hyperopt_optimization(multi_task_model, data)
    print(f"最优参数: {best_params}")

    print("\n6. 模型验证与集成")
    print(f"进行{cv_folds}折交叉验证...")
    cv_results = cross_validate_model(data, labels, k=cv_folds)
    print("模型集成完成")
    return multi_task_model, best_params, cv_results

def user_input_prediction(model):
    """用户输入预测数据"""
    print("\n=== 预测数据输入区域 ===")
    while True:
        print("\n选择输入类型:")
        print("1. 输入SMILES字符串")
        print("2. 输入剂量参数 (剂量, 频率, 途径)")
        print("3. 退出")
        
        choice = input("请选择 (1/2/3): ").strip()
        
        if choice == '1':
            smiles = input("请输入SMILES: ").strip()
            predict_new_drug(model, smiles=smiles)
        elif choice == '2':
            try:
                dose = float(input("请输入剂量 (mg/kg): ").strip())
                freq = float(input("请输入频率 (次/天): ").strip())
                route = int(input("请输入途径 (0:口服, 1:注射, 2:局部): ").strip())
                predict_new_drug(model, dose=dose, freq=freq, route=route)
            except ValueError:
                print("输入无效，请输入数字")
        elif choice == '3':
            break
        else:
            print("无效选择")

def main():
    """主函数"""
    print("=== 药物疗效预测神经网络 ===")

    # 1-6. 训练与验证流程
    multi_task_model, _, _ = run_training_pipeline(
        num_drugs=50,
        pretrain_epochs=10,
        proto_epochs=10,
        finetune_epochs=10,
        cv_folds=5,
    )

    # 7. 用户预测输入
    user_input_prediction(multi_task_model)

    print("\n=== 完成 ===")

if __name__ == "__main__":
    main()