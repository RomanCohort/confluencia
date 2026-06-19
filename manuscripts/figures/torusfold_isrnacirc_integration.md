# TorusFold物理模式改进方案（借鉴IsRNAcirc）

## 1. 多副本采样 (REMD-inspired)

**IsRNAcirc做法**: 运行8-16个副本，温度300K-500K，定期交换构象

**TorusFold改进**:
```python
# 在 physics_structure_head.py 中添加 REMDSampler

class REMDSampler:
    """Replica Exchange Molecular Dynamics-inspired sampler."""
    
    def __init__(
        self,
        n_replicas: int = 8,
        temp_min: float = 300.0,  # K
        temp_max: float = 450.0,  # K
        exchange_interval: int = 100,
    ):
        self.temps = np.linspace(temp_min, temp_max, n_replicas)
        self.n_replicas = n_replicas
        
    def sample(self, constraints: ConstraintSet) -> List[np.ndarray]:
        """Run parallel sampling at multiple temperatures."""
        conformations = []
        for T in self.temps:
            # Higher temperature = more flexible sampling
            conf = self._sample_at_temp(constraints, T)
            conformations.append(conf)
        
        # Exchange step: swap conformations between adjacent temps
        for i in range(self.n_replicas - 1):
            if self._should_exchange(conformations[i], conformations[i+1]):
                conformations[i], conformations[i+1] = conformations[i+1], conformations[i]
        
        return conformations
```

## 2. 模拟退火末端闭合 (End Closure with Annealing)

**IsRNAcirc做法**: 从线性RNA出发，通过模拟退火将5'和3'端连接

**TorusFold改进**:
```python
# 在 constraint_solver.py 中添加 AnnealingClosure

class AnnealingClosure:
    """Simulated annealing for BSJ closure."""
    
    def __init__(
        self,
        initial_temp: float = 500.0,  # K
        final_temp: float = 300.0,    # K
        cooling_rate: float = 0.95,
        n_steps_per_temp: int = 50,
    ):
        pass
        
    def close_bsj(
        self,
        coords: np.ndarray,  # Initial linear-like coords
        bsj_idx: int,        # Index of back-splice junction
        target_distance: float = 0.5,  # Å tolerance
    ) -> np.ndarray:
        """Anneal the ends together."""
        T = self.initial_temp
        while T > self.final_temp:
            for step in range(self.n_steps_per_temp):
                # Perturb positions near BSJ
                perturbed = self._perturb_near_bsj(coords, bsj_idx, T)
                # Compute closure distance
                dist = np.linalg.norm(perturbed[0] - perturbed[-1])
                # Accept if closer or with temperature-dependent probability
                if dist < target_distance or np.random.random() < np.exp(-dist/T):
                    coords = perturbed
            T *= self.cooling_rate
        return coords
```

## 3. 5-bead粗粒化表示

**IsRNAcirc做法**: 每个核苷酸用5个珠子表示（磷酸、糖环、碱基3个珠子）

**TorusFold改进**: 在physics_ba模式中支持5-bead CG

```python
class FiveBeadCGModel:
    """5-bead coarse-grained model per nucleotide."""
    
    # Bead positions (relative to nucleotide center):
    # - P: phosphate backbone
    # - S: sugar ring center
    # - B1, B2, B3: base beads (for orientation)
    
    BEAD_POSITIONS = {
        'P': (0.0, 0.0, -5.9/2),   # ~3Å from sugar toward backbone
        'S': (0.0, 0.0, 0.0),       # sugar center
        'B1': (0.0, 2.5, 0.0),      # base major groove
        'B2': (0.0, -2.5, 0.0),     # base minor groove
        'B3': (0.0, 0.0, 3.4),      # base stacking direction
    }
    
    def to_all_atom(self, cg_coords: np.ndarray) -> np.ndarray:
        """Convert 5-bead CG to full atomic coordinates."""
        # Use template matching for each nucleotide
        pass
```

## 4. 整合到PhysicsStructureHead

```python
class PhysicsStructureHead(nn.Module):
    def __init__(self, ...):
        # Add REMD sampler option
        self.remd_sampler = REMDSampler(
            n_replicas=kwargs.get('n_remd_replicas', 4),
        )
        
        # Add annealing closure
        self.annealing_closure = AnnealingClosure(
            initial_temp=kwargs.get('annealing_temp_init', 500.0),
        )
        
        # Add 5-bead CG option
        self.use_5bead_cg = kwargs.get('use_5bead_cg', False)
        
    def forward(self, pair_repr: torch.Tensor, seq_len: int):
        # Extract constraints
        constraints = self.constraint_extractor(pair_repr)
        
        # REMD sampling (if enabled)
        if self.use_remd:
            conformations = self.remd_sampler.sample(constraints)
            best_conf = self._select_lowest_energy(conformations)
        else:
            best_conf = self.constraint_solver.solve(constraints)
        
        # Annealing closure for BSJ
        best_conf = self.annealing_closure.close_bsj(best_conf, seq_len)
        
        # Optional CGMD refinement
        if self.structure_mode == "physics_ba":
            best_conf = self.cgmd_refiner.refine(best_conf)
        
        return best_conf
```

## 5. 配置更新

```python
@dataclass
class TorusFoldConfig:
    # ... existing fields ...
    
    # REMD sampling (IsRNAcirc-inspired)
    use_remd: bool = False
    n_remd_replicas: int = 4
    remd_temp_min: float = 300.0
    remd_temp_max: float = 450.0
    
    # Annealing closure (IsRNAcirc-inspired)
    use_annealing_closure: bool = True
    annealing_temp_init: float = 500.0
    annealing_temp_final: float = 300.0
    
    # 5-bead CG model (IsRNAcirc-inspired)
    use_5bead_cg: bool = False  # More accurate but slower
```

## 实现优先级

| 改进 | 优先级 | 原因 |
|------|--------|------|
| 模拟退火末端闭合 | 🔴 高 | 直接解决BSJ闭合精度问题 |
| REMD多副本采样 | 🟡 中 | 提高构象覆盖率，但计算量增加 |
| 5-bead CG表示 | 🟢 低 | 更精细但实现复杂度高 |

## 论文中的描述

"Under data scarcity, TorusFold's physics mode borrows concepts from IsRNAcirc: simulated annealing for BSJ closure (ensuring circular topology), optional REMD-inspired multi-replica sampling for conformation diversity, and 5-bead coarse-grained representation for higher structural accuracy. This hybrid approach combines DL-based pair prediction with physics-based refinement, complementary to IsRNAcirc's ab initio MD approach."