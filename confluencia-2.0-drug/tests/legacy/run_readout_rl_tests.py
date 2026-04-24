import torch
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))

from gnn import AttentionReadout
from rl_sampling import AtomPolicyNet, sample_atoms, reinforce_update


def main():
    # test attention readout
    node_emb = torch.randn(5, 16)
    readout = AttentionReadout(16)
    out = readout(node_emb)
    assert out.shape == (16,)
    print('test_attention_readout_basic: OK')

    # test RL policy
    node_emb = torch.randn(8, 12)
    policy = AtomPolicyNet(12)
    logits = policy(node_emb)
    assert logits.shape == (8,)

    idx, logp = sample_atoms(policy, node_emb, k=2)
    assert idx.numel() == 2
    assert logp.numel() == 2

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rewards = torch.tensor([1.0, 0.5])
    loss_val = reinforce_update(opt, logp, rewards, baseline=0.0)
    assert isinstance(loss_val, float)
    print('test_rl_policy_and_sampling: OK')


if __name__ == '__main__':
    main()
