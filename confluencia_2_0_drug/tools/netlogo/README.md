# NetLogo Immune ABM

This folder contains a simplified agent-based immune simulation model.

## Files

- `immune_response_abm.nlogo`: ready-to-open NetLogo project with interface
- `immune_response_abm.nls`: NetLogo code-only version (paste into NetLogo Code tab)
- Trigger CSV: generated from Python via `core.immune_abm.export_netlogo_trigger_csv`

## Agent definitions

- APC (`apcs`): capture and present antigens
- T cells (`t-cells`): activate after APC contact and expand
- B cells (`b-cells`): become plasma-like with T-cell help
- Antibodies (`antibodies`): neutralize antigens
- Antigens (`antigens`): introduced from epitope triggers

## Trigger input

Expected CSV columns:

- `sample_id`
- `tick`
- `epitope_seq`
- `immunogenicity`
- `antigen_input`

If using `.nlogo`, open file directly and run buttons.

If using code-only `.nls`, in NetLogo Command Center:

```netlogo
setup
load-trigger-events "D:/path/to/epitope_triggers.csv" 0
repeat 120 [ go ]
```

## Typical outputs to monitor

- `antigen-pool`
- `activated-t-count`
- `plasma-b-count`
- `antibody-titer`
- `cytokine-level`
