import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

/** Simulate small-molecule PK (CTM 6-compartment model) */
export async function simulateCTM(bridge: PythonBridge): Promise<void> {
    const dose = await vscode.window.showInputBox({ prompt: 'Dose (mg)', value: '200' });
    if (!dose) return;
    const freq = await vscode.window.showInputBox({ prompt: 'Frequency (per day)', value: '2' });
    if (!freq) return;
    const binding = await vscode.window.showInputBox({ prompt: 'Binding score (0-1)', value: '0.72' });
    if (!binding) return;
    const immune = await vscode.window.showInputBox({ prompt: 'Immune score (0-1)', value: '0.65' });
    if (!immune) return;
    const inflammation = await vscode.window.showInputBox({ prompt: 'Inflammation score (0-1)', value: '0.12' });
    if (!inflammation) return;

    try {
        const result = await bridge.call('ctm_simulate', {
            dose: parseFloat(dose), freq: parseFloat(freq),
            binding: parseFloat(binding), immune: parseFloat(immune),
            inflammation: parseFloat(inflammation), horizon: 72,
        });
        showPKResult(result, 'CTM PK Simulation');
    } catch (e: any) {
        vscode.window.showErrorMessage(`CTM simulation failed: ${e}`);
    }
}

/** Simulate circRNA PK (RNA-CTM 6-compartment model) */
export async function simulateRNACTM(bridge: PythonBridge): Promise<void> {
    const dose = await vscode.window.showInputBox({ prompt: 'Dose (mg)', value: '5' });
    if (!dose) return;
    const freq = await vscode.window.showInputBox({ prompt: 'Frequency (per day)', value: '1' });
    if (!freq) return;
    const mod = await vscode.window.showQuickPick(
        ['none', 'm6A', 'pseudouridine', '5mC', 'ms2m6A'],
        { placeHolder: 'Select nucleotide modification' });
    if (!mod) return;

    try {
        const result = await bridge.call('rna_ctm_simulate', {
            dose: parseFloat(dose), freq: parseFloat(freq),
            modification: mod, horizon: 168,
        });
        showPKResult(result, `circRNA PK (${mod})`);
    } catch (e: any) {
        vscode.window.showErrorMessage(`RNA-CTM simulation failed: ${e}`);
    }
}

/** Show PK curve in a webview panel using Plotly.js */
function showPKResult(data: Record<string, number[]>, title: string): void {
    const panel = vscode.window.createWebviewPanel('confluencia.pkCurve', title,
        vscode.ViewColumn.One, { enableScripts: true });

    const time = data.time_h;
    const efficacy = data.efficacy_signal;
    const toxicity = data.toxicity_signal;

    panel.webview.html = `<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body>
<div id="plot" style="width:100%;height:400px;"></div>
<script>
Plotly.newPlot('plot', [
  {x: ${JSON.stringify(time)}, y: ${JSON.stringify(efficacy)},
   name: 'Efficacy Signal', mode: 'lines', line: {color: '#2196F3'}},
  {x: ${JSON.stringify(time)}, y: ${JSON.stringify(toxicity)},
   name: 'Toxicity Signal', mode: 'lines', line: {color: '#F44336'}}
], {
  title: '${title}',
  xaxis: {title: 'Time (h)'},
  yaxis: {title: 'Signal'},
  template: 'plotly_dark'
});
</script></body></html>`;
}