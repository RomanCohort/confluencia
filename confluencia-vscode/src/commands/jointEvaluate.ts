import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';
import { ScoreTreeProvider } from '../views/scoreTreeView';

/** Run 5D joint evaluation */
export async function jointEvaluate(bridge: PythonBridge, scoreProvider: ScoreTreeProvider): Promise<void> {
    const smiles = await vscode.window.showInputBox({ prompt: 'SMILES', value: 'CC(=O)Oc1ccccc1C(=O)O' });
    if (!smiles) return;
    const epitopeSeq = await vscode.window.showInputBox({ prompt: 'Epitope sequence', value: 'SLYNTVATL' });
    if (!epitopeSeq) return;
    const mhcAllele = await vscode.window.showInputBox({ prompt: 'MHC allele', value: 'HLA-A*02:01' });
    if (!mhcAllele) return;
    const dose = await vscode.window.showInputBox({ prompt: 'Dose (mg)', value: '200' });
    if (!dose) return;
    const freq = await vscode.window.showInputBox({ prompt: 'Frequency (per day)', value: '2' });
    if (!freq) return;
    const treatTime = await vscode.window.showInputBox({ prompt: 'Treatment time (h)', value: '72' });
    if (!treatTime) return;

    try {
        const result = await bridge.call('joint_evaluate', {
            smiles, epitope_seq: epitopeSeq, mhc_allele: mhcAllele,
            dose_mg: parseFloat(dose), freq_per_day: parseFloat(freq),
            treatment_time: parseFloat(treatTime),
        });

        // Update score tree
        scoreProvider.setResult(result);
        vscode.commands.executeCommand('setContext', 'confluencia.hasResult', true);

        // Show result in output channel
        const composite = result.composite ?? 'N/A';
        const recommendation = result.recommendation ?? 'N/A';
        vscode.window.showInformationMessage(
            `Confluencia: Composite ${composite.toFixed ? composite.toFixed(3) : composite} — ${recommendation}`);

    } catch (e: any) {
        vscode.window.showErrorMessage(`Joint evaluation failed: ${e}`);
    }
}