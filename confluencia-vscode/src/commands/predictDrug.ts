import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

export async function predictDrug(bridge: PythonBridge): Promise<void> {
    const bundle = await vscode.window.showInputBox({ prompt: 'Model bundle path (.joblib)' });
    if (!bundle) return;
    const smiles = await vscode.window.showInputBox({ prompt: 'SMILES' });
    if (!smiles) return;

    try {
        const result = await bridge.call('drug_predict', { bundle_path: bundle, smiles });
        vscode.window.showInformationMessage(`Drug efficacy: ${result}`);
    } catch (e: any) {
        vscode.window.showErrorMessage(`Prediction failed: ${e}`);
    }
}