import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

export async function predictEpitope(bridge: PythonBridge): Promise<void> {
    const bundle = await vscode.window.showInputBox({ prompt: 'Model bundle path (.joblib)' });
    if (!bundle) return;
    const sequence = await vscode.window.showInputBox({ prompt: 'Epitope sequence' });
    if (!sequence) return;

    try {
        const result = await bridge.call('epitope_predict', { bundle_path: bundle, sequence });
        vscode.window.showInformationMessage(`Epitope binding: ${result}`);
    } catch (e: any) {
        vscode.window.showErrorMessage(`Prediction failed: ${e}`);
    }
}