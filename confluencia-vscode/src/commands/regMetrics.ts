import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

export async function regMetrics(bridge: PythonBridge): Promise<void> {
    const yTrueStr = await vscode.window.showInputBox({ prompt: 'True values (comma-separated)', value: '1,2,3,4,5' });
    if (!yTrueStr) return;
    const yPredStr = await vscode.window.showInputBox({ prompt: 'Predicted values (comma-separated)', value: '1.1,1.9,3.1,3.8,5.2' });
    if (!yPredStr) return;

    const yTrue = yTrueStr.split(',').map(Number);
    const yPred = yPredStr.split(',').map(Number);

    try {
        const result = await bridge.call('reg_metrics', { y_true: yTrue, y_pred: yPred });
        vscode.window.showInformationMessage(
            `MAE=${result.mae?.toFixed(4)}, RMSE=${result.rmse?.toFixed(4)}, R²=${result.r2?.toFixed(4)}`);
    } catch (e: any) {
        vscode.window.showErrorMessage(`Metrics failed: ${e}`);
    }
}