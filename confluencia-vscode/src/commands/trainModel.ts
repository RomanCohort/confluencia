import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

export async function trainDrug(bridge: PythonBridge): Promise<void> {
    const csvPath = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        filters: { 'CSV Files': ['csv'] },
        title: 'Select training CSV for drug model',
    });
    if (!csvPath || csvPath.length === 0) return;

    const modelNames = ['ridge', 'rf', 'gbr', 'hgb', 'mlp'];
    const modelPick = await vscode.window.showQuickPick(modelNames, {
        placeHolder: 'Select model type (default: gbr)',
    });
    if (!modelPick) return;

    try {
        const result = await bridge.call('drug_train', {
            csv_path: csvPath[0].fsPath,
            model_name: modelPick,
        });
        const msg = `Drug model trained: R²=${result.r2?.toFixed(3)}, MAE=${result.mae?.toFixed(3)}, saved to ${result.bundle_path}`;
        vscode.window.showInformationMessage(msg);
    } catch (e: any) {
        vscode.window.showErrorMessage(`Training failed: ${e}`);
    }
}

export async function trainEpitope(bridge: PythonBridge): Promise<void> {
    const csvPath = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        filters: { 'CSV Files': ['csv'] },
        title: 'Select training CSV for epitope model',
    });
    if (!csvPath || csvPath.length === 0) return;

    const modelNames = ['ridge', 'rf', 'hgb', 'mlp'];
    const modelPick = await vscode.window.showQuickPick(modelNames, {
        placeHolder: 'Select model type (default: hgb)',
    });
    if (!modelPick) return;

    try {
        const result = await bridge.call('epitope_train', {
            csv_path: csvPath[0].fsPath,
            model_name: modelPick,
        });
        const msg = `Epitope model trained: R²=${result.r2?.toFixed(3)}, MAE=${result.mae?.toFixed(3)}, saved to ${result.bundle_path}`;
        vscode.window.showInformationMessage(msg);
    } catch (e: any) {
        vscode.window.showErrorMessage(`Training failed: ${e}`);
    }
}
