import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

export async function predictImmunogenicity(bridge: PythonBridge): Promise<void> {
    const sequence = await vscode.window.showInputBox({
        prompt: 'circRNA sequence (ACGU)', value: 'ACGUACGUACGUACGUACGUACGU',
    });
    if (!sequence) return;

    try {
        const result = await bridge.call('circrna_immunogenicity', { sequence });
        const overall = result.overall_immunogenicity ?? 'N/A';
        vscode.window.showInformationMessage(
            `Immunogenicity: overall=${typeof overall === 'number' ? overall.toFixed(4) : overall}`);
    } catch (e: any) {
        vscode.window.showErrorMessage(`Prediction failed: ${e}`);
    }
}