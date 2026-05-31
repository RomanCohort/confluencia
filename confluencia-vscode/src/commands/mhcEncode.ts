import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

export async function mhcEncode(bridge: PythonBridge): Promise<void> {
    const peptide = await vscode.window.showInputBox({ prompt: 'Peptide sequence' });
    if (!peptide) return;
    const allele = await vscode.window.showInputBox({ prompt: 'MHC allele (e.g. HLA-A*02:01)' });
    if (!allele) return;

    try {
        const result = await bridge.call('mhc_encode', { peptide, allele });
        vscode.window.showInformationMessage(`MHC features: ${result.length}-dim vector`);
    } catch (e: any) {
        vscode.window.showErrorMessage(`Encoding failed: ${e}`);
    }
}