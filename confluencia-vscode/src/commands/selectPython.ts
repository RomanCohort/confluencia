import * as vscode from 'vscode';
import { PythonBridge } from '../pythonBridge';

export async function selectPython(bridge: PythonBridge): Promise<void> {
    const result = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        filters: { 'Python Executable': ['exe', 'py'] },
        title: 'Select Python executable with Confluencia installed',
    });
    if (result && result[0]) {
        const pythonPath = result[0].fsPath;
        const cfg = vscode.workspace.getConfiguration('confluencia');
        cfg.update('pythonPath', pythonPath, vscode.ConfigurationTarget.Workspace);
        bridge.setPython(pythonPath);
        vscode.window.showInformationMessage(`Confluencia Python set to: ${pythonPath}`);
    }
}