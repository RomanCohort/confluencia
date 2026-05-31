import * as vscode from 'vscode';
import { PythonBridge } from './pythonBridge';
import { simulateCTM, simulateRNACTM } from './commands/simulateCTM';
import { predictImmunogenicity } from './commands/predictImmunogenicity';
import { jointEvaluate } from './commands/jointEvaluate';
import { predictDrug } from './commands/predictDrug';
import { predictEpitope } from './commands/predictEpitope';
import { mhcEncode } from './commands/mhcEncode';
import { regMetrics } from './commands/regMetrics';
import { selectPython } from './commands/selectPython';
import { trainDrug, trainEpitope } from './commands/trainModel';
import { ScoreTreeProvider } from './views/scoreTreeView';

let bridge: PythonBridge;

export function activate(context: vscode.ExtensionContext) {
    bridge = new PythonBridge(context);

    // Score tree view
    const scoreProvider = new ScoreTreeProvider();
    vscode.window.registerTreeDataProvider('confluencia.scoreTree', scoreProvider);

    // Register commands
    const cmds: [string, (...args: any[]) => any][] = [
        ['confluencia.simulateCTM', () => simulateCTM(bridge)],
        ['confluencia.simulateRNACTM', () => simulateRNACTM(bridge)],
        ['confluencia.predictImmunogenicity', () => predictImmunogenicity(bridge)],
        ['confluencia.jointEvaluate', () => jointEvaluate(bridge, scoreProvider)],
        ['confluencia.predictDrug', () => predictDrug(bridge)],
        ['confluencia.predictEpitope', () => predictEpitope(bridge)],
        ['confluencia.mhcEncode', () => mhcEncode(bridge)],
        ['confluencia.regMetrics', () => regMetrics(bridge)],
        ['confluencia.selectPython', () => selectPython(bridge)],
        ['confluencia.trainDrug', () => trainDrug(bridge)],
        ['confluencia.trainEpitope', () => trainEpitope(bridge)],
    ];

    for (const [id, fn] of cmds) {
        context.subscriptions.push(vscode.commands.registerCommand(id, fn));
    }

    vscode.commands.executeCommand('setContext', 'confluencia.hasResult', false);
}

export function deactivate() {
    bridge?.dispose();
}