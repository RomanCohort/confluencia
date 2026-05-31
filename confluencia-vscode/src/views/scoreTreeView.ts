import * as vscode from 'vscode';

interface ScoreEntry {
    name: string;
    score: number;
    children?: ScoreEntry[];
}

export class ScoreTreeProvider implements vscode.TreeDataProvider<ScoreItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<ScoreItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private result: Record<string, any> | null = null;

    setResult(result: Record<string, any>): void {
        this.result = result;
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: ScoreItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ScoreItem): ScoreItem[] {
        if (!this.result) return [];

        const composite = this.result.composite ?? 0;
        const recommendation = this.result.recommendation ?? 'N/A';

        if (!element) {
            // Root: composite + recommendation
            const icon = recommendation === 'Go' ? 'check' :
                         recommendation === 'No-Go' ? 'error' : 'warning';
            return [
                new ScoreItem(`Composite: ${typeof composite === 'number' ? composite.toFixed(3) : composite} [${recommendation}]`,
                    vscode.TreeItemCollapsibleState.Expanded, icon),
            ];
        }

        // Dimension items
        if (this.result) {
            const dims = [
                { label: 'Clinical', key: 'clinical_score', fields: ['efficacy', 'binding', 'safety_penalty'] },
                { label: 'Binding', key: 'binding_score', fields: ['epitope_efficacy', 'affinity_class'] },
                { label: 'Kinetics', key: 'kinetics_score', fields: ['auc_effect', 'therapeutic_index'] },
                { label: 'Gene Signature', key: 'gene_signature_score', fields: ['risk_score', 'predicted_response'] },
                { label: 'CircRNA', key: 'circrna_score', fields: ['immunotherapy_score', 'tide_score'] },
            ];
            return dims.map(d => {
                const dimData = this.result?.[d.key];
                const score = dimData?.overall ?? dimData?.score ?? 'N/A';
                return new ScoreItem(
                    `${d.label} (${typeof score === 'number' ? score.toFixed(2) : score})`,
                    vscode.TreeItemCollapsibleState.Collapsed,
                    typeof score === 'number' && score >= 0.65 ? 'check' :
                    typeof score === 'number' && score < 0.4 ? 'error' : 'warning'
                );
            });
        }
        return [];
    }
}

class ScoreItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        icon?: string,
    ) {
        super(label, collapsibleState);
        if (icon === 'check') this.iconPath = new vscode.ThemeIcon('check', new vscode.ThemeColor('charts.green'));
        else if (icon === 'error') this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
        else if (icon === 'warning') this.iconPath = new vscode.ThemeIcon('warning', new vscode.ThemeColor('list.warningForeground'));
    }
}