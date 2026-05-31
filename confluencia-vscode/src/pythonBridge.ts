import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Python bridge using JSON-RPC over stdin/stdout to confluencia_bridge.py.
 * Mirrors the kernel-server.py pattern used in the Electron Studio.
 */
export class PythonBridge {
    private proc: cp.ChildProcessWithoutNullStreams | null = null;
    private pending: Map<number, { resolve: (v: any) => void; reject: (e: any) => void }> = new Map();
    private nextId = 1;
    private buffer = '';
    private pythonPath: string = '';
    private disposed = false;

    constructor(private context: vscode.ExtensionContext) {
        this.pythonPath = this.findPython();
    }

    /** Find Python executable following priority chain */
    findPython(): string {
        // 1. Extension setting
        const cfg = vscode.workspace.getConfiguration('confluencia');
        const cfgPath = cfg.get<string>('pythonPath', '');
        if (cfgPath && fs.existsSync(cfgPath)) return cfgPath;

        // 2. Environment variable
        const envPath = process.env.CONFLUENCIA_PYTHON || '';
        if (envPath && fs.existsSync(envPath)) return envPath;

        // 3. .venv in workspace
        const wsRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (wsRoot) {
            const venvPy = path.join(wsRoot, '.venv',
                process.platform === 'win32' ? 'Scripts\\python.exe' : 'bin/python');
            if (fs.existsSync(venvPy)) return venvPy;
        }

        // 4. System Python
        return process.platform === 'win32' ? 'python' : 'python3';
    }

    /** Set Python path and restart bridge */
    setPython(pythonPath: string) {
        this.pythonPath = pythonPath;
        this.stop();
    }

    /** Ensure the JSON-RPC subprocess is running */
    private async ensureStarted(): Promise<void> {
        if (this.proc && !this.proc.killed) return;

        const bridgeScript = path.join(this.context.extensionPath, 'src', 'python', 'confluencia_bridge.py');
        const projectRoot = vscode.workspace.getConfiguration('confluencia').get<string>('projectRoot', '') || '';

        const env: Record<string, string> = { ...process.env as Record<string, string> };
        env.PYTHONIOENCODING = 'utf-8';
        if (projectRoot) env.CONFLUENCIA_ROOT = projectRoot;

        this.proc = cp.spawn(this.pythonPath, ['-X', 'utf8', bridgeScript, '--mode', 'rpc'], {
            env,
            stdio: ['pipe', 'pipe', 'pipe'],
            windowsHide: true,
        });

        this.proc.stdout!.on('data', (data: Buffer) => {
            this.buffer += data.toString('utf-8');
            const lines = this.buffer.split('\n');
            this.buffer = lines.pop() || '';
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    if ('event' in msg && msg.event === 'ready') {
                        // Bridge is ready
                    } else if ('id' in msg) {
                        const pending = this.pending.get(msg.id);
                        if (pending) {
                            this.pending.delete(msg.id);
                            if ('result' in msg) pending.resolve(msg.result);
                            else pending.reject(msg.error || 'Unknown error');
                        }
                    }
                } catch { /* ignore non-JSON lines */ }
            }
        });

        this.proc.stderr!.on('data', (data: Buffer) => {
            console.error('[Confluencia bridge]', data.toString('utf-8'));
        });

        // Wait briefly for ready event
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    /** Call a bridge method via JSON-RPC */
    async call(method: string, params: Record<string, any> = {}): Promise<any> {
        await this.ensureStarted();
        const id = this.nextId++;
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            const msg = JSON.stringify({ id, method, params }) + '\n';
            this.proc!.stdin!.write(msg);
            // Timeout after 60s
            setTimeout(() => {
                if (this.pending.has(id)) {
                    this.pending.delete(id);
                    reject(new Error(`Bridge call timed out: ${method}`));
                }
            }, 60000);
        });
    }

    /** Stop the bridge subprocess */
    stop() {
        if (this.proc && !this.proc.killed) {
            this.proc.kill();
            this.proc = null;
        }
    }

    dispose() {
        this.disposed = true;
        this.stop();
    }
}