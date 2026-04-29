import { app, BrowserWindow, ipcMain, Menu, shell, net } from 'electron';
import { spawn, ChildProcess, execSync } from 'child_process';
import path from 'path';
import fs from 'fs';

// Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
}

// Global references
let mainWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;
const streamlitProcesses: Map<number, ChildProcess> = new Map();

// Kernel restart tracking
let kernelRestartAttempts = 0;
let lastKernelExitTime = 0;
const MAX_RESTART_ATTEMPTS = 5;
const RESTART_COOLDOWN_MS = 5000; // 5 seconds between restarts
const RESTART_RESET_MS = 60000; // Reset attempt counter after 1 minute of stability

const isDev = !app.isPackaged;
const ROOT_DIR = isDev
  ? path.join(__dirname, '..', '..')
  : path.join(process.resourcesPath, '..');
const CONF_DIR = path.join(process.env.APPDATA || process.env.HOME || '', '.confluencia');

// ---------------------------------------------------------------------------
// Python Runtime Management
// ---------------------------------------------------------------------------

const PYTHON_DIR = isDev
  ? path.join(ROOT_DIR, '.python-embed')
  : path.join(process.resourcesPath, 'python');

const PYTHON_EMBED_URL = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip';
const GET_PIP_URL = 'https://bootstrap.pypa.io/get-pip.py';

function getPythonExe(): string {
  return process.platform === 'win32'
    ? path.join(PYTHON_DIR, 'python.exe')
    : 'python3';
}

function isPythonReady(): boolean {
  return fs.existsSync(getPythonExe());
}

function findPython(): string {
  if (isPythonReady()) return getPythonExe();
  try {
    execSync('python --version 2>&1', { windowsHide: true });
    return 'python';
  } catch {}
  try {
    execSync('python3 --version 2>&1', { windowsHide: true });
    return 'python3';
  } catch {}
  return 'python';
}

// ---------------------------------------------------------------------------
// Window & Menu
// ---------------------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'Confluencia Studio',
    backgroundColor: '#1e1e2e',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false,
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  setupMenu();
}

function setupMenu() {
  const template: Electron.MenuItemConstructorOptions[] = [
    {
      label: 'File',
      submenu: [
        { label: 'New Script', accelerator: 'CmdOrCtrl+N', click: () => mainWindow?.webContents.send('menu:new') },
        { label: 'Open...', accelerator: 'CmdOrCtrl+O', click: () => mainWindow?.webContents.send('menu:open') },
        { label: 'Save', accelerator: 'CmdOrCtrl+S', click: () => mainWindow?.webContents.send('menu:save') },
        { type: 'separator' },
        { label: 'Exit', accelerator: 'CmdOrCtrl+Q', click: () => app.quit() },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { label: 'Undo', accelerator: 'CmdOrCtrl+Z', role: 'undo' },
        { label: 'Redo', accelerator: 'CmdOrCtrl+Shift+Z', role: 'redo' },
        { type: 'separator' },
        { label: 'Cut', accelerator: 'CmdOrCtrl+X', role: 'cut' },
        { label: 'Copy', accelerator: 'CmdOrCtrl+C', role: 'copy' },
        { label: 'Paste', accelerator: 'CmdOrCtrl+V', role: 'paste' },
        { type: 'separator' },
        { label: 'Find', accelerator: 'CmdOrCtrl+F', click: () => mainWindow?.webContents.send('menu:find') },
      ],
    },
    {
      label: 'View',
      submenu: [
        { label: 'Reload', accelerator: 'CmdOrCtrl+R', role: 'reload' },
        { label: 'Toggle DevTools', accelerator: 'F12', role: 'toggleDevTools' },
        { type: 'separator' },
        { label: 'Zoom In', accelerator: 'CmdOrCtrl+Plus', role: 'zoomIn' },
        { label: 'Zoom Out', accelerator: 'CmdOrCtrl+-', role: 'zoomOut' },
        { label: 'Reset Zoom', accelerator: 'CmdOrCtrl+0', role: 'resetZoom' },
      ],
    },
    {
      label: 'Apps',
      submenu: [
        { label: 'Drug Discovery App', click: async () => { if (mainWindow) await launchStreamlitApp('drug'); } },
        { label: 'Epitope Prediction App', click: async () => { if (mainWindow) await launchStreamlitApp('epitope'); } },
        { label: 'circRNA Analysis App', click: async () => { if (mainWindow) await launchStreamlitApp('circrna'); } },
        { label: 'Joint Evaluation App', click: async () => { if (mainWindow) await launchStreamlitApp('joint'); } },
      ],
    },
    {
      label: 'Help',
      submenu: [
        { label: 'Documentation', click: () => shell.openExternal('https://github.com/igem-fbh/confluencia') },
        { label: 'About', click: () => mainWindow?.webContents.send('menu:about') },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// ---------------------------------------------------------------------------
// Python Kernel
// ---------------------------------------------------------------------------

function startPythonKernel() {
  const pythonScript = isDev
    ? path.join(__dirname, '..', 'electron', 'python', 'kernel-server.py')
    : path.join(process.resourcesPath, 'python', 'kernel-server.py');

  const pythonExe = findPython();

  const modulePaths: string[] = [];
  if (isDev) {
    modulePaths.push(
      path.join(ROOT_DIR, 'confluencia_cli'),
      path.join(ROOT_DIR, 'confluencia_shared'),
      path.join(ROOT_DIR, 'confluencia-2.0-drug'),
      path.join(ROOT_DIR, 'confluencia-2.0-epitope'),
      path.join(ROOT_DIR, 'confluencia_circrna'),
      path.join(ROOT_DIR, 'confluencia_joint'),
      path.join(ROOT_DIR, 'confluencia_studio'),
      ROOT_DIR,
    );
  } else {
    modulePaths.push(
      path.join(process.resourcesPath, 'confluencia_cli'),
      path.join(process.resourcesPath, 'confluencia_shared'),
      path.join(process.resourcesPath, 'confluencia-2.0-drug'),
      path.join(process.resourcesPath, 'confluencia-2.0-epitope'),
      path.join(process.resourcesPath, 'confluencia_circrna'),
      path.join(process.resourcesPath, 'confluencia_joint'),
      path.join(process.resourcesPath, 'confluencia_studio'),
      process.resourcesPath,
    );
  }

  const env: Record<string, string> = { ...process.env as Record<string, string> };
  env.PYTHONPATH = modulePaths.join(path.delimiter) + path.delimiter + (env.PYTHONPATH || '');
  env.ELECTRON_RESOURCES_PATH = process.resourcesPath;

  pythonProcess = spawn(pythonExe, [pythonScript], {
    cwd: isDev ? ROOT_DIR : path.dirname(process.execPath),
    env,
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });

  let buffer = '';
  // Buffer early events before renderer is ready
  const earlyEvents: any[] = [];
  let rendererReady = false;

  pythonProcess.stdout?.on('data', (data: Buffer) => {
    buffer += data.toString('utf-8');
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.trim()) {
        try {
          const msg = JSON.parse(line);
          if (rendererReady) {
            mainWindow?.webContents.send('kernel:event', msg);
          } else {
            earlyEvents.push(msg);
          }
        } catch {
          const text = line;
          if (rendererReady) {
            mainWindow?.webContents.send('kernel:output', text);
          } else {
            earlyEvents.push({ event: 'output', data: { text } });
          }
        }
      }
    }
  });

  pythonProcess.stderr?.on('data', (data: Buffer) => {
    mainWindow?.webContents.send('kernel:error', data.toString('utf-8'));
  });

  pythonProcess.on('error', (err) => {
    console.error('Python process error:', err);
    mainWindow?.webContents.send('kernel:error', `Python process error: ${err.message}`);
  });

  pythonProcess.on('exit', (code, signal) => {
    const now = Date.now();
    console.error('Python process exited with code', code, 'signal', signal);
    pythonProcess = null;

    // Reset attempt counter if kernel was stable for a while
    if (now - lastKernelExitTime > RESTART_RESET_MS) {
      kernelRestartAttempts = 0;
    }
    lastKernelExitTime = now;

    // Notify renderer
    mainWindow?.webContents.send('kernel:event', { event: 'disconnected', data: { code, signal } });

    // Attempt restart if within limits
    if (kernelRestartAttempts < MAX_RESTART_ATTEMPTS) {
      kernelRestartAttempts++;
      const delay = Math.min(RESTART_COOLDOWN_MS * kernelRestartAttempts, 30000); // Exponential backoff, max 30s

      mainWindow?.webContents.send('kernel:event', {
        event: 'reconnecting',
        data: { attempt: kernelRestartAttempts, maxAttempts: MAX_RESTART_ATTEMPTS, delayMs: delay }
      });

      console.log(`Restarting kernel in ${delay}ms (attempt ${kernelRestartAttempts}/${MAX_RESTART_ATTEMPTS})`);

      setTimeout(() => {
        if (!pythonProcess && mainWindow) {
          startPythonKernel();
        }
      }, delay);
    } else {
      mainWindow?.webContents.send('kernel:event', {
        event: 'restart_failed',
        data: { attempts: kernelRestartAttempts, maxAttempts: MAX_RESTART_ATTEMPTS }
      });
      console.error('Max kernel restart attempts reached');
    }
  });

  // Flush buffered events once renderer is ready
  mainWindow?.webContents.on('did-finish-load', () => {
    rendererReady = true;
    for (const msg of earlyEvents) {
      mainWindow?.webContents.send('kernel:event', msg);
    }
    earlyEvents.length = 0;
  });
}

function restartPythonKernel(): boolean {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
  kernelRestartAttempts = 0; // Manual restart resets counter
  startPythonKernel();
  return true;
}

function sendToPython(method: string, params: Record<string, unknown> = {}) {
  if (!pythonProcess || !pythonProcess.stdin) {
    mainWindow?.webContents.send('kernel:error', 'Kernel not running. Attempting to reconnect...\n');
    return false;
  }
  const msg = JSON.stringify({ method, params, id: Date.now() });
  pythonProcess.stdin.write(msg + '\n');
  return true;
}

// ---------------------------------------------------------------------------
// Streamlit App Management
// ---------------------------------------------------------------------------

const STREAMLIT_APPS: Record<string, { file: string; port: number }> = {
  main: { file: 'src/frontend.py', port: 8500 },  // Confluencia 1.0 main app
  drug: { file: 'confluencia-2.0-drug/app.py', port: 8501 },
  epitope: { file: 'confluencia-2.0-epitope/epitope_frontend.py', port: 8502 },
  circrna: { file: 'confluencia_circrna/circrna_streamlit.py', port: 8503 },
  joint: { file: 'confluencia_joint/joint_streamlit.py', port: 8504 },
};

function resolveStreamlitApp(appName: string): string | null {
  const app = STREAMLIT_APPS[appName];
  if (!app) return null;

  const searchDirs = isDev
    ? [ROOT_DIR]
    : [process.resourcesPath, path.join(process.resourcesPath, '..')];

  for (const dir of searchDirs) {
    const p = path.join(dir, app.file);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

async function launchStreamlitApp(appName: string): Promise<{ name: string; pid: number; path: string; url: string } | null> {
  const appPath = resolveStreamlitApp(appName);
  if (!appPath) {
    mainWindow?.webContents.send('kernel:error', `App not found: ${appName}\n`);
    return null;
  }

  const appConfig = STREAMLIT_APPS[appName]!;
  const pythonExe = findPython();

  const proc = spawn(pythonExe, [
    '-m', 'streamlit', 'run', appPath,
    '--server.headless', 'true',
    '--server.port', String(appConfig.port),
    '--browser.gatherUsageStats', 'false',
  ], {
    cwd: path.dirname(appPath),
    windowsHide: true,
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  streamlitProcesses.set(proc.pid!, proc);

  const url = `http://localhost:${appConfig.port}`;

  proc.stderr?.on('data', (data: Buffer) => {
    const msg = data.toString();
    if (msg.includes('Network URL') || msg.includes('External URL') || msg.includes('You can now view')) {
      mainWindow?.webContents.send('kernel:output', `[Streamlit] ${msg.trim()}\n`);
    }
  });

  proc.on('exit', (code) => {
    streamlitProcesses.delete(proc.pid!);
    mainWindow?.webContents.send('kernel:output', `[Streamlit] ${appName} exited (code ${code})\n`);
  });

  setTimeout(() => {
    shell.openExternal(url);
  }, 3000);

  return { name: appName, pid: proc.pid!, path: appPath, url };
}

// ---------------------------------------------------------------------------
// IPC Handlers
// ---------------------------------------------------------------------------

ipcMain.handle('kernel:execute', (_, cmd: string) => {
  sendToPython('execute', { command: cmd });
});

ipcMain.handle('kernel:execute_python', (_, code: string) => {
  sendToPython('execute_python', { code });
});

ipcMain.handle('kernel:set_module', (_, module: string) => {
  sendToPython('set_module', { module });
});

ipcMain.handle('kernel:get_variables', () => {
  sendToPython('get_variables', {});
});

ipcMain.handle('kernel:save_image', (_, filePath?: string) => {
  sendToPython('save_image', { path: filePath });
});

ipcMain.handle('kernel:load', (_, filePath: string) => {
  sendToPython('load', { path: filePath });
});

ipcMain.handle('kernel:get_history', () => {
  sendToPython('get_history', {});
});

ipcMain.handle('kernel:run_script', (_, code: string) => {
  sendToPython('run_script', { code });
});

ipcMain.handle('kernel:check_syntax', (_, code: string) => {
  sendToPython('check_syntax', { code });
});

ipcMain.handle('kernel:execute_mixed', (_, code: string) => {
  sendToPython('execute_mixed', { code });
});

ipcMain.handle('kernel:get_symbols', (_, code: string) => {
  sendToPython('get_symbols', { code });
});

// File operations
ipcMain.handle('dialog:open_file', async (_, options: Electron.OpenDialogOptions) => {
  const { dialog } = await import('electron');
  return dialog.showOpenDialog(mainWindow!, options);
});

ipcMain.handle('dialog:save_file', async (_, options: Electron.SaveDialogOptions) => {
  const { dialog } = await import('electron');
  return dialog.showSaveDialog(mainWindow!, options);
});

ipcMain.handle('file:read', async (_, filePath: string) => {
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch {
    return null;
  }
});

ipcMain.handle('file:write', async (_, filePath: string, content: string) => {
  try {
    fs.writeFileSync(filePath, content, 'utf-8');
    return true;
  } catch {
    return false;
  }
});

ipcMain.handle('shell:run', async (_, cmd: string) => {
  const result = await new Promise<{ stdout: string; stderr: string; code: number }>((resolve) => {
    const proc = spawn('cmd', ['/c', cmd], { shell: true, windowsHide: true });
    let stdout = '', stderr = '';
    proc.stdout?.on('data', (d: Buffer) => stdout += d.toString());
    proc.stderr?.on('data', (d: Buffer) => stderr += d.toString());
    proc.on('exit', (code) => resolve({ stdout, stderr, code: code || 0 }));
  });
  return result;
});

ipcMain.handle('app:get_conf_dir', () => CONF_DIR);
ipcMain.handle('app:get_root_dir', () => ROOT_DIR);

ipcMain.handle('app:python_status', () => ({
  pythonReady: isPythonReady(),
  pipReady: fs.existsSync(path.join(PYTHON_DIR, 'Scripts', 'pip.exe')),
  depsInstalled: fs.existsSync(path.join(PYTHON_DIR, '.deps_installed')),
  pythonDir: PYTHON_DIR,
}));

ipcMain.handle('kernel:restart', () => restartPythonKernel());

ipcMain.handle('app:bootstrap_python', async () => {
  mainWindow?.webContents.send('kernel:output', '[Setup] Python bootstrap not implemented in this version - using system Python\n');
  return true;
});

// Streamlit app management
ipcMain.handle('streamlit:launch', async (_, appName: string) => {
  return await launchStreamlitApp(appName);
});

ipcMain.handle('streamlit:stop', async (_, pid: number) => {
  const proc = streamlitProcesses.get(pid);
  if (proc) {
    proc.kill();
    streamlitProcesses.delete(pid);
    return true;
  }
  return false;
});

ipcMain.handle('streamlit:apps', async () => {
  return Object.keys(STREAMLIT_APPS);
});

ipcMain.handle('shell:openExternal', async (_, url: string) => {
  await shell.openExternal(url);
  return true;
});

// ---------------------------------------------------------------------------
// Studio Enhancement IPC Handlers (Phase 0-5)
// ---------------------------------------------------------------------------

// Task Queue
ipcMain.handle('task:submit', async (_, label: string, fnName: string, args: unknown[]) => {
  sendToPython('task_submit', { label, fn_name: fnName, args });
});

ipcMain.handle('task:list', async () => {
  sendToPython('task_list', {});
});

ipcMain.handle('task:get', async (_, taskId: string) => {
  sendToPython('task_get', { task_id: taskId });
});

ipcMain.handle('task:cancel', async (_, taskId: string) => {
  sendToPython('task_cancel', { task_id: taskId });
});

// RAG Knowledge Base
ipcMain.handle('rag:query', async (_, query: string, k: number = 5) => {
  sendToPython('rag_query', { query, k });
});

ipcMain.handle('rag:index_docs', async (_, docsDir: string) => {
  sendToPython('rag_index_docs', { docs_dir: docsDir });
});

ipcMain.handle('rag:status', async () => {
  sendToPython('rag_status', {});
});

// Function/Tool Execution
ipcMain.handle('tool:list', async () => {
  sendToPython('tool_list', {});
});

ipcMain.handle('tool:execute', async (_, toolName: string, args: Record<string, unknown>) => {
  sendToPython('tool_execute', { tool_name: toolName, args });
});

// LLM Context (for inline completions)
ipcMain.handle('llm:completions', async (_, code: string, cursor: { line: number; col: number }) => {
  sendToPython('llm_completions', { code, cursor });
});

// Plotly Charts
ipcMain.handle('plotly:generate', async (_, spec: { data: unknown[]; layout: unknown; title?: string }) => {
  sendToPython('plotly_generate', { spec });
});

ipcMain.handle('plotly:list', async () => {
  sendToPython('plotly_list', {});
});

ipcMain.handle('plotly:clear', async () => {
  sendToPython('plotly_clear', {});
});

// 3D Molecule Viewer
ipcMain.handle('molecule:generate3d', async (_, smiles: string) => {
  sendToPython('molecule_generate3d', { smiles });
});

ipcMain.handle('molecule:smiles2svg', async (_, smiles: string, size: number = 300) => {
  sendToPython('molecule_smiles2svg', { smiles, size });
});

// Experiment Tracker
ipcMain.handle('experiment:start', async (_, name: string, module: string, params: Record<string, unknown>, tags?: string[]) => {
  sendToPython('experiment_start', { name, module, params, tags });
});

ipcMain.handle('experiment:log_metric', async (_, expId: string, key: string, value: unknown) => {
  sendToPython('experiment_log_metric', { exp_id: expId, key, value });
});

ipcMain.handle('experiment:log_artifact', async (_, expId: string, filePath: string) => {
  sendToPython('experiment_log_artifact', { exp_id: expId, file_path: filePath });
});

ipcMain.handle('experiment:finish', async (_, expId: string, status: string = 'completed') => {
  sendToPython('experiment_finish', { exp_id: expId, status });
});

ipcMain.handle('experiment:list', async (_, filter?: { module?: string; status?: string; tags?: string[] }) => {
  sendToPython('experiment_list', { filter });
});

ipcMain.handle('experiment:get', async (_, expId: string) => {
  sendToPython('experiment_get', { exp_id: expId });
});

ipcMain.handle('experiment:compare', async (_, expIds: string[]) => {
  sendToPython('experiment_compare', { exp_ids: expIds });
});

// Model Registry
ipcMain.handle('model:register', async (_, path: string, name: string, version: string, modelType: string, metrics?: Record<string, number>, params?: Record<string, unknown>) => {
  sendToPython('model_register', { path, name, version, model_type: modelType, metrics, params });
});

ipcMain.handle('model:list', async (_, nameFilter?: string) => {
  sendToPython('model_list', { name_filter: nameFilter });
});

ipcMain.handle('model:get', async (_, name: string, version?: string) => {
  sendToPython('model_get', { name, version });
});

ipcMain.handle('model:set_production', async (_, name: string, version: string) => {
  sendToPython('model_set_production', { name, version });
});

// Hyperopt Visualization
ipcMain.handle('hyperopt:list_studies', async () => {
  sendToPython('hyperopt_list_studies', {});
});

ipcMain.handle('hyperopt:get_trials', async (_, studyName: string) => {
  sendToPython('hyperopt_get_trials', { study_name: studyName });
});

ipcMain.handle('hyperopt:get_importance', async (_, studyName: string) => {
  sendToPython('hyperopt_get_importance', { study_name: studyName });
});

// Report Generation
ipcMain.handle('report:generate', async (_, expId: string, sections?: string[]) => {
  sendToPython('report_generate', { exp_id: expId, sections });
});

ipcMain.handle('report:export_latex', async (_, markdown: string, outputPath: string) => {
  sendToPython('report_export_latex', { markdown, output_path: outputPath });
});

ipcMain.handle('report:export_docx', async (_, markdown: string, outputPath: string) => {
  sendToPython('report_export_docx', { markdown, output_path: outputPath });
});

// Bundle Export
ipcMain.handle('bundle:create', async (_, expId: string, outputPath: string, options?: { include_data?: boolean; include_models?: boolean }) => {
  sendToPython('bundle_create', { exp_id: expId, output_path: outputPath, options });
});

// Residual Analysis
ipcMain.handle('residual:generate', async (_, yTrue: number[], yPred: number[], outputDir?: string, format?: string) => {
  sendToPython('residual_generate', { y_true: yTrue, y_pred: yPred, output_dir: outputDir, format });
});

ipcMain.handle('shap:generate', async (_, modelPath: string, XPath: string, outputPath?: string) => {
  sendToPython('shap_generate', { model_path: modelPath, X_path: XPath, output_path: outputPath });
});

// Export Utilities
ipcMain.handle('export:dataframe', async (_, data: unknown[], format: string, outputPath: string) => {
  sendToPython('export_dataframe', { data, format, output_path: outputPath });
});

// Plugin System
ipcMain.handle('plugin:register', async (_, plugin: { id: string; name: string; version: string; panels?: unknown[]; commands?: unknown[] }) => {
  sendToPython('plugin_register', { plugin });
});

ipcMain.handle('plugin:unregister', async (_, pluginId: string) => {
  sendToPython('plugin_unregister', { plugin_id: pluginId });
});

ipcMain.handle('plugin:list', async () => {
  sendToPython('plugin_list', {});
});

// ---------------------------------------------------------------------------
// Toxicity Analysis
// ---------------------------------------------------------------------------

ipcMain.handle('toxicity:admet', async (_, smiles: string) => {
  sendToPython('toxicity_admet', { smiles });
});

ipcMain.handle('toxicity:toxicophore', async (_, smiles: string) => {
  sendToPython('toxicity_toxicophore', { smiles });
});

ipcMain.handle('toxicity:dose', async (_, smiles: string, ed50Mgkg?: number, expectedDoseMgkg?: number) => {
  sendToPython('toxicity_dose', { smiles, ed50_mgkg: ed50Mgkg, expected_dose_mgkg: expectedDoseMgkg });
});

ipcMain.handle('toxicity:full', async (_, smiles: string, ed50Mgkg?: number, expectedDoseMgkg?: number) => {
  sendToPython('toxicity_full', { smiles, ed50_mgkg: ed50Mgkg, expected_dose_mgkg: expectedDoseMgkg });
});

// ---------------------------------------------------------------------------
// App Lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(async () => {
  createWindow();
  startPythonKernel();
});

app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.on('window-all-closed', () => {
  pythonProcess?.kill();
  streamlitProcesses.forEach(proc => proc.kill());
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
