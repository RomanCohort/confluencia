import { contextBridge, ipcRenderer } from 'electron';

export interface KernelEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface ShellResult {
  stdout: string;
  stderr: string;
  code: number;
}

export interface PythonStatus {
  pythonReady: boolean;
  pipReady: boolean;
  depsInstalled: boolean;
  pythonDir: string;
}

export interface StreamlitApp {
  name: string;
  pid: number;
  path: string;
  url: string;
}

contextBridge.exposeInMainWorld('api', {
  // Kernel commands
  execute: (cmd: string) => ipcRenderer.invoke('kernel:execute', cmd),
  executePython: (code: string) => ipcRenderer.invoke('kernel:execute_python', code),
  setModule: (module: string) => ipcRenderer.invoke('kernel:set_module', module),
  getVariables: () => ipcRenderer.invoke('kernel:get_variables'),
  saveImage: (path?: string) => ipcRenderer.invoke('kernel:save_image', path),
  load: (path: string) => ipcRenderer.invoke('kernel:load', path),
  getHistory: () => ipcRenderer.invoke('kernel:get_history'),
  runScript: (code: string) => ipcRenderer.invoke('kernel:run_script', code),
  restartKernel: () => ipcRenderer.invoke('kernel:restart'),

  // Syntax checking / diagnostics
  checkSyntax: (code: string) => ipcRenderer.invoke('kernel:check_syntax', code),

  // Symbol extraction for outline view
  getSymbols: (code: string) => ipcRenderer.invoke('kernel:get_symbols', code),

  // Mixed execution (auto-detect Python vs confluencia)
  executeMixed: (code: string) => ipcRenderer.invoke('kernel:execute_mixed', code),

  // Kernel events
  onKernelEvent: (callback: (event: KernelEvent) => void) => {
    const handler = (_: unknown, data: KernelEvent) => callback(data);
    ipcRenderer.on('kernel:event', handler);
    return () => ipcRenderer.removeListener('kernel:event', handler);
  },
  onKernelOutput: (callback: (text: string) => void) => {
    const handler = (_: unknown, text: string) => callback(text);
    ipcRenderer.on('kernel:output', handler);
    return () => ipcRenderer.removeListener('kernel:output', handler);
  },
  onKernelError: (callback: (text: string) => void) => {
    const handler = (_: unknown, text: string) => callback(text);
    ipcRenderer.on('kernel:error', handler);
    return () => ipcRenderer.removeListener('kernel:error', handler);
  },

  // File dialogs
  openFile: (options: Record<string, unknown>) => ipcRenderer.invoke('dialog:open_file', options),
  saveFile: (options: Record<string, unknown>) => ipcRenderer.invoke('dialog:save_file', options),
  readFile: (path: string) => ipcRenderer.invoke('file:read', path),
  writeFile: (path: string, content: string) => ipcRenderer.invoke('file:write', path, content),

  // Shell
  runShell: (cmd: string): Promise<ShellResult> => ipcRenderer.invoke('shell:run', cmd),
  openExternal: (url: string) => ipcRenderer.invoke('shell:openExternal', url),

  // App info
  getConfDir: () => ipcRenderer.invoke('app:get_conf_dir'),
  getRootDir: () => ipcRenderer.invoke('app:get_root_dir'),

  // Python runtime management
  getPythonStatus: (): Promise<PythonStatus> => ipcRenderer.invoke('app:python_status'),
  bootstrapPython: (): Promise<boolean> => ipcRenderer.invoke('app:bootstrap_python'),

  // Streamlit app management
  launchStreamlit: (appName: string): Promise<StreamlitApp> => ipcRenderer.invoke('streamlit:launch', appName),
  stopStreamlit: (pid: number) => ipcRenderer.invoke('streamlit:stop', pid),
  getAvailableApps: (): Promise<string[]> => ipcRenderer.invoke('streamlit:apps'),

  // Menu events
  onMenu: (callback: (action: string) => void) => {
    ipcRenderer.on('menu:new', () => callback('new'));
    ipcRenderer.on('menu:open', () => callback('open'));
    ipcRenderer.on('menu:save', () => callback('save'));
    ipcRenderer.on('menu:find', () => callback('find'));
    ipcRenderer.on('menu:about', () => callback('about'));
    return () => {
      ipcRenderer.removeAllListeners('menu:new');
      ipcRenderer.removeAllListeners('menu:open');
      ipcRenderer.removeAllListeners('menu:save');
      ipcRenderer.removeAllListeners('menu:find');
      ipcRenderer.removeAllListeners('menu:about');
    };
  },

  // ===================== PHASE 0: Task Queue =====================
  taskSubmit: (label: string, fnName: string, args: unknown[]) => ipcRenderer.invoke('task:submit', label, fnName, args),
  taskList: () => ipcRenderer.invoke('task:list'),
  taskGet: (taskId: string) => ipcRenderer.invoke('task:get', taskId),
  taskCancel: (taskId: string) => ipcRenderer.invoke('task:cancel', taskId),

  // ===================== PHASE 1: LLM Enhancement =====================
  ragQuery: (query: string, k?: number) => ipcRenderer.invoke('rag:query', query, k),
  ragIndexDocs: (docsDir: string) => ipcRenderer.invoke('rag:index_docs', docsDir),
  ragStatus: () => ipcRenderer.invoke('rag:status'),
  toolList: () => ipcRenderer.invoke('tool:list'),
  toolExecute: (toolName: string, args: Record<string, unknown>) => ipcRenderer.invoke('tool:execute', toolName, args),
  llmCompletions: (code: string, cursor: { line: number; col: number }) => ipcRenderer.invoke('llm:completions', code, cursor),

  // ===================== PHASE 2: Visualization =====================
  plotlyGenerate: (spec: { data: unknown[]; layout: unknown; title?: string }) => ipcRenderer.invoke('plotly:generate', spec),
  plotlyList: () => ipcRenderer.invoke('plotly:list'),
  plotlyClear: () => ipcRenderer.invoke('plotly:clear'),
  moleculeGenerate3D: (smiles: string) => ipcRenderer.invoke('molecule:generate3d', smiles),
  moleculeSmiles2Svg: (smiles: string, size?: number) => ipcRenderer.invoke('molecule:smiles2svg', smiles, size),
  residualGenerate: (yTrue: number[], yPred: number[], outputDir?: string, format?: string) => ipcRenderer.invoke('residual:generate', yTrue, yPred, outputDir, format),
  shapGenerate: (modelPath: string, XPath: string, outputPath?: string) => ipcRenderer.invoke('shap:generate', modelPath, XPath, outputPath),

  // ===================== PHASE 3: Experiment Management =====================
  experimentStart: (name: string, module: string, params: Record<string, unknown>, tags?: string[]) => ipcRenderer.invoke('experiment:start', name, module, params, tags),
  experimentLogMetric: (expId: string, key: string, value: unknown) => ipcRenderer.invoke('experiment:log_metric', expId, key, value),
  experimentLogArtifact: (expId: string, filePath: string) => ipcRenderer.invoke('experiment:log_artifact', expId, filePath),
  experimentFinish: (expId: string, status?: string) => ipcRenderer.invoke('experiment:finish', expId, status),
  experimentList: (filter?: { module?: string; status?: string; tags?: string[] }) => ipcRenderer.invoke('experiment:list', filter),
  experimentGet: (expId: string) => ipcRenderer.invoke('experiment:get', expId),
  experimentCompare: (expIds: string[]) => ipcRenderer.invoke('experiment:compare', expIds),
  modelRegister: (path: string, name: string, version: string, modelType: string, metrics?: Record<string, number>, params?: Record<string, unknown>) => ipcRenderer.invoke('model:register', path, name, version, modelType, metrics, params),
  modelList: (nameFilter?: string) => ipcRenderer.invoke('model:list', nameFilter),
  modelGet: (name: string, version?: string) => ipcRenderer.invoke('model:get', name, version),
  modelSetProduction: (name: string, version: string) => ipcRenderer.invoke('model:set_production', name, version),
  hyperoptListStudies: () => ipcRenderer.invoke('hyperopt:list_studies'),
  hyperoptGetTrials: (studyName: string) => ipcRenderer.invoke('hyperopt:get_trials', studyName),
  hyperoptGetImportance: (studyName: string) => ipcRenderer.invoke('hyperopt:get_importance', studyName),

  // ===================== PHASE 4: Export & Report =====================
  reportGenerate: (expId: string, sections?: string[]) => ipcRenderer.invoke('report:generate', expId, sections),
  reportExportLatex: (markdown: string, outputPath: string) => ipcRenderer.invoke('report:export_latex', markdown, outputPath),
  reportExportDocx: (markdown: string, outputPath: string) => ipcRenderer.invoke('report:export_docx', markdown, outputPath),
  bundleCreate: (expId: string, outputPath: string, options?: { include_data?: boolean; include_models?: boolean }) => ipcRenderer.invoke('bundle:create', expId, outputPath, options),
  exportDataframe: (data: unknown[], format: string, outputPath: string) => ipcRenderer.invoke('export:dataframe', data, format, outputPath),

  // ===================== PHASE 5: Plugin System =====================
  pluginRegister: (plugin: { id: string; name: string; version: string; panels?: unknown[]; commands?: unknown[] }) => ipcRenderer.invoke('plugin:register', plugin),
  pluginUnregister: (pluginId: string) => ipcRenderer.invoke('plugin:unregister', pluginId),
  pluginList: () => ipcRenderer.invoke('plugin:list'),

  // ===================== TOXICITY ANALYSIS =====================
  toxicityAdmet: (smiles: string) => ipcRenderer.invoke('toxicity:admet', smiles),
  toxicityToxicophore: (smiles: string) => ipcRenderer.invoke('toxicity:toxicophore', smiles),
  toxicityDose: (smiles: string, ed50Mgkg?: number, expectedDoseMgkg?: number) => ipcRenderer.invoke('toxicity:dose', smiles, ed50Mgkg, expectedDoseMgkg),
  toxicityFull: (smiles: string, ed50Mgkg?: number, expectedDoseMgkg?: number) => ipcRenderer.invoke('toxicity:full', smiles, ed50Mgkg, expectedDoseMgkg),
});
