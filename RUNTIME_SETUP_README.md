# Universal PyInstaller Runtime Setup

通用的 VC++ 运行库安装器，适用于任何 PyInstaller 打包的程序。

## 文件清单

| 文件 | 用途 |
|------|------|
| `run_setup.py` | Python 源码（跨平台，需要 Python） |
| `run.bat` | Windows 批处理版（无需 Python） |
| `build_setup_generic.ps1` | 打包脚本（生成独立 exe） |

## 使用方法

### 方法 1：批处理版（最简单，无需 Python）

把 `run.bat` 放到目标程序目录，双击运行即可。

```
myapp/
├── myapp.exe       ← PyInstaller 打包的程序
└── run.bat         ← 复制这个文件
```

运行后会自动：
1. 检测 VC++ 运行库
2. 下载安装缺失项
3. 启动程序

### 方法 2：打包为独立 exe

```powershell
# 在当前目录运行
.\build_setup_generic.ps1

# 输出：RuntimeSetup_dist\RuntimeSetup.exe
```

把这个 exe 复制到任意电脑上运行即可。

### 方法 3：直接运行 Python（需要 Python 3.10+）

```bash
python run_setup.py              # GUI 模式
python run_setup.py --cli        # 命令行模式
python run_setup.py --check      # 仅检查，不安装
python run_setup.py --silent     # 静默安装
```

## 参数说明

```
run.bat [app.exe] [--no-install] [--silent]

  不带参数   自动查找当前目录的 .exe 文件并启动
  app.exe    指定要启动的程序
  --no-install  跳过运行库检查，直接启动
  --silent      静默模式，不显示安装窗口
```

```
run_setup.py [选项]

  --cli         使用命令行界面
  --check       仅检查，不安装
  --silent      静默安装
  --exe xxx.exe 指定目标程序
  --version     显示版本
```

## 安装的运行库

| 运行库 | 覆盖的 DLL |
|--------|-----------|
| VC++ 2015-2022 (x64/x86) | vcruntime140.dll, msvcp140.dll 等 |
| VC++ 2010 SP1 (x64/x86) | MSVCR100.dll 等 |

## 手动下载链接

如果自动安装失败：
- https://aka.ms/vs/17/release/vc_redist.x64.exe
- https://aka.ms/vs/17/release/vc_redist.x86.exe
- https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x64.exe

## 注意事项

1. 需要管理员权限才能安装运行库
2. 如果杀毒软件拦截，请放行
3. 安装后可能需要重启电脑
