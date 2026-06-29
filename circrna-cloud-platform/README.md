# CircRNA Cloud Platform

## AutoDL风格算力租赁平台

面向科研团队的circRNA 3D结构预测算力租赁服务，运行于DGX Spark (Blackwell GB10 GPU)。

---

## 功能特性

- **预置镜像模板**: ViennaRNA + OpenMM + RoseTTAFold2NA + PyTorch
- **按时计费**: 按小时计费，用完即止
- **Web控制台**: Streamlit界面，一键创建/启动/停止实例
- **SSH访问**: 每实例独立密码，数据隔离
- **端口映射**: Jupyter/Streamlit/TensorBoard直接访问

---

## 镜像模板

| 模板 | 价格 | 描述 |
|------|------|------|
| circrna-base | $0.50/h | ViennaRNA + OpenMD基础版 |
| circrna-full | $1.20/h | RoseTTAFold2NA完整管线 |
| bio-ml | $0.80/h | PyTorch + rdkit药物发现版 |

---

## 目录结构

```
circrna-cloud-platform/
├── app/
│   ├── Cloud.py              # AutoDL风格Web界面
│   └── Home.py               # 旧版（保留）
│
├── backend/
│   ├── instance_manager.py   # 实例管理（镜像/启动/停止）
│   ├── user_billing.py       # 用户+计费
│   ├── job_scheduler.py      # 任务队列（单人模式）
│   ├── container_manager.py  # Docker容器编排
│   └── user_manager.py       # 用户认证
│
├── config/
│   └── settings.py           # 配置参数
│
├── docker/
│   └── Dockerfile.base       # 基础镜像（生信工具预置）
│
└── data/                     # 运行时数据（自动创建）
    ├── users/                 # 用户数据隔离
    ├── instances/             # 实例记录
    └── mirrors/               # 镜像缓存
```

---

## 快速部署

### 1. 构建基础镜像

```bash
cd docker
docker build -t circrna-cloud-base:latest -f Dockerfile.base .
```

### 2. 启动Web服务

```bash
cd app
streamlit run Cloud.py --server.port 8501 --server.address 0.0.0.0
```

### 3. 访问平台

浏览器打开: http://your-server-ip:8501

---

## 使用流程

1. **注册账号** → 充值余额
2. **选择镜像** → 创建实例
3. **启动实例** → 获取SSH密码和端口
4. **连接使用** → SSH/Web/Jupyter
5. **停止实例** → 自动计费扣款

---

## 计费说明

- 按小时计费，精确到秒
- 实例停止后自动扣款
- 余额不足时实例自动停止
- 企业用户可开具发票

---

## 技术栈

- **前端**: Streamlit
- **后端**: Python + Docker SDK
- **GPU**: NVIDIA Blackwell GB10
- **工具链**: ViennaRNA, OpenMM, RoseTTAFold2NA, PyTorch