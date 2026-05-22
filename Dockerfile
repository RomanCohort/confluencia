FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Chinese apt mirror for AutoDL
RUN sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y \
    build-essential \
    librdkit-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Chinese pip mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

WORKDIR /app

# Copy dependency files first (for layer caching)
COPY requirements-shared-full.txt .
RUN pip install --no-cache-dir -r requirements-shared-full.txt

# Copy source code
COPY confluencia_shared/ ./confluencia_shared/
COPY confluencia-2.0-drug/ ./confluencia-2.0-drug/
COPY confluencia-2.0-epitope/ ./confluencia-2.0-epitope/
COPY confluencia_joint/ ./confluencia_joint/
COPY confluencia_circrna/ ./confluencia_circrna/
COPY confluencia_cli/ ./confluencia_cli/
COPY confluencia_studio/ ./confluencia_studio/
COPY benchmarks/ ./benchmarks/
COPY tests/ ./tests/
COPY data/ ./data/
COPY output/ ./output/
COPY pyproject.toml .
COPY LICENSE .
COPY CITATION.cff .
COPY README.md .
COPY INTEGRATION_SUMMARY.md .
COPY Dockerfile .
COPY docker-compose.yml .

# Install package
RUN pip install --no-cache-dir -e .

# Expose Streamlit ports
EXPOSE 8501 8502

# Default: run joint frontend
CMD ["streamlit", "run", "confluencia_joint/joint_streamlit.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
