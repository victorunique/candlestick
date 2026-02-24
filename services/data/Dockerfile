FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml .
# We can't install dependecies cleanly without lockfile unless we just run uv init
# So we run uv sync or uv pip install. Let's create a requirements.txt using uv 
RUN uv pip install --system fastapi uvicorn grpcio grpcio-tools duckdb pandas httpx loguru

COPY proto /proto
COPY services/data /app
RUN python -m grpc_tools.protoc -I/proto --python_out=src/generated --grpc_python_out=src/generated /proto/trading.proto

ENV DATABASE_PATH=/app/data/market_data.duckdb

# Expose both
EXPOSE 8001 50051

CMD ["python", "src/main.py"]
