FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

WORKDIR /app

RUN uv pip install --system pandas numpy fastapi uvicorn grpcio grpcio-tools loguru

COPY proto /proto
COPY services/backtest /app
RUN python -m grpc_tools.protoc -I/proto --python_out=src/generated --grpc_python_out=src/generated /proto/trading.proto

EXPOSE 8003

CMD ["python", "src/main.py"]
