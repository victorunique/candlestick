FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

WORKDIR /app

RUN uv pip install --system fastapi uvicorn httpx grpcio grpcio-tools torch numpy loguru stable-baselines3 

COPY proto /proto
COPY services/inference /app
RUN python -m grpc_tools.protoc -I/proto --python_out=src/generated --grpc_python_out=src/generated /proto/trading.proto

EXPOSE 8000

CMD ["python", "src/main.py"]
