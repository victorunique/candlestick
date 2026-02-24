import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "generated"))

import asyncio
import grpc
import io
import torch
from loguru import logger

import src.generated.trading_pb2 as trading_pb2
import src.generated.trading_pb2_grpc as trading_pb2_grpc
from src.chart_generator import ChartGenerator

class ChartServiceServicer(trading_pb2_grpc.ChartServiceServicer):
    def __init__(self):
        # Default generator creates 1000x64 tensors, but adapts to input length
        self.generator = ChartGenerator(width=1000, height=64)
        
    async def GenerateChart(self, request, context):
        try:
            # Reconstruct lists from request
            o = list(request.open)
            h = list(request.high)
            l = list(request.low)
            c = list(request.close)
            v = list(request.volume)
            
            # Use dynamic width based on input to support testing and smaller windows
            self.generator.width = len(o)
            
            tensor = self.generator.generate_chart(o, h, l, c, v)
            
            # Serialize tensor to bytes to send over gRPC
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            image_bytes = buffer.getvalue()
            
            return trading_pb2.ChartImage(
                image_data=image_bytes,
                width=self.generator.width,
                height=self.generator.height
            )
        except Exception as e:
            logger.error(f"Error in GenerateChart: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

async def serve():
    server = grpc.aio.server()
    trading_pb2_grpc.add_ChartServiceServicer_to_server(ChartServiceServicer(), server)
    server.add_insecure_port('[::]:50052')
    await server.start()
    logger.info("Chart Service gRPC server started on port 50052")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
