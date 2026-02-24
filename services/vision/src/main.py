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
from src.vision_model import VisionPipeline

class VisionServiceServicer(trading_pb2_grpc.VisionServiceServicer):
    def __init__(self):
        self.pipeline = VisionPipeline()
        
    async def ExtractFeatures(self, request, context):
        try:
            # Deserialize tensor from bytes
            buffer = io.BytesIO(request.image_data)
            tensor = torch.load(buffer, map_location="cpu")
            
            features = self.pipeline.extract_features(tensor)
            # Flatten to 1D list
            features_list = features.flatten().tolist()
            
            return trading_pb2.FeatureVector(
                features=features_list
            )
        except Exception as e:
            logger.error(f"Error in ExtractFeatures: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

async def serve():
    server = grpc.aio.server()
    trading_pb2_grpc.add_VisionServiceServicer_to_server(VisionServiceServicer(), server)
    server.add_insecure_port('[::]:50053')
    await server.start()
    logger.info("Vision Service gRPC server started on port 50053")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
