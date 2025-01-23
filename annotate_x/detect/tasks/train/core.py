import logging
from common_utils.model.base import BaseModels
from common_utils.detection.core import Detections

params = {
    "weights": "/home/appuser/src/data/models/base.segment.pt",
    "mlflow": {
        "active": False
    }
}

model = BaseModels(
    weights=params.get('weights'), 
    task='segments', 
    mlflow=params.get('mlflow', {}).get('active', False),
)

def execute(params):
    try:
        results = model.train(
            data=params['data'],
            imgsz=params.get('imgsz'),
            lr0=params.get('lr0'),
            lrf=params.get('lrf'),
            epochs=params.get('epochs'),
            batch=params.get('batch'),
            augment=params.get('augment', False),
            name=params.get('name', 'train'),
        )
        
        return results
    except Exception as err:
        raise ValueError(f"Error in Training: {err}")