import logging
from common_utils.model.base import BaseModels
from common_utils.detection.core import Detections


params = {
    "conf": 0.1,
    "mode": "detect",
    "weights": "iserlohn.amk.want:waste.segments",
    "mlflow": {
        "active": True
    }
}

model = BaseModels(
    weights=params.get('weights'), 
    task='segments', 
    mlflow=params.get('mlflow', {}).get('active', False),
)

def predict(image):
    detections = Detections.from_dict({})
    try:
        assert not image is None, f'Image is None'
        assert not model is None, f'Model is None'
        results = model.classify_one(image=image, conf=float(params.get('conf')), mode=params.get('mode'))
        detections = Detections.from_dict(results=results)
        
    except Exception as err:
        logging.error(f'Unexpected Error in Segmentation: {err}')
    
    return detections