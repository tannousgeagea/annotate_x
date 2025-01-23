import logging
from common_utils.model.base import BaseModels
from common_utils.detection.core import Detections


params = {
    "conf": 0.2,
    "mode": "detect",
    "weights": "/home/appuser/src/data/models/amk.front.impurity.v1.pt",
    "mlflow": {
        "active": False
    }
}

model = BaseModels(
    weights=params.get('weights'), 
    task='segments', 
    mlflow=params.get('mlflow', {}).get('active', False),
)

roi = [
    (0.0966796875, 0),
    (0.0966796875, 0.8971354166666666),
    (0.9326171875, 0.8971354166666666),
    (0.9326171875, 0),
    (0.0966796875, 0)
]

def predict(image):
    detections = Detections.from_dict({})
    try:
        assert not image is None, f'Image is None'
        assert not model is None, f'Model is None'
        
        h, w, _ = image.shape
        roi_pixels = [(int(x * w), int(y * h)) for x, y in roi]

        # Find the bounding box around the ROI
        x_coords, y_coords = zip(*roi_pixels)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        cropped_image = image[y_min:y_max, x_min:x_max]
        results = model.classify_one(image=cropped_image, conf=float(params.get('conf')), mode=params.get('mode'))
        detections = Detections.from_dict(results=results)
        
        detections = detections.adjust_to_roi(
            offset=(x_min, y_min),
            crop_size=cropped_image.shape[:2],
            original_size=image.shape[:2],
        )
        
    except Exception as err:
        logging.error(f'Unexpected Error in Segmentation: {err}')
    
    return detections