

import os
import cv2
import logging
import numpy as np
from common_utils.annotate.core import Annotator
from common_utils.filters.core import FilterEngine
from common_utils.model.base import BaseModels
from common_utils.detection.core import Detections
from common_utils.detection.convertor import polygon_to_mask, rescale_polygon
from segments.tasks import (
    segment,
    train,
)

VALID_FORMAT = ['jpeg', 'png', 'jpg', 'webp']

filter_config = {
    # "crane": {
    #     "model_path": "iserlohn.amk.crane.want:crane.detect",
    #     "conf": 0.5,
    #     "mlflow": True
    #     },
}


def load_image(path):
    if not os.path.exists(path):
        print('Image %s does not exist')
        return None

    return cv2.imread(path)

def find_images(source):
    
    try:
        if os.path.isdir(source):
            images = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(tuple(VALID_FORMAT))])
        
        elif os.path.isfile(source):
            images = [source] if source.endswith(tuple(VALID_FORMAT)) else []
        
        else:
            images = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(tuple(VALID_FORMAT))])
            
        return images
    except Exception as err:
        raise ValueError(f"Failed to find images: {err}")    
    

def annotate(params, model, output_dir = "/home/appuser/src/data"):
    try:
        filter_engine = FilterEngine()
        if filter_config:
            for obj_type, model in filter_config.items():
                filter_engine.add_model(
                    object_type=obj_type,
                    detection_model=model["model_path"],
                    mlflow=model["mlflow"]
                )
            
        assert 'source' in params, f"Missing arg in annotate: source"
        images = find_images(source=params['source'])
        for image in images:
            cv_image = load_image(image)
            if cv_image is None:
                continue
            
            detections = segment.predict.predict(
                image=cv_image, model=model, conf=float(params.get('conf')), mode=params.get('mode')
            )
            
            if filter_config:
                filtered_results = filter_engine.filter_objects(
                    image=cv_image,
                    segmentation_results=detections,
                    filter_types=list(filter_config.keys()),
                )
                detections = detections[filtered_results]
            
            
            detections.to_txt(
                txt_file=f'{output_dir}/labels/{os.path.basename(image).split(".")[0]}.txt',
                task='segment',
            )
            annotator = Annotator(
                im=cv_image.copy(),
                line_width=3
            )
            
            
            os.makedirs(
                f"{output_dir}/predictions",
                exist_ok=True
            )
            for i, xyxy in enumerate(detections.xyxy):
                annotator.box_label(
                    box=xyxy,
                )
                
                
                kernel = np.ones((5, 5), np.uint8)
                
                
                polygon = detections.xy[i].astype(np.int32)
                epsilon = 0.01 * cv2.arcLength(polygon, True)
                polygon = cv2.approxPolyDP(polygon, epsilon, True)
                w0, h0 = 416, 416
                
                polygon = rescale_polygon(polygon, wh0=(cv_image.shape[1], cv_image.shape[0]), wh=(w0, h0))
                mask = polygon_to_mask(polygon, resolution_wh=(w0, h0))
                mask = cv2.dilate(mask, kernel, iterations=5) 
                
                resized = cv2.resize(cv_image.copy(), (w0, h0))
                extracted = cv2.bitwise_and(resized, resized, mask=mask)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                object_cropped = extracted[y:y+h, x:x+w]
                mask_cropped = mask[y:y+h, x:x+w]
                
                background = np.ones((h0, w0, 3), dtype=np.uint8) * 255
                
                center_x = (w0 - w) // 2
                center_y = (h0 - h) // 2
                
                background[center_y:center_y+h, center_x:center_x+w] = cv2.bitwise_and(
                    background[center_y:center_y+h, center_x:center_x+w],
                    background[center_y:center_y+h, center_x:center_x+w],
                    mask=cv2.bitwise_not(mask_cropped)
                )
                
                background[center_y:center_y+h, center_x:center_x+w] += object_cropped
            
                cv2.imwrite(
                    f"{output_dir}/predictions/{os.path.basename(image)}_{str(i).zfill(3)}.jpg",
                    background
                )
                
                
    except Exception as err:
        raise ValueError(f"Failed to annotate: {err}")

def train_model(params):
    try:
        assert 'data' in params, f"Missing argument in tain: data"
        results = train.core.execute(params)
        
    except Exception as err:
        raise ValueError(f"Failed to train: {err}")
    
task_mapper = {
    "train": train_model,
    "annotate": annotate,
}    

def main(
    task,
    params,
    model,
):
    try:
        assert task in task_mapper, f"unsupported task: {task} ! Supported Task are: {list(task_mapper.keys())}"
        func = task_mapper[task]
        func(params, model)
    except Exception as err:
        logging.error(f"Error: {err}")
        
if __name__ == "__main__":
    
    # params = {
    #     "conf": 0.2,
    #     "mode": "detect",
    #     "weights": "/home/appuser/src/data/models/amk.bunker.segmentation.v1.pt",
    #     "mlflow": {
    #         "active": False
    #     },
    #     "source": "/home/appuser/src/data/images/025.jpg",
    # }
    
    # model = BaseModels(
    #     weights=params.get('weights'), 
    #     task='segments', 
    #     mlflow=params.get('mlflow', {}).get('active', False),
    # )
    
    # main(
    #     task='annotate',
    #     params=params,
    #     model=model
    # )
    
    from inference_sdk import InferenceHTTPClient

    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="qm0YWCLBTWPm82O4QfuH"
    )

    result = CLIENT.infer("/home/appuser/src/data/predictions/025.jpg_039.jpg", model_id="garbage-classification-3/2")
    
    print(result)