

import os
import cv2
import logging
from common_utils.annotate.core import Annotator
from detect.tasks import (
    detect,
    train,
)

VALID_FORMAT = ['jpeg', 'png', 'jpg', 'webp']

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
    

def annotate(params):
    try:
        
        assert 'source' in params, f"Missing arg in annotate: source"
        images = find_images(source=params['source'])
        for image in images:
            cv_image = load_image(image)
            if cv_image is None:
                continue
            
            detections = detect.predict.predict(
                image=cv_image
            )
            
            detections.to_txt(
                txt_file=f'/home/appuser/src/data/labels/{os.path.basename(image).split(".")[0]}.txt',
                task='detect',
            )
            annotator = Annotator(
                im=cv_image.copy(),
                line_width=3
            )
            
            for i, xyxy in enumerate(detections.xyxy):
                annotator.box_label(
                    box=xyxy,
                )
                
            
            os.makedirs(
                "/home/appuser/src/data/predictions",
                exist_ok=True
            )
            
            cv2.imwrite(
                f"/home/appuser/src/data/predictions/{os.path.basename(image)}",
                annotator.im.data
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
):
    
    try:
        assert task in task_mapper, f"unsupported task: {task} ! Supported Task are: {list(task_mapper.keys())}"
        func = task_mapper[task]
        func(params) 
    except Exception as err:
        logging.error(f"Error: {err}")
        
if __name__ == "__main__":
    params = {
        "source": "/home/appuser/src/data/images",
    }
    
    main(
        task='annotate',
        params=params,
    )