

import os
import cv2
import logging
from common_utils.annotate.core import Annotator
from segments.tasks import (
    segment
)

VALID_FORMAT = ['jpeg', 'png', 'jpg', 'webp']

def load_image(path):
    if not os.path.exists(path):
        print('Image %s does not exist')
        return None

    return cv2.imread(path)

def main(
    source,
):
    
    try:
        images = []
        print(os.path.isdir(source))
        if os.path.isdir(source):
            images = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(tuple(VALID_FORMAT))])
        
        elif os.path.isfile(source):
            images = [source] if source.endswith(tuple(VALID_FORMAT)) else []
        
        else:
            images = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(tuple(VALID_FORMAT))])
        
        for image in images:
            cv_image = load_image(image)
            if cv_image is None:
                continue
            
            detections = segment.predict.predict(
                image=cv_image
            )
            
            detections.to_txt(
                txt_file=f'/home/appuser/src/data/labels/{os.path.basename(image).split(".")[0]}.txt',
                task='segment',
            )
            annotator = Annotator(
                im=cv_image.copy(),
                line_width=3
            )
            
            for xyxy in detections.xyxy:
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
        logging.error(f"Error: {err}")
        
        
if __name__ == "__main__":
    main(
        source='/home/appuser/src/data/images'
    )