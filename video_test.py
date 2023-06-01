# Changing the the directory where all modules are saved.
import os
os.chdir(r'C:\Users\vladi\OneDrive\Desktop\Sigmoid\CV\Object detection\object detection\object detection\utils')

# Importing all needed libraries.
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from PIL.ExifTags import TAGS, GPSTAGS


def get_model(num_classes):
    '''
        This function is used to create a pretrained FastRCNN Model.
    :num_classes: int
        The number of classes. Should be setted n+1 wher n is the number of object to detect.
    '''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    #get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


import numpy as np
import cv2
from PIL import ImageDraw

loaded_model = get_model(num_classes=2)
loaded_model.load_state_dict(torch.load(r'C:\Users\vladi\OneDrive\Desktop\Sigmoid\CV\Logo bluring\logo_detection_model\fix1_ports_model.pth'))

cap = cv2.VideoCapture(r'C:\Users\vladi\OneDrive\Desktop\Sigmoid\CV\Logo bluring\video_test\Times Square - Manhattan, New York [HQ].mp4')
print(cap)
while cap.isOpened():
    ret, frame = cap.read()
    
    img = Image.fromarray(frame).convert('RGB')

    frame = T.Compose([T.ToTensor()])(img,'0')
    
    img, _ = frame
    
    loaded_model.eval()
    # Making the prediction.
    with torch.no_grad():
        prediction = loaded_model([img])

    # Getting an drawing the image.
    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)

    # Drawing the real box around the object.
#     for elem in range(len(label_boxes)):
#         draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
#                        (label_boxes[elem][2], label_boxes[elem][3])],
#                       outline='green', width=3)
    # Drawing the predicted box around the object.
    for element in range(len(prediction[0]['boxes'])):
        boxes = prediction[0]['boxes'][element].cpu().numpy()
        score = np.round(prediction[0]['scores'][element].cpu().numpy(), decimals=4)

        if score > 0.3:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                         outline='red', width=3)
            draw.text((boxes[0], boxes[1]), text=str(score))
    

    frame = np.array(image)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()