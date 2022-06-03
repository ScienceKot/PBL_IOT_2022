# Importing all needed libraries.
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from PIL.ExifTags import TAGS
from torchvision import transforms as T

class ApplyFlowerDetector:
    def __init__(self, **kwargs) -> None:
        '''
            The function constructor.
        :param kwargs: dict
            The setting arguments of the class.
        '''
        # Setting up the number of classes.
        self.num_classes = int(kwargs['num_classes'])

        # Setting up the model.
        self.model = self.get_model()
        self.model.load_state_dict(torch.load(kwargs['model_path']))
        self.model.eval()

        # Setting upe the transformation function.
        self.transformations = T.Compose([T.ToTensor()])

        # Setting up the threshold.
        self.threshold = float(kwargs['threshold'])

    def get_model(self):
        '''
            This function creates a pretrained model of FasterRCNN.
        :return: FastRCNNPredictor
            The model that will be used by the class.
        '''
        # Getting the model.
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Locking the parameters of the model.
        for param in model.parameters():
            param.required_grad = False

        # Getting the size of the incoming array for the model.
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Setting the type of the model.
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model

    def predict_apple_flowers(self, img_path):
        '''
            This function takes the path of the image and returns the number of apple flowers in the image.
        :param img_path: str
            The path to the image to make predictions on.
        :return: int
            The number of flowers on the image.
        '''
        # Loading the image.
        img = Image.open(img_path).convert('RGB')

        # Getting the orientation tags.
        for orientation in TAGS.keys():
            if TAGS[orientation] == 'Orientation' : break

        # Getting the exif tags.
        exif = dict(img.getexif().items())

        # Changing the orientation of the image if the exif tags changes the image rotations.
        if len(exif) != 0:
            if orientation in exif.keys():
                if exif[orientation] == 3:
                    img = img.rotate(180, expand = True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand = True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand = True)

        # Transforming the image.
        img = self.transformations(img)

        # Making the prediction.
        with torch.no_grad():
            prediction = self.model([img])

        # Defining the empty counter.
        detected_flowers = 0

        # Drawing the predicted box around the object.
        for element in range(len(prediction[0]['boxes'])):
            score = np.round(prediction[0]['scores'][element].cpu().numpy(), decimals=4)

            if score > self.threshold:
                detected_flowers+=1
        return detected_flowers
