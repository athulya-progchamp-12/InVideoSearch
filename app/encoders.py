import torch
import torchvision
import timm
from transformers import BertModel
from yolov5.models.yoloE import BaseModel, DetectionModel, DetectMultiBackend

# InceptionV3 Encoder
class InceptionV3Encoder(torch.nn.Module):
    def __init__(self):
        super(InceptionV3Encoder, self).__init__()
        self.incep3 = torchvision.models.inception_v3(pretrained=True)
        self.incep3.aux_logits = False
        self.output_size = self.incep3.fc.in_features
        self.incep3.fc = torch.nn.Identity()
        for parameter in self.incep3.parameters():
            parameter.requires_grad = False
        self.num_parameters = sum([torch.prod(torch.tensor(params.size())) for params in self.incep3.parameters()])

    def forward(self, x):
        return self.incep3(x)

# Xception Encoder
class XceptionEncoder(torch.nn.Module):
    def __init__(self):
        super(XceptionEncoder, self).__init__()
        self.xception = timm.create_model("xception", pretrained=True)
        self.output_size = self.xception.fc.in_features
        self.xception.fc = torch.nn.Identity()
        for parameter in self.xception.parameters():
            parameter.requires_grad = False
        self.num_parameters = sum([torch.prod(torch.tensor(params.size())) for params in self.xception.parameters()])

    def forward(self, x):
        return self.xception(x)

# ResNet Encoder
class ResNetEncoder(torch.nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True, progress=False)
        self.output_size = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False
        self.num_parameters = sum([torch.prod(torch.tensor(params.size())) for params in self.resnet.parameters()])

    def forward(self, x):
        return self.resnet(x)

# YOLO Encoder
class YoloEncoder(BaseModel):
    def __init__(self, cfg='yolov5s.yaml', model=None):
        super().__init__()
        model = DetectionModel()
        self._from_detection_model(model) if model is not None else self._from_yaml(cfg)
        self.output_size = self.model[-1].cv2.conv.out_channels * 10 * 10
        self.num_parameters = sum([torch.prod(torch.tensor(params.size())) for params in self.model.parameters()])

    def _from_detection_model(self, model):
        if isinstance(model, DetectMultiBackend):
            model = model.model
        self.model = model.model

    def forward(self, x):
        return self.model(x)

# BERT Encoder
class BertEncoder(torch.nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.output_size = self.bert.config.hidden_size
        for parameter in self.bert.parameters():
            parameter.requires_grad = False
        self.num_parameters = sum([torch.prod(torch.tensor(params.size())) for params in self.bert.parameters()])

    def forward(self, input_ids, attention_mask):
        last_hidden, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return last_hidden[:, 0]

# Choose Encoder based on name
def Encoder(name):
    if name == "InceptionV3":
        return InceptionV3Encoder()
    elif name == "Xception":
        return XceptionEncoder()
    elif name == "ResNet":
        return ResNetEncoder()
    elif name == "YOLO":
        return YoloEncoder()
    elif name == "BERT":
        return BertEncoder()
    else:
        raise ValueError(name + " has not been implemented!")
