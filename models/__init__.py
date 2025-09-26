from .allcnn import *
from .resnet import *
from .vgg import *
from .vision_transformer import *
from .swin_transformer import *
import torch.nn as nn
import torch
from torchvision import models
from torchvision.models.vision_transformer import ViT_B_16_Weights

def _reset_classifier(model, num_classes: int):
    """
    Replace the final classifier layer with `num_classes` for common backbones:
    - ResNet: model.fc
    - Swin/timm-like: model.head or model.head.fc
    - torchvision ViT: model.heads[-1]
    """
    # ResNet
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Swin / timm-style
    if hasattr(model, "head"):
        # plain Linear head
        if isinstance(model.head, nn.Linear):
            in_f = model.head.in_features
            model.head = nn.Linear(in_f, num_classes)
            return model
        # nested .head.fc
        if hasattr(model.head, "fc") and isinstance(model.head.fc, nn.Linear):
            in_f = model.head.fc.in_features
            model.head.fc = nn.Linear(in_f, num_classes)
            return model

    # torchvision ViT: heads is an nn.Sequential; last module is Linear
    if hasattr(model, "heads"):
        try:
            last = model.heads[-1]
            if isinstance(last, nn.Linear):
                in_f = last.in_features
                model.heads[-1] = nn.Linear(in_f, num_classes)
                return model
        except Exception:
            pass

    # Fallback: if there's an attribute named 'classifier'
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, num_classes)
        return model

    return model


def get_model(model_name, num_classes, use_pretrained):
    if model_name == 'resnet18':
        w = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet18(weights=w)
        model = _reset_classifier(model, num_classes)
        
    elif model_name == 'resnet50':
        w = models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet50(weights=w)
        model = _reset_classifier(model, num_classes)
        
    elif model_name == "allcnn":
        model = AllCNN(n_channels=3, num_classes=num_classes, filters_percentage=0.5)
        
    elif model_name == "my-resnet18":
        model = resnet18(num_classes=num_classes)   
        
    elif model_name == "vgg16":
        model = vgg16_bn(num_classes=num_classes)  
        
    elif model_name == "vgg11":
        model = vgg11_bn(num_classes=num_classes)
        
    elif model_name == "vit-s-16":  
        model = _vision_transformer(
            num_classes=num_classes,
            patch_size=16,
            num_layers=12, 
            num_heads=6,   
            hidden_dim=384, 
            mlp_dim=1536, 
            progress=False,
            weights=None, 
        )
    
    elif model_name == "vit-b-16":  
        vit_w = ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
        
        if vit_w is not None:
            model = _vision_transformer(
                patch_size=16,
                num_layers=12,     # base
                num_heads=12,      # base
                hidden_dim=768,    # base
                mlp_dim=3072,      # base
                progress=False,
                weights=vit_w,
            )
            model = _reset_classifier(model, num_classes)
        else:
            model = _vision_transformer(
                num_classes=num_classes,
                patch_size=16,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                mlp_dim=3072,
                progress=False,
                weights=None,
            )
            
    elif model_name == "swin-t":
        model = swin_tiny_patch4_window7_224(pretrained=use_pretrained, num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def load_model(model_path, model_name, num_classes):
    model_ckpt = torch.load( model_path , map_location="cuda")
    if isinstance(model_ckpt, dict):  
        model = get_model(model_name, num_classes, use_pretrained=False)
        model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
        model.load_state_dict(model_ckpt)
    else:
        model = model_ckpt
    model = model.to("cuda")
    return model