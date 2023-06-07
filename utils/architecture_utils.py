from torchvision.models import swin_b, Swin_B_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

# from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
from torch import nn
import torch


def get_architecture(
    model_name: str, dropout_rate: float, num_classes: int, fine_tune: bool = True
):
    if model_name == "trans_vit":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        model_head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(768, num_classes)
        )
    elif model_name == "trans_swin_t":
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        preprocess = Swin_T_Weights.IMAGENET1K_V1.transforms()
        model_head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(768, num_classes)
        )
        # model_head = nn.Sequential(nn.Dropout(p=dropout_rate, inplace=True),
        #                           nn.Linear(768, num_classes))
    elif model_name == "trans_swin_v2_t":
        model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        preprocess = Swin_V2_T_Weights.IMAGENET1K_V1.transforms()
        model_head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(768, num_classes)
        )
    elif model_name == "trans_swin_l":
        model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        preprocess = Swin_B_Weights.IMAGENET1K_V1.transforms()
        model_head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(1024, num_classes)
        )
        # model_head = nn.Sequential(nn.Dropout(p=dropout_rate, inplace=True),
        #                           nn.Linear(768, num_classes))
    elif model_name == "trans_swin_v2_l":
        model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        preprocess = Swin_V2_B_Weights.IMAGENET1K_V1.transforms()
        model_head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(1024, num_classes)
        )
    # model_head = nn.Sequential(nn.Dropout(p=dropout_rate, inplace=True),
    #                           nn.Linear(768, num_classes))
    elif model_name == "cnn_efficientnet_v2":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        preprocess = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
        model_head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(1280, num_classes)
        )
    elif model_name == "regnet":
        model = regnet_y_32gf(weights=RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
        preprocess = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        model_head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(3712, num_classes)
        )

    for param in model.parameters():
        param.requires_grad = fine_tune

    if model_name.startswith("trans"):
        model.head = model_head
    elif model_name.startswith("regnet"):
        model.fc = model_head
    else:
        model.classifier = model_head

    torch.compile(model, mode="default", backend="inductor")

    return model, preprocess
