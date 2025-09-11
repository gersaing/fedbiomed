# extractor_caracteristicas.py
from __future__ import annotations
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from typing import Union

class ExtractorCaracteristicas:
    """
    Extractor de características con ResNet-18 preentrenada (ImageNet).
    - Conserva convoluciones + GAP y elimina la capa totalmente conectada (FC).
    - Usa las transformaciones por defecto ligadas a los pesos (224x224 + normalización).
    """

    def __init__(
        self,
        pesos: ResNet18_Weights = ResNet18_Weights.DEFAULT,
        dispositivo: Union[str, torch.device, None] = None,
        modo_transform: str = "compat",  # "compat" normaliza a 224X224 "imagenet" = transforms() oficial (recorta la imagen)
    ):
        if dispositivo is None:
            dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.dispositivo = torch.device(dispositivo)
            modelo = resnet18(weights=pesos).to(self.dispositivo).eval()
        except Exception as e:
            print(f"[Extractor] CUDA no disponible ({e}); usando CPU.")
            self.dispositivo = torch.device("cpu")
            modelo = resnet18(weights=pesos).to(self.dispositivo).eval()

        self.extractor = torch.nn.Sequential(*list(modelo.children())[:-1])
        self.dim_salida = 512

        if modo_transform == "imagenet":
            # Puede incluir Resize(232/256) + CenterCrop(224) + Normalize
            self.transformaciones = pesos.transforms()
        else:  # "compat": igual a tu código anterior
            self.transformaciones = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    @torch.inference_mode()
    def extraer(self, ruta_imagen: str):
        """Devuelve un vector (512,) float32 (numpy) para la imagen indicada."""
        with Image.open(ruta_imagen) as im:
            img = im.convert("RGB")
        x = self.transformaciones(img).unsqueeze(0).to(self.dispositivo)  # [1,3,224,224]
        f = self.extractor(x)                                             # [1,512,1,1]
        v = torch.flatten(f, 1).squeeze(0).to("cpu", dtype=torch.float32).numpy()
        return v
