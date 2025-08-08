import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image

# Preparación del modelo ResNet-18
modelo_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
modelo_resnet.eval()
extractor = torch.nn.Sequential(*list(modelo_resnet.children())[:-1])

transformar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extraer_caracteristicas(ruta_imagen):
    """
    Extrae el vector de características de la imagen indicada usando ResNet-18 preentrenada.
    Retorna un vector numpy.
    """
    imagen = Image.open(ruta_imagen).convert("RGB")
    imagen = transformar(imagen).unsqueeze(0)  # Batch size = 1
    with torch.no_grad():
        vector = extractor(imagen)
    return vector.squeeze().numpy()

