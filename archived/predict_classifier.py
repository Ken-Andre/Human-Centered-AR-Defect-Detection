
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import mobilenet_v3_small
import torch.nn as nn

# Load model
model = mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("../models/mobilenet_screw_classifier.pt", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Predict function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    _, pred = torch.max(output, 1)
    label = "good" if pred.item() == 0 else "defect"
    print(f"[PREDICT] {image_path} → {label}")

# Exemple
predict("../dataset/classification/defect/000.png")
