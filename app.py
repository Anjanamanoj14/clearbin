import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Bin colors and emojis for each class
class_info = {
    'cardboard' : {'emoji': '📦', 'bin': 'Brown bin',  'color': '#8B4513'},
    'glass'     : {'emoji': '🍶', 'bin': 'Green bin',  'color': '#2E8B57'},
    'metal'     : {'emoji': '🥫', 'bin': 'Blue bin',   'color': '#4169E1'},
    'paper'     : {'emoji': '📄', 'bin': 'Blue bin',   'color': '#4169E1'},
    'plastic'   : {'emoji': '🧴', 'bin': 'Yellow bin', 'color': '#FFD700'},
    'trash'     : {'emoji': '🗑️', 'bin': 'Black bin',  'color': '#333333'},
}

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 6)
)
model.load_state_dict(torch.load("clearbin_best.pth",
                      map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image):
    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    # Get top prediction
    top_idx = probabilities.argmax().item()
    top_class = classes[top_idx]
    top_conf = float(probabilities[top_idx]) * 100
    info = class_info[top_class]

    # Result message
    result = f"{info['emoji']} {top_class.upper()} — {top_conf:.1f}% confident\n"
    result += f"♻️ Put this in the {info['bin']}"

    # All class probabilities
    scores = {f"{class_info[classes[i]]['emoji']} {classes[i]}": 
              float(probabilities[i]) for i in range(len(classes))}

    return result, scores

with gr.Blocks(theme=gr.themes.Soft(), title="ClearBin") as app:
    gr.Markdown("""
    # ♻️ ClearBin — AI Waste Classifier
    ### Helping you sort waste correctly with AI
    Upload a photo of any waste item and ClearBin will tell you exactly which bin it belongs to.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload waste image")
            submit_btn = gr.Button("🔍 Classify", variant="primary")

        with gr.Column():
            result_text = gr.Textbox(label="Result", lines=2)
            confidence_chart = gr.Label(
                num_top_classes=6, 
                label="Confidence scores"
            )

    gr.Markdown("""
    ### 🗑️ Bin guide
    📦 Cardboard → Brown bin | 🍶 Glass → Green bin | 
    🥫 Metal → Blue bin | 📄 Paper → Blue bin | 
    🧴 Plastic → Yellow bin | 🗑️ Trash → Black bin
    """)

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_text, confidence_chart]
    )

app.launch()