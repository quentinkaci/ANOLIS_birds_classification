import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import torch
import torchvision
import pandas as pd

st.header("Classification d'oiseaux français")

img_file = st.file_uploader(label='Charger un fichier', type=['png', 'jpg'])
st.set_option('deprecation.showfileUploaderEncoding', False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load('artifacts/model.pth')
    model = model.to(device)
    model.eval()
    return model

with st.spinner('Le modèle est en chargement..'):
    model = load_model()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean.tolist(), std.tolist())
unnormalize = torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize
])

class_names = np.load('artifacts/class_names.npy')
fr_species_csv = pd.read_csv('artifacts/fr_species.csv')

if img_file:
    img = Image.open(img_file)
    cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF', aspect_ratio=None)
    cropped_img = transform(cropped_img)
    cropped_img = cropped_img.to(device)
    cropped_img = cropped_img.unsqueeze(0)

    with torch.no_grad():
        output = model(cropped_img)

    output = torch.softmax(output, 1)
    scores, preds = output.topk(3, 1, True, True)
    for score, pred in zip(scores[0], preds[0]):
        fr_species_name = fr_species_csv[fr_species_csv['id'] == class_names[pred]]['french'].item()
        st.write(f'L\'image est classifiée en tant que {fr_species_name} à {100 * score.item():0.2f}\%')