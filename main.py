import os
import glob
import csv

import pandas as pd
import numpy as np

from wand.image import Image as wImg

from skimage import transform, io

import torch
from torchvision.transforms import v2
import torchvision.models as models
from PIL import Image

from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA


from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt


print("\n--------------------------------------------------------------------------------------\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print("\n--------------------------------------------------------------------------------------\n")

def load_icons_from_db(db_path, target_icon_path):

    icons = []
    caminhos = []

    for filename in os.listdir(db_path):
        path = os.path.join(db_path, filename)
        icon = Image.open(path).convert('RGB')
        icons.append(icon)
        caminhos.append(path)

    target_icon = Image.open(target_icon_path).convert('RGB')
    
    icons.append(target_icon)
    caminhos.append(target_icon_path)
    

    return icons, caminhos

def icon_pre_process(icons, height = 224, width = 224):
    pre_processed_icons = []

    
    img = v2.Compose([
                    v2.Resize((height, width)),
                    v2.ToImage(), 
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    for i in icons:
        icon = img(i)
        pre_processed_icons.append(icon)

    return pre_processed_icons



def characteristics_extractor(icons, batch_size=10):
    torch.cuda.empty_cache()
    #modelo ResNet50 pré-treinado
    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    # Remover a última camada totalmente conectada (classificação)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Modo de avaliação
    model.to(device).eval()
    
    # Criando o DataLoader

    dataloader = DataLoader(icons, batch_size=batch_size, shuffle=False)

    characteristics_extracted = []

    for batch in dataloader:
            imgs = batch.to(device)  # Ensure the batch is on the GPU
            if imgs.dim() == 3:  # Check if input is 3D and add batch dimension
                imgs = imgs.unsqueeze(0)
            with torch.no_grad():
                try:
                    features = model(imgs).squeeze()  # Remove batch and singleton dimensions if necessary
                    characteristics_extracted.extend(features.cpu().numpy())  # Move back to CPU for numpy conversion
                except RuntimeError as e:
                    print(f"RuntimeError: {e}")
                    # Reduce batch size if running out of memory
                    batch_size = max(1, batch_size // 2)
                    dataloader = DataLoader(icons, batch_size=batch_size, shuffle=False)
                    characteristics_extracted = []
                    break

    return np.array(characteristics_extracted)


def reducing_dimension(characteristics, variancia_desejada=0.99):

    pca = PCA()
    pca.fit(characteristics)
    
    # Calcular a variância acumulada
    variancia_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # Encontrar o número de componentes que explicam pelo menos a variância desejada
    n_componentes = np.argmax(variancia_acumulada >= variancia_desejada) + 1
    
    # Ajustar PCA com o número de componentes determinado
    pca = PCA(n_components=n_componentes)
    reduced_characteristics = pca.fit_transform(characteristics)
    
    return reduced_characteristics


def finding_alikes(characteristcs_given_icon, characteristics_bank, top_n=5):

    distances = [euclidean(characteristcs_given_icon, c) for c in characteristics_bank]
    alike_indexes = np.argsort(distances)
    top_alike_indexes = alike_indexes[:top_n]

    return top_alike_indexes



def visualizar_similares(indices_similares, caminhos):
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices_similares):
        img = Image.open(caminhos[idx])
        plt.subplot(1, len(indices_similares), i + 1)
        plt.imshow(img)
        plt.title(f'Imagem {idx}')
        plt.axis('off')
    plt.show()



def shoot(db_path, target_icon_path, height = 224, width = 224, variancia_desejada=0.99, top_n = 5):

    icons_db, paths = load_icons_from_db(db_path = db_path, target_icon_path = target_icon_path)

    preprocessed_icons = icon_pre_process(icons = icons_db, height = height, width = width)

    characteristics = characteristics_extractor(preprocessed_icons)

    reduced_characteristcs = reducing_dimension(characteristics = characteristics, variancia_desejada = variancia_desejada)
    
    characteristcs_given_icon = reduced_characteristcs[-1]

    characteristics_bank = reduced_characteristcs[:-1]

    alike_indexes = finding_alikes(characteristcs_given_icon = characteristcs_given_icon, characteristics_bank = characteristics_bank, top_n = top_n)

    teste = paths[:-1]
    alike_icons_paths = [teste[idx] for idx in alike_indexes]

    
    print("Path dos icones similares:")
    for path in alike_icons_paths:
        print("\n" + path +"\n")
    
    visualizar_similares(alike_indexes, paths[:-1])                                                                                                                      

    return alike_indexes




shoot("C:\\Users\\IgorG\\OneDrive\\Área de Trabalho\\coisas\\python\\icon-recognizer\\icons-png",
       "C:\\Users\\IgorG\\OneDrive\\Área de Trabalho\\coisas\\python\\icon-recognizer\\teste.png")
