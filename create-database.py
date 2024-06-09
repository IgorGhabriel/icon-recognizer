import os
import json

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from wand.image import Image as wImg



import torch
from torchvision.transforms import v2
from PIL import Image


import torchvision.models as models


from torch.utils.data import DataLoader

from sqlalchemy import create_engine

def ler_dataframe(nome_arquivo_sqlite='icons-database.sqlite'):

    if os.path.isfile(nome_arquivo_sqlite):
        engine = create_engine(f'sqlite:///{nome_arquivo_sqlite}')
        nome_data_base = pd.read_sql('data', engine)

    else:
        nome_data_base = pd.DataFrame()
        return nome_data_base

    return nome_data_base

def ler_svg(path_to_svgs='icons-base'):

    db_original = ler_dataframe(nome_arquivo_sqlite='icons-database.sqlite')

    path_svgs = list()

    for modulo in os.listdir(path_to_svgs):
        for estilo in os.listdir(os.path.join(path_to_svgs, modulo)):
            path_modulo_estilo = os.path.join(path_to_svgs, modulo, estilo)
            for icone in os.listdir(path_modulo_estilo):
                path_svgs.append(os.path.join(path_modulo_estilo, icone))

    nome_svgs = [os.path.splitext(os.path.basename(path))[0] for path in path_svgs]
    modulos = [os.path.split(os.path.dirname(os.path.dirname(path)))[1] for path in path_svgs]
    estilos = [os.path.split(os.path.dirname(path))[1] for path in path_svgs]

    data_base = pd.DataFrame({'Nome': nome_svgs, 'Svg Path': path_svgs, 'Modulo': modulos, 'Estilo': estilos})

    if db_original.empty:
        print('\nAinda não existe um database chamado icons-database.sqlite nesse diretório.\n')
        return data_base
    else:
        print('Database já existe no diretório, lendo-o.\n')
        data_base = data_base[~data_base['Svg Path'].isin(db_original['Svg Path'])]
        return data_base
    

def converter_svg_para_png(data_base):

    data_base['Png Path'] = (data_base['Svg Path'].str.replace('icons-base', 'icons-png')).str.replace('.svg', '.png')

    data_base['criando dirs pngs'] = data_base['Png Path'].str.rsplit('\\', n=1).str[0]


    for path in data_base['criando dirs pngs'].unique():
        os.makedirs(path, exist_ok=True)



    def process_image(svg_path, png_path):
        with wImg(filename=svg_path) as img:
            img.format = 'png'
            img.save(filename=png_path)

        icon = Image.open(png_path).convert('RGB')
        return icon


    with ThreadPoolExecutor() as executor:
        data_base['PIL Image'] = None
        data_base['PIL Image'] = data_base['PIL Image'].astype(object)
        
        futures = {executor.submit(process_image, row['Svg Path'], row['Png Path']): index for index, row in data_base.iterrows()}
        for future in futures:
            index = futures[future]
            data_base.at[index, 'PIL Image'] = future.result()

    data_base.drop(columns=['criando dirs pngs'], inplace = True)

    return data_base


def tensorizar(data_base):


    tensorizar = v2.Compose([
        v2.Resize((250, 250)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    lista_tensores = data_base['PIL Image'].apply(lambda img: tensorizar(img))

    return lista_tensores

def analisar_tensores(data_base, lista_tensores, batch_size=150):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Torch device: {device}.\n")

    torch.cuda.empty_cache()
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    model.to(device).eval()

    dataloader = DataLoader(lista_tensores.to_list(), batch_size=batch_size, shuffle=False)
    array_caracteristicas = list()

    for batch in dataloader:
        imgs = batch.to(device)
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        with torch.no_grad():
            features = model(imgs).squeeze()
            array_caracteristicas.extend(features.cpu().numpy())

    data_base['Array características'] = array_caracteristicas
    data_base['Array características'] = data_base['Array características'].apply(lambda x: json.dumps(x.tolist()))

    torch.cuda.empty_cache()

    return data_base


def salvar_dataframe(data_base, nome_arquivo_sqlite):

    data_base_formatado = data_base.drop(columns = ['PIL Image'])

    engine = create_engine(f'sqlite:///{nome_arquivo_sqlite}')
    data_base_formatado.to_sql('data', engine, if_exists='replace', index=False)


def main(path_to_svgs = 'icons-base', nome_arquivo_sqlite = 'icons-database.sqlite', batch_size=150):

    data_base = ler_svg(path_to_svgs)
    if data_base.empty:
        print('Nenhum módulo ou estilo novo adicionado a icons-base.')
        return None

    data_base = converter_svg_para_png(data_base)
    tensores = tensorizar(data_base)
    data_base = analisar_tensores(data_base, tensores, batch_size)
    data_base_original = ler_dataframe(nome_arquivo_sqlite)

    if os.path.isfile(nome_arquivo_sqlite):
        data_base = data_base[data_base_original.columns.tolist() + list(set(data_base.columns) - set(data_base_original.columns))]
        data_base = pd.concat([data_base_original, data_base], ignore_index=True)
        salvar_dataframe(data_base, nome_arquivo_sqlite)
        print(f'Arquivo SQLite com o database chamado {nome_arquivo_sqlite} atualizado com sucesso.\n')
    else:
        salvar_dataframe(data_base, nome_arquivo_sqlite)
        print(f'Arquivo SQLite com o database chamado {nome_arquivo_sqlite} criado com sucesso.\n')

    print(f"Diretório 'icons-png' criada/atualizada com sucesso.\n")

    main()