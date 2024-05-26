import glob
import os
import csv

import pandas as pd
from wand.image import Image as wImg

diretorio = 'material-design-icons/svg/filled'

padrao = '*.svg'

lista_arquivos = glob.glob(os.path.join(diretorio, padrao))

csv_dict = []

for arquivo in lista_arquivos:

    icon_file = arquivo.rsplit(("\\"))[-1]
    icon_name = icon_file.replace(".svg", "")

    icon_data = {
        'name': icon_name,
        'path': arquivo.replace("\\", "/"),
    }

    csv_dict.append(icon_data)

filename = 'icons.csv'
fields = ['name', 'path']

with open(filename, 'w') as csvfile:
    
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_dict)


df_icons = pd.read_csv("icons.csv")


def svg_to_png_converter(path_svg, png_name_file):

    path_png = "C:\\Users\\IgorG\\OneDrive\\Área de Trabalho\\coisas\\python\\icon-recognizer\\icons-png\\" + png_name_file
    with wImg(filename=path_svg) as img:
        img.format = 'png'
        img.save(filename=path_png)

df_icons.apply(lambda row: svg_to_png_converter(row['path'], row['name'] + ".png"), axis=1)
        

