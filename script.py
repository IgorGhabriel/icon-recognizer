import glob
import os
import csv

diretorio = '/home/gust/projects/icon-recognizer/material-design-icons/svg/filled'

padrao = '*.svg'

lista_arquivos = glob.glob(os.path.join(diretorio, padrao))

csv_dict = []

for arquivo in lista_arquivos:
    icon_file = arquivo.split('/')[-1]
    icon_name = icon_file.replace(".svg", "")
    icon_data = {
        'name': icon_name,
        'path': arquivo,
    }
    csv_dict.append(icon_data)

filename = 'icons.csv'
fields = ['name', 'path']

with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_dict)

