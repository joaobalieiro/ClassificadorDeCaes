import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from  PIL import Image

plt.rcParams['figure.figsize'] = (20.0, 10.0)

# recupera o caminho das imagens
path = "C:\\Users\\fonso\\Downloads\\projetosGitHub\\projetoCaes\\dataset\\images"

imagens = os.listdir(path)

print('{} folders in img_folder'.format(len(imagens)))
print('\n'.join(imagens))

# ============================
# visualizacao de uma classe
# ============================

Doberman_folder = os.path.join(path,'n02107142-Doberman')

Doberman_images = os.listdir(Doberman_folder)
print(f"{len(Doberman_images)} images in Doberman_folder")

os.chdir(Doberman_folder)

im = Image.open(Doberman_images[12],'r')
plt.imshow(im)
plt.axis("off")
plt.show()
print(im.format, im.size, im.mode)

# ============================
# ANALISE DE DESBALANCEAMENTO
# ============================

data = []

for breed_folder in os.listdir(path):
    breed_path = os.path.join(path, breed_folder)

    if os.path.isdir(breed_path):
        images = os.listdir(breed_path)

        for img in images:
            data.append({
                "breed": breed_folder,
                "image": img
            })

df = pd.DataFrame(data)

print("\nDataFrame preview:")
print(df.head())

# contagem de imagens por raca
class_counts = df["breed"].value_counts()

print("\nNumero de imagens por raca:")
print(class_counts)

# ============================
# VISUALIZACAO DO DESBALANCEAMENTO
# ============================

class_counts.plot(kind="bar")
plt.title("Distribuicao de imagens por raca")
plt.xlabel("Raca")
plt.ylabel("Numero de imagens")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ============================
# dados de treino e teste
# ============================

train_data = loadmat(r'''C:\\Users\\fonso\\Downloads\\projetosGitHub\\projetoCaes\\dataset\\train_data.mat''')
test_data = loadmat(r'''C:\\Users\\fonso\\Downloads\\projetosGitHub\\projetoCaes\\dataset\\test_data.mat''')
print(train_data.keys())

print("=" * 60)
print("INFORMACOES DO CONJUNTO DE TREINAMENTO")
print("=" * 60)

print("\nMetadados do treino (train_info):")
print(train_data['train_info'])

print("\nQuantidade de amostras em train_fg_data:")
print(len(train_data['train_fg_data'][0]))

print("\nQuantidade de amostras em train_data:")
print(len(train_data['train_data'][0]))

print("\n" + "=" * 60)
print("INFORMACOES DO CONJUNTO DE TESTE")
print("=" * 60)

print("\nMetadados do teste (train_info):")
print(test_data['train_info'])

print("\nQuantidade de amostras em train_fg_data:")
print(len(test_data['train_fg_data'][0]))

print("\nQuantidade de amostras em train_data:")
print(len(test_data['train_data'][0]))

print("\n" + "=" * 60)
print("FIM DA INSPECAO DOS DADOS")
print("=" * 60)

# define arrays numpy para construir o DataFrame
def mat_para_df(caminho_arquivo, rotulo_estrutura):
    matfile = loadmat(caminho_arquivo)

    lista_arquivos = matfile[rotulo_estrutura][0][0][0]
    lista_anotacoes = matfile[rotulo_estrutura][0][0][1]
    labels = matfile[rotulo_estrutura][0][0][2]
    fg_id = matfile[rotulo_estrutura][0][0][3]

    dados = np.array([lista_anotacoes, labels, fg_id])

    df = pd.DataFrame(dados)
    return df

