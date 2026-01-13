import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

# ============================
# DEFINICAO DE CAMINHOS
# ============================

ROOT_PROJETO = r"C:\\Users\\fonso\\Downloads\\projetosGitHub\\projetoCaes"
CAMINHO_DATASET = os.path.join(ROOT_PROJETO, "dataset")
CAMINHO_IMAGENS = os.path.join(ROOT_PROJETO, "dataset", "images")

# ============================
# VALIDACAO DE DIRETORIOS
# ============================

print("Verificando diretorios...\n")

for caminho in [CAMINHO_DATASET, CAMINHO_IMAGENS]:
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Diretorio nao encontrado: {caminho}")
    else:
        print(f"OK: {caminho}")

# ============================
# CARREGAMENTO DOS ARQUIVOS .MAT
# ============================

train_info = loadmat(os.path.join(CAMINHO_DATASET, "train_data.mat"))
test_info  = loadmat(os.path.join(CAMINHO_DATASET, "test_data.mat"))

train_list = loadmat(os.path.join(CAMINHO_DATASET, "train_list.mat"))["file_list"]
test_list  = loadmat(os.path.join(CAMINHO_DATASET, "test_list.mat"))["file_list"]

print("\nArquivos .mat carregados com sucesso")

# ============================
# INSPECAO DAS ESTRUTURAS
# ============================

print("\nChaves em train_data.mat:")
print(train_info.keys())

print("\nChaves em test_data.mat:")
print(test_info.keys())

print("\nNumero de imagens:")
print("Treino:", len(train_list))
print("Teste :", len(test_list))

# ============================
# CONVERSAO PARA DATAFRAME
# ============================

def lista_para_df(lista_arquivos, nome_conjunto):
    arquivos = [str(x[0]) for x in lista_arquivos]

    df = pd.DataFrame({
        "arquivo": arquivos,
        "conjunto": nome_conjunto
    })

    return df

df_train = lista_para_df(train_list, "treino")
df_test  = lista_para_df(test_list, "teste")

df = pd.concat([df_train, df_test], ignore_index=True)

print("\nPreview do DataFrame:")
print(df.head())

# ============================
# VERIFICACAO DE IMAGENS FALTANTES
# ============================

faltantes = []

for arquivo in df["arquivo"]:
    caminho_img = os.path.join(CAMINHO_IMAGENS, arquivo)
    if not os.path.exists(caminho_img):
        faltantes.append(arquivo)

print("\nImagens ausentes:", len(faltantes))

if len(faltantes) > 0:
    print("Exemplo:", faltantes[:5])

# ============================
# VALIDACAO FINAL
# ============================

print("\nResumo final do dataset:")
print(df["conjunto"].value_counts())

print("\nDirectory setup concluido com sucesso.")
