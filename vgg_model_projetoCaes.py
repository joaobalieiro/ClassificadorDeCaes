"""
Classificador de racas de caes (VGG16)

Este script:
1) Le os splits stratificados (train_list.mat / test_list.mat) do Stanford Dogs Dataset
2) Cria uma estrutura de pastas compativel com Keras (splits_vgg/train e splits_vgg/test)
   usando hardlink/symlink/copia (nessa ordem) para evitar duplicacao desnecessaria
3) Treina um modelo com Transfer Learning (VGG16 ImageNet)
4) Avalia no conjunto de teste e salva resultados

Requisitos:
- Python 3.10+ (Windows recomendado instalar do python.org, nao Microsoft Store)
- numpy < 2
- scipy
- pillow
- tensorflow (2.12+ normalmente; depende do seu ambiente)
- scikit-learn (opcional para relatorio detalhado)

"""

from __future__ import annotations

import argparse
import os
import shutil
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# Utilitarios para .mat
# ----------------------------

def _loadmat(path: Path) -> dict:
    """Carrega .mat com configuracao segura."""
    return loadmat(str(path), squeeze_me=False, struct_as_record=False)


def _get_first_existing_key(mat: dict, candidates: List[str]) -> str:
    """Retorna a primeira chave existente dentre as candidatas."""
    for k in candidates:
        if k in mat:
            return k
    raise KeyError(f"Nenhuma das chaves {candidates} foi encontrada. Chaves disponiveis: {sorted(mat.keys())}")


def _normalizar_relpath(s: str) -> str:
    s = str(s).strip()

    # remove wrappers do tipo: "['...']"
    m = re.match(r"^\[\s*'(.+)'\s*\]$", s)
    if m:
        s = m.group(1)

    # remove aspas externas
    s = s.strip("\"'")

    # corrige separadores e quebras de linha
    s = s.replace("\r", "")
    s = s.replace("\n", "/")
    s = s.replace("\\", "/")

    # remove barras duplicadas
    s = re.sub(r"/+", "/", s)

    return s


def _mat_file_list_to_strings(file_list_arr: np.ndarray) -> List[str]:
    """
    Converte 'file_list' do .mat para lista de strings limpas.
    Suporta arrays aninhados e entradas em bytes.
    """
    out: List[str] = []

    for item in file_list_arr:
        s = item

        # desempacota arrays aninhados ate virar escalar/string
        while isinstance(s, np.ndarray):
            s = s[0]

        # bytes -> str
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8", errors="ignore")

        s = _normalizar_relpath(s)
        out.append(s)

    return out


def _ensure_jpg_suffix(rel_path: str) -> str:
    p = _normalizar_relpath(rel_path)

    if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return p

    return p + ".jpg"


# ----------------------------
# Criacao de splits em pastas
# ----------------------------

def _try_link_or_copy(src: Path, dst: Path) -> None:
    """
    Tenta criar hardlink, depois symlink, depois copia.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Se ja existe, nao refaz
    if dst.exists():
        return

    # Hardlink (rapido, ocupa pouco espaco)
    try:
        os.link(str(src), str(dst))
        return
    except Exception:
        pass

    # Symlink (pode exigir permissao no Windows)
    try:
        os.symlink(str(src), str(dst))
        return
    except Exception:
        pass

    # Copia (fallback)
    shutil.copy2(str(src), str(dst))


def criar_splits_por_listas(
    images_dir: Path,
    train_list: List[str],
    test_list: List[str],
    out_root: Path,
    limpar: bool = False,
) -> Tuple[Path, Path]:
    """
    Cria:
      out_root/train/<classe>/<arquivo>.jpg
      out_root/test/<classe>/<arquivo>.jpg
    """
    train_dir = out_root / "train"
    test_dir = out_root / "test"

    if limpar and out_root.exists():
        shutil.rmtree(out_root)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def _populate(split_dir: Path, rel_paths: List[str]) -> None:
        for rel in rel_paths:
            rel2 = _ensure_jpg_suffix(rel)
            # rel costuma ser algo como "n02107142-Doberman/n02107142_10074"
            rel_path = Path(rel2.replace("/", os.sep))
            classe = rel_path.parts[0]
            nome_arquivo = rel_path.name

            src = images_dir / rel_path
            if not src.exists():
                # tenta o caminho original sem .jpg
                alt = images_dir / Path(rel.replace("/", os.sep))
                if alt.exists():
                    src = alt
                else:
                    raise FileNotFoundError(f"Imagem nao encontrada: {src}")

            dst = split_dir / classe / nome_arquivo
            _try_link_or_copy(src, dst)

    print("\n[Split] Criando estrutura de treino...")
    _populate(train_dir, train_list)

    print("[Split] Criando estrutura de teste...")
    _populate(test_dir, test_list)

    return train_dir, test_dir


# ----------------------------
# Treinamento VGG16
# ----------------------------

def construir_modelo_vgg16(num_classes: int, input_shape=(224, 224, 3), dropout=0.3) -> tf.keras.Model:
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_vgg16(model: tf.keras.Model, fine_tune_at: int = 15, lr: float = 1e-5) -> tf.keras.Model:
    """
    Descongela parte do backbone VGG16 para ajuste fino.
    fine_tune_at: indice de camada a partir da qual descongela.
    """
    # backbone e o segundo "layer" grande no Model (VGG16)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("vgg16"):
            backbone = layer
            break
    if backbone is None:
        # fallback: encontra por nome
        for layer in model.layers:
            if "vgg16" in layer.name.lower():
                backbone = layer
                break
    if backbone is None:
        print("[Aviso] Nao foi possivel localizar o backbone VGG16 para fine-tuning.")
        return model

    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------
# Avaliacao
# ----------------------------

def salvar_relatorios(
    out_dir: Path,
    history: tf.keras.callbacks.History,
    test_generator,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Historico
    hist_path = out_dir / "history.csv"
    pd = __import__("pandas")
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(hist_path, index=False)

    # Relatorio
    report_path = out_dir / "classification_report.txt"
    cm_path = out_dir / "confusion_matrix.npy"

    if SKLEARN_OK:
        rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(rep)

        cm = confusion_matrix(y_true, y_pred)
        np.save(cm_path, cm)
    else:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("scikit-learn nao instalado; relatorio detalhado indisponivel.\n")

    # Pequeno resumo
    summary_path = out_dir / "summary.txt"
    acc = (y_pred == y_true).mean()
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Acuracia em teste: {acc:.6f}\n")
        f.write(f"Total de exemplos: {len(y_true)}\n")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Treino VGG16 para classificacao de racas de caes (Stanford Dogs)")
    parser.add_argument("--root_projeto", type=str, required=True,
                        help="Diretorio raiz do projeto (contendo pastas dataset/ e images/images/)")
    parser.add_argument("--images_subdir", type=str, default=r"images\images",
                        help="Subdiretorio onde estao as imagens (relativo ao root_projeto)")
    parser.add_argument("--dataset_subdir", type=str, default="dataset",
                        help="Subdiretorio onde estao os .mat (relativo ao root_projeto)")

    parser.add_argument("--train_list", type=str, default="train_list.mat", help="Arquivo .mat com file_list e labels do treino")
    parser.add_argument("--test_list", type=str, default="test_list.mat", help="Arquivo .mat com file_list e labels do teste")

    parser.add_argument("--splits_dir", type=str, default="splits_vgg", help="Diretorio de saida para os splits (relativo ao root_projeto)")
    parser.add_argument("--criar_splits", action="store_true", help="Cria (ou recria) os splits em pastas")
    parser.add_argument("--limpar_splits", action="store_true", help="Limpa a pasta de splits antes de criar")

    parser.add_argument("--img_size", type=int, default=224, help="Tamanho (quadrado) para resize das imagens")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Epocas (fase 1 - backbone congelado)")
    parser.add_argument("--fine_tune", action="store_true", help="Executa ajuste fino (fine-tuning) apos fase 1")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Epocas adicionais de fine-tuning")
    parser.add_argument("--fine_tune_at", type=int, default=15, help="Camada inicial para destravar no backbone")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    parser.add_argument("--output_dir", type=str, default="outputs_vgg", help="Diretorio para salvar modelo e relatorios (relativo ao root_projeto)")

    args = parser.parse_args()

    root = Path(args.root_projeto)
    images_dir = root / Path(args.images_subdir)
    dataset_dir = root / args.dataset_subdir
    splits_root = root / args.splits_dir
    out_dir = root / args.output_dir

    # Validacoes basicas
    if not images_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {images_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Pasta de dataset (.mat) nao encontrada: {dataset_dir}")

    train_mat_path = dataset_dir / args.train_list
    test_mat_path = dataset_dir / args.test_list

    if not train_mat_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {train_mat_path}")
    if not test_mat_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {test_mat_path}")

    # Seed
    tf.keras.utils.set_random_seed(args.seed)

    # Leitura dos .mat
    train_mat = _loadmat(train_mat_path)
    test_mat = _loadmat(test_mat_path)

    train_file_key = _get_first_existing_key(train_mat, ["file_list", "train_list", "list"])
    test_file_key = _get_first_existing_key(test_mat, ["file_list", "test_list", "list"])

    train_list_raw = train_mat[train_file_key]
    test_list_raw = test_mat[test_file_key]

    train_list = _mat_file_list_to_strings(train_list_raw)
    test_list = _mat_file_list_to_strings(test_list_raw)

    print(f"[Info] Itens em treino: {len(train_list)}")
    print(f"[Info] Itens em teste : {len(test_list)}")

    # Criacao de splits
    train_dir = splits_root / "train"
    test_dir = splits_root / "test"

    if args.criar_splits or (not train_dir.exists()) or (not test_dir.exists()):
        train_dir, test_dir = criar_splits_por_listas(
            images_dir=images_dir,
            train_list=train_list,
            test_list=test_list,
            out_root=splits_root,
            limpar=args.limpar_splits,
        )
        print(f"[Split] Pronto: {train_dir}")
        print(f"[Split] Pronto: {test_dir}")
    else:
        print("[Split] Reutilizando estrutura existente (use --criar_splits para recriar).")

    # Geradores
    img_size = (args.img_size, args.img_size)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.15,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=args.seed,
    )
    val_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=args.seed,
    )
    test_generator = test_datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = train_generator.num_classes
    print(f"[Info] Numero de classes: {num_classes}")

    # Modelo
    model = construir_modelo_vgg16(num_classes=num_classes, input_shape=(args.img_size, args.img_size, 3))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "vgg16_dogs.keras"

    callbacks = [
        ModelCheckpoint(filepath=str(model_path), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1),
    ]

    # Treino fase 1
    history1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tuning
    history = history1
    if args.fine_tune:
        print("\n[Fine-tuning] Iniciando ajuste fino do backbone VGG16...")
        model = fine_tune_vgg16(model, fine_tune_at=args.fine_tune_at, lr=1e-5)

        history2 = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Mescla historicos (para salvar)
        merged = {}
        for k, v in history1.history.items():
            merged[k] = list(v) + list(history2.history.get(k, []))
        history.history = merged

    # Avaliacao em teste
    print("\n[Avaliacao] Avaliando no conjunto de teste...")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"[Avaliacao] loss={test_loss:.6f}  acc={test_acc:.6f}")

    # Predicoes
    probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    salvar_relatorios(
        out_dir=out_dir,
        history=history,
        test_generator=test_generator,
        y_pred=y_pred,
        y_true=y_true,
        class_names=class_names,
    )

    print(f"\n[Saida] Modelo salvo em: {model_path}")
    print(f"[Saida] Relatorios salvos em: {out_dir}")
    if not SKLEARN_OK:
        print("[Aviso] Para relatorio detalhado e matriz de confusao, instale scikit-learn.")


if __name__ == "__main__":
    main()
