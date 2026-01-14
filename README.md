# Classificação de Raças de Cães com Transfer Learning (VGG16)

Neste projeto eu construí um pipeline **reprodutível** para treinar e avaliar um classificador de **raças de cães** a partir do Stanford Dogs Dataset, usando **Transfer Learning com VGG16 (ImageNet)**.

## O que eu fiz

- **Automatizei o preparo do dataset**: leio as listas `.mat` (train/test) e **monto os splits em pastas** no formato compatível com Keras (`train/<classe>/`, `test/<classe>/`).
- **Otimizei armazenamento/tempo** na criação dos splits: tento **hardlink → symlink → cópia**, evitando duplicar imagens quando possível.
- **Modelei o treinamento com Transfer Learning**: backbone VGG16 congelado + cabeça densa para classificação, com **data augmentation** para melhorar generalização.
- **Implementei treinamento robusto** com callbacks (checkpoint do melhor modelo, early stopping e ajuste automático de learning rate).
- **Incluí fine-tuning opcional**: possibilidade de destravar camadas finais do backbone para ganhar performance.
- **Gerei artefatos para análise**: modelo salvo e relatórios (histórico, métricas e, se disponível, matriz de confusão/relatório por classe).

## Por que eu fiz assim

- **Transfer Learning (VGG16)**: reduz custo computacional e acelera convergência, aproveitando features já aprendidas em ImageNet.
- **Splits em diretórios + CLI**: facilita reproduzir experimentos e integrar em ambientes diferentes (Windows/Linux) sem etapas manuais.
- **Links antes de copiar**: evita desperdício de espaço em disco, algo crítico em datasets de imagens.
- **Callbacks e logs**: garantem rastreabilidade (saber “qual foi o melhor modelo” e “como o treino evoluiu”) e evitam overfitting.

## Como rodar

### Treinar (criando splits do zero)
```bash
python vgg_model_projetoCaes.py --root_projeto "<CAMINHO_DO_PROJETO>" \
  --images_subdir "dataset/images" --dataset_subdir "dataset" \
  --criar_splits --limpar_splits --epochs 10 --batch_size 32

### Fine-tuning
python vgg_model_projetoCaes.py --root_projeto "<CAMINHO_DO_PROJETO>" \
  --images_subdir "dataset/images" --dataset_subdir "dataset" \
  --fine_tune --fine_tune_epochs 5 --fine_tune_at 15
