import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from  PIL import Image

plt.rcParams['figure.figsize'] = (20.0, 10.0)

# recupera o caminho das imagens
path = "C:\\Users\\fonso\\Downloads\\images\\images"

imagens = os.listdir(path)

print('{} folders in img_folder'.format(len(imagens)))
print('\n'.join(imagens))

# recupera as imagens de dobermans
Doberman_folder = os.path.join(path,'n02107142-Doberman')

Doberman_images = os.listdir(Doberman_folder)
print(f"{len(Doberman_images)} images in Doberman_folder")

os.chdir(Doberman_folder)

im = Image.open(Doberman_images[12],'r')
plt.imshow(im)
plt.axis("off")
plt.show()
print(im.format, im.size, im.mode)