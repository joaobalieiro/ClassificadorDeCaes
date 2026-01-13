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

