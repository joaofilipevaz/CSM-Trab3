# This Python file uses the following encoding: utf-8

# Trabalho 3 CSM
# João Filipe Vaz - 40266
# João Ventura - 38950

import cv2
from time import time
from os import path
import numpy as np
import matplotlib.pyplot as plt
import Queue


# 1

"""
Construa uma função (codiﬁcador) que para cada bloco de 8×8 da imagem original efectue a DCT bidimensional.
Use por exemplo a instrução
import cv2
# DCT2D direta
bloco_dct = cv2.dct(bloco)
Veja a imagem com o conjunto dos blocos após a transformada.
Construa uma função (descodiﬁcador) que faça a DCT inversa.
#DCT2D inversa (IDCT2D)
bloco_rec = cv2.dct(bloco_dct,[],cv2.DCT_INVERSE)
Veriﬁque que a imagem é igual à original.
"""

def codificador(bloco):
    bloco_dct = cv2.dct(bloco)



def descodificador(bloco_dct):
    bloco_rec = cv2.dct(bloco_dct, [], cv2.DCT_INVERSE)
