# This Python file uses the following encoding: utf-8

# Trabalho 3 CSM
# João Filipe Vaz - 40266
# João Ventura - 38950

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Tables_jpeg as jpeg

# 1

"""
Construa uma função (codiﬁcador) que para cada bloco de 8×8 da imagem original efectue a DCT bidimensional.
Veja a imagem com o conjunto dos blocos após a transformada.
Construa uma função (descodiﬁcador) que faça a DCT inversa.
Veriﬁque que a imagem é igual à original.
"""


def codificador(bloco, k1, alfa):
    # DCT2D direta
    bloco_dct = cv2.dct(bloco)
    return (bloco_dct / k1) * alfa


def descodificador(bloco_dct, k1, alfa):
    # DCT2D inversa (IDCT2D)
    bloco_rec = (bloco_dct * k1) / alfa
    return cv2.dct(bloco_rec, np.zeros((8, 8)), cv2.DCT_INVERSE)


# 2

"""
Construa uma função (codiﬁcador) que para cada bloco de 8 × 8 de coeﬁcientes da transformação efectuada faça
a divisão pela matriz de quantiﬁcação (tabela K1 no anexo da norma) multiplicada por um factor de qualidade
q (ver pág. 230 do livro "Tecnologias de Compressão Multimédia").
Veja a imagem com o conjunto dos blocos após a quantiﬁcação.
Construa uma função (descodiﬁcador) que realize a operação inversa da quantiﬁcação.
Junte estas funções às já realizadas e veriﬁque para diferentes factores de qualidade qual a SNR e veja a imagem
descodiﬁcada.
"""

# 3

"""
Construa uma função (codiﬁcador) que faça a codiﬁcação diferencial dos coeﬁcientes DC após a quantiﬁcação.
Construa a função inversa para o descodiﬁcador.
"""

# 4

"""
 Construa uma função (codiﬁcador) que crie um array com a indexação em zig-zag dos coeﬁcientes AC após a
quantiﬁcação e crie um array com os pares (zero run length, nonzero value).
Construa a função inversa para o descodiﬁcador.
Junte estas funções às já realizadas e veja a imagem descodiﬁcada.
"""

# 5

"""
Construa uma função que dados os arrays das alíneas anteriores use as tabelas do código de Huﬀman (tabela K3
e K5) e grave num ﬁcheiro a sequência de bits correspondente. (não é necessário usar o formato JFIF)
"""

# 6

"""
Construa uma função que leia o ﬁcheiro gravado e retorne os arrays com os coeﬁcientes AC e DC.
"""

# 7

"""
Junte estas funções às já realizadas e veja a imagem descodiﬁcada.
Para diferentes factores de qualidade meça a relação sinal-ruído e a taxa de compressão obtida. Represente um
gráﬁco onde se apresente a taxa de compressão em função do SNR.
"""

# 8

"""
No mesmo gráﬁco compare o seu compressor de imagem com outros existentes para várias qualidades.
"""

# 9

"""
O relatório deve conter uma descrição breve das funções realizadas e uma tabela com todos os resultados da
SNR, taxa de compressão, tempo de compressão e descompressão.
"""


def create_8x8block(array):
    # resultado da divisao modulo 8 pelo comprimento do array
    mod8 = len(array) % 8

    # Lista de blocos 8x8
    lista_blocos = []

    if mod8 != 0:
        print "Dimensão do array não é multipla de 8"
        # diff = np.zeros(8 - mod8)
        # diff.fill(array[-1])
        # array = np.append(array, diff)

    nblocos = len(array) / 64

    for i in xrange(nblocos):
        block = array[0+(i*64):64+(i*64)].reshape(8, 8)
        lista_blocos.append(block.astype(np.float32))

    return lista_blocos


def quality_factor(q_factor):
    if q_factor <= 50:
        factor = 50.0 / q_factor
    else:
        factor = 2.0 - (q_factor * 2.0) / 100.0
    return factor


def main():
    print "========================================================================================================"
    print "================================Analise Ficheiro lena================================" \
          "======="

    # np.random.seed(68)
    # bloco = np.random.randint(-10, 10, size=(8, 8)) * 1.0

    # factor de qualidade q
    q = 50

    #
    alfa = quality_factor(q)

    # Matriz de quantificação K1
    # table K1 - Luminance quantize Matrix
    k1 = np.zeros((8, 8))
    k1[0] = [16, 11, 10, 16, 24, 40, 51, 61]
    k1[1] = [12, 12, 14, 19, 26, 58, 60, 55]
    k1[2] = [14, 13, 16, 24, 40, 57, 69, 56]
    k1[3] = [14, 17, 22, 29, 51, 87, 80, 62]
    k1[4] = [18, 22, 37, 56, 68, 109, 103, 77]
    k1[5] = [24, 35, 55, 64, 81, 104, 113, 92]
    k1[6] = [49, 64, 78, 87, 103, 121, 120, 101]
    k1[7] = [72, 92, 95, 98, 112, 100, 103, 99]

    x = cv2.imread("samples/Lena.tiff", cv2.IMREAD_GRAYSCALE)

    xi = x.ravel()

    lista_blocos = create_8x8block(xi)

    # le o ficheiro especifico
    # x_lena = np.fromfile("samples/lena_gray_scale.bmp", 'uint8')

    for bloco in lista_blocos:

        bloco_dct = codificador(bloco, k1, alfa)

        bloco_rec = descodificador(bloco_dct, k1, alfa)

        print np.all(np.rint(bloco_rec) == bloco)

    # for i in xrange(len(bloco)):
    # for z in xrange(len(bloco[i])):
    # (x_lena[0:64].reshape((8, 8)) / Q) * 50

    print "========================================================================================================"
    print "========================================================================================================"
    print "========================================================================================================"
    print
    print

main()