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

    # quantificação
    dct_quant = np.round(bloco_dct / (k1 * alfa))

    return dct_quant


def descodificador(bloco_dct, k1, alfa):
    # DCT2D inversa (IDCT2D)
    bloco_rec = bloco_dct * (k1 * alfa)
    return cv2.idct(bloco_rec)


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


def dpcm(bloco_dct):

    # lista de controlo com os valores originais
    dc_original = [bloco_dct[i][0][0] for i in xrange(len(bloco_dct))]
    #dc_original[:] = [x - 128 for x in dc_original]

    # lista de controlo com os valores DC diferenciais
    dc = [bloco_dct[0][0][0]]

    # copia do array original para um novo array
    bloco_dct_dpcm = np.copy(bloco_dct)

    # DPCM da componente DC
    for i in xrange(1, len(bloco_dct)):
        diff = bloco_dct[i][0][0] - bloco_dct[i-1][0][0]
        bloco_dct_dpcm[i][0][0] = diff
        dc.append(diff)

    # print dc_original
    # print dc

    return bloco_dct_dpcm


def desc_dpcm(bloco_dct_dpcm):

    dc = [bloco_dct_dpcm[0][0][0]]

    bloco_dct = np.copy(bloco_dct_dpcm)

    # DPCM da componente DC
    for i in xrange(1, len(bloco_dct_dpcm)):
        dc_value = bloco_dct[i-1][0][0] + bloco_dct_dpcm[i][0][0]
        bloco_dct[i][0][0] = dc_value
        dc.append(dc_value)

    # print dc

    return bloco_dct

# 4

"""
 Construa uma função (codiﬁcador) que crie um array com a indexação em zig-zag dos coeﬁcientes AC após a
quantiﬁcação e crie um array com os pares (zero run length, nonzero value).
Construa a função inversa para o descodiﬁcador.
Junte estas funções às já realizadas e veja a imagem descodiﬁcada.
"""


def zig_zag(bloco_dct_dpcm, zigzag):

    zigzag_order = zigzag.ravel().astype(np.int8)

    # array com os pares (zero run length, nonzero value)
    ac = []
    bloco_dct_dpcm_zz = []
    temp = np.zeros(64)

    for i in xrange(0, len(bloco_dct_dpcm)):
        bloco_1D = bloco_dct_dpcm[i][:][:].ravel()
        # print bloco_1D
        for z in xrange(0, len(bloco_1D)):
            temp[zigzag_order[z]] = bloco_1D[z]

        count = 0
        # print temp
        for z in xrange(1, len(temp)):
            if (temp[z] == 0) and (z == 63):
                ac.append((0, 0))
            elif temp[z] == 0:
                count += 1
            else:
                ac.append((count, int(temp[z])))
                count = 0
        # print ac
        bloco_dct_dpcm_zz.append(ac)
        ac = []

    print bloco_dct_dpcm_zz


def zag_zig(bloco_dct_dpcm_zigzag):
    return None


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
    mod8 = (array.shape[0] % 8) == 0 and (array.shape[1] % 8) == 0

    # Lista de blocos 8x8
    lista_blocos = []

    if mod8 != True:
        print "Dimensão do array não é multipla de 8"

    for i in xrange(0,array.shape[0], 8):
        for z in xrange(0, array.shape[1], 8):
            block = array[i:(i+8),z:(z+8)]
            lista_blocos.append(block.astype(np.float32))

    return lista_blocos


def revert_to_original_block(lista_blocos, original_shape):

    array_original = np.zeros(original_shape)

    count = 0

    for i in xrange(0, array_original.shape[0], 8):
        for z in xrange(0, array_original.shape[1], 8):
            array_original[i:(i+8),z:(z+8)] = lista_blocos[count]
            count+=1

    return array_original


def quality_factor(q_factor):
    if q_factor <= 50:
        factor = 50.0 / q_factor
    else:
        factor = 2.0 - (q_factor * 2.0) / 100.0
    return factor


# função auxiliar para calcular o SNR
def snr(x, sinal_desquant):
    # erro de quantificação
    erro_quant = x - sinal_desquant

    # potencia do sinal original
    p_sinal = np.sum((x ** 2.0) / len(x))

    # potencia do ruido
    p_erro_quant = np.sum((erro_quant ** 2.0) / len(erro_quant))

    # Signal to Noise Ratio
    sig_noise_ratio = 10. * np.log10(p_sinal / p_erro_quant)

    return round(sig_noise_ratio, 2)


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

    # zig-zag order
    zigzag = np.zeros((8, 8))
    zigzag[0] = [0, 1, 5, 6, 14, 15, 27, 28]
    zigzag[1] = [2, 4, 7, 13, 16, 26, 29, 42]
    zigzag[2] = [3, 8, 12, 17, 25, 30, 41, 43]
    zigzag[3] = [9, 11, 18, 24, 31, 40, 44, 53]
    zigzag[4] = [10, 19, 23, 32, 39, 45, 52, 54]
    zigzag[5] = [20, 22, 33, 38, 46, 51, 55, 60]
    zigzag[6] = [21, 34, 37, 47, 50, 56, 59, 61]
    zigzag[7] = [35, 36, 48, 49, 57, 58, 62, 63]




    x = cv2.imread("samples/Lena.tiff", cv2.IMREAD_GRAYSCALE)

    # xi = x.ravel()*1.0

    lista_blocos = create_8x8block(x)

    # le o ficheiro especifico
    # x_lena = np.fromfile("samples/lena_gray_scale.bmp", 'uint8')

    bloco_dct = []
    bloco_rec = []

    #DCT e Quantificação
    for i in xrange(len(lista_blocos)):

        bloco = codificador(lista_blocos[i], k1, alfa)

        bloco_dct.append(bloco)

    # codificação parametro DC
    bloco_dct_dpcm = dpcm(bloco_dct)

    bloco_dct_dpcm_zigzag = zig_zag(bloco_dct_dpcm, zigzag)

    x_desc = revert_to_original_block(bloco_dct_dpcm_zigzag, x.shape)
    #print snr(x, x_desc.astype(np.uint8))
    cv2.imshow("Lena cod alfa = 0", x_desc.astype(np.uint8))
    k = cv2.waitKey(0) & 0xFF

    bloco_dct_dpcm = zag_zig(bloco_dct_dpcm_zigzag)

    # Descodificação parametro DC
    bloco_dct = desc_dpcm(bloco_dct_dpcm)

    # descodificação
    for i in xrange(len(lista_blocos)):

        bloco = descodificador(bloco_dct[i], k1, alfa)

        bloco_rec.append(bloco)

    print np.all(np.rint(bloco_rec) == lista_blocos)

    #print np.all(np.rint() == )

    x_rec = revert_to_original_block(bloco_rec, x.shape)

    print snr(x, x_rec.astype(np.uint8))

    print np.all(x == np.rint(x_rec))
    cv2.imshow("Lena desc alfa=0", x_rec.astype(np.uint8))
    k = cv2.waitKey(0) & 0xFF

    cv2.imwrite("lena_output.png", x_rec.astype(np.uint8))


    # for i in xrange(len(bloco)):
    # for z in xrange(len(bloco[i])):
    # (x_lena[0:64].reshape((8, 8)) / Q) * 50

    print "========================================================================================================"
    print "========================================================================================================"
    print "========================================================================================================"
    print
    print

main()