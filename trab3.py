# This Python file uses the following encoding: utf-8

# Trabalho 3 CSM
# João Filipe Vaz - 40266
# João Ventura - 38950


import cv2
import numpy as np
# import matplotlib.pyplot as plt
from Tables_jpeg import K3, K5
from time import time
from os import path

# 1

"""
Construa uma função (codiﬁcador) que para cada bloco de 8×8 da imagem original efectue a DCT bidimensional.
Veja a imagem com o conjunto dos blocos após a transformada.
Construa uma função (descodiﬁcador) que faça a DCT inversa.
Veriﬁque que a imagem é igual à original.
"""


def codificador(bloco, k1, alfa):
    # DCT2D direta
    bloco_dct = cv2.dct(bloco-128)

    # quantificação
    dct_quant = np.round(bloco_dct / (k1 * alfa))

    return dct_quant


def descodificador(bloco_desc_dct, k1, alfa):
    # DCT2D inversa (IDCT2D)
    bloco_rec = np.round((k1 * alfa) * bloco_desc_dct)
    return np.round(cv2.idct(bloco_rec)+128)


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

    # lista de controlo com os valores DC diferenciais
    dc = [bloco_dct[0][0][0]]

    # copia do array original para um novo array
    bloco_dct_dpcm = np.copy(bloco_dct)

    # DPCM da componente DC
    for i in xrange(1, len(bloco_dct)):
        diff = bloco_dct[i][0][0] - bloco_dct[i-1][0][0]
        bloco_dct_dpcm[i][0][0] = diff
        dc.append(diff)

    return bloco_dct_dpcm, dc


def desc_dpcm(bloco_dct_dpcm, dc):

    bloco_dct = np.copy(bloco_dct_dpcm)

    dcanterior = 0

    # DPCM da componente DC
    for i in xrange(0, len(dc)):
        bloco_dct[i][0][0] = dc[i] + dcanterior
        dcanterior = bloco_dct[i][0][0]

    return bloco_dct

# 4

"""
Construa uma função (codiﬁcador) que crie um array com a indexação em zig-zag dos coeﬁcientes AC após a
quantiﬁcação e crie um array com os pares (zero run length, nonzero value).
Construa a função inversa para o descodiﬁcador.
Junte estas funções às já realizadas e veja a imagem descodiﬁcada.
"""


def zig_zag(bloco_dct_dpcm, zigzag, debug, test_block):

    zigzag_order = zigzag.ravel().astype(np.int8)

    # lista bidimensional com os ac de cada bloco
    bloco_dct_dpcm_zz = []

    # array temporario para guardar os valores ordenaados pela order do zigzag
    temp = np.zeros(64)

    for i in xrange(len(bloco_dct_dpcm)):
        # captura o primeiro bloco 8x8
        bloco_1d = bloco_dct_dpcm[i][:][:].ravel()

        # lista com os pares (zero run length, nonzero value)
        ac = []

        if i == test_block and debug:
            print bloco_1d
            print zigzag_order

        for z in xrange(0, len(bloco_1d)):
            # guarda o valor no indice correspondente pela ordem do zigzag
            temp[zigzag_order[z]] = bloco_1d[z]

        # variavel auxiliar para contar o numero de zeros
        zeros = 0

        if i == test_block and debug:
            print temp

        for t in xrange(1, len(temp), 1):
            # valida o fim do bloco
            if (temp[t] == 0) and (t == 63):
                ac.append((0, 0))
            # aplica o limita máximo de 15 zeros consecutivos para nao criar conflitos com a cod huff
            elif temp[t] == 0 and zeros < 15:
                zeros += 1
            else:
                # adiciona o um tuplo (run length code, value)
                ac.append((zeros, int(temp[t])))
                zeros = 0

        if i == test_block and debug:
            print ac

        bloco_dct_dpcm_zz.append(ac)

    return bloco_dct_dpcm_zz


def zag_zig(bloco_dct_dpcm_zz, zigzag, debug, test_block):

    zigzag_order = zigzag.ravel().astype(np.int8)

    # lista de output 8x8
    bloco_dct_dpcm = []

    for i in xrange(len(bloco_dct_dpcm_zz)):
        ac = bloco_dct_dpcm_zz[i]

        if i == test_block and debug:
            print ac

        temp = np.zeros(64)

        ultima_pos = 0

        for z in xrange(len(ac)):

            zeros = ac[z][0]
            value = ac[z][1]

            if value != 0:
                temp[zeros+1+ultima_pos] = value
                ultima_pos += zeros+1

        if i == test_block and debug:
            print temp
            print zigzag_order

        bloco_1d_ordenado = np.zeros(64)

        for t in xrange(1, len(temp)):
            if temp[t] != 0:
                # guarda o valor no indice correspondente pela ordem do zigzag
                bloco_1d_ordenado[np.where(zigzag_order == t)[0][0]] = temp[t]

        bloco_1d_ordenado = bloco_1d_ordenado.reshape((8, 8))

        if i == test_block and debug:
            print bloco_1d_ordenado

        bloco_dct_dpcm.append(bloco_1d_ordenado)

    return bloco_dct_dpcm


# 5

"""
Construa uma função que dados os arrays das alíneas anteriores use as tabelas do código de Huﬀman (tabela K3
e K5) e grave num ﬁcheiro a sequência de bits correspondente. (não é necessário usar o formato JFIF)
"""


def codifica_huff(bloco_dct_dpcm_zz, bloco_dct_dpcm, debug):

    # stream de bits de saida
    bit_stream = ""

    # insere informação sobre a o numero de blocos 8x8 a ler
    bit_stream += '{0:032b}'.format(len(bloco_dct_dpcm_zz))

    for i in xrange(len(bloco_dct_dpcm)):
        # valor componente DC
        dc = int(bloco_dct_dpcm[i][0][0])

        if dc != 0:
            # O campo Size indica quantos bits codificam o campo amplitude
            size = len('{0:b}'.format(abs(dc)))
        else:
            size = 0

        # adiciona o size ao bitstream recorrendo à codificação de huffman
        bit_stream += K3[size]  # + " "

        if size != 0:
            # amplitude é o valor em binario do componente dc
            amp_dc = ones_complement(dc, size)

            # adiciona o valor directamente ao bitstream sem codificação de huffman
            bit_stream += amp_dc  # + " "

        # analise da componente ac
        for z in xrange(len(bloco_dct_dpcm_zz[i])):

            # quantidade de 0s consecutivos
            runlength = bloco_dct_dpcm_zz[i][z][0]

            # valor do coeficiente nao nulo
            value = bloco_dct_dpcm_zz[i][z][1]

            if value != 0:
                # o valor é ainda subdividido em size e amp como no dc
                size = len('{0:b}'.format(abs(value)))
                amp_ac = ones_complement(value, size)
                # o tuplo (runlength, size) é codificado recorrendo a tabela K5 com codigo de Huffman
                bit_stream += K5[(runlength, size)]  # + " "
                # o valor é codificado sem huffman
                bit_stream += amp_ac  # + " "

            else:
                size = 0
                # o tuplo (runlength, size) é codificado recorrendo a tabela K5 com codigo de Huffman
                bit_stream += K5[(runlength, size)]  # + " "

    if debug:
        print bit_stream
        print len(bit_stream)

    # utiliza a função desenvolvida no trab anterior para escrever para ficheiro
    escrever(bit_stream, "Lena_Cod.huf")


# 6

"""
Construa uma função que leia o ﬁcheiro gravado e retorne os arrays com os coeﬁcientes AC e DC.
"""


def le_huff():

    # lista com os coeficientes dc
    dc = []

    # lista com os coeficientes ac por bloco
    ac = []

    # Sequencia de bits com a codificação da mensagem
    seqbits = ler("Lena_Cod.huf")

    # le o primeiro bloco de 32 bits com a informação sobre o numero de blocos 8x8 a ler
    n_blocos = int(seqbits[0:32], 2)
    seqbits = seqbits[32:]

    # lista bidimensional com a totalidade dos ac dos blocos
    bloco_dct_dpcm_zz = []

    # loops de (15,0) para serem retirados
    zero_run_loops = 0

    # lê os bits codificados enquanto houver dados para leitura
    for z in xrange(n_blocos):
        # flag end of block
        eob = False
        # le o dc
        for k in K3:
            # avalia o prefixo inicial de acordo com a chave do dicionario
            if seqbits.startswith(K3[k]):
                # slice da mensagem de bits para lermos sempre a partir do inicio
                seqbits = seqbits[len(K3[k]):]
                if k > 0:
                    # adiciona o valor a lista dc
                    dc.append(read_ones_complement(seqbits[0:k]))
                    # remove o valor lido da mensagem
                    seqbits = seqbits[k:]
                else:
                    dc.append(0)
                # print "DC =" + str(dc)
                break
        while not eob:
            # print "loop"
            for y in K5:
                # avalia o prefixo inicial de acordo com a chave do dicionario
                if seqbits.startswith(K5[y]):
                    if K5[y] == "1010":
                        eob = True

                    # obtemos runleght e size
                    runlength = y[0]
                    size = y[1]

                    # e slice da mensagem de bits para lermos sempre a partir do inicio
                    seqbits = seqbits[len(K5[y]):]

                    if size != 0:
                        # obtemos o valor
                        value = read_ones_complement(seqbits[0:size])

                        # remove o valor lido da mensagem
                        seqbits = seqbits[size:]

                        # teste para perceber se superamos o limite de runlenght do dicionario
                        if zero_run_loops > 0:
                            # se sim temos que levar em conta os zeros que "ficaram para tras"
                            ac.append((runlength+(15*zero_run_loops), value))
                            zero_run_loops = 0
                        else:
                            ac.append((runlength, value))
                    elif eob:
                        if zero_run_loops > 0:
                            zero_run_loops = 0
                        ac.append((0, 0))
                        bloco_dct_dpcm_zz.append(ac)
                        ac = []
                    else:
                        zero_run_loops += 1
                    break
        # print "AC = " + str(bloco_dct_dpcm_zz)
    print "done"
    return dc, bloco_dct_dpcm_zz, n_blocos


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


# converte a imagem num array de blocos 8x8
def create_8x8block(array):
    # resultado da divisao modulo 8 pelo comprimento do array
    mod8 = (array.shape[0] % 8) == 0 and (array.shape[1] % 8) == 0

    # Lista de blocos 8x8
    lista_blocos = []

    if not mod8:
        print "Dimensão do array não é multipla de 8"

    for i in xrange(0, array.shape[0], 8):
        for z in xrange(0, array.shape[1], 8):
            block = array[i:(i+8), z:(z+8)]
            lista_blocos.append(block.astype(np.float32))

    return lista_blocos


# efectua o processo inverso devolvendo a imagem a shape original
def revert_to_original_block(lista_blocos, original_shape):

    array_original = np.zeros(original_shape)

    count = 0

    for i in xrange(0, array_original.shape[0], 8):
        for z in xrange(0, array_original.shape[1], 8):
            array_original[i:(i+8), z:(z+8)] = lista_blocos[count]
            count += 1

    return array_original


def quality_factor(q_factor):
    if q_factor <= 50:
        factor = 50.0 / q_factor
    else:
        factor = 2.0 - (q_factor * 2.0) / 100.0
    return factor


# calcula o valor em binario com recurso ao inverso (complemento de uns) para binario negativo
def ones_complement(value, size):
    if value >= 0:
        return '{0:b}'.format(value)
    else:
        bit_lenght = "{" + "0:0{}b".format(str(size)) + "}"
        return bit_lenght.format(2**size - 1 - abs(value))


# efectua o procedimento inverso retornado o numero que corresponde ao binario
def read_ones_complement(bin_number):
    bin_number = list(bin_number)
    if bin_number[0] == "1":
        bin_number = ''.join(bin_number)
        return int(bin_number, 2)
    else:
        for i in xrange(len(bin_number)):
            if bin_number[i] == "0":
                bin_number[i] = "1"
            else:
                bin_number[i] = "0"
        bin_number = ''.join(bin_number)
        return -int(bin_number, 2)

# funções de leitura e escrita obtidas do trabalho anterior
"""
Elabore uma função ("escrever") que dada uma sequência de bits (mensagem codiﬁcada) e o nome do ﬁcheiro,
escreva a sequência de bits para o ﬁcheiro.
"""


def escrever(seqbits, nomeficheiro):

    # array de bytes que irá ser escrito para ficheiro
    array_bytes = bytearray()

    # assegura que o numero de bits é multiplo de 8 adicionando os bits necessarios
    # avalia o modulo da divisao por 8 para saber quantos bits estão "livres"
    n_bits_livres = len(seqbits) % 8

    if n_bits_livres != 0:
        # enche o resto do byte de 1s
        seqbits += '1' * (8 - n_bits_livres)

    # insere informação sobre a quantidade de bits de stuffing para permitir a sua remoçao na leitura
    seqbits += '{0:08b}'.format((8 - n_bits_livres))

    # converte os bits para bytes
    for i in range(len(seqbits) / 8):
        # segmento de 8 bits = 1 byte
        substring = seqbits[i * 8: i * 8 + 8]
        # adiciona o segmento ao array
        array_bytes.append(int(substring, base=2))

    # inicializa o ficheiro em modo de escrita
    f = open("{}".format(nomeficheiro), "wb")

    # escreve os bytes para ficheiro
    for byte in bytes(array_bytes):
        f.write(byte)

    # fecha o stream de escrita
    f.close()

    print "Foram escritos {} bits para ficheiro".format(len(seqbits))


"""
Elabore uma função ("ler") que dado o nome do ﬁcheiro, leia uma sequência de bits (mensagem codiﬁcada)
contida no ﬁcheiro.
"""


def ler(nomeficheiro):

    # Sequencia de bits com a codificação da mensagem
    seqbits = ""

    # with garante tratamento de exepções e close integrado
    with open("{}".format(nomeficheiro), "rb") as f:
        # le o byte
        byte = f.read(1)
        while byte:
            # adciona os bits correspondentes do byte à seq de bits
            seqbits += '{0:08b}'.format(ord(byte))
            byte = f.read(1)

    print "Foram lidos {} bits do ficheiro".format(len(seqbits))

    # verifica quantos bits foram utilizados para stuffing
    bits_stuffing = int(seqbits[-8:], 2)

    # remove o campo de informação sobre os bits de stuffing e esses bits
    seqbits = seqbits[:-8-bits_stuffing]

    print len(seqbits)

    return seqbits


# função auxiliar para calcular o SNR entre a imagem original e a comprimida
def calculoSNR(imgOrig, imgComp):
    PSinal = np.sum(imgComp**2.0)
    PRuido = np.sum((imgComp - imgOrig)**2.0)
    args = (PSinal/PRuido)
    return np.round(10.0*np.log10(args), 3)



def main():
    print "========================================================================================================"
    print "================================Analise Ficheiro lena================================" \
          "======="

    # variavel que controla o modo de impressao de dados de teste
    debug = True
    test_block = 1000

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

    x = cv2.imread("samples/lena.tiff", cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite("samples/lena_gray.jpeg", x.astype(np.uint8))

    lista_blocos = create_8x8block(x)

    bloco_dct = []

    t0 = time()

    # DCT e Quantificação
    for i in xrange(len(lista_blocos)):

        bloco = codificador(lista_blocos[i], k1, alfa)

        bloco_dct.append(bloco)

    t1 = time()
    print "O tempo necessário para efectuar a DCT e a Quantificação foi de {} segundos".format(round(t1 - t0, 3))

    # codificação parametro DC
    bloco_dct_dpcm, dc_cod = dpcm(bloco_dct)

    t2 = time()
    print "O tempo necessário para efectuar a codificação DC foi de {} segundos".format(round(t2 - t1, 3))

    if debug:
        print lista_blocos[test_block]
        print bloco_dct_dpcm[test_block]

    # codificacao ac
    bloco_dct_dpcm_zz = zig_zag(bloco_dct_dpcm, zigzag, debug, test_block)

    t3 = time()
    print "O tempo necessário para efectuar a codificação AC foi de {} segundos".format(round(t3 - t2, 3))

    # codificação huffman e escrita para ficheiro
    codifica_huff(bloco_dct_dpcm_zz, bloco_dct_dpcm, debug)

    t4 = time()
    print "O tempo necessário para o bloco de entropy coding (huffman) foi de {} segundos".format(round(t4 - t3, 3))

    # imprime imagem
    x_desc = revert_to_original_block(bloco_dct_dpcm, x.shape)
    # print snr(x, x_desc.astype(np.uint8))
    cv2.imshow("Lena cod alfa = 0", x_desc.astype(np.uint8))
    cv2.waitKey(0) & 0xFF

    # leitura do ficheiro e reconstrução do ac e dc
    dc, bloco_desc_dct_dpcm_zz, n_blocos = le_huff()

    if debug:
        print "o valor do DC descodificado é igual ao codificado = " + str(dc == dc_cod)

    t5 = time()
    print "O tempo necessário para a leitura do ficheiro e reconstrução do ac e dc foi de {} " \
          "segundos".format(round(t5 - t4, 3))

    # descodificacao ac
    bloco_desc_dct_dpcm = zag_zig(bloco_desc_dct_dpcm_zz, zigzag, debug, test_block)

    t6 = time()
    print "O tempo necessário para a descodificacao ac foi de {} segundos".format(round(t6 - t5, 3))

    # Descodificação parametro DC
    bloco_desc_dct = desc_dpcm(bloco_desc_dct_dpcm, dc)

    t7 = time()
    print "O tempo necessário para a descodificacao dc foi de {} segundos".format(round(t7 - t6, 3))

    if debug:
        print bloco_desc_dct_dpcm[test_block]
        print bloco_dct[test_block]
        print np.all(np.rint(bloco_dct_dpcm[test_block]) == np.rint(bloco_desc_dct[test_block]))

    bloco_rec = []

    # descodificação
    for i in xrange(n_blocos):

        bloco = descodificador(bloco_desc_dct[i], k1, alfa)

        bloco_rec.append(bloco)

    t8 = time()
    print "O tempo necessário para a dct inversa dc foi de {} segundos".format(round(t8 - t7, 3))

    if debug:
        print lista_blocos[test_block]
        print np.rint(bloco_rec[test_block])
        print lista_blocos[test_block]-bloco_rec[test_block]

    x_rec = revert_to_original_block(bloco_rec, x.shape)

    cv2.imshow("Lena descodificada alfa = {}".format(alfa), x_rec.astype(np.uint8))
    cv2.waitKey(0) & 0xFF

    cv2.imwrite("Lena descodificada alfa = {}".format(alfa), x_rec.astype(np.uint8))

    print "factor q = " + str(q)
    print "alfa = " + str(alfa)
    print "SNR = " + str(calculoSNR(x, x_rec))
    size_ini = path.getsize("samples/lena_gray.jpeg")
    size_end = path.getsize("lena_output.jpeg")
    print "A dimensão do ficheiro original é de {} Kb".format(round(size_ini / 1024., 2))
    print "A dimensão do ficheiro codificado é de {} Kb".format(round(size_end / 1024., 2))
    print "A taxa de compressão conseguida foi de {}".format(1. * size_ini / size_end)
    print "O saldo da compressão foi de {} Kb".format(round((size_ini - size_end) / 1024., 2))

    print "========================================================================================================"
    print "========================================================================================================"
    print "========================================================================================================"
    print
    print

main()
