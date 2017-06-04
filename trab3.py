# This Python file uses the following encoding: utf-8

# Trabalho 3 CSM
# João Filipe Vaz - 40266
# João Ventura - 38950


import cv2
import numpy as np
import matplotlib.pyplot as plt
from Tables_jpeg import K3, K5

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
    # dc_original = [bloco_dct[i][0][0] for i in xrange(len(bloco_dct))]
    # dc_original[:] = [x - 128 for x in dc_original]

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


def zig_zag(bloco_dct_dpcm, zigzag, debug):

    zigzag_order = zigzag.ravel().astype(np.int8)

    # lista bidimensional com os ac de cada bloco
    bloco_dct_dpcm_zz = []
    # array temporario para guardar os valores ordenaados pela order do zigzag
    temp = np.zeros(64)

    for i in xrange(0, len(bloco_dct_dpcm)):
        # captura o primeiro bloco 8x8
        bloco_1d = bloco_dct_dpcm[i][:][:].ravel()

        # lista com os pares (zero run length, nonzero value)
        ac = []

        if i == 0 and debug:
            print bloco_1d

        for z in xrange(0, len(bloco_1d)):
            # guarda o valor no indice correspondente pela ordem do zigzag
            temp[zigzag_order[z]] = bloco_1d[z]

        # variavel auxiliar para contar o numero de zeros
        zeros = 0

        if i == 0 and debug:
            print temp

        for t in xrange(1, len(temp)):
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

        if i == 0 and debug:
            print ac

        bloco_dct_dpcm_zz.append(ac)

    return bloco_dct_dpcm_zz


def zag_zig(bloco_dct_dpcm_zz, zigzag, debug):

    zigzag_order = zigzag.ravel().astype(np.int8)

    # lista de output 8x8
    bloco_dct_dpcm = []

    for i in xrange(0, len(bloco_dct_dpcm_zz)):
        ac = bloco_dct_dpcm_zz[i]

        if i == 0 and debug:
            print ac

        temp = np.zeros(64)

        ultima_pos = 0

        for z in xrange(0, len(ac)):

            zeros = ac[z][0]
            value = ac[z][1]
            if value != 0:
                temp[zeros+1+ultima_pos] = value
                ultima_pos = zeros+1

        if i == 0 and debug:
            print temp
            print zigzag_order

        bloco_1d_ordenado = np.zeros(64)

        for t in xrange(1, len(temp)):
            if temp[t] != 0:
                # guarda o valor no indice correspondente pela ordem do zigzag
                bloco_1d_ordenado[np.where(zigzag_order == t)[0][0]] = temp[t]

        bloco_1d_ordenado = bloco_1d_ordenado.reshape((8, 8))

        if i == 0 and debug:
            print bloco_1d_ordenado

        bloco_dct_dpcm.append(bloco_1d_ordenado)

    return bloco_dct_dpcm


# 5

"""
Construa uma função que dados os arrays das alíneas anteriores use as tabelas do código de Huﬀman (tabela K3
e K5) e grave num ﬁcheiro a sequência de bits correspondente. (não é necessário usar o formato JFIF)
"""


def codifica_huff(bloco_dct_dpcm_zz, bloco_dct_dpcm):

    # stream de bits de saida
    bit_stream = ""

    # print bloco_dct_dpcm_zz[2]
    # print bloco_dct_dpcm[2]

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
        # amplitude é o valor em binario do componente dc
        amp = ones_complement(dc, size)
        # adiciona o valor directamente ao bitstream sem codificação de huffman
        bit_stream += amp  # + " "

        # analise da componente ac
        for z in xrange(len(bloco_dct_dpcm_zz[i])):
            # quantidade de 0s consecutivos
            runlength = bloco_dct_dpcm_zz[i][z][0]
            # valor do coeficiente nao nulo
            value = bloco_dct_dpcm_zz[i][z][1]

            if value != 0:
                # o valor é ainda subdividido em size e amp como no dc
                size = len('{0:b}'.format(abs(value)))
                amp = ones_complement(value, size)
            else:
                size = 0

            # o tuplo (runlength, size) é codificado recorrendo a tabela K5 com codigo de Huffman
            bit_stream += K5[(runlength, size)]  # + " "

            if value != 0:
                # o valor é codificado sem huffman
                bit_stream += amp  # + " "

    print bit_stream

    # utiliza a função desenvolvida no trab anterior para escrever para ficheiro
    escrever(bit_stream, "Lena_Cod.huf")


# 6

"""
Construa uma função que leia o ﬁcheiro gravado e retorne os arrays com os coeﬁcientes AC e DC.
"""


def le_huff():

    # lista com os coeficientes dc
    dc = []

    # lista com os coeficientes ac
    ac = []

    # Sequencia de bits com a codificação da mensagem
    seqbits = ler("Lena_Cod.huf")

    # flag end of block
    eob = False

    # lê os bits codificados enquanto houver dados para leitura
    while len(seqbits) != 0:
        # le o dc
        for k in K3:
            # avalia o prefixo inicial de acordo com a chave do dicionario
            if seqbits.startswith(K3[k]):
                # slice da mensagem de bits para lermos sempre a partir do inicio
                seqbits = seqbits[len(K3[k]):]
                print int(seqbits[0:k])
                # adiciona o valor a lista dc
                dc.append(int(seqbits[0:k], 2))
                # remove o valor lido da mensagem
                seqbits = seqbits[k:]
        while not eob:
            for y in K5:
                # avalia o prefixo inicial de acordo com a chave do dicionario
                if seqbits.startswith(K5[y]):
                    if K5[y] == "1010":
                        eob = True
                    print y
                    print K5[y]
                    # quando encontramos a chave correta adicionamos o valor ao array de simbolos
                    ac.append(y)
                    # e slice da mensagem de bits para lermos sempre a partir do inicio
                    seqbits = seqbits[len(y):]

    return dc, ac


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


def ones_complement(value, size):
    if value >= 0:
        return '{0:b}'.format(value)
    else:
        bit_lenght = "{" + "0:0{}b".format(str(size)) + "}"
        return bit_lenght.format(2**size - 1 - abs(value))


def read_ones_complement(bin_number):
    bin_number = list(bin_number)
    if bin_number[0] == "1":
        bin_number = str(bin_number)
        ''.join(bin_number)
        return int(bin_number, 2)
    else:
        for i in xrange(len(bin_number)):
            if bin_number[i] == "0":
                bin_number[i] = "1"
            else:
                bin_number[i] = "0"
        int(str(bin_number), 2)


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
    seqbits = seqbits[:-8 - bits_stuffing]

    return seqbits


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

    print bloco_dct_dpcm[0]

    # codificacao ac
    bloco_dct_dpcm_zz = zig_zag(bloco_dct_dpcm, zigzag, False)

    # codificação huffman e escrita para ficheiro
    codifica_huff(bloco_dct_dpcm_zz, bloco_dct_dpcm)


    # imprime imagem
    #x_desc = revert_to_original_block(bloco_dct_dpcm_zz, x.shape)
    #print snr(x, x_desc.astype(np.uint8))
    #cv2.imshow("Lena cod alfa = 0", x_desc.astype(np.uint8))
    #k = cv2.waitKey(0) & 0xFF

    # leitura do ficheiro e reconstrução do ac e dc
    dc, ac = le_huff()

    # descodificacao ac
    bloco_dct_dpcm = zag_zig(bloco_dct_dpcm_zz, zigzag, False)

    # Descodificação parametro DC
    bloco_dct = desc_dpcm(bloco_dct_dpcm)

    print bloco_dct_dpcm[0]

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

    #cv2.imwrite("lena_output.png", x_rec.astype(np.uint8))


    # for i in xrange(len(bloco)):
    # for z in xrange(len(bloco[i])):
    # (x_lena[0:64].reshape((8, 8)) / Q) * 50

    print "========================================================================================================"
    print "========================================================================================================"
    print "========================================================================================================"
    print
    print

main()