import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading

import scipy.linalg

def lu_decomposition(A):
    lu, piv = scipy.linalg.lu_factor(A)
    return lu

class IdentificaImagemDeFundo:
    def __init__(self, pathVideo="videos/video.mp4", pathImgFundo="images/img_de_fundo.jpg", resultado="videos/video_sem_fundo.mp4"):
        self.pathVideo = pathVideo
        self.pathImg = pathImgFundo
        self.pathResultado = resultado

#-----------------------------------------------------------------------------------------------------------------

    def calcularFundoSVD(self, lote_de_imagens=50):
        video = cv2.VideoCapture(self.pathVideo)
        frames = []
        while True:
            sucesso, imagem = video.read()
            if not sucesso:
                break
            frames.append(imagem)

        frames = np.array(frames)
        num_frames = frames.shape[0]

        altura, largura, _ = frames[0].shape
        background = np.zeros((altura, largura, 3))

        def processando_canal(j, batch_frames):
            dados_do_canal = batch_frames[:, :, :, j].reshape(batch_frames.shape[0], -1).T
            u, s, v = np.linalg.svd(dados_do_canal, full_matrices=False)
            Ar = np.dot(u[:, :1], np.dot(np.diag(s[:1]), v[:1, :]))
            background[:, :, j] = Ar.mean(axis=1).reshape(altura, largura)

        for i in range(num_frames // lote_de_imagens):
            batch_frames = frames[i * lote_de_imagens: (i + 1) * lote_de_imagens]
            threads = []
            for j in range(3):
                thread = threading.Thread(target=processando_canal, args=(j, batch_frames))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        frames_faltando = num_frames % lote_de_imagens
        if frames_faltando > 0:
            batch_frames = frames[(num_frames // lote_de_imagens) * lote_de_imagens:]
            threads = []
            for j in range(3):
                thread = threading.Thread(target=processando_canal, args=(j, batch_frames))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        cv2.imwrite(self.pathImg, background.astype(np.uint8))
        plt.imshow(background.astype(np.uint8))

#-----------------------------------------------------------------------------------------------------------------

    def calcularFundoMedia(self):
        # Inicializar o vídeo
        video = cv2.VideoCapture(self.pathVideo)
        frames = []
        sucesso, imagem = video.read()
        while sucesso:
            # Coletar os frames do vídeo
            frames.append(imagem)
            sucesso, imagem = video.read()

        frames = np.array(frames)

        # Calcular o background pela média dos pixels de todos os frames
        background = np.mean(frames, axis=0).astype(np.uint8)

        # Salvar a imagem de fundo em um arquivo e exibi-la usando Matplotlib
        cv2.imwrite(self.pathImg, background)
        plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

#-----------------------------------------------------------------------------------------------------------------

    def calcularFundoEliminacaoGaussiana(self, lote_de_imagens=50):
        video = cv2.VideoCapture(self.pathVideo)
        frames = []
        while True:
            sucesso, imagem = video.read()
            if not sucesso:
                break
            frames.append(imagem)

        frames = np.array(frames)
        num_frames = frames.shape[0]

        altura, largura, _ = frames[0].shape
        background = np.zeros((altura, largura, 3))

        def processar_canal(j, batch_frames):
            for frame in batch_frames:
                A = frame[:, :, j].reshape(-1, 1)  # Definir a forma correta da matriz A
                b = np.mean(frame[:, :, j]) * np.ones(A.shape[0])  # Criar o vetor b
                if A.shape[0] == A.shape[1]:
                    # Aplicar a eliminação gaussiana
                    x = np.linalg.solve(A, b)
                    background[:, :, j] = x.reshape(altura, largura)
                else:
                    print("Matriz A não é quadrada, não é possível resolver o sistema.")

        for i in range(num_frames // lote_de_imagens):
            batch_frames = frames[i * lote_de_imagens: (i + 1) * lote_de_imagens]
            threads = []
            for j in range(3):
                thread = threading.Thread(target=processar_canal, args=(j, batch_frames))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        frames_faltando = num_frames % lote_de_imagens
        if frames_faltando > 0:
            batch_frames = frames[(num_frames // lote_de_imagens) * lote_de_imagens:]
            threads = []
            for j in range(3):
                thread = threading.Thread(target=processar_canal, args=(j, batch_frames))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        cv2.imwrite(self.pathImg, background.astype(np.uint8))
        cv2.imshow('Background', background.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#-----------------------------------------------------------------------------------------------------------------

    def removerFundo(self):
        # Carregar o vídeo
        video = cv2.VideoCapture(self.pathVideo)
        # Carregar a imagem de fundo
        background = cv2.imread(self.pathImg)

        # Configurar o gravador de vídeo com as mesmas configurações do vídeo original
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(self.pathResultado, fourcc, video.get(cv2.CAP_PROP_FPS),
        (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:
            # Ler um frame do vídeo
            verificado, frame = video.read()
            if not verificado:
                break

            # Subtrair o fundo do frame do vídeo para detectar objetos em movimento
            frame_sem_fundo = cv2.absdiff(frame, background)

            # Converter o frame resultante para escala de cinza para simplificar a detecção
            frame_sem_fundo_cinza = cv2.cvtColor(frame_sem_fundo, cv2.COLOR_BGR2GRAY)

            # Aplicar um limiar adaptativo para identificar áreas diferentes de zero na subtração
            _, thresholded = cv2.threshold(frame_sem_fundo_cinza, 40, 255, cv2.THRESH_BINARY)

            # Utilizar a função bitwise_and para manter a parte original do frame onde não há movimento
            frame_final = cv2.bitwise_and(frame, frame, mask=thresholded)

            # Escrever o frame processado no novo vídeo
            output_video.write(frame_final)

        # Liberar recursos
        video.release()
        output_video.release()
        cv2.destroyAllWindows()
        print("Terminou!")

#-----------------------------------------------------------------------------------------------------------------

identificaImagem = IdentificaImagemDeFundo()
identificaImagem.calcularFundoEliminacaoGaussiana()
identificaImagem.removerFundo()