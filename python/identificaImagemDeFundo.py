import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
import scipy.linalg

class IdentificaImagemDeFundo:
    def __init__(self, pathVideo="videos/video.mp4", pathImgFundo="images/img_de_fundo.jpg", resultado="videos/video_sem_fundo.mp4"):
        self.pathVideo = pathVideo
        self.pathImg = pathImgFundo
        self.pathResultado = resultado

#-----------------------------------------------------------------------------------------------------------------

    #265 segundos - boa qualidade de imagem - a qualidade da reoção que ta ruim
    def calcularFundoSVD(self, batch_size=50):  # Define um tamanho de lote padrão
        video = cv2.VideoCapture(self.pathVideo)
        frames = []
        while True:
            sucesso, imagem = video.read()
            if not sucesso:
                break
            frames.append(imagem)

        if len(frames) == 0:
            print("Não foi possível ler nenhum frame do vídeo.")
            return

        frames = np.array(frames)
        num_frames = frames.shape[0]
        print("Número de frames do vídeo:", num_frames)

        height, width, _ = frames[0].shape
        background = np.zeros((height, width, 3))

        def process_channel(j, batch_frames):
            channel_data = batch_frames[:, :, :, j].reshape(batch_frames.shape[0], -1).T
            print(f"Forma dos dados do canal {j}: {channel_data.shape}")
            u, s, v = np.linalg.svd(channel_data, full_matrices=False)
            Ar = np.dot(u[:, :1], np.dot(np.diag(s[:1]), v[:1, :]))
            background[:, :, j] = Ar.mean(axis=1).reshape(height, width)

        num_batches = num_frames // batch_size
        for i in range(num_batches):
            batch_frames = frames[i * batch_size: (i + 1) * batch_size]
            threads = []
            for j in range(3):
                thread = threading.Thread(target=process_channel, args=(j, batch_frames))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        remaining_frames = num_frames % batch_size
        if remaining_frames > 0:
            batch_frames = frames[num_batches * batch_size:]
            threads = []
            for j in range(3):
                thread = threading.Thread(target=process_channel, args=(j, batch_frames))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        cv2.imwrite(self.pathImg, background.astype(np.uint8))
        plt.imshow(background.astype(np.uint8))

#-----------------------------------------------------------------------------------------------------------------

    #8 segundos - qualidade de imagem boa - qualidade da remoção de fundo ruim
    def calcularFundoMedia(self):
        # Inicializar o vídeo
        video = cv2.VideoCapture(self.pathVideo)
        frames = []
        sucesso, imagem = video.read()
        while sucesso:
            # Coletar os frames do vídeo
            frames.append(imagem)
            sucesso, imagem = video.read()

        if len(frames) == 0:
            print("Não foi possível ler nenhum frame do vídeo.")
            return

        frames = np.array(frames)
        print("Número de frames do vídeo:", frames.shape[0])

        # Calcular o background pela média dos pixels de todos os frames
        background = np.mean(frames, axis=0).astype(np.uint8)

        # Salvar a imagem de fundo em um arquivo e exibi-la usando Matplotlib
        cv2.imwrite(self.pathImg, background)
        plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

#-----------------------------------------------------------------------------------------------------------------
    #334 segundos - qualidade de imagem boa - qualidade da remoção de fundo ruim
    def calcularFundoEliminacaoGaussiana(self):
        # Inicializar o vídeo
        video = cv2.VideoCapture(self.pathVideo)
        frames = []
        sucesso, imagem = video.read()

        # Coletar os frames do vídeo
        while sucesso:
            frames.append(imagem)
            sucesso, imagem = video.read()

        if len(frames) == 0:
            print("Não foi possível ler nenhum frame do vídeo.")
            return

        frames = np.array(frames)
        print("Número de frames do vídeo:", frames.shape[0])

        # Inicializar a matriz para armazenar os componentes da eliminação gaussiana de cada canal
        height, width, _ = frames[0].shape
        background = np.zeros((height, width, 3))

        # Função para processar cada canal de cor
        def process_channel(j):
            channel_data = frames[:, :, :, j].reshape(frames.shape[0], -1)
            print(f"Forma dos dados do canal {j}: {channel_data.shape}")

            # Aplicar eliminação gaussiana para encontrar a matriz triangular superior
            U, s, Vt = np.linalg.svd(channel_data, full_matrices=False)
            # Rearranjar os fatores L e U para formar uma matriz equivalente à SVD
            Ar = np.dot(U, Vt)
            background[:, :, j] = Ar.mean(axis=0).reshape(height, width)

        # Processar cada canal de cor em paralelo
        for j in range(3):
            process_channel(j)

        # Salvar a imagem de fundo em um arquivo e exibi-la usando Matplotlib
        cv2.imwrite(self.pathImg, background.astype(np.uint8))
        plt.imshow(background.astype(np.uint8))

#-----------------------------------------------------------------------------------------------------------------

    def removerFundo(self):
        # Carregar o vídeo
        video = cv2.VideoCapture(self.pathVideo)

        # Verificar se o vídeo foi carregado corretamente
        if not video.isOpened():
            print("Erro ao abrir o vídeo!")
            exit()

        # Variáveis para acumular frames
        contadorDeFrames = 0
        acumulador = None

        # Ler frames do vídeo para calcular o fundo
        while True:
            ret, frame = video.read()

            # Verificar se o frame foi lido corretamente
            if not ret:
                break
            
            # Aplicar filtro de suavização para reduzir o ruído
            frame = cv2.GaussianBlur(frame, (1, 1), 0)

            # Converter o frame para float64 para evitar overflow durante a soma
            frame = frame.astype(np.float64)

            # Acumular os frames para calcular a média
            if acumulador is None:
                acumulador = frame
            else:
                acumulador += frame
                contadorDeFrames += 1

        # Verificar se algum frame foi lido
        if contadorDeFrames == 0:
            print("Nenhum frame foi lido do vídeo.")
            exit()

        # Calcular a média dos frames acumulados para obter o fundo
        background = (acumulador / contadorDeFrames).astype(np.uint8)

        # Salvar a imagem de fundo
        cv2.imwrite(self.pathImg, background)

        # Recarregar a imagem de fundo
        background = cv2.imread(self.pathImg)

        # Verificar se a imagem de fundo foi carregada corretamente
        if background is None:
            print("Erro ao carregar a imagem de fundo!")
            exit()

        # Configurar o gravador de vídeo com as mesmas configurações do vídeo original
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(self.pathResultado, fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

        # Reiniciar a captura de vídeo para processar do início
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            # Ler um frame do vídeo
            ret, frame = video.read()
            if not ret:
                break

            # Aplicar filtro de suavização para reduzir o ruído
            frame = cv2.GaussianBlur(frame, (1, 1), 0)

            # Subtrair o fundo do frame do vídeo para detectar objetos em movimento
            frame_sem_fundo = cv2.absdiff(frame, background)

            # Converter o frame resultante para escala de cinza para simplificar a detecção
            frame_sem_fundo_gray = cv2.cvtColor(frame_sem_fundo, cv2.COLOR_BGR2GRAY)

            # Aplicar um limiar adaptativo para identificar áreas diferentes de zero na subtração
            _, thresholded = cv2.threshold(frame_sem_fundo_gray, 25, 255, cv2.THRESH_BINARY)

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
identificaImagem.calcularFundoSVD()
identificaImagem.removerFundo()