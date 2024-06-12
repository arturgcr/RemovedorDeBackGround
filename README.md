# Documentação do BackgroundRemover

## Visão Geral
A classe `BackgroundRemover` foi projetada para remover o fundo de um vídeo usando uma imagem estática de fundo. Este processo envolve interpolação de quadros, diferenciação de imagens, limiarização e operações morfológicas. Abaixo, explicamos a lógica matemática e as fórmulas por trás de cada etapa.

## Método: `remove_background`

### Parâmetros
- `background_img_path` (str): Caminho para a imagem de fundo.
- `video_path` (str): Caminho para o vídeo de entrada.
- `result_path` (str): Caminho para salvar o vídeo de saída.
- `threshold_value` (int): Valor para a limiarização para segmentar o primeiro plano.
- `morph_kernel_size` (tuple): Tamanho do kernel morfológico.
- `min_size` (int): Tamanho mínimo dos componentes conectados a serem mantidos.
- `alpha` (float): Fator de ponderação para a interpolação de quadros.

### Etapas e Formulação Matemática

1. **Inicialização e Verificação de Entrada**
    - Verifique a existência dos arquivos da imagem de fundo e do vídeo.
    - Leia a imagem de fundo e o vídeo de entrada.
    - Redimensione a imagem de fundo para corresponder às dimensões dos quadros do vídeo.

2. **Loop de Processamento do Vídeo**
    - Inicialize variáveis e configure o escritor de vídeo de saída.
    - Faça um loop através de cada quadro do vídeo até o final.

3. **Interpolação de Quadros**
    - Se um quadro anterior existir, interpole entre o quadro anterior e o quadro atual:
      \[
      \text{interpolated\_frame} = \alpha \cdot \text{prev\_frame} + (1 - \alpha) \cdot \text{frame}
      \]
    - Se não houver quadro anterior, use diretamente o quadro atual.

4. **Subtração de Fundo**
    - Calcule a diferença absoluta entre o quadro interpolado e a imagem de fundo redimensionada:
      \[
      \text{frame\_no\_background} = |\text{interpolated\_frame} - \text{background\_resized}|
      \]
    - Converta a imagem resultante para escala de cinza.

5. **Limiarização**
    - Aplique limiarização binária na imagem em escala de cinza:
      \[
      \text{thresholded} = 
      \begin{cases} 
      255 & \text{se } \text{frame\_no\_background\_gray} > \text{threshold\_value} \\
      0 & \text{caso contrário}
      \end{cases}
      \]

6. **Operações Morfológicas**
    - Crie um kernel morfológico (elíptico):
      \[
      \text{kernel} = \text{cv2.getStructuringElement(cv2.MORPH\_ELLIPSE, morph\_kernel\_size)}
      \]
    - Aplique fechamento morfológico (dilatação seguida de erosão) para remover pequenos buracos:
      \[
      \text{thresholded} = \text{cv2.morphologyEx(thresholded, cv2.MORPH\_CLOSE, kernel)}
      \]
    - Aplique abertura morfológica (erosão seguida de dilatação) para remover pequenos ruídos:
      \[
      \text{thresholded} = \text{cv2.morphologyEx(thresholded, cv2.MORPH\_OPEN, kernel)}
      \]

7. **Análise de Componentes Conectados**
    - Identifique componentes conectados na imagem limiarizada:
      \[
      \text{num\_labels}, \text{labels}, \text{stats}, \text{centroids} = \text{cv2.connectedComponentsWithStats(thresholded, connectivity=8)}
      \]
    - Extraia os tamanhos dos componentes conectados:
      \[
      \text{sizes} = \text{stats}[1:, -1]
      \]

8. **Filtragem de Componentes Pequenos**
    - Crie uma máscara para manter apenas componentes maiores que o tamanho mínimo especificado:
      \[
      \text{mask} = np.zeros(\text{labels.shape}, np.uint8)
      \]
    - Faça um loop através dos componentes conectados e atualize a máscara:
      \[
      \text{if sizes[j]} \geq \text{min\_size} \implies \text{mask[labels == j + 1]} = 255
      \]

9. **Aplicar Máscara ao Quadro Original**
    - Combine a máscara com o quadro original para isolar o primeiro plano:
      \[
      \text{frame\_final} = \text{cv2.bitwise_and(frame, frame, mask=mask)}
      \]

10. **Escrever Quadro Processado no Vídeo de Saída**
    - Escreva o quadro final no vídeo de saída.

11. **Liberar Recursos**
    - Libere os objetos de vídeo e vídeo de saída e destrua todas as janelas do OpenCV.
