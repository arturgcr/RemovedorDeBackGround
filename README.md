# Identificação e Remoção de Imagem de Fundo em Vídeos

## Introdução

A identificação e remoção de imagem de fundo em vídeos é um processo essencial em visão computacional, utilizado para destacar objetos em movimento. Existem três técnicas principais para realizar essa tarefa: Decomposição de Valores Singulares (SVD), média de frames e eliminação gaussiana. Cada uma dessas técnicas utiliza uma abordagem matemática distinta para calcular e remover o fundo, aprimorando a identificação dos objetos em movimento.

## Técnica 1: Decomposição de Valores Singulares (SVD)

A Decomposição de Valores Singulares (SVD) é uma técnica poderosa na álgebra linear, usada para decompor uma matriz em três outras matrizes:

\[ A = U \Sigma V^T \]

- **A**: Matriz original (número de pixels por número de frames)
- **U**: Matriz ortogonal (contém informações sobre os pixels)
- **\Sigma**: Matriz diagonal (contém os valores singulares ordenados)
- **V^T**: Matriz ortogonal (contém informações sobre os frames)

### Passos Abordados Na Função:

1. **Formatação dos Dados**: Cada canal de cor (R, G, B) é reorganizado em uma matriz de pixels por frames.
2. **Aplicação do SVD**: A decomposição SVD é aplicada à matriz de cada canal de cor.
3. **Reconstrução**: Utiliza-se apenas o primeiro valor singular (principal componente) para reconstruir a matriz, minimizando o ruído.
4. **Média**: Calcula-se a média dos valores reconstruídos para obter o fundo do canal de cor.

## Técnica 2: Média de Frames

A média de frames é uma técnica simples, mas eficaz, para calcular o fundo. Consiste em calcular a média de cada pixel ao longo de todos os frames.

### Fórmula:

\[ \text{Fundo}(i, j) = \frac{1}{N} \sum_{k=1}^{N} \text{Frame}_k(i, j) \]

- **N**: Número total de frames
- **i, j**: Coordenadas do pixel
- **Frame_k(i, j)**: Valor do pixel no frame \( k \)

### Técnica 3: Eliminação Gaussiana
A eliminação gaussiana é um método algébrico que transforma uma matriz em uma forma triangular superior. Este processo é usado para resolver sistemas de equações lineares e, neste contexto, para manipular os valores dos pixels de cada canal de cor de um vídeo, a fim de extrair o fundo estático da cena.

## Passos Abordados Na Função:
### Reformatação dos Dados:
Inicialmente, cada canal de cor (vermelho, verde e azul) é separado e reorganizado em uma matriz apropriada para processamento.Os frames do vídeo são lidos e armazenados em uma estrutura de dados que facilita a manipulação.

## Eliminação Gaussiana:
Em vez de aplicar diretamente a eliminação gaussiana tradicional, o código utiliza uma abordagem baseada na Decomposição em Valores Singulares (SVD).

Esta é uma técnica robusta que permite decompor a matriz de pixels em componentes que podem ser reorganizados.

A matriz de cada canal é decomposta em três matrizes: ( U ) (matriz ortogonal), ( s ) (vetor de valores singulares) e ( V^T ) (outra matriz ortogonal).

Utilizando ( U ) e ( V^T ), a matriz é reorganizada de modo que os elementos significativos sejam isolados, facilitando a média dos valores para obter uma representação do fundo estático.

### Rearranjo e Média:
Após a decomposição, os fatores resultantes são rearranjados e utilizados para formar uma matriz que se aproxima da matriz original.
A média dos valores ao longo do tempo é calculada para cada posição de pixel, resultando em uma imagem média que representa o fundo do vídeo.

## Aplicação: Remoção de Fundo

Após calcular o fundo utilizando uma das técnicas, subtrai-se esse fundo de cada frame do vídeo para destacar os objetos em movimento. Em seguida, utilizam-se operações de suavização e limiarização para refinar a detecção.

## Conclusão

Cada técnica utilizada tem abordagem matemática única para a identificação do fundo em vídeos. A SVD é eficaz na redução de ruído devido à sua capacidade de decomposição. A média de frames oferece uma solução simples e direta, enquanto a eliminação gaussiana, embora mais complexa, proporciona uma alternativa robusta para cenários específicos. Portanto, é possível que apesar dos resultados apresentados para os três métodos serem muito similares, o algoritmo mais eficiente será uma mistura, o melhor de cada abordagem, para gerar o vídeo mais preciso. Por isso, na próxima etapa desse trabalho abordarei formas de junção desses algoritmos junto com tecnicas de interpolação e regressão, assim como estava na proposta.