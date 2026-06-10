import cv2
import numpy as np
from sklearn.svm import SVC
import joblib

class ClassificadorParametrosSVM:
    def __init__(self, caminho_modelo=None):
        """
        Gerencia o modelo SVM para predição dinâmica de hiperparâmetros.
        """
        if caminho_modelo:
            self.model = joblib.load(caminho_modelo)
        else:
            self.model = SVC(kernel='rbf', C=1.0)

    def extrair_atributos_imagem(self, imagem_cinza):
        """
        Extrai vetor de características baseado em textura, brilho e contraste
        conforme a metodologia da dissertação.
        """
        # 1. Estatísticas globais de intensidade
        media = np.mean(imagem_cinza)
        variancia = np.var(imagem_cinza)
        
        # 2. Gradiente (Sobel) para avaliar nível de textura/bordas
        sobelx = cv2.Sobel(imagem_cinza, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(imagem_cinza, cv2.CV_64F, 0, 1, ksize=3)
        magnitude_gradiente = np.sqrt(sobelx**2 + sobely**2)
        media_gradiente = np.mean(magnitude_gradiente)
        
        # Retorna o vetor de características plano (1D)
        return np.array([media, variancia, media_gradiente]).reshape(1, -1)

    def predizer_classe_cenario(self, imagem_cinza):
        """
        Retorna a classe do cenário predita pelo SVM.
        """
        caracteristicas = self.extrair_atributos_imagem(imagem_cinza)
        # Retorna a classe (ex: 0 para Baixo Contraste, 1 para Alto Contraste/Textura)
        return int(self.model.predict(caracteristicas)[0])

    def mapear_classe_para_parametros(self, classe_predita):
        """
        Mapeia a saída do SVM para dicionários específicos de thresholds.
        """
        if classe_predita == 0:  # Ex: Solo homogêneo / Pouca textura
            return {
                'akaze_threshold': 0.00005,  # Menor threshold para capturar mais pontos
                'orb_nfeatures': 2000
            }
        else:  # Ex: Área urbana / Muita textura
            return {
                'akaze_threshold': 0.001,    # Maior threshold para filtrar ruído
                'orb_nfeatures': 1000
            }