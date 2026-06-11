# treinar_svm.py
import os
import cv2 as cv
import numpy as np
import joblib
from sklearn.svm import SVC
from classificador_svm import ClassificadorParametrosSVM

def treinar_classificador_rota(diretorio_imagens):
    svm_aux = ClassificadorParametrosSVM()
    X = []
    valores_gradiente = []

    # Filtragem e ordenação dos frames
    arquivos = sorted([os.path.join(diretorio_imagens, f) for f in os.listdir(diretorio_imagens) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not arquivos:
        print("Dados insuficientes para precisão: Nenhuma imagem encontrada.")
        return

    print(f"Extraindo características de {len(arquivos)} frames da rota...")

    # Primeiro passo: Extração das features e isolamento do indicador de textura
    for caminho_img in arquivos:
        img = cv.imread(caminho_img, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        caracteristicas = svm_aux.extrair_atributos_imagem(img).flatten()
        X.append(caracteristicas)
        valores_gradiente.append(caracteristicas[2])  # Índice 2 corresponde à média_gradiente

    if len(valores_gradiente) < 2:
        print("Dados insuficientes para precisão: Imagens válidas insuficientes.")
        return

    X = np.array(X)
    
    # Segundo passo: Definição do limiar adaptativo usando a mediana da própria rota
    limiar_mediana = np.median(valores_gradiente)
    
    # Rotulação balanceada: Garante a existência de ambas as classes (0 e 1)
    y = np.where(np.array(valores_gradiente) < limiar_mediana, 0, 1)

    # Validação estrita de segurança para o SVC
    if len(np.unique(y)) < 2:
        print("Erro: Variabilidade de textura insuficiente na rota para treinar o SVM. Forçando classes artificiais.")
        y[0] = 0
        y[-1] = 1

    # Treinamento do modelo
    modelo_final = SVC(kernel='rbf', C=1.0)
    modelo_final.fit(X, y)

    # Salvamento do arquivo de parametrização exigido pelo pipeline
    joblib.dump(modelo_final, 'modelo_treinado_svm.pkl')
    print(f"Módulo SVM treinado com limiar adaptativo (Mediana: {limiar_mediana:.2f}) e salvo com sucesso.")

if __name__ == "__main__":
    caminho_da_sua_rota = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\PMG\Google earth\1500_05_QUADRADO\Resized"
    treinar_classificador_rota(caminho_da_sua_rota)