# api/main.py

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2 # ESTA LINHA DEVE ESTAR PRESENTE E SEM NENHUM COMENTÁRIO TIPO REMOVER!
import sys # Importado para print no stderr

# --- Configurações Globais ---
MODEL_PATH = 'modelo_digitos_mnist.tflite'
OUTPUT_FILE = 'resultado_predicao.txt'
DEFAULT_IMAGE_NAME = 'meus_digitos.png' # Nome que o app.py salva a imagem temporária

# --- Carregamento do Modelo TFLite ---
def carregar_modelo_tflite():
    # Caminho do modelo relativo ao main.py (que está em api/)
    model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_full_path):
        print(f"Erro: Modelo TFLite '{model_full_path}' não encontrado.", file=sys.stderr)
        # sys.exit(1) # Pode forçar a saída se o modelo for crítico
        return None
    print(f"--- Carregando modelo TFLite de: {model_full_path} ---", file=sys.stderr)
    try:
        # Use tflite_runtime.interpreter se TensorFlow não estiver instalado no ambiente
        # from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        # interpreter = TFLiteInterpreter(model_path=model_full_path)
        interpreter = tf.lite.Interpreter(model_path=model_full_path)
        interpreter.allocate_tensors()
        print("Modelo TFLite carregado com sucesso!", file=sys.stderr)
        return interpreter
    except Exception as e:
        print(f"Erro ao carregar modelo TFLite: {e}", file=sys.stderr)
        return None

# --- Função Auxiliar para Salvar Resultados ---
def salvar_resultado(texto):
    # O arquivo de resultado também é salvo na mesma pasta do main.py (api/)
    result_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    try:
        with open(result_filepath, 'w') as f:
            f.write(texto)
        print(f"Resultado salvo em: {result_filepath}", file=sys.stderr)
    except Exception as e:
        print(f"Erro ao salvar resultado em {result_filepath}: {e}", file=sys.stderr)

# --- Função para Encontrar Bounding Boxes de Dígitos ---
def find_digit_bounding_boxes(image_path, min_area=50, max_area=5000, aspect_ratio_min=0.2, aspect_ratio_max=2.0):
    try:
        # Imagem deve ser carregada da pasta 'api/' onde app.py a salvou
        if not os.path.exists(image_path):
            print(f"Erro: Imagem '{image_path}' não encontrada para detecção de bounding boxes.", file=sys.stderr)
            return [] # Retorna lista vazia se a imagem não existe

        img_pil = Image.open(image_path).convert('L') # Carrega e converte para tons de cinza (L mode)
        img_np = np.array(img_pil) # Converte para array NumPy para OpenCV

        # --- NOVA FORMA DE INVERTER CORES USANDO NUMPY ---
        # Inverte cores: 0 vira 255, 255 vira 0 (preto no branco -> branco no preto)
        img_np_inverted = 255 - img_np
        # --- FIM NOVA FORMA ---

        # --- DEBUG: INFORMAÇÕES DA IMAGEM ANTES DO THRESHOLD ---
        print(f"DEBUG main.py - ANTES DO THRESHOLD: dtype={img_np_inverted.dtype}, shape={img_np_inverted.shape}, "
              f"min_val={img_np_inverted.min()}, max_val={img_np_inverted.max()}", file=sys.stderr)
        # --- FIM DEBUG ---

        # Aplica a binarização. O valor 128 é um limiar inicial.
        # Experimente outros valores (ex: 80, 100, 150) se o fundo não ficar puro.
        # Agora usamos img_np_inverted que já tem as cores invertidas
        _, binary_img = cv2.threshold(img_np_inverted, 128, 255, cv2.THRESH_BINARY)

        # --- DEBUG: INFORMAÇÕES DA IMAGEM DEPOIS DO THRESHOLD ---
        print(f"DEBUG main.py - DEPOIS DO THRESHOLD: dtype={binary_img.dtype}, shape={binary_img.shape}, "
              f"min_val={binary_img.min()}, max_val={binary_img.max()}", file=sys.stderr)
        # --- FIM DEBUG ---

        # Salva a imagem binarizada para depuração (verifique em api/debug_binary_image.png)
        debug_filepath = os.path.join(os.path.dirname(__file__), 'debug_binary_image.png')
        cv2.imwrite(debug_filepath, binary_img)
        print(f"DEBUG main.py: Imagem binarizada salva para depuração: {debug_filepath}", file=sys.stderr)


        # Encontra contornos na imagem binarizada
        # cv2.RETR_EXTERNAL: Recupera apenas os contornos externos (evita buracos dentro dos dígitos)
        # cv2.CHAIN_APPROX_SIMPLE: Compacta pontos redundantes, economiza memória
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filtra contornos baseados na área para evitar ruído.
            # Estes valores podem precisar de ajuste dependendo do tamanho e clareza dos dígitos.
            # Tente min_area=20, max_area=10000 se os dígitos não forem detectados.
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filtro adicional para proporção (digitos tendem a ser mais quadrados)
                # Tente aspect_ratio_min=0.1, aspect_ratio_max=10.0 se os dígitos não forem detectados.
                aspect_ratio = float(w) / h
                if aspect_ratio_min < aspect_ratio < aspect_ratio_max:
                    bounding_boxes.append((x, y, w, h))

        # Ordena as bounding boxes da esquerda para a direita (útil para números com múltiplos dígitos)
        bounding_boxes.sort(key=lambda b: b[0])

        print(f"DEBUG main.py: {len(bounding_boxes)} bounding boxes detectadas.", file=sys.stderr)
        return bounding_boxes

    except Exception as e:
        print(f"Erro durante a detecção de bounding boxes: {e}", file=sys.stderr)
        return []


# --- Função Principal para Predição de Múltiplos Dígitos ---
def prever_multiplos_digitos_em_imagem(interpreter, caminho_imagem=DEFAULT_IMAGE_NAME):
    try:
        bounding_boxes = find_digit_bounding_boxes(caminho_imagem,
                                                   min_area=50, max_area=5000,
                                                   aspect_ratio_min=0.2, aspect_ratio_max=2.0)

        if not bounding_boxes:
            result_text = f"IMAGEM: '{caminho_imagem}' -> Não foi possível detectar nenhum dígito."
            print(result_text, file=sys.stderr)
            salvar_resultado(result_text)
            return

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2] # Geralmente (1, 28, 28, 1)

        original_img_pil = Image.open(caminho_imagem).convert('L') # Converte para tons de cinza
        original_img_np = np.array(original_img_pil) # Converte para NumPy array

        predicted_number_str = ""
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            digit_img_np_cropped = original_img_np[y:y+h, x:x+w]
            digit_img_np_inverted = 255 - digit_img_np_cropped # Inverte as cores
            digit_img_pil_inverted = Image.fromarray(digit_img_np_inverted)

            # --- INÍCIO DA NOVA LÓGICA DE REDIMENSIONAMENTO E CENTRALIZAÇÃO ---
            # Define o tamanho alvo para o dígito, deixando um padding de 4 pixels de cada lado no 28x28
            target_digit_size = 20 # Ex: 20x20 pixels dentro do 28x28

            # Calcula a nova largura e altura mantendo a proporção
            # O lado maior do dígito será redimensionado para 'target_digit_size'
            if w > h: # Se a largura for maior
                new_w = target_digit_size
                new_h = int(h * (target_digit_size / w))
            else: # Se a altura for maior ou igual
                new_h = target_digit_size
                new_w = int(w * (target_digit_size / h))

            # Redimensiona o dígito recortado mantendo a proporção
            digit_img_resized_prop = digit_img_pil_inverted.resize((new_w, new_h), Image.LANCZOS)

            # Cria um novo canvas preto 28x28
            final_canvas = Image.new('L', (input_width, input_height), color='black') # 'L' para tons de cinza, fundo preto

            # Calcula a posição para colar o dígito centralizado
            paste_x = (input_width - new_w) // 2
            paste_y = (input_height - new_h) // 2

            # Cola o dígito redimensionado no centro do canvas
            final_canvas.paste(digit_img_resized_prop, (paste_x, paste_y))

            # Converte o canvas final para array NumPy
            digit_img_for_prediction = np.array(final_canvas, dtype=np.float32)
            # --- FIM DA NOVA LÓGICA ---

            # Adiciona as dimensões necessárias para o modelo (1, 28, 28, 1)
            digit_img_for_prediction = digit_img_for_prediction[np.newaxis, :, :, np.newaxis]

            # Normaliza os pixels para o intervalo [0, 1]
            digit_img_for_prediction /= 255.0

            # Salva o dígito processado para depuração (MANTENHA ISSO POR ENQUANTO!)
           # debug_digit_path = os.path.join(os.path.dirname(__file__), f'debug_digit_{i}.png')
            # Precisamos salvar o array NumPy como imagem: converte de volta para 0-255 e uint8
           # Image.fromarray(np.uint8(digit_img_for_prediction[0,:,:,0] * 255)).save(debug_digit_path)
           # print(f"DEBUG main.py: Dígito processado {i} salvo para depuração: {debug_digit_path}", file=sys.stderr)


            # Realiza a predição
            interpreter.set_tensor(input_details[0]['index'], digit_img_for_prediction)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_digit = np.argmax(predictions[0])
            predicted_number_str += str(predicted_digit)
            print(f"DEBUG main.py: Dígito {i} previsto como: {predicted_digit}", file=sys.stderr)

        result_text = f"IMAGEM: '{caminho_imagem}' -> Número Previsto: {predicted_number_str}"
        print(result_text, file=sys.stderr)
        salvar_resultado(result_text)

    except Exception as e:
        print(f"Ocorreu um erro durante a predição de múltiplos dígitos: {e}", file=sys.stderr)
        salvar_resultado(f"IMAGEM: '{caminho_imagem}' -> Erro: {e}")



# --- Execução Principal ---
if __name__ == "__main__":
    tflite_interpreter = carregar_modelo_tflite()
    if tflite_interpreter:
        print("\n--- Modo de Análise Automático para Múltiplos Dígitos (TFLite) ---", file=sys.stderr)
        print(f"O programa tentará ler a imagem '{DEFAULT_IMAGE_NAME}'.", file=sys.stderr)
        print("Por favor, certifique-se de que a imagem esteja na mesma pasta do script.", file=sys.stderr)
        print("Desenhe os dígitos em um fundo claro (branco) e os números em cor escura (preto).", file=sys.stderr)
        print("Deixe um pequeno espaço entre os dígitos para uma melhor detecção.", file=sys.stderr)
        # Chama a função principal passando o interpreter e o nome da imagem padrão
        prever_multiplos_digitos_em_imagem(tflite_interpreter, DEFAULT_IMAGE_NAME)