# api/app.py

# --- IMPORTS NECESSÁRIOS ---
import os
import subprocess
import sys
import io
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # Para permitir requisições de origens diferentes (se precisar)
from PIL import Image # Usado aqui para converter para PNG se for HEIC
import pillow_heif # Importa pillow_heif para dar suporte a arquivos HEIC/HEIF
from pillow_heif import register_heif_opener
register_heif_opener()
# --- CONFIGURAÇÕES DO APLICATIVO FLASK ---

# Define o caminho base para a pasta raiz do projeto ('seu_projeto/')
# Isso é crucial porque app.py está em 'api/' e index.html está em 'seu_projeto/'
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
Image.register_mime('image/heic', 'heif') # Isso pode ser necessário se for uma versão mais antiga do Pillow/Pillow-HEIF
# Inicializa o aplicativo Flask
# - template_folder: Diz ao Flask onde encontrar os arquivos de template (como index.html).
#   Aponta para a pasta 'seu_projeto/'.
# - static_folder: Opcional, para servir arquivos estáticos (CSS, JS) se não estiverem no template_folder.
#   Assume que há uma pasta 'static' dentro de 'seu_projeto/'.
app = Flask(__name__, template_folder=base_dir, static_folder=os.path.join(base_dir, 'static'))

# Habilita CORS para permitir requisições do frontend
CORS(app)

# Configura o limite máximo de tamanho de upload (em bytes)
# 50 MB como exemplo, ajuste conforme a necessidade.
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# --- CAMINHOS DE ARQUIVOS (Relativos à pasta 'api/', onde app.py está) ---
# UPLOAD_FOLDER aponta para a pasta 'api/' (onde app.py está e onde main.py espera a imagem)
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
TEMP_IMAGE_FILENAME = 'meus_digitos.png' # Nome fixo para a imagem temporária
RESULT_FILE_PATH = os.path.join(UPLOAD_FOLDER, 'resultado_predicao.txt') # Caminho para o arquivo de resultado
MAIN_SCRIPT_PATH = os.path.join(UPLOAD_FOLDER, 'main.py') # Caminho para o script main.py


# --- ROTA PARA A PÁGINA INICIAL ---
# Esta rota serve o arquivo index.html quando alguém acessa http://127.0.0.1:5000/
@app.route('/')
def index():
    return render_template('index.html')

# --- ROTA DE PREDIÇÃO DA IMAGEM ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("DEBUG app.py: Nenhuma chave 'image' no request.files", file=sys.stderr)
        return jsonify({'prediction': 'Erro: Nenhuma imagem enviada.'}), 400

    file = request.files['file']
    if file.filename == '':
        print("DEBUG app.py: Nenhum arquivo selecionado.", file=sys.stderr)
        return jsonify({'prediction': 'Erro: Nenhum arquivo selecionado.'}), 400

    print(f"DEBUG app.py: Recebeu arquivo: {file.filename} (MIME: {file.mimetype})", file=sys.stderr)

    # Define o caminho completo onde a imagem temporária será salva
    filepath_to_save = os.path.join(UPLOAD_FOLDER, TEMP_IMAGE_FILENAME)

    try:
        # Tenta abrir o arquivo diretamente com Pillow.
        # Pillow com pillow_heif instalado deve ser capaz de abrir HEIC, JPG, PNG etc.
        file_stream = io.BytesIO(file.read())
        img = Image.open(file_stream) # Pillow agora deve lidar com HEIC, JPG, PNG automaticamente

        # Garante que a imagem é convertida para PNG e salva
        print(f"DEBUG app.py: Salvando imagem original como PNG: {filepath_to_save}", file=sys.stderr)
        img.save(filepath_to_save, format="PNG")

        # ... (o resto do código, como a confirmação de que o arquivo existe)
        print(f"DEBUG app.py: Arquivo salvo com sucesso em: {filepath_to_save}", file=sys.stderr)
        if os.path.exists(filepath_to_save):
            print(f"DEBUG app.py: Confirmação: O arquivo existe no disco após salvar. Tamanho: {os.path.getsize(filepath_to_save)} bytes", file=sys.stderr)
        else:
            print(f"DEBUG app.py: ERRO CRÍTICO: Arquivo NÃO EXISTE no disco após file.save()!", file=sys.stderr)
            return jsonify({'prediction': 'Erro interno: Falha ao salvar imagem.'}), 500


        # Chama o main.py como um subprocesso
        # Chama o main.py como um subprocesso
        print(f"DEBUG app.py: Executando main.py em: {MAIN_SCRIPT_PATH} com CWD: {UPLOAD_FOLDER}", file=sys.stderr)
        process_result = subprocess.run(
            [sys.executable, MAIN_SCRIPT_PATH], # Usa sys.executable para garantir o python correto
            capture_output=True,
            text=True,
            timeout=30,
            cwd=UPLOAD_FOLDER # <--- ADICIONE ESTA LINHA! Define o diretório de trabalho para main.py
        )

        # Imprime a saída do main.py (stdout e stderr) para depuração
        if process_result.stdout:
            print(f"DEBUG app.py - Saída do main.py (stdout):\n{process_result.stdout}", file=sys.stderr)
        if process_result.stderr:
            print(f"DEBUG app.py - Saída do main.py (stderr):\n{process_result.stderr}", file=sys.stderr)

        # Tenta ler o resultado do arquivo gerado por main.py
        if os.path.exists(RESULT_FILE_PATH):
            with open(RESULT_FILE_PATH, 'r') as f:
                prediction_result = f.read().strip()
            print(f"DEBUG app.py: Resultado lido de {RESULT_FILE_PATH}: {prediction_result}", file=sys.stderr)

            # Analisa o resultado para ver se houve erro ou predição
            if "Erro:" in prediction_result or "Não foi possível detectar" in prediction_result:
                 return jsonify({'success': False, 'message': prediction_result}), 500
            else:
                return jsonify({'success': True, 'prediction': prediction_result}) # Retorna 200 OK
        else:
            print(f"DEBUG app.py: Arquivo de resultado '{RESULT_FILE_PATH}' não encontrado.", file=sys.stderr)
            return jsonify({'prediction': 'Erro: Resultado da predição não disponível.'}), 500

    except subprocess.TimeoutExpired:
        print("DEBUG app.py: Timeout ao executar main.py", file=sys.stderr)
        return jsonify({'success': False, 'message': 'Tempo limite excedido ao processar a imagem.'}), 504
    except Exception as e:
        print(f"DEBUG app.py: Erro inesperado no Flask durante o processamento: {e}", file=sys.stderr)
        return jsonify({'success': False, 'message': f'Erro interno do servidor: {e}'}), 500

# --- INÍCIO DA EXECUÇÃO DO APLICATIVO ---
if __name__ == '__main__':
    # Adicionado io para o tratamento HEIC
    import io
    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' permite acesso de outros dispositivos na rede, '127.0.0.1' é só local