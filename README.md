# 🎯 Face Extractor

Aplicação para identificação e extração automática de faces de vídeos de vigilância. Dado um vídeo, o sistema detecta todos os rostos presentes, permite que o usuário selecione o alvo e extrai automaticamente todas as aparições dessa pessoa ao longo do vídeo.

---

## Como funciona

O pipeline opera em três etapas:

**1. Detecção de candidatos**
O MTCNN escaneia os primeiros 15 segundos do vídeo em resolução reduzida (360p) para máxima velocidade. Cada rosto único encontrado é apresentado em uma galeria para seleção.

**2. Seleção do alvo**
O usuário escolhe qual pessoa deseja extrair. O embedding ArcFace desse rosto se torna a identidade de referência para toda a extração.

**3. Extração completa**
O vídeo inteiro é varrido frame a frame. Para cada rosto detectado, o sistema calcula a distância coseno entre o embedding ArcFace e a identidade de referência. Rostos abaixo do limiar são salvos; duplicatas são eliminadas por comparação de embeddings.

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| Interface | Streamlit |
| Detecção facial | MTCNN (via DeepFace) |
| Reconhecimento | ArcFace (via DeepFace) |
| Similaridade | Distância coseno (SciPy) |
| Processamento de imagem | OpenCV, Pillow |

---

## Instalação

**Pré-requisitos:** Python 3.9+

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/face-extractor.git
cd face-extractor

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Instale as dependências
pip install -r requirements.txt
```

Na primeira execução, o DeepFace baixará automaticamente os pesos do modelo ArcFace (~500MB). Isso ocorre uma única vez.

---

## Uso

```bash
streamlit run app.py
```

1. Faça upload do vídeo (MP4, MOV ou AVI)
2. Clique em **Analisar Vídeo** para detectar os candidatos
3. Selecione a pessoa que deseja extrair na galeria
4. Escolha o modo de captura e clique em **Iniciar Extração**
5. As imagens são salvas na pasta `output/`

**Dica:** Para acelerar o processo, edite o vídeo para que a pessoa de interesse apareça já nos primeiros 15 segundos. Isso garante que ela seja incluída na galeria de candidatos.

---

## Modos de captura

| Modo | FPS analisados | Indicado para |
|---|---|---|
| Velocidade | 0.5 fps | Vídeos longos, resultado rápido |
| Equilíbrio | 3 fps | Uso geral |
| Máxima Extração | 5 fps | Vídeos curtos, maior cobertura |

---

## Parâmetros de configuração

Os principais parâmetros estão no topo de `engine.py`:

```python
# Limiar de reconhecimento — distância coseno ArcFace
# Abaixe para ser mais restrito (menos falsos positivos)
# Suba para ser mais permissivo (menos falsos negativos)
if dist_ia >= 0.45: continue

# Limiar de deduplicação por embedding
# 0.10 → salva mais variações de pose
# 0.20 → elimina mais duplicatas
MIN_DIST_DUPLICATA = 0.15

# Raio de rastreamento por posição (pixels no frame 360p)
raio_ajustado = 200
```

---

## Estrutura do projeto

```
face-extractor/
├── app.py          # Interface Streamlit
├── engine.py       # Pipeline de detecção e extração
├── requirements.txt
├── .gitignore
├── output/         # Imagens extraídas (gerado automaticamente)
└── .face_cache/    # Cache de embeddings (gerado automaticamente)
```

---

## requirements.txt

```
streamlit
opencv-python
deepface
scipy
pillow
numpy
tf-keras
```

---

## Limitações conhecidas

- **Óculos escuros:** o filtro de pose frontal é desativado quando o MTCNN não detecta landmarks oculares, pois óculos escuros remove o principal sinal de estimativa de pose.
- **Baixa resolução:** faces menores que 30×30 pixels no frame original são descartadas.
- **Múltiplas pessoas próximas:** o filtro de rastreamento por posição pode rejeitar o alvo em cenas com movimentação rápida. Nesse caso, aumente `raio_ajustado`.
- **Processamento:** o pipeline roda inteiramente em CPU. Vídeos longos em modo de Máxima Extração podem levar vários minutos.

---

## Licença

MIT