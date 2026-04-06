import streamlit as st
import os
import shutil
from engine import get_candidate_faces, run_extraction

# 1. Configuração de Página
st.set_page_config(page_title="Extrator de Faces", layout="wide")

# FUNÇÃO PARA LIMPAR TUDO (Garante que o ArcFace não brigue com lixo do Facenet)
def reset_total():
    # Limpa as variáveis de controle da interface
    st.session_state.target_emb = None
    st.session_state.imgs = None
    st.session_state.embs = None
    # Opcional: deletar o arquivo temporário se quiser economizar espaço
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")

st.title("🎯 Extrator de Faces (ArcFace Edition)")


# Upload do vídeo
video_file = st.file_uploader(
    "Suba seu vídeo (MP4, MOV, AVI)", 
    type=['mp4', 'mov', 'avi'],
    on_change=reset_total 
)

if video_file:
    video_path = "temp_video.mp4"
    
    # Se o imgs é None, significa que acabamos de subir o arquivo ou resetamos
    if st.session_state.get('imgs') is None:
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        # Garante que as pastas de saída também sumam
        if os.path.exists("extraidos"):
            import shutil
            shutil.rmtree("extraidos")

    # --- PASSO 1: BUSCAR ROSTOS ---
    if st.session_state.target_emb is None and st.session_state.imgs is None:
        if st.button("🔍 1. Analisar Vídeo e Buscar Candidatos"):
            with st.spinner("O motor ArcFace está aquecendo..."):
                imgs, embs = get_candidate_faces(video_path)
                if imgs:
                    st.session_state.imgs = imgs
                    st.session_state.embs = embs                    
                else:
                    st.error("Nenhuma face encontrada. Verifique a iluminação nos primeiros 15s.")

    # --- PASSO 2: MOSTRAR GALERIA ---
    if st.session_state.imgs and st.session_state.target_emb is None:
        st.divider()
        st.write("### Escolha quem você deseja extrair:")
        
        # Definimos um grid fixo de 6 colunas para as fotos ficarem pequenas
        num_cols = 6 
        for i in range(0, len(st.session_state.imgs), num_cols):
            cols = st.columns(num_cols)
            # Pegamos o lote de candidatos para esta linha
            lote = st.session_state.imgs[i : i + num_cols]
            
            for j, img in enumerate(lote):
                idx_real = i + j
            with cols[j]:
                st.image(img, channels="BGR", width='stretch')
                if st.button(f"Extrair este 👤", key=f"btn_sel_{idx_real}"):
                    st.session_state.target_emb = st.session_state.embs[idx_real]
                    st.rerun()

    # --- PASSO 3: CONFIGURAÇÃO E EXTRAÇÃO TOTAL ---
    if st.session_state.target_emb is not None:
        st.success("🎯 Candidato alvo selecionado!")
        
        st.write("### ⚙️ Configuração de Captura")
        modo_escolhido = st.select_slider(
            "Selecione o detalhamento (FPS):",
            options=["Velocidade", "Equilíbrio", "Máxima Extração"],
            value="Equilíbrio"
        )
        
        mapa_divisor = {"Velocidade": 0.5, "Equilíbrio": 3, "Máxima Extração": 5}
        divisor = mapa_divisor[modo_escolhido]

        c1, c2, _ = st.columns([2, 2, 6])
        with c1:
            disparar = st.button("🚀 INICIAR EXTRAÇÃO", key="btn_run_total")
        with c2:
            if st.button("🔄 Trocar Candidato"):
                st.session_state.target_emb = None
                st.rerun()

        status_texto = st.empty() 
        barra_progresso = st.progress(0)
        
        if disparar:
            # Garante que a pasta de saída esteja limpa
            if os.path.exists("output"): shutil.rmtree("output")
            os.makedirs("output")

            fotos_salvas = run_extraction(
                video_path, 
                st.session_state.target_emb, 
                "output", 
                barra_progresso,
                status_texto,
                divisor 
            )
            
            if fotos_salvas:
                st.success(f"✅ Concluído! {len(fotos_salvas)} fotos salvas na pasta /output.")
                st.balloons()
            else:
                st.error("❌ Alvo não encontrado com os parâmetros atuais.")