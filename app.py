import streamlit as st
import os
import io
import zipfile
from engine import get_candidate_faces, run_extraction

# -------------------------------------------------------------------
# Configuração de Página
# -------------------------------------------------------------------
st.set_page_config(page_title="Extrator de Faces", layout="wide")

st.title("🎯 Extrator de Faces")
st.info("""
Este app identifica faces em vídeos e extrai apenas a pessoa escolhida.

**Dica:** Para acelerar o processo, edite o vídeo para conter apenas o período em que a
pessoa de interesse aparece, garantindo que ela esteja presente já nos primeiros 15 segundos.
""")


def reset_state():
    st.session_state.imgs = None
    st.session_state.embs = None
    st.session_state.target_emb = None


def gerar_zip(output_dir: str) -> bytes:
    """Empacota todas as fotos da pasta output em um ZIP em memória."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            zf.write(fpath, arcname=fname)
    buf.seek(0)
    return buf.read()


# -------------------------------------------------------------------
# Upload do vídeo
# -------------------------------------------------------------------
video_file = st.file_uploader(
    "Suba seu vídeo (MP4, MOV, AVI)",
    type=["mp4", "mov", "avi"],
    on_change=reset_state,
)

if video_file:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    for key in ("imgs", "embs", "target_emb"):
        if key not in st.session_state:
            st.session_state[key] = None

    # -------------------------------------------------------------------
    # PASSO 1: BUSCAR ROSTOS
    # -------------------------------------------------------------------
    if st.session_state.target_emb is None:
        if st.button("🔍 1. Analisar Vídeo e Buscar Candidatos", key="btn_analisar"):
            with st.spinner("O motor MTCNN está escaneando os primeiros segundos..."):
                imgs, embs = get_candidate_faces(video_path)
                if imgs:
                    st.session_state.imgs = imgs
                    st.session_state.embs = embs
                    st.rerun()
                else:
                    st.error("Nenhuma face encontrada no início do vídeo.")

    # -------------------------------------------------------------------
    # PASSO 2: GALERIA DE CANDIDATOS
    # -------------------------------------------------------------------
    if st.session_state.imgs and st.session_state.target_emb is None:
        st.divider()
        st.write("### Escolha quem você deseja extrair:")
        cols = st.columns(len(st.session_state.imgs))
        for idx, img in enumerate(st.session_state.imgs):
            with cols[idx]:
                st.image(img, channels="BGR", width=150)
                if st.button(f"Extrair este 👤", key=f"btn_sel_{idx}"):
                    st.session_state.target_emb = st.session_state.embs[idx]
                    st.rerun()

    # -------------------------------------------------------------------
    # PASSO 3: CONFIGURAÇÃO E EXTRAÇÃO
    # -------------------------------------------------------------------
    if st.session_state.target_emb is not None:
        st.success("🎯 Candidato alvo selecionado!")

        st.write("---")
        st.write("### ⚙️ Configuração de Captura")

        modo_escolhido = st.select_slider(
            "Selecione o nível de detalhamento (FPS):",
            options=["Velocidade", "Equilíbrio", "Máxima Extração"],
            value="Equilíbrio",
            help="Velocidade (0.5 fps), Equilíbrio (3 fps), Máxima (5 fps)",
        )

        mapa_divisor = {"Velocidade": 0.5, "Equilíbrio": 3, "Máxima Extração": 5}
        divisor = mapa_divisor[modo_escolhido]

        c1, c2, _ = st.columns([2, 2, 6])
        with c1:
            disparar_extracao = st.button("🚀 INICIAR EXTRAÇÃO TOTAL", key="btn_run_total")
        with c2:
            if st.button("🔄 Trocar Candidato", key="btn_reset_target"):
                st.session_state.target_emb = None
                st.rerun()

        status_texto = st.empty()
        barra_progresso = st.progress(0)
        msg_final = st.empty()

        if disparar_extracao:
            msg_final.info(f"Modo {modo_escolhido} ativado. Processando...")

            fotos_salvas = run_extraction(
                video_path,
                st.session_state.target_emb,
                "output",
                barra_progresso,
                status_texto,
                divisor,
            )

            barra_progresso.empty()
            status_texto.empty()

            if fotos_salvas:
                msg_final.success(f"✅ Concluído! {len(fotos_salvas)} fotos salvas.")
                st.balloons()

                # --- Preview ---
                st.write("---")
                st.write(f"### 🖼️ Prévia — {len(fotos_salvas)} imagens extraídas")
                preview_files = sorted(fotos_salvas)[:12]
                cols_prev = st.columns(min(len(preview_files), 4))
                for i, fname in enumerate(preview_files):
                    with cols_prev[i % 4]:
                        st.image(
                            os.path.join("output", fname),
                            use_container_width=True,
                            caption=fname,
                        )
                if len(fotos_salvas) > 12:
                    st.caption(f"… e mais {len(fotos_salvas) - 12} imagens na pasta /output.")

                # --- Botão de download ZIP ---
                st.write("---")
                zip_bytes = gerar_zip("output")
                st.download_button(
                    label=f"⬇️ Baixar todas as {len(fotos_salvas)} fotos (.zip)",
                    data=zip_bytes,
                    file_name="faces_extraidas.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            else:
                msg_final.error("❌ Alvo não encontrado no vídeo.")
