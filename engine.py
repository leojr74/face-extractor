import cv2
import os
import numpy as np
import shutil
from deepface import DeepFace
from scipy.spatial import distance
from PIL import Image


def is_frontal(face_data: dict, max_asymmetry: float = 0.30) -> bool:
    """
    Retorna True se a pose for suficientemente frontal.

    Prioridade:
      1. Olhos + nariz  → referência mais precisa
      2. Boca + nariz   → fallback para óculos escuros
      3. Sem landmarks  → False (descarta, não deixa passar)
    """
    left_eye    = face_data.get("left_eye")
    right_eye   = face_data.get("right_eye")
    nose        = face_data.get("nose")
    mouth_left  = face_data.get("left_mouth")
    mouth_right = face_data.get("right_mouth")

    # Caso 1: tem olhos — usa assimetria olho/nariz (mais preciso)
    if all([left_eye, right_eye, nose]):
        eye_dist = abs(right_eye[0] - left_eye[0])
        if eye_dist >= 1:
            mid_x = (left_eye[0] + right_eye[0]) / 2
            return abs(nose[0] - mid_x) / eye_dist < max_asymmetry

    # Caso 2: sem olhos mas tem boca + nariz (óculos escuros)
    if all([nose, mouth_left, mouth_right]):
        mouth_dist = abs(mouth_right[0] - mouth_left[0])
        if mouth_dist >= 1:
            mid_x = (mouth_left[0] + mouth_right[0]) / 2
            nose_centered = abs(nose[0] - mid_x) / mouth_dist < max_asymmetry
            nose_above = nose[1] < (mouth_left[1] + mouth_right[1]) / 2
            return nose_centered and nose_above

    # Caso 3: sem landmarks suficientes → deixa passar
    # (óculos escuros frequentemente não retorna landmarks — não podemos descartar)
    return True


def is_good_quality(image):
    """Filtro de qualidade ultra-leve (CPU Friendly)."""
    if image is None or image.size == 0:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 10:
        return False
    if gray.std() < 15:
        return False
    return True


def get_candidate_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    v_fps = int(cap.get(cv2.CAP_PROP_FPS) or 15)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates, candidate_embs = [], []

    limit_frames = min(total_f, v_fps * 15)
    step = max(1, v_fps // 2)

    for i in range(0, limit_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        target_h = 360
        scale = target_h / frame.shape[0]
        low_res = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        try:
            faces = DeepFace.extract_faces(
                low_res,
                detector_backend='mtcnn',
                enforce_detection=False,
                align=False
            )

            for f in faces:
                if f.get("confidence", 0) < 0.85:
                    continue

                area = f["facial_area"]
                x = int(area["x"] / scale)
                y = int(area["y"] / scale)
                w = int(area["w"] / scale)
                h = int(area["h"] / scale)

                face_img = frame[max(0, y):y+h, max(0, x):x+w]
                if face_img.size == 0:
                    continue

                emb_res = DeepFace.represent(
                    face_img,
                    model_name="ArcFace",
                    enforce_detection=False,
                    align=False
                )

                if emb_res:
                    emb = emb_res[0]["embedding"]
                    if not any(distance.cosine(emb, c) < 0.60 for c in candidate_embs):
                        face_zoom = cv2.resize(face_img, (150, 150), interpolation=cv2.INTER_LANCZOS4)
                        candidates.append(face_zoom)
                        candidate_embs.append(emb)

            if len(candidates) >= 12:
                break
        except Exception:
            continue

    cap.release()
    return candidates, candidate_embs


def run_extraction(video_path, target_emb, output_dir, progress_bar, status_text, divisor):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    v_fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, int(v_fps // divisor))

    count = 0
    saved_embs = []   # deduplicação por embedding — invariante a translação
    last_pos = None

    # Raio fixo conservador — evita aceitar faces de outras pessoas no frame
    raio_ajustado = 200

    for i in range(0, total_f, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        target_h = 360
        scale = target_h / frame.shape[0]
        low_res = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        try:
            faces = DeepFace.extract_faces(
                low_res,
                detector_backend='mtcnn',
                enforce_detection=False,
                align=False
            )

            for f in faces:
                if f.get("confidence", 0) < 0.98:
                    continue

                # --- FILTRO DE POSE FRONTAL ---
                # False no fallback: sem landmarks = descarta (corrige perfis passando)
                if not is_frontal(f, max_asymmetry=0.30):
                    continue

                area = f["facial_area"]
                curr_x = int(area["x"] / scale)
                curr_y = int(area["y"] / scale)
                curr_w = int(area["w"] / scale)
                curr_h = int(area["h"] / scale)

                # --- FILTRO DE RASTREAMENTO ---
                if last_pos is not None:
                    last_x, last_y, _, _ = last_pos
                    dist_pixel = ((curr_x - last_x)**2 + (curr_y - last_y)**2)**0.5
                    if dist_pixel > raio_ajustado:
                        continue

                face_para_ia = frame[max(0, curr_y):curr_y+curr_h, max(0, curr_x):curr_x+curr_w]
                if face_para_ia.size == 0:
                    continue

                if not is_good_quality(face_para_ia):
                    continue

                # --- RECONHECIMENTO ARCFACE ---
                emb_res = DeepFace.represent(
                    face_para_ia,
                    model_name="ArcFace",
                    enforce_detection=False,
                    align=False
                )
                if not emb_res:
                    continue

                emb_atual = emb_res[0]["embedding"]
                dist_ia = distance.cosine(emb_atual, target_emb)

                if dist_ia >= 0.45:
                    continue

                last_pos = (curr_x, curr_y, curr_w, curr_h)

                # --- DEDUPLICAÇÃO POR EMBEDDING ---
                # Mais robusta que dHash: invariante a pequenas translações e luz.
                # 0.15 = mesma pose/frame  → aumente para 0.20 se salvar demais
                #                          → diminua para 0.10 se descartar poses diferentes
                MIN_DIST_DUPLICATA = 0.15
                if any(distance.cosine(emb_atual, e) < MIN_DIST_DUPLICATA for e in saved_embs):
                    continue

                saved_embs.append(emb_atual)

                # --- SALVAMENTO ---
                margin = int(curr_w * 0.8)
                y1 = max(0, curr_y - margin)
                y2 = min(frame.shape[0], curr_y + curr_h + margin)
                x1 = max(0, curr_x - margin)
                x2 = min(frame.shape[1], curr_x + curr_w + margin)
                face_final = frame[y1:y2, x1:x2]

                img_save = Image.fromarray(cv2.cvtColor(face_final, cv2.COLOR_BGR2RGB))
                img_save = img_save.resize((500, 500), Image.LANCZOS)
                img_save.save(
                    os.path.join(output_dir, f"face_{count}.jpg"),
                    dpi=(500, 500),
                    quality=95
                )
                count += 1

        except Exception:
            continue

        if progress_bar and i % (step * 5) == 0:
            prog = i / total_f
            progress_bar.progress(min(prog, 1.0))
            status_text.text(f"Modo Turbo: {int(prog*100)}% | Achados: {count}")

    cap.release()
    return os.listdir(output_dir)


def save_final_image(img_np, output_dir, idx):
    rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img = pil_img.resize((250, 250), resample=Image.LANCZOS)
    path = os.path.join(output_dir, f"face_{idx}.jpg")
    pil_img.save(path, dpi=(500, 500), quality=90)