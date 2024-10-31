import argparse 
import os
import torch
import torchaudio
import numpy as np
from wav2vec2_vib_gelu import Model
# from wav2vec2_linear_nll import Model
from torchinfo import summary
import librosa
from audio_cam import AudioGradCAM  # visualize_cam 제거
import pandas as pd
from collections import defaultdict
from tqdm import tqdm  # 진행 상황 표시용

# 디바이스 설정 (CUDA 사용 가능 시 사용)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 모델 초기화
model = Model(
    device=device
).to(device)

# 모델 가중치 로드
try:
    model.load_state_dict(torch.load('/datad/pretrained/AudioDeepfakeCMs/vib/vib_conf-5_gelu_acmccs_apr3_moreko_telephone_epoch22.pth', map_location=device))
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("Model weights file not found. Please check the file path.")
    exit(1)  # 가중치 로드 실패 시 종료
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

# 타겟 레이어 설정 (모델의 마지막 레이어 예시)
target_layer = model.LL  # 실제 모델의 레이어에 따라 변경 필요

# Grad-CAM 객체 생성
grad_cam = AudioGradCAM(model=model, target_layers=[target_layer])  # target_class는 스푸핑 클래스 예시

# eval_seglab_0.16.txt 파일 경로
eval_seglab_path = '/home/woonj/grad-cam/ps/PartialSpoof/database/segment_labels/eval_seglab_0.16.txt'

# eval_seglab_0.16.txt 파일 로드
try:
    labels_df = pd.read_csv(eval_seglab_path, sep=' ', header=None, names=['file_path', 'subset', 'label'])
    print(f"Number of labels loaded: {labels_df.shape[0]}")
    print(labels_df.head())
except FileNotFoundError:
    print("Labels file not found. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading labels file: {e}")
    exit(1)

# 파일 ID 추출 (파일 경로에서 'eval/' 접두사 제거 및 확장자 제거)
labels_df['file_id'] = labels_df['file_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

# Grad-CAM 결과를 저장할 텍스트 파일 경로 설정
output_txt_path = '/home/woonj/grad-cam/experiment_results/grad_cam_bonafide_score.txt'

# 출력 디렉터리 존재 여부 확인
output_dir = os.path.dirname(output_txt_path)
if not os.path.exists(output_dir):
    print(f"Output directory does not exist: {output_dir}")
    exit(1)

# 데이터셋 폴더 경로
dataset_dir = '/home/woonj/grad-cam/ps/PartialSpoof/database/eval/con_wav'

# 텍스트 파일 열기 (쓰기 모드)
with open(output_txt_path, 'w') as f:
    # 헤더 작성 (선택 사항)
    f.write("file_id label cam_values\n")
    
    # 모든 오디오 파일에 대해 Grad-CAM 계산 및 저장
    for idx, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0]):
        file_id = row['file_id']
        label = row['label']
        
        # 오디오 파일 전체 경로 설정
        audio_filename = f"{file_id}.wav"
        full_audio_path = os.path.join(dataset_dir, audio_filename)
        
        # 오디오 파일 존재 여부 확인
        if not os.path.exists(full_audio_path):
            print(f"File not found: {full_audio_path}")
            continue
        
        # 오디오 파일 로드
        try:
            waveform, sr = librosa.load(full_audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading {full_audio_path}: {e}")
            continue
        
        # 텐서로 변환 및 디바이스 이동
        waveform_tensor = torch.Tensor(waveform).unsqueeze(0).to(device)  # (1, N)
        
        # Grad-CAM 계산
        try:
            cam = grad_cam(waveform_tensor)
            cam = np.array(cam, dtype=np.float32).squeeze()  # NumPy 배열로 변환
        except Exception as e:
            print(f"Error computing Grad-CAM for {full_audio_path}: {e}")
            continue
        
        # Grad-CAM 값을 리스트 형식으로 변환 (예: 0.1,0.2,0.3,...)
        cam_values_str = ','.join(map(str, cam))
        
        # 텍스트 파일에 저장 (파일명, 레이블, Grad-CAM 값)
        f.write(f"{file_id} {label} {cam_values_str}\n")
        # print(f"Processed {file_id}: label={label}, cam_values_str={cam_values_str[:50]}...")  # 첫 50자 출력

print(f"Grad-CAM results saved to {output_txt_path}")
