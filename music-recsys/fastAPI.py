from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Static 파일 서빙 설정
app.mount("/album_covers", StaticFiles(directory="album_covers"), name="album_covers")

# 데이터 로딩
df = pd.read_csv('final_dataset.csv')
df['id'] = df.index  # ID 컬럼 추가

# 수치형 데이터 선택 (모델 입력에 사용될 특성들)
numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
numeric_data = df[numeric_features]

# 데이터 정규화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)


# 모델 로드 함수
def load_models(encoder_path, kmeans_path):
    try:
        autoencoder_model = load_model(encoder_path, compile=False)
        kmeans_model = joblib.load(kmeans_path)
        return autoencoder_model, kmeans_model
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return None, None


# 모델 로드
encoder_model, kmeans_model = load_models(
    "./autoencoder_model_dims_2048_1024_512_256_128_64_32_16_8_4.keras",
    "./kmeans_model_clusters_15_dims_2048_1024_512_256_128_64_32_16_8_4.pkl"
)


# 입력 데이터 형태 조정 함수
def adjust_input_shape(data, target_shape):
    current_shape = data.shape[1]
    if current_shape < target_shape:
        padding = np.zeros((data.shape[0], target_shape - current_shape))
        return np.hstack((data, padding))
    elif current_shape > target_shape:
        return data[:, :target_shape]
    else:
        return data


# 입력 데이터 형태 조정
if encoder_model is not None:
    target_shape = encoder_model.input_shape[1]  # 모델이 기대하는 입력 형태
    adjusted_scaled_data = adjust_input_shape(scaled_data, target_shape)
    logger.info(f"Original numeric data shape: {numeric_data.shape}")
    logger.info(f"Adjusted scaled data shape: {adjusted_scaled_data.shape}")
    logger.info(f"Model expected input shape: {encoder_model.input_shape}")

    # 전체 데이터에 대해 인코딩된 벡터 생성
    encoded_data = encoder_model.predict(adjusted_scaled_data)
    logger.info(f"Encoded data shape: {encoded_data.shape}")
else:
    logger.error("Encoder model failed to load. Recommendation features will not be available.")
    encoded_data = None


# 트랙 모델
class Track(BaseModel):
    id: int
    title: str
    album: str
    artist: str
    cover_url: str
    duration: int


# 데이터프레임을 Track 모델에 맞게 변환하는 함수
def df_to_track(row):
    return Track(
        id=int(row['id']),
        title=row['track_name'],
        album=row['album_name'],
        artist=row['artist_name'],
        cover_url=f"/album_covers/{row['album_id']}.jpg",
        duration=int(row['duration_ms'] / 1000)  # ms를 초로 변환
    )


# 트랙 목록 가져오기
@app.get("/tracks", response_model=List[Track])
async def get_tracks(page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100)):
    start = (page - 1) * limit
    end = start + limit
    tracks = df.iloc[start:end].apply(df_to_track, axis=1).tolist()
    return tracks


# 트랙 검색
@app.get("/search", response_model=List[Track])
async def search_tracks(query: str, search_type: str = "곡", page: int = Query(1, ge=1),
                        limit: int = Query(10, ge=1, le=100)):
    if search_type == "곡":
        mask = df['track_name'].str.contains(query, case=False, na=False)
        column = 'track_name'
    elif search_type == "아티스트":
        mask = df['artist_name'].str.contains(query, case=False, na=False)
        column = 'artist_name'
    else:
        raise HTTPException(status_code=400, detail="Invalid search type")

    matching_tracks = df[mask]

    if matching_tracks.empty:
        return []

    # 유사도 점수 계산
    matching_tracks['similarity'] = matching_tracks[column].apply(lambda x: fuzz.ratio(query.lower(), x.lower()))

    # 유사도 점수로 정렬
    matching_tracks = matching_tracks.sort_values('similarity', ascending=False)

    start = (page - 1) * limit
    end = start + limit

    return matching_tracks.iloc[start:end].apply(df_to_track, axis=1).tolist()


# 트랙 상세 정보
@app.get("/track/{track_id}", response_model=Track)
async def get_track(track_id: int):
    try:
        track = df_to_track(df.loc[df['id'] == track_id].iloc[0])
        return track
    except IndexError:
        raise HTTPException(status_code=404, detail="Track not found")


# 유사한 트랙 추천
@app.get("/similar/{model_type}/{track_id}", response_model=List[Track])
async def get_similar_tracks(model_type: str, track_id: int, top_n: int = Query(10, ge=1, le=100)):
    if encoder_model is None or encoded_data is None:
        raise HTTPException(status_code=500, detail="Model or encoded data not available")

    try:
        track_index = df.index[df['id'] == track_id].tolist()[0]
        input_song_vector = scaled_data[track_index].reshape(1, -1)
        input_song_vector_adjusted = adjust_input_shape(input_song_vector, target_shape)
        encoded_input_song = encoder_model.predict(input_song_vector_adjusted)

        if model_type == "cosine":
            similarities = cosine_similarity(encoded_input_song, encoded_data)
            top_n_indices = np.argsort(similarities[0])[-top_n - 1:][::-1][1:]  # 자기 자신 제외
        elif model_type == "euclidean":
            distances = euclidean_distances(encoded_input_song, encoded_data)
            top_n_indices = np.argsort(distances[0])[1:top_n + 1]  # 자기 자신 제외
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        similar_tracks = df.iloc[top_n_indices].apply(df_to_track, axis=1).tolist()
        return similar_tracks
    except Exception as e:
        logger.error(f"Error in get_similar_tracks: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# 앨범 커버 이미지 제공
@app.get("/album_covers/{album_id}.jpg")
async def get_album_cover(album_id: str):
    cover_path = f"album_covers/{album_id}.jpg"
    if os.path.exists(cover_path):
        return FileResponse(cover_path)
    else:
        return FileResponse("album_covers/default_cover.jpg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# uvicorn fastAPI:app --reload