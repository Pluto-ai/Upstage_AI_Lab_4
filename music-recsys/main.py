import boto3
from botocore.exceptions import NoCredentialsError
from io import BytesIO

# S3 클라이언트 설정
s3 = boto3.client('s3')
BUCKET_NAME = 'hyunaebucket'

# S3에서 파일 읽기 함수
def read_file_from_s3(file_name):
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=file_name)
        return BytesIO(response['Body'].read())
    except NoCredentialsError:
        print("Credentials not available")
        return None

# S3에서 데이터 로딩
def load_data_from_s3():
    file = read_file_from_s3('final_dataset.csv')
    if file:
        return pd.read_csv(file)
    return None

# S3에서 모델 로드 함수
def load_models_from_s3(encoder_path, kmeans_path):
    try:
        encoder_file = read_file_from_s3(encoder_path)
        kmeans_file = read_file_from_s3(kmeans_path)
        
        if encoder_file and kmeans_file:
            autoencoder_model = load_model(encoder_file)
            kmeans_model = joblib.load(kmeans_file)
            return autoencoder_model, kmeans_model
    except Exception as e:
        logger.error(f"Failed to load models from S3: {str(e)}")
    return None, None

# S3에서 앨범 커버 가져오기
@app.get("/album_covers/{album_id}.jpg")
async def get_album_cover(album_id: str):
    try:
        file_key = f"album_covers/{album_id}.jpg"
        response = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
        return StreamingResponse(response['Body'].iter_chunks(), media_type="image/jpeg")
    except:
        # 기본 커버 이미지 반환
        default_key = "album_covers/default_cover.jpg"
        response = s3.get_object(Bucket=BUCKET_NAME, Key=default_key)
        return StreamingResponse(response['Body'].iter_chunks(), media_type="image/jpeg")

# 데이터 로딩 부분 수정
df = load_data_from_s3()
if df is None:
    logger.error("Failed to load data from S3")
    # 에러 처리 로직 추가

# 모델 로드 부분 수정
encoder_model, kmeans_model = load_models_from_s3(
    "models/autoencoder_model.keras",
    "models/kmeans_model.pkl"
)
if encoder_model is None or kmeans_model is None:
    logger.error("Failed to load models from S3")
    # 에러 처리 로직 추가

# StaticFiles 마운트 제거 (S3를 사용하므로 더 이상 필요 없음)
# app.mount("/album_covers", StaticFiles(directory="album_covers"), name="album_covers")