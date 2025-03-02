import argparse
import os
import pandas as pd
import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from datetime import datetime
import pytz


def define_custom_sessions(df):
    """시간 기반으로 세션 재정의"""
    df['event_time'] = pd.to_datetime(df['event_time'])
    df = df.sort_values(['user_id', 'event_time'])
    df['time_diff'] = df.groupby('user_id')['event_time'].diff().dt.total_seconds()
    df['new_session'] = (df['time_diff'] > 1800) | (df['time_diff'].isna())
    df['custom_session_id'] = df.groupby('user_id')['new_session'].cumsum()
    df['session_key'] = df['user_id'] + '_' + df['custom_session_id'].astype(str)
    return df

def apply_event_time_weight(df):
    """시간대별 가중치 적용"""
    df['hour'] = df['event_time'].dt.hour
    
    time_weights = {
        (0, 6): 1.0,    # 새벽
        (6, 9): 1.2,    # 아침
        (9, 12): 1.3,   # 오전
        (12, 14): 1.4,  # 점심
        (14, 18): 1.3,  # 오후
        (18, 21): 1.4,  # 저녁
        (21, 24): 1.2   # 밤
    }
    
    df['time_weight'] = df['hour'].apply(
        lambda x: next((weight for (start, end), weight in time_weights.items() 
                       if start <= x < end), 1.0)
    )
    return df

def apply_category_weight(df):
    """카테고리 기반 가중치 적용"""
    category_count = df['category_code'].value_counts()
    df['category_weight'] = df['category_code'].map(
        np.log1p(category_count) / np.log1p(category_count.max())
    ).fillna(0.5)
    
    # 사용자별 카테고리 선호도
    user_category = df.groupby(['user_id', 'category_code']).size().reset_index(name='count')
    user_category['preference'] = user_category.groupby('user_id')['count'].transform(
        lambda x: x / x.sum()
    )
    
    category_pref_dict = dict(zip(
        zip(user_category['user_id'], user_category['category_code']),
        user_category['preference']
    ))
    
    df['category_preference'] = df.apply(
        lambda row: category_pref_dict.get((row['user_id'], row['category_code']), 0),
        axis=1
    )
    
    df['category_weight'] = df['category_weight'] * (1 + df['category_preference'])
    return df

def apply_brand_weight(df):
    """브랜드 기반 가중치 적용"""
    brand_count = df['brand'].value_counts()
    df['brand_weight'] = df['brand'].map(
        np.log1p(brand_count) / np.log1p(brand_count.max())
    ).fillna(0.5)
    
    # 사용자별 브랜드 선호도
    user_brand = df.groupby(['user_id', 'brand']).size().reset_index(name='count')
    user_brand['preference'] = user_brand.groupby('user_id')['count'].transform(
        lambda x: x / x.sum()
    )
    
    brand_pref_dict = dict(zip(
        zip(user_brand['user_id'], user_brand['brand']),
        user_brand['preference']
    ))
    
    df['brand_preference'] = df.apply(
        lambda row: brand_pref_dict.get((row['user_id'], row['brand']), 0),
        axis=1
    )
    
    df['brand_weight'] = df['brand_weight'] * (1 + df['brand_preference'])
    return df

def apply_price_weight(df):
    """가격대 기반 가중치 적용"""
    df['price_range'] = pd.qcut(df['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # 사용자별 선호 가격대
    user_price = df.groupby(['user_id', 'price_range'])['event_type'].count().reset_index()
    user_price['preference'] = user_price.groupby('user_id')['event_type'].transform(
        lambda x: x / x.sum()
    )
    
    price_pref_dict = dict(zip(
        zip(user_price['user_id'], user_price['price_range']),
        user_price['preference']
    ))
    
    df['price_weight'] = df.apply(
        lambda row: 1 + price_pref_dict.get((row['user_id'], row['price_range']), 0),
        axis=1
    )
    return df

def apply_recency_weight(df):
    """최신성 가중치 적용"""
    df['days_diff'] = (df['event_time'].max() - df['event_time']).dt.total_seconds() / (24*60*60)
    df['recency_weight'] = np.exp(-df['days_diff'] / 30)  # 30일 기준
    return df

def calculate_session_weights(df):
    """세션 기반 가중치 계산"""
    session_stats = df.groupby('session_key').agg({
        'event_type': ['count', lambda x: (x == 'purchase').sum()],
        'price': 'mean'
    }).reset_index()
    
    session_stats.columns = ['session_key', 'event_count', 'purchase_count', 'avg_price']
    
    # 세션 품질 점수
    session_stats['session_weight'] = 1.0 + (
        0.2 * np.clip(session_stats['event_count'] / session_stats['event_count'].quantile(0.95), 0, 1) +
        0.3 * (session_stats['purchase_count'] > 0).astype(float)
    )
    
    return df.merge(session_stats[['session_key', 'session_weight']], on='session_key', how='left')

def apply_event_weights(df):
    """이벤트 타입별 가중치 적용"""
    event_weights = {
        'purchase': 4.0,
        'cart': 2.0,
        'view': 1.0
    }
    df['event_weight'] = df['event_type'].map(event_weights)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="train.parquet", type=str)
    parser.add_argument("--dir_path", default="/data/ephemeral/home/data/", type=str)
    parser.add_argument("--output_dir", default="/data/ephemeral/home/output/", type=str)
    parser.add_argument("--num_factor", type=int, default=256)
    parser.add_argument("--regularization", type=float, default=0.008)
    parser.add_argument("--alpha", type=float, default=25)
    parser.add_argument("--seed", default=42, type=int)
    
    args = parser.parse_args()
    
    # 데이터 로드
    train_df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))
    
    # 기본 가중치 적용
    train_df = define_custom_sessions(train_df)
    train_df = apply_event_time_weight(train_df)
    train_df = apply_category_weight(train_df)
    train_df = apply_brand_weight(train_df)
    train_df = apply_price_weight(train_df)
    train_df = apply_recency_weight(train_df)
    
    # 세션 및 이벤트 가중치 적용
    train_df = calculate_session_weights(train_df)
    train_df = apply_event_weights(train_df)
    
    # 최종 가중치 계산 (각 가중치의 영향력 조절)
    train_df['final_weight'] = (
        train_df['event_weight'] * 
        (1.0 +
         0.2 * (train_df['session_weight'] - 1) +    # 세션 영향력
         0.2 * (train_df['time_weight'] - 1) +       # 시간대 영향력
         0.2 * (train_df['category_weight'] - 1) +   # 카테고리 영향력
         0.2 * (train_df['brand_weight'] - 1) +      # 브랜드 영향력
         0.1 * (train_df['price_weight'] - 1) +      # 가격대 영향력
         0.1 * (train_df['recency_weight'] - 1)      # 최신성 영향력
        )
    )
    
    # 인덱스 매핑
    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}
    
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)
    
    # 사용자-아이템 행렬 생성
    user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["final_weight"].sum().reset_index()
    
    sparse_user_item = sparse.csr_matrix(
        (user_item_matrix["final_weight"].values,
         (user_item_matrix["user_idx"].values,
          user_item_matrix["item_idx"].values)),
        shape=(len(user2idx), len(item2idx))
    )
    
    # 모델 학습
    model = AlternatingLeastSquares(
        factors=args.num_factor,
        regularization=args.regularization,
        alpha=args.alpha,
        use_gpu=False
    )
    
    model.fit(sparse_user_item)
    
    # 추천 생성
    test_users_idx = np.array(train_df['user_idx'].unique())
    test_users_idx_li = [num for num in test_users_idx for _ in range(10)]
    
    public_outputs = model.recommend(
        test_users_idx,
        sparse_user_item[test_users_idx],
        N=10,
        filter_already_liked_items=False
    )
    
    recommend_items = public_outputs[0]
    sub_df = pd.DataFrame({
        'user_id': test_users_idx_li,
        'item_id': recommend_items.flatten()
    })
    
    sub_df['user_id'] = sub_df['user_id'].map(idx2user)
    sub_df['item_id'] = sub_df['item_id'].map(idx2item)
    
    # 결과 저장
    kst = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    
    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    sub_df.to_csv(os.path.join(outdir, f"output_{current_time}.csv"), index=False)

if __name__ == "__main__":
    main()