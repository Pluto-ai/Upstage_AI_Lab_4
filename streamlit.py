import streamlit as st
import requests
import pandas as pd

BASE_URL = "http://localhost:8000"  # FastAPI 서버 주소

st.set_page_config(page_title="트랙 추천 서비스", layout="wide")

st.title("음악 추천 서비스")
st.write('선택한 곡과 유사한 곡들을 추천하는 서비스입니다.')

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'view' not in st.session_state:
    st.session_state.view = 'list'
if 'last_search_query' not in st.session_state:
    st.session_state.last_search_query = ""
if 'last_search_type' not in st.session_state:
    st.session_state.last_search_type = "곡"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "cosine"  # 기본값으로 코사인 유사도 모델 선택

# 시간을 '분:초' 형식으로 변환하는 함수
def format_duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"

# 앨범 커버 URL 처리 함수
def get_cover_url(cover_url):
    return f"{BASE_URL}{cover_url}"

# API 요청 함수
def api_request(endpoint, params=None):
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 중 오류가 발생했습니다: {str(e)}")
        return None

# 트랙 목록 표시 함수
def display_track_list(tracks):
    if not tracks:
        st.write("표시할 트랙이 없습니다.")
        return

    for track in tracks:
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
        with col1:
            st.image(get_cover_url(track['cover_url']), width=50)
        with col2:
            if st.button(track['title'], key=f"track_{track['id']}"):
                st.session_state.selected_track = track['id']
                st.session_state.view = 'detail'
                st.rerun()
        with col3:
            st.write(track['album'])
        with col4:
            st.write(track['artist'])

# 검색 수행 함수
def perform_search(query, search_type, page=1, limit=10):
    return api_request("search", params={"query": query, "search_type": search_type, "page": page, "limit": limit})

# 목록 보기 함수
def show_track_list():
    # 검색 기능
    col1, col2 = st.columns([1, 5])
    with col1:
        search_type = st.selectbox("", ["곡", "아티스트"], index=0, key="search_type_select")
    with col2:
        search_query = st.text_input("", placeholder="곡 또는 아티스트 이름을 입력하세요", key="search_query_input")

    # 검색어나 검색 타입이 변경되었을 때만 새로운 검색 수행
    if search_query != st.session_state.last_search_query or search_type != st.session_state.last_search_type:
        st.session_state.last_search_query = search_query
        st.session_state.last_search_type = search_type
        st.session_state.page = 1  # 새 검색 시 페이지 초기화

    if st.session_state.last_search_query:
        tracks = perform_search(st.session_state.last_search_query, st.session_state.last_search_type,
                                st.session_state.page)
    else:
        tracks = api_request(f"tracks?page={st.session_state.page}&limit=10")

    if tracks is not None:
        display_track_list(tracks)

        # 페이지네이션 (한 줄에 가운데 정렬)
        st.write("")  # 약간의 간격 추가
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col2:
            st.button("이전", disabled=st.session_state.page == 1, key="prev_button",
                      on_click=lambda: setattr(st.session_state, 'page', st.session_state.page - 1))
        with col3:
            st.write(f"Page {st.session_state.page}")
        with col4:
            st.button("다음", disabled=len(tracks) < 10, key="next_button",
                      on_click=lambda: setattr(st.session_state, 'page', st.session_state.page + 1))

        # 페이지 변경 시 재실행
        if st.session_state.page != st.session_state.get('last_page', 0):
            st.session_state.last_page = st.session_state.page
            st.rerun()


# 상세 보기 함수
def show_track_detail():
    # 상단에 "목록으로 돌아가기" 버튼 배치
    if st.button("< 목록으로 돌아가기"):
        st.session_state.view = 'list'
        st.rerun()

    track = api_request(f"track/{st.session_state.selected_track}")
    if track is None:
        return

    st.header("선택된 트랙")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(get_cover_url(track['cover_url']), width=200, output_format="auto")
    with col2:
        st.write(f"제목: {track['title']}")
        st.write(f"앨범: {track['album']}")
        st.write(f"아티스트: {track['artist']}")
        st.write(f"재생 시간: {format_duration(track['duration'])}")

    # 유사한 트랙 목록 헤더와 모델 선택 버튼
    col1, col2, col3, col4, col5 = st.columns([3, 3, 0.1, 0.8, 0.8])
    with col1:
        st.subheader("이 트랙과 유사한 트랙")
    with col4:
        if st.button("코사인 유사도", key="cosine_button",
                     type="primary" if st.session_state.selected_model == "cosine" else "secondary"):
            st.session_state.selected_model = "cosine"
            st.rerun()
    with col5:
        if st.button("유클리드 거리", key="euclidean_button",
                     type="primary" if st.session_state.selected_model == "euclidean" else "secondary"):
            st.session_state.selected_model = "euclidean"
            st.rerun()

    # 유사한 트랙 목록
    similar_tracks = api_request(f"similar/{st.session_state.selected_model}/{st.session_state.selected_track}")
    if similar_tracks is not None:
        display_track_list(similar_tracks)

# 메인 앱 로직
if st.session_state.view == 'list':
    show_track_list()
elif st.session_state.view == 'detail':
    show_track_detail()

# streamlit run streamlit.py