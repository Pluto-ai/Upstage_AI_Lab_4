import api_handler
import db_operations
from datetime import datetime

from rag_retriever import RAGRetriever
from llm_handler import query_to_sql, refine_sql, final_query
import config

from fastapi import FastAPI
import uvicorn
import subprocess

def fetch_and_store_data():
    # 오늘 날짜 가져오기
    today = datetime.today().strftime('%Y%m%d')
    
    # 김포공항 출발 및 도착 데이터 가져오기
    kac_departure = api_handler.KAC_Departure_api(today)
    kac_arrival = api_handler.KAC_Arrival_api(today)
    
    # 인천공항 출발 및 도착 데이터 가져오기
    iiac_departure = api_handler.IIAC_Departure_api()
    iiac_arrival = api_handler.IIAC_Arrival_api()
    
    # 모든 데이터 합치기
    all_flights = kac_departure + kac_arrival + iiac_departure + iiac_arrival
    
    # 데이터베이스에 저장
    db_operations.insert_data_bulk(all_flights)
    
    print(f"Total flights fetched and stored: {len(all_flights)}")

def initialize_rag():
    embedding_model = api_handler.embed_model_init()
    llm = api_handler.llm_init()
    rag_retriever = RAGRetriever(embedding_model)
    return rag_retriever, llm

def handle_user_query(query, rag_retriever):
    try:
        sql = query_to_sql(config.API_KEY, query)
        refined_sql = refine_sql(sql)
        
        print("Executing SQL query:", refined_sql) # 디버깅을 위한 것
    
        result = rag_retriever.execute_sql_query(refined_sql)
        
        if result.startswith("Database error"):
            return f"죄송합니다. 데이터베이스 조회 중 오류가 발생했습니다: {result}"
        
        query_with_result = query + " Data:" + result
        response = final_query(query_with_result, config.API_KEY)
        return response
    except Exception as e:
        return f"죄송합니다. 쿼리 처리 중 오류가 발생했습니다: {str(e)}"

app = FastAPI()

@app.get('/')
async def root():
    return {'message':'Hello World!'}

@app.get('/solar')
async def rag_solar_get(user_message):
    return {'message': handle_user_query(user_message, app.state.rag_retriever)}

@app.post('/solar')
async def rag_solar_get(param: dict={}):
    user_message = param.get('user_message', ' ')
    return {'message': handle_user_query(user_message, app.state.rag_retriever)}

def main():
    fetch_and_store_data()
    
    rag_retriever, _ = initialize_rag()
    
    app.state.rag_retriever = rag_retriever
    
    subprocess.Popen(["streamlit", "run", "streamlit.py"])

    uvicorn.run(app, host="0.0.0.0", port=8080)
    # while True:
    #     user_query = input("Enter your query (or 'quit' to exit): ")
    #     if user_query.lower() == 'quit':
    #         break
        
    #     response = handle_user_query(user_query, rag_retriever)
    #     print("Response:", response)

if __name__ == "__main__":
    main()