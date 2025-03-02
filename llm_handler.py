#RDB에 해당 user sql문을 조회하는 파일
import rag_retriever
# Query to SQL library
import config
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Response를 얻기 위한 library
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage

def query_to_sql(api_key,user_query):
    llm_1 = ChatUpstage(api_key=api_key)
    template = """
You only have to use only fixed columns' name: [flight_number,airline,flight_date,STD,Departure,Arrival,IO]
You will use the following rules to convert the input to SQL:
1. Variable for flight Number: Use flight_number as the fixed variable for the flight number.
2. Variable for firline: Use airline as the fixed variable for the airline.
3. Variable for flight Date: Ensure that flight date is enclosed in quotation marks when used.
4. Variable for STD: Use STD as the fixed variable for the estimated departure time or estimated arrival time depends on IO.
5. Variable for IO: Use IO as the fixed variable for distinguishing between departure and arrival times. 
6. Variable for Airport: Use Departure for the departing airport and Arrival for the arriving airport. 
If query includes '에서', that word will be Departure
If query includes '로', that word will be Arrival
If query includes '출발', it means that IO has to be '출발'
If query includes '도착', it means that IO has to be '도착'
if query is "2024년 8월 16일 08시부터 11시 사이에 청주에서 제주로 가는 항공편 알려줘.", return should be
SELECT flight_number, airline, flight_date, STD, Departure, Arrival, IO
FROM flights
WHERE flight_date = "2024-08-16" AND STD >= "08:00:00" AND STD < "11:00:00" AND Departure = "청주" AND Arrival = "제주" AND IO = "출발";
And make sure Aiport's name is in Korean
Default year is 2024.
Make sure table name is flights
SELECT clause, FROM clause and WHERE clause should be in each line
Ensure that WHERE clauses and AND conditions are written on a single line without any line breaks.
Ensure that  SQL query that you make does not contain unintended text
Just make sure that only return SQL statements, do not include "SQL:"
You are a data scientist. Convert the following natural language query to "ONLY" SQL query
:
Query: {query}
SQL:
"""
    messages = [
    SystemMessage(content=template),
    HumanMessage(content=user_query)
    ]
    response = llm_1.invoke(messages)
    return response.content
def refine_sql(sql_query):
    # sql문을 rag_retriever로 보내는 역할
    # original_string = "여기는 작은따옴표(')가 포함된 문자열입니다."
    # 작은따옴표를 큰따옴표로 바꾸기
    changed_sql_query = sql_query.replace("'", '"')  
    changed_sql_query = changed_sql_query.replace("```sql", "")
    changed_sql_query = changed_sql_query.replace("```", "")
    changed_sql_query = changed_sql_query.replace("SQL:", "")
    return str(changed_sql_query)
def final_query(user_query,api_key):
    llm_2= ChatUpstage(api_key=api_key)
    messages = [
    SystemMessage(
        content=  """
            Based on Given data, Give me Information in Korean
            If query includes '항공편', and the given data is 'ZE702', '이스타항공', '2024-08-16', '7:50:00', '제주', '청주', '출발',
            response has to be following example
            Example : "제주에서 청주로 출발하는 항공편입니다 \n "항공편: ZE706, 항공사: 이스타항공, 출발날짜: 2024-08-16, 출발시간: 7시 50분, 출발: 제주, 도착: 청주 "
          
            If the IO is '출발', consider STD as the departure time. And return STD starts with '출발 시간은:'
            If the IO is '도착', consider STD as the arrival time. And return STD starts with '도착 시간은:'
            
            And I will append chat history you can use this making response
            """
    ),
    HumanMessage(
        content= user_query
    )
    ]
    response = llm_2.invoke(messages)
    return response.content
def history_chat(query):
    history =[]
    history+=query
    return history

def handle_user_query(query, rag_retriever):
    try:
        sql = query_to_sql(config.API_KEY, query)
        refined_sql = refine_sql(sql)
        
        print("Executing SQL query:", refined_sql) # 디버깅을 위한 것
    
        result = rag_retriever.execute_sql_query(refined_sql)
        
        if result.startswith("Database error"):
            return f"죄송합니다. 데이터베이스 조회 중 오류가 발생했습니다: {result}"
        
        query_with_result = query + " Data:" + result +history_chat()
        response = final_query(query_with_result, config.API_KEY)
        history_chat(query+response)  # for debugging purpose
        return response
    except Exception as e:
        return f"죄송합니다. 쿼리 처리 중 오류가 발생했습니다: {str(e)}"

if __name__== "__main__":
    user_query="2024년 8월 16일 청주에서 제주로 출발하는 항공편 알려줘"
    llm_api_key="up_CxnSYwc4TYOD487y9mbEKJdeVUjDy"
   
    # User query to User SQL query
    sql = query_to_sql(llm_api_key,user_query)
    print(refine_sql(sql))
    # # Get result 
    result = rag_retriever.execute_sql_query(refine_sql(sql))
    print(result)
    user_query += "Data:" + str(result)
    print(final_query(user_query,llm_api_key))

 
 
 


