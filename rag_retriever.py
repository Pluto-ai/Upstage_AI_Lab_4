import mysql.connector
from db_operations import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

class RAGRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.db_config = {
            'host': DB_HOST,
            'user': DB_USER,
            'password': DB_PASSWORD,
            'database': DB_NAME
        }

    def execute_sql_query(self, query):
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchall()
        
            # 결과를 보기 좋게 포맷팅
            formatted_result = []
            for row in result:
                formatted_row = ", ".join([f"{k}: {v}" for k, v in row.items()])
                formatted_result.append(formatted_row)
        
            return "\n".join(formatted_result)
        except mysql.connector.Error as e:
            return f"Database error: {e}"
        finally: # ??
            if conn.is_connected():
                cursor.close()
                conn.close()