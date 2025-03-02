# api_handler.py가 만들어지기 전에 db_handler.py를 테스트 해보기 위해 만든 파일이므로 실제 구동에서는 필요 없는 모듈
from db_operations import insert_data_bulk, select_data
from datetime import datetime, timedelta
import random

def generate_sample_data(num_records=100):
    airlines = ["AirKorea", "SkyJapan", "ChinaFly", "USAir", "EuroWings"]
    airports = ["ICN", "NRT", "PEK", "LAX", "LHR", "CDG", "SIN", "SYD", "JFK", "FRA"]
    
    base_date = datetime.now()
    
    data = []
    for i in range(num_records):
        flight_date = (base_date + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
        std = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00"
        departure = random.choice(airports)
        arrival = random.choice([a for a in airports if a != departure])
        io = random.choice(['I', 'O'])
        
        record = {
            "flight_number": f"FL{i:04d}",
            "airline": random.choice(airlines),
            "flight_date": flight_date,
            "STD": std,
            "Departure": departure,
            "ARRIVAL": arrival,
            "IO" : io
        }
        data.append(record)
    
    return data

def main():
    # 샘플 데이터 생성
    sample_data = generate_sample_data()
    print(f"Generated {len(sample_data)} sample records")

    # 디버깅을 위한 샘플 데이터 출력
    print("Sample record:")
    print(sample_data[0])

    # DB에 데이터 삽입
    try:
        insert_data_bulk(sample_data)
        print("Data inserted successfully")
    except Exception as e:
        print(f"Error during data insertion: {e}")
        print("Sample of problematic data:")
        print(sample_data[0])

    # 쿼리 받아서 내보내는 것 예시 코드
    select_query = "SELECT * FROM flights LIMIT 10"
    results = select_data(select_query)
    print("Sample data from database:")
    for row in results:
        print(row)

if __name__ == "__main__":
    main()