from datetime import datetime

from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
import requests

import config   # API_KEY, DATA_API_KEY

def embed_model_init():
    return UpstageEmbeddings(api_key=config.API_KEY, model='solar-embedding-1-large')

def llm_init():
    return ChatUpstage(api_key=config.API_KEY)


def KAC_Departure_api(flight_date=datetime.today().strftime('%Y%m%d')):
    KAC_api_key = config.DATA_API_KEY
    flight_date = flight_date
    url = f'https://api.odcloud.kr/api/FlightStatusListDTL/v1/getFlightStatusListDetail?perPage=1000&cond%5BFLIGHT_DATE%3A%3AEQ%5D={flight_date}&cond%5BIO%3A%3AEQ%5D=o&serviceKey={KAC_api_key}'
    flight_dict = requests.get(url).json()
    flight_list = flight_dict['data']

    m_flight_list = []

    for flight in flight_list:
        flight_tmp = {
            'flight_number': flight.get('AIR_FLN'),
            'airline': flight.get('AIRLINE_KOREAN'),
            'flight_date': flight.get('FLIGHT_DATE'),
            'STD': flight.get('STD') + '00',
            'Departure': flight.get('BOARDING_KOR'),
            'ARRIVAL': flight.get('ARRIVED_KOR'),
            'IO': '출발'
        }
        m_flight_list.append(flight_tmp)

    return m_flight_list

def KAC_Arrival_api(flight_date=datetime.today().strftime('%Y%m%d')):
    KAC_api_key = config.DATA_API_KEY
    flight_date = flight_date
    url = f'https://api.odcloud.kr/api/FlightStatusListDTL/v1/getFlightStatusListDetail?serviceKey={KAC_api_key}&perPage=1000&cond%5BFLIGHT_DATE%3A%3AEQ%5D={flight_date}&cond%5BIO%3A%3AEQ%5D=i'
    flight_dict = requests.get(url).json()
    flight_list = flight_dict['data']

    m_flight_list = []

    for flight in flight_list:
        flight_tmp = {
            'flight_number': flight.get('AIR_FLN'),
            'airline': flight.get('AIRLINE_KOREAN'),
            'flight_date': flight.get('FLIGHT_DATE'),
            'STD': flight.get('STD') + '00',
            'Departure': flight.get('BOARDING_KOR'),
            'ARRIVAL': flight.get('ARRIVED_KOR'),
            'IO': '도착'
        }
        m_flight_list.append(flight_tmp)

    return m_flight_list

def IIAC_Arrival_api():
    IIAC_api_key = config.DATA_API_KEY
    url = 'http://apis.data.go.kr/B551177/StatusOfPassengerFlightsOdp/getPassengerArrivalsOdp'
    params = {
        'serviceKey' : IIAC_api_key,
        'from_time' : '0000',
        'to_time' : '2400',
        'airport' : '',
        'flight_id' : '',
        'airline' : '',
        'lang' : 'K',
        'type' : 'json'
    }
    flight_dict = requests.get(url=url, params=params).json()
    flight_list = flight_dict['response']['body']['items']

    m_flight_list = []

    for flight in flight_list:
        flight_tmp = {
            'flight_number': flight.get('flightId'),
            'airline': flight.get('airline'),
            'flight_date': datetime.today().strftime('%Y%m%d'),
            'STD': flight.get('scheduleDateTime') + '00',
            'Departure': flight.get('airport'),
            'ARRIVAL': '인천',
            'IO': '도착'
        }
        m_flight_list.append(flight_tmp)

    return m_flight_list

def IIAC_Departure_api():
    IIAC_api_key = config.DATA_API_KEY
    url = 'http://apis.data.go.kr/B551177/StatusOfPassengerFlightsOdp/getPassengerDeparturesOdp'
    params = {
        'serviceKey' : IIAC_api_key,
        'from_time' : '0000',
        'to_time' : '2400',
        'airport' : '',
        'flight_id' : '',
        'airline' : '',
        'lang' : 'K',
        'type' : 'json'
    }
    flight_dict = requests.get(url=url, params=params).json()
    flight_list = flight_dict['response']['body']['items']

    m_flight_list = []

    for flight in flight_list:
        flight_tmp = {
            'flight_number': flight.get('flightId'),
            'airline': flight.get('airline'),
            'flight_date': datetime.today().strftime('%Y%m%d'),
            'STD': flight.get('scheduleDateTime') + '00',
            'Departure': '인천',
            'ARRIVAL': flight.get('airport'),
            'IO': '출발'
        }
        m_flight_list.append(flight_tmp)

    return m_flight_list