from fastapi import FastAPI
import main

app = FastAPI()

@app.get('/')
async def root():
    return {'message':'Hello World!'}

@app.get('/solar')
async def rag_solar_get(user_message):
    return {'message': main.main(user_message)}

@app.post('/solar')
async def rag_solar_get(param: dict={}):
    user_message = param.get('user_message', ' ')
    return {'message': main.main(user_message)}