import json
import requests

import streamlit as st


class rag_web():
    def __init__(self):
        self.URL = 'http://127.0.0.1:8080/solar'

    def get_answer(self, message):
        param = {'user_message': message}
        resp = requests.post(self.URL, json=param)
        output = json.loads(resp.content)['message']

        return output

    def window(self):
        st.title('항공편 조회 챗봇')
        st.caption('developed by QA Engine Team2')

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for content in st.session_state.chat_history:
            with st.chat_message(content['role']):
                st.markdown(content['message'])


        if prompt := st.chat_input('메시지를 입력하세요.'):
            with st.chat_message('user'):
                st.markdown(prompt)
                st.session_state.chat_history.append({'role': 'user', 'message': prompt})

            with st.chat_message('ai'):
                response = self.get_answer(prompt)
                st.markdown(response)
                st.session_state.chat_history.append({'role': 'ai', 'message': response})


if __name__ == '__main__':
    st.set_page_config(
        page_title='항공편 조회 챗봇 Demo',
        page_icon='✈️'
    )
    web = rag_web()
    web.window()