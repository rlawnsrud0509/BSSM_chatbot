import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")


@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model


@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('bsg_chat.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df


model = cached_model()
df = get_dataset()

st.header('부산소프트웨어마이스터고 챗봇')
st.subheader("안녕하세요 소마고 챗봇입니다.")

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('물어볼 말!!!', '')
    submitted = st.form_submit_button('질문하기')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(
        lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] >= 0.7:
        st.session_state.generated.append(answer['대답'])
    elif answer['distance'] < 0.7:
        st.session_state.generated.append(
            '저도 잘 모르겠어요. 학교로 문의해보세요 (051-971-2153)')

for i in range(len(st.session_state['past'])):
    st.markdown(
        """
        <div class="Rcontainer">
            <div class="right">
                <div class="msg-img"></div>
                    {0}
                </div>
            </div>
        </div>
        <div class="Lcontainer">
            <div class="left">
                <div class="msg-img" style="background-color: cadetblue"></div>
                    <div class="msg-bubble-l">
                        <div class="msg-info"></div>
                        <div class="msg-info-name"></div>
                    </div>
                    {1}
                </div>
            </div>
        </div>
        
    """.format(st.session_state['past'][i], st.session_state['generated'][i]),
        unsafe_allow_html=True)


st.sidebar.header('BSSM 챗봇')
st.sidebar.markdown(
    'BSSM 챗봇은 물어보면 뭐든지 알려줘용')

st.sidebar.header('만든 목적')
st.sidebar.markdown('''
    신입생, 학부모님! BSSM 챗봇과 함께 좋은 시간 되세용
''')

st.sidebar.header('TEL -> 051-971-2153')
st.sidebar.header('Instagram -> https://www.instagram.com/bssm.hs/')
st.sidebar.header('Youtube -> https://www.youtube.com/@user-jl3rs8zp4w')
