from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate, load_prompt


load_dotenv()
llm1= HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     task="text-generation",pipeline_kwargs=dict(temperature=0.4,max_new_tokens=512)
         )
model1 = ChatHuggingFace(llm=llm1)
st.header('Quiz Preparation Tool')
topic_input = st.selectbox('Select Research Paper Name',['Select','Supervised Learning','Decision Trees', 'SVM',"Knn","Logistic and Linear regression"])
#style_input = st.multiselect("Select Explanation Style",["Select","Code-heavy","Math heavy","Analogies","Simple layman terms beginner friendly","Technical"])
#length_input = st.selectbox("Select Explanation Length",["1-2","3-5","more than 5"])
#template1 = load_prompt("template1.json")
template1 = PromptTemplate(template="""
Give exactly only 4 questions on {topic_input}. Each question will have 4 options to select from, out of which only 1 should be the right answer to the question.
Each multi choice option must be different from each other.
Focus either on:
- intuition or applications or conceptual understanding


Include correct answers.
""", input_variables=['topic_input'],
validate_template=True
)
if st.button("Questions"):
    chain1 = template1 | model1
    result1 = chain1.invoke({
    'topic_input':topic_input})
    
    st.write(result1.content)
    #static prompt is when you always have to write a new query for a new question
    #Based on prompt, llm output changes a lot, so if the user writes a weak prompt they might get a wrong answer.
    #Hence we need dynamic prompting- a template so that users get the same style of response everytime.
    #using chains, only one invoke will suffice
    #'style_input':style_input,
    #'length_input':length_input
