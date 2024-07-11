import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import sentencepiece
#from transformers import  T5Tokenizer, T5ForConditionalGeneration 
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

from peft import PeftModel, PeftConfig

from PIL import Image
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def extract_triples(sentences):
   
    dict_op = {'great': 'POS', 'bad': 'NEG', 'ok': 'NEU'}
    lista_aspectos = []
    lista_opinion = []
    lista_polarity = []
    lista_triple = []
    lista_erros = []
    lista_ok = []
    lista_status = []


    try:
        for i in sentences:
            temp = i.split(',') #[sep]')
        for sent in temp:
            try:
              part1,part2 = sent.split('because')
              #print("Parte 1:", part1)
              #print("Parte 2:", part2)
              #st.write("Parte 1: "+ part1)
              #st.write("Parte 2: "+  part2)
              part1 = part1.replace('isnt', "is not").replace("isn't", "is not").replace("isn 't", "is not")
              asp1, pol1 = part1.split(' is ')
              #print("Aspecto 1:", asp1)
              #print("Polaridade:", pol1)
              lista_aspectos.append(asp1.strip())
              lista_polarity.append(dict_op.get(pol1.strip()))

              part2 = part2.replace('isnt', "is not").replace("isn't", "is not").replace("isn 't", "is not")
              #print("Parte2:", part2)
              #print(part2.split(' is '))
              asp2, op1 = part2.split(' is ')
              asp2 = asp2.strip()
              op1 = op1.split()[-1]
              #print("Opiniao:", op1)
              if asp2 not in lista_aspectos:
                lista_aspectos.append(asp2)
              lista_opinion.append(op1)
              #print("Parts ok:", sent)
              lista_status.append('ok') # Quando a triple na sentença estiver ok não há erro.

            except:
              #print("Parts bad", sent)
              lista_erros.append(sent)
              lista_status.append('error')

    except:
        #print(" - Checar split < is > :", i)
        print("Sentença extraida com problema :",i )
        st.error('Sorry. Try again or wait a moment', icon="🚨")



    #print('\nAspectos:',lista_aspectos)
    size = len(lista_aspectos)
    try:
      for i in range(size):
        lista_triple.append((lista_aspectos[i].lower(), lista_opinion[i].lower(), lista_polarity[i]))
    except:
      #print(" - Checar lista_triple ou lista_aspecto")
      pass

    return lista_aspectos, lista_opinion, lista_polarity
    

def get_data(df,column):
    l_data = df[column]
    return l_data
    
    
# App title
st.set_page_config(page_title="🦙 ABSA")    

html_page_llm = """
     <div style="background-color:black;padding=50px">
         <p style='text-align:center;font-size:30px;font-weight:bold'>LLM</p>
     </div>
               """

html_page_title = """
     <div style="background-color:black;padding=50px">
         <p style='text-align:center;font-size:45px;font-weight:bold'>Aspect Based Sentiment Analysis</p>
     </div>
               """               
st.markdown(html_page_title, unsafe_allow_html=True)

html_page_title2 = """
     <div style="background-color:tomato;padding=40px">
         <p style='text-align:center;font-size:30px;font-weight:bold'>Aspect / Opinion / Polarity</p>
     </div>
               """
st.markdown(html_page_title2, unsafe_allow_html=True)

image3 = Image.open("imgs/fig2.png")      # Prompt Reviews
image4 = Image.open("imgs/logo_absa.png") # Home
image2 = Image.open("imgs/about.png")   # About
image1 = Image.open("imgs/fig1.png")      # Prompt Reviews
image5 = Image.open("imgs/logo2.png")      # Domain Reviews
image6 = Image.open("imgs/positive.jpeg")      # POS
image7 = Image.open("imgs/negative.jpeg")      # NEG
image8 = Image.open("imgs/neutral.jpeg")      # NEU

max_length = 512
 
#hf_model = "SilvioLima/absa_parafrase"
hf_model = "SilvioLima/absa_domain_parafrase_v512"


prefix_fs = "Follow the example, rewrite the sentence and answer in the same format of example."
example_task_parafrase = "\nExample Sentence: 'Excellent atmosphere , delicious dishes good, friendly service but location is dangerous.\nAnswer = 'atmosphere is great because atmosphere is Excellent , dishes is great because dishes is delicious , service is great because service is good , service is great because service is friendly, location is bad because location is dangerous. </s>'"
answer = "\nAnswer ="
sentence = "\nSentence: " 

################

instruction_zs = "Aspects are nouns in reviews and opinions are adjectives in reviews. Identify all aspects and opinions in this review and format the output as (aspect, opinion). Review: "


# PEFT
#peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
#tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

#parafrase_peft = PeftModel.from_pretrained(peft_model_base, 
#                        'SilvioLima/absa_parafrase_peft', 
#                        torch_dtype=torch.bfloat16,
#                        is_trainable=False)                                     
#model = parafrase_peft 
 
repo_model = "SilvioLima/absa_1_domain_lora_large"
model_name_or_path = repo_model
config = PeftConfig.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(repo_model)


def plot1(df):
    # Agrupar os dados por 'Aspect' e contar a quantidade de cada aspecto
    aspect_counts = df['Aspect'].value_counts().reset_index()
    aspect_counts.columns = ['Aspect', 'Counts']

    # Agrupar os dados por 'Aspect' e 'Polarity' e contar a quantidade de cada combinação
    #aspect_polarity_counts = df.groupby(['Aspect', 'Polarity']).size().reset_index(name='Counts')

    # Plotar a quantidade de aspectos
    #st.write('Count per Aspect:')
    fig1, ax1 = plt.subplots()
    sns.barplot(data=aspect_counts, x='Aspect', y='Counts', ax=ax1)
    ax1.set_title('Count per Aspect')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)
 

def plot2(df):
    # Agrupar os dados por 'Aspect' e contar a quantidade de cada aspecto
    aspect_counts = df['Aspect'].value_counts().reset_index()
    aspect_counts.columns = ['Aspect', 'Counts']

    # Agrupar os dados por 'Aspect' e 'Polarity' e contar a quantidade de cada combinação
    aspect_polarity_counts = df.groupby(['Aspect', 'Polarity']).size().reset_index(name='Counts')

    # Passo 2: Calcular o percentual de cada polaridade em relação ao total de polaridades para cada aspecto
    total_counts = aspect_polarity_counts.groupby('Aspect')['Counts'].transform('sum')
    aspect_polarity_counts['Percentage'] = aspect_polarity_counts['Counts'] / total_counts * 100

    # Passo 3: Plotar o gráfico empilhado com essas porcentagens
    fig3, ax = plt.subplots()
    sns.barplot(data=aspect_polarity_counts, x='Aspect', y='Percentage', hue='Polarity', ax=ax, ci=None, palette={"POS": "green", "NEG": "red", "NEU": "yellow"})
    ax.set_title('Percentage of Polarity per Aspect')
    ax.legend(bbox_to_anchor= (1.2,1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Adicionar porcentagens no gráfico
    #for container in ax.containers:
    #    ax.bar_label(container, fmt='%.1f%%', label_type='center')

    st.pyplot(fig3)
    
def plot3(df):
    # Agrupar os dados por 'Opinion' e contar a quantidade de cada opiniao
    opinion_counts = df['Opinion'].value_counts().reset_index()
    opinion_counts.columns = ['Opinion', 'Counts']
    
    # Agrupar os dados por 'Aspect' e 'Polarity' e contar a quantidade de cada combinação
    opinion_polarity_counts = df.groupby(['Opinion', 'Polarity']).size().reset_index(name='Counts')


    # Plotar a quantidade de opinioes
    #st.write('Count per Opinion:')
    fig1, ax1 = plt.subplots()
    sns.barplot(data=opinion_polarity_counts, x='Opinion', y='Counts', ax=ax1, hue='Polarity', ci=None, palette={"POS": "green", "NEG": "red", "NEU": "yellow"})
    
    ax1.set_title('Count per Opinion')
    ax1.legend(bbox_to_anchor= (1.2,1))
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)
 
# Função para explodir apenas colunas com listas contendo mais de um valor
def conditional_explode(df, columns):
    for column in columns:
        if df[column].apply(lambda x: isinstance(x, list) and len(x) > 1).any():
            df = df.explode(column)
        else:
            # Garantir que valores únicos sejam mantidos
            df[column] = df[column].apply(lambda x: x if isinstance(x, list) else [x])
            df = df.explode(column)    
    return df 

def color_polarity(val):
    color = ''
    if val == 'POS':
        color = 'background-color: green'
    elif val == 'NEG':
        color = 'background-color: red'
    elif val == 'NEU':
        color = 'background-color: yellow'
    return color

#def apply_colors(row):
#    polarity_color = color_polarity(row['Polarity'])
#    return [polarity_color, polarity_color, polarity_color, '', '', '', '', '']

# Função para colorir texto usando HTML
def color_text(aspect, polarity):
    colors = {
        'POS': 'green',
        'NEG': 'red',
        'NEU': 'yellow'
    }
    return f"<span style='color:{colors[polarity]}; font-size:30px'>{aspect}</span>"

# Função para aplicar cores aos aspectos nas reviews
def apply_colors_to_review(row):
    review = row['Review']
    aspect = row['Aspect']
    opinion = row['Opinion']
    polarity = row['Polarity']
    #if polarity == 'POS':
    #   emoji = '👍'
    #else:
    #   emoji = '👎'    
    colored_aspect = color_text(aspect, polarity)
    colored_opinion = color_text(opinion, polarity)
    review = review.replace(aspect, colored_aspect)
    review = review.replace(opinion, colored_opinion)
    return review # emoji + review
  
    

    
def apply_colors(row):
    return [color_polarity(row['Polarity']) if col in ['Aspect', 'Opinion', 'Polarity'] else '' for col in row.index]
    

def chatbot(input_text):
    # Adicionando o token de separação de contexto para a pergunta
    #input_text = instruction + question

    # Tokenizar a entrada
    #inputs = tokenizer.encode(input_text, return_tensors="pt")
    inputs  = tokenizer(input_text, max_length=max_length, truncation=True ,\
                return_tensors="pt", padding="max_length")

    input_ids = inputs['input_ids']
    
    # Gerar resposta
    #output = model.generate(input_ids)
    outputs = model.generate(input_ids = input_ids, max_new_tokens=200)

    # Decodificar a saída
    #response = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_part = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_part
 

#tokenizer, model = load_model()
#st.sidebar.write("System: done")

ds = load_dataset('SilvioLima/absa', streaming=True)

df = pd.DataFrame.from_dict(ds['test'])

df['domain'] = df['domain'].str.upper() 

st.sidebar.image(image4,caption="", use_column_width=True)

st.sidebar.markdown(html_page_llm, unsafe_allow_html=True)

with st.sidebar:
    st.header("Menu")
    activities = ["Domain Reviews","Prompt Reviews", "Read Dataset", "About"]
    choice = st.radio(" menu ",activities, label_visibility='collapsed')

 
# Define custom CSS for button colors
#button_color = "#0000FF"  # Blue color for Button 

#with st.sidebar:
if choice == 'Domain Reviews': 
    st.subheader("Domain Reviews")
    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
        
    #st.sidebar.markdown("#### Choose a domain")
    st.sidebar.subheader("Choose a Domain")
    option_domain = st.sidebar.selectbox('Choose a Domain',set(get_data(df, 'domain')), label_visibility = 'collapsed')
  
    df_domain = df.loc[df['domain'] == option_domain]
    st.subheader("Review")
    review = st.selectbox('Review',get_data(df_domain, 'sentence'), label_visibility = 'collapsed')
    st.subheader(review)
    
    #with st.sidebar:    
    if st.button("Extract  Triples"):
        try:
            with st.spinner('Wait for it...we are processing'):
 
               
                #input_text = instruction + " " + review
                input_text = prefix_fs + example_task_parafrase + sentence + review + answer
                inputs = tokenizer(input_text, max_length=max_length, truncation=True ,\
                return_tensors="pt", padding="max_length")
                
                input_ids = inputs['input_ids']
                
                outputs = model.generate(input_ids = input_ids, max_new_tokens=200)
                
                #outputs = model.generate(inputs_ids, max_length=max_length) #do_sample=True, temperature=(temperature).astype('float'))
                generated_part = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                aspectos, opinions, polarities  = extract_triples([generated_part])
                
                bar = st.progress(0)
                for i in range(11):
                    bar.progress(i * 10)
                    # wait
                    time.sleep(0.1)
                    
                col1, col2, col3 = st.columns(3)
    
                with col1:
                    st.image(image5,caption="", use_column_width=False)
                  
                with col2:
                    st.image(image5,caption="", use_column_width=False)
                    
                with col3:
                    st.image(image5,caption="", use_column_width=False)

                data = {'ASPECTS': aspectos,
                        'OPINIONS': opinions,
                        'POLARITY': polarities}
                df = pd.DataFrame(data)
                st.table(df)
                
                
        except:
            st.error('Sorry. Try again or wait a moment', icon="🚨")

elif choice == 'Prompt Reviews':

    with st.sidebar:
        st.header("Process Review")
        activities = ["Zero-shot", "Few-shot"]
        proc = st.radio(" menu ",activities, label_visibility='collapsed')   
        
    
    # Interface do Streamlit
    st.subheader("Prompt Reviews")
    st.markdown("##### Example: Pizza is good, drinks are excellents, the price is very expensive")

    #st.sidebar.image(image1,caption="", use_column_width=True)


    # Entrada de texto para a pergunta do usuário
    #st.markdown("#### Faça um comentário:")
    user_review = st.text_input(label = "Write a review:")
    
    
    
    if proc == "Basic":
        instruction = "Tell me something about this review: "
        st.sidebar.markdown('#### instruction = "Tell me something about this review:')
        st.sidebar.markdown('#### inputs = instruction +user_review')
        inputs = instruction + user_review
    elif proc == "Zero-shot":
        inputs = instruction_zs + user_review
        st.sidebar.markdown('#### instruction_zs = "Aspects are nouns in reviews and opinions are adjectives in reviews. Identify all aspects and opinions in this review and format the output as (aspect, opinion). Review: "')
        st.sidebar.markdown('#### inputs = instruction_zs + user_review')
    elif proc == "Few-shot":
        inputs = prefix_fs + example_task_parafrase + sentence + user_review + answer
        st.sidebar.markdown('#### prefix_fs = "Follow the example, rewrite the sentence and answer in the same format of example."')
        st.sidebar.markdown("#### example_task = 'Example Sentence: The pizza was delicious because pizza came hot. Answer = Pizza is great because pizza came hot.")
        #st.sidebar.markdown("#### answer = 'Answer = '")
        #st.sidebar.markdown("#### sentence = 'Sentence = '")
        st.sidebar.markdown('#### inputs = prefix_fs + example_task + user_review + answer')   
        

    # Botão para enviar a pergunta
    if st.button("Enviar"):
        try:
            with st.spinner('Wait for it...we are processing'):
            # Verificando se a pergunta não está vazia
                if user_review.strip() != "":
                    # Obtendo a resposta do chatbot
                    bot_response = chatbot(inputs)
                    # Exibindo a resposta
                    st.subheader(proc)
                    st.markdown("#### Answer:")
                    st.subheader(bot_response)
                    aspect, opinion, polarity = extract_triples([bot_response])
                    
                    col1, col2, col3 = st.columns(3)
    
                    #with col1:
                    #    st.image(image3,caption="", use_column_width=False)
                  
                    with col2:
                        st.image(image3,caption="", use_column_width=False)
                 
                    #with col3:
                    #    st.image(image3,caption="", use_column_width=False)

                    data = {'ASPECTS': aspect,
                        'OPINIONS': opinion,
                        'POLARITY': polarity}
                    df = pd.DataFrame(data)
                    st.table(df)
        
                else:
                    st.markdown("#### Please, write a review:.")
        except:
            st.error('Sorry. Try again or wait a moment', icon="🚨")
       

elif choice == "Read Dataset":
    try:
        uploaded_file = st.file_uploader("Choose a csv file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            lista_review = df['review'].to_list()
            l_parafrase = []
            l_aspect = []
            l_opinion = []
            l_polarity = []
            l_review = []
            size = len(lista_review)
            st.write('Size: '+str(size))
            number = st.number_input(label = "Number of reviews to analyse", min_value=1, max_value=15, value=3 )
            st.write("Number of reviews ", number)
            if st.button("Start"):
                with st.spinner('Wait for it...we are processing'):
                    for i in range(0,number):
                        user_review = lista_review[i]
                        st.write('Review ',i+1, ': ', user_review)
                        l_review.append(user_review)
                        #st.write('Review: '+user_review)
                        #st.write(user_review)
                        inputs = prefix_fs + example_task_parafrase + sentence + user_review + answer
                        #st.write(inputs)
                        bot_response = chatbot(inputs)
                        time.sleep(1)
                        l_parafrase.append(bot_response)
                        aspect, opinion, polarity = extract_triples([bot_response])
                        #aspect = aspect.lower()
                        l_aspect.append(aspect)
                        l_opinion.append(opinion)
                        l_polarity.append(polarity)
                    
                #st.write('review', len(l_review))
                #st.write('parafrase', len(l_parafrase))
                #st.write('aspect', len(l_aspect))
                #st.write('opinion', len(l_opinion))
                #st.write('polarity', len(l_polarity))
                
                data = {'Review': l_review,
                    'Parafrase': l_parafrase,
                    'Aspect': l_aspect,
                    'Opinion': l_opinion,
                    'Polarity': l_polarity}
                df1 = pd.DataFrame(data)
                
                
                #st.markdown("### Dataset Final")
                # Aplicar a função no DataFrame
                df2 = df1[['Review', 'Aspect', 'Opinion', 'Polarity']]
                df_expanded = conditional_explode(df2, ['Aspect', 'Opinion', 'Polarity']).reset_index(drop=True)
                # Alguns registros mostram 'doctor' e outros 'doctors', definiu-se pela forma no plural.
                df_expanded['Aspect'] = np.where(df_expanded['Aspect'] == 'doctors', 'doctor', df_expanded['Aspect'])
                #df_expanded['Aspect'] = np.where(df_expanded['Aspect'] == 'doctors', 'doctor', df_expanded['Aspect'])
                
                df_expanded = df_expanded.drop_duplicates()
                #st.write('df4')              
                df3 = df_expanded.reset_index(drop=True)
                df4 = df3
                #st.table(df4)
                # Aplicar coloração aos aspectos dentro das reviews
                df4['Colored_Review'] = df4.apply(apply_colors_to_review, axis=1)
                #st.table(df4)
                # Converter o DataFrame estilizado para HTML
                #st.write('df4')
                #st.table(df4)
                #html_reviews_raw = df4['Review'].to_list()
                #html_reviews_raw = "<br><br>".join(html_reviews_raw)
                
                html_reviews = df4['Colored_Review'].to_list()
                html_reviews = "<br><br>".join(html_reviews)
                # Mostrar as reviews coloridas no Streamlit
                st.markdown('### Identify Aspects and Opinions using Colors')
             
                col1, col2, col3 = st.columns(3)
    
                with col1:
                    st.markdown("### Positive")
                    st.image(image6,caption="", use_column_width=False)
                  
                with col2:
                    st.markdown("### Negative")
                    st.image(image7,caption="", use_column_width=False)
                    
                with col3:
                    st.markdown("### Neutral")
                    st.image(image8,caption="", use_column_width=False)
           
                # Aplicar a coloração ao DataFrame
                styled_df = df3.style.apply(apply_colors, axis=1)
                # Mostrar o DataFrame estilizado no Streamlit
                st.markdown("### Dataset Final - Reviews 👍 👎")
                st.dataframe(styled_df)
                #df2 = df_expanded
                #st.table(df2.style.applymap(color_polarity, subset=['Aspect', 'Opinion','Polarity']))
                #st.table(df_expanded)
                # Salvar o dataframe
                csv = df_expanded.to_csv(index=False)
                # Graph
                st.markdown("### Aspects, Opinions and Polarities in reviews")
                plot1(df3)
                plot2(df3)
                plot3(df3)
                
                
                st.markdown("### O dataset pode ser salvo com nome df_review_absa.csv")
                # Adicionar um botão para baixar o CSV
                st.download_button(
                       label="Baixar CSV",
                       data=csv,
                       file_name='df_review_absa.csv',
                       mime='text/csv'
)                                             
               
    except:
        st.write("\nOps...Hostoun we have a problem")
        st.error('Sorry. Try again or wait a moment', icon="🚨")        
else:
        st.subheader("About:")
        st.markdown("#### Identify aspects, opinions and polarities in product reviews.")
        st.markdown("#### Example:  ")
        st.markdown("  ")
       
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            st.image(image2,caption="", use_column_width=False)

        st.sidebar.markdown("### Configuração")
        st.sidebar.write("Model: google/flan-t5-base")
        st.sidebar.write("Tokenizer: T5Tokenizer / T5ForConditionalGeneration")
        st.sidebar.write("LORA applied ")
        st.sidebar.write("Train: 10810")
        st.sidebar.write("Valid/Test: 1352")
        st.sidebar.write("Batch-size: 4")
        st.sidebar.write("Max_length: 512")
        st.sidebar.write("Num_Epochs: 10")
