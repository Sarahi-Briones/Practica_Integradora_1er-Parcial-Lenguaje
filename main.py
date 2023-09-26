import pandas as pd
import limpieza_texto
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize

# df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')
# df['Opinion']= limpieza_texto.procesar_texto(df['Opinion'])
#df.to_excel('archivo_procesado.xls', index=False, engine='openpyxl')

#--------------------------Extraemos el dataset----------------------------
df = pd.read_excel('archivo_procesado.xls')
opiniones = df['Opinion']
lista_opiniones = opiniones.values.tolist()
tagged_data = [TaggedDocument(word_tokenize(_d),str(i)) for i,_d in enumerate(lista_opiniones)]

model = Doc2Vec(window=10,dm=1)
model.build_vocab(tagged_data)

#Entrenar el modelo
print("empieza el entrenamiento")
max_epochs = 100 
i=0
for epoch in range(max_epochs):
    model.train(tagged_data,total_examples = model.corpus_count, epochs = model.epochs)
    model.alpha =-0.0002
    model.min_alpha = model.alpha
    print(f"epoca: {i}")
    i=i+1
model.save("d2v_01.model")