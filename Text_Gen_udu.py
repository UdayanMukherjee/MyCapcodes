import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
import tensorflow.keras.utils
from tensorflow.keras.callbacks import ModelCheckpoint
file =open("frankenstein-2.txt", encoding="utf-8").read()
#tokenization
#standardization
def tokenize_words(input):
    input=input.lower
    tk=RegexpTokenizer(r'\w+')
    tokens=tk.tokenize(input)
    fd=filter(lambda token: token not in stopwords.words('english'), tokens )
    return " ".join(fd)
processed_inputs=tokenize_words(file)
#chars to numbers
chars=sorted(list(set(processed_inputs)))
char_to_num= dict((c,i) for i,c in enumerate(chars))
#check if words to num or chars to num has worked
input_len=len(processed_inputs)
vocab_len=len(chars)
print("Total number of characters:", input_len)
print("Total vocab:", vocab_len)
#seq_length
seq_length=[]
x_data= []
y_data= []
#loop thru the sequence 
for i in range(0,input_len = seq_length,1):
    in_seq= processed_input(i:1+seq_length)
    out_seq= processed_input(i+seq_length)
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append([char_to_num[out_seq]])
n_patterns=len(x_data)
print("Total Patterns:", n_patterns)
#convert input sequence to np array
X=numpy.reshape(x_data,(n_patterns,seq_length,1))
X=X/float(vocab_len)
#one-hot encoding
y=np_utils.to_categorical(y_data)
model=Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2]),return_sequences=True)
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True)
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
#compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam')
#
filepath="model_weights_saved.hdf5"
ck=ModelCheckpoint(filepath,monitor='loss',verbose='1',save_best_only=True,mode='min')
desired_callbacks=[ck]
model.fit(X,y,epochs=4,batch_size=256, callbacks=desired_callbacks)
filename="model_weights_saved.hdf5"
model.load_weigths(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')
num_to_char= dict((i,c) for i,c in enumerate(chars))
start=numpy.random.randint(0,len(x_data)-1)
pattern=x.data[start]
print("Random Seed :")
print("\"", ''.join([num_to_char[value]for value in pattern]),"\"")
for i in range(1000):
    x=numpy.reshape(pattern,(1,len(pattern),1))
    x=x/float(vocab_len)
    pred=model.predict(x,verbose=0)
    index=numpy.argmax(pred)
    result=num_to_char[index]
    seq_in=[num_to_char[value]for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern=pattern[1:len(pattern)]
          
