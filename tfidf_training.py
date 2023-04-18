from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import string
import joblib
import pandas as pd
from string import digits


df_merge_drop = pd.read_csv("data.csv")



def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

for i in range(0,len(df_merge_drop)):
    remove_digits = str.maketrans('', '', digits)
    ini_string=str(df_merge_drop["text"][i])
    df_merge_drop["text"][i] = ini_string.translate(remove_digits)


df_merge_drop["text"] = df_merge_drop["text"].apply(wordopt)
x = df_merge_drop["text"]
y = df_merge_drop["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)



vectorization = TfidfVectorizer(max_features=100)
vectorization = vectorization.fit(x_train)
print('Vectoriser toh hogya train')
joblib.dump(vectorization,'vectoriser.pkl')

