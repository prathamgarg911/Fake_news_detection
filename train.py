from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import joblib
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from string import digits
from sklearn.linear_model import LogisticRegression


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



vectorization = joblib.load('vectoriser.pkl')
xv_train = vectorization.transform(x_train)
xv_test = vectorization.transform(x_test)

print('vectoriser saved')
pac = PassiveAggressiveClassifier()
LR = LogisticRegression()
LR.fit(xv_train,y_train)
joblib.dump(LR,'LR_latest.pkl' )
pred_lr=LR.predict(xv_test)
# x_test.to_csv("xtest.csv")
# pac.score(xv_test, y_test)
print(classification_report(y_test, pred_lr))

def output_lable(n):
    if n == 1:
        return "Fake News hai Bhai" 
    elif n == 0:
        return "Nahi Fake News Nahi hai Bhai"
    
def manual_testing(news):
	
    print('Batata hun ruko zara')	
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)

    return print("\n\n Prediction: {}".format(output_lable(pred_LR[0])))

print('hanji batao konsi news puchhoge')
news = str(input())
manual_testing(news)






