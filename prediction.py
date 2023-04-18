import pandas as pd 
import re
import string
import pandas as pd
import joblib
from string import digits

LR = joblib.load('LR_latest.pkl')
vectorization = joblib.load('vectoriser.pkl')


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

def output_lable(n):
    if n == 1:
        return "Fake News hai Bhai"
    elif n == 0:
        return "Nahi Fake News nahi hai Bhai"
    
def manual_testing(news):

    remove_digits = str.maketrans('', '', digits)
    ini_string=str(news)
    news = ini_string.translate(remove_digits)
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)

    return print("\n\nLR Prediction: {}".format(output_lable(pred_LR[0])))


def main() : 
    print('Batao kis news ke baare mein btaun :')
    news = str(input())
    print('Ruko Abhi Batata hun.....')
    manual_testing(news)



if __name__ == '__main__':
    main()


