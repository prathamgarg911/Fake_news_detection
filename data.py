import pandas as pd 


print('started')


df = pd.read_csv("fake_real_news_78k (1).csv")
df_True, df_False = [x for _, x in df.groupby(df['label'] == "TRUE")]
df_False["class"] = 1
df_True["class"] = 0
df_merge = pd.concat([df_False, df_True], axis =0 )
df_merge.head(10)
df_merge.columns
df_merge_drop = df_merge.drop(["Unnamed: 0","title","label"], axis=1)
df_merge_drop = df_merge_drop.sample(frac = 1)
df_merge_drop.reset_index(inplace = True)
df_merge_drop.drop(["index"], axis = 1, inplace = True)


df_merge_drop.to_csv('data.csv',index=False)
