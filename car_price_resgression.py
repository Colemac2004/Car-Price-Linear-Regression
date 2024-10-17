import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

#define objects
#minmax
scaler=MinMaxScaler()
#linear regression
lm=LinearRegression()
#label encoder
label_encoder=LabelEncoder()

#function for getting objects then encoding them
def hot_one_encoding(df):
    #first get objects
    df=df.select_dtypes(include='object')
    #now encode objects
    df=pd.get_dummies(df,dtype=int)
    return df

#funciton for normalizing data
def normalize(df):
    df_scaled=scaler.fit_transform(df)
    df=pd.DataFrame(df_scaled,columns=df.columns)
    return df

#function for getting other things
def get_int_float(df):
    #get ints
    df_int=df.select_dtypes(include='int')
    df_float=df.select_dtypes(include='float')
    df=concat_dataframes(df_int,df_float)
    return df

#function fo rconcating to dataframes
def concat_dataframes(frame1,frame2):
    df=pd.concat([frame1,frame2],axis=1)
    return df

#normalize data
def label_encode(df):
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    return df_encoded

#function for splitting data
def sklearn_split_data(df,test_ratio,y_df):
    #split data
    X_train,X_test,y_train,y_test=train_test_split(df,y_df,test_size=test_ratio,random_state=42)
    return X_train,X_test,y_train,y_test


#function for seaborn plots
def seasborn_plots(plot,x_name,y_name,df_data):
    if plot=="jointplot":
        sns.jointplot(x=x_name,y=y_name,data=df_data,alpha=0.5)
    elif plot=="pairplot":
        sns.pairplot(df_data,kind="scatter",plot_kws={'alpha':0.5})
    elif plot=="lmplot":
        sns.lmplot(x=x_name,y=y_name,data=df_data,scatter_kws={'alpha':0.5})
    elif plot=="displot":
        sns.displot(df_data,bins=30)
    #show data
    plt.show()



#read csv
df=pd.read_csv('CarPrice_Assignment.csv')

#define price
y_df=df[['price']]
df=df.drop('price',axis=1)

#get objects encoded
df_encoded=label_encode(df)

#get ints and floats
df_numeric=get_int_float(df)

#normalize ints and floats
df_normalized=normalize(df_numeric)

#now concat normalized and encoded
data_df=concat_dataframes(df_encoded,df_normalized)

#now we need to split data
X_train,X_test,y_train,y_test=sklearn_split_data(data_df,0.2,y_df)

#fit data
lm.fit(X_train,y_train)
#print(lm.coef_)


#now we predict data
predictions=lm.predict(X_test)
print(mean_absolute_error(y_test,predictions))


#now we can get restricitons
restrictions=y_test-predictions
#print(restrictions)

#make plot
plt.hist(restrictions,bins=30)
plt.show()
