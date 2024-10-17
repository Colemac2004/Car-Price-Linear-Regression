import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


#define objects
#label encoder
label_encoder=LabelEncoder()
#minmax scaler
scaler=MinMaxScaler()
#linear regression
lm=LinearRegression()


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

#function for getting all objects
def get_objects(df):
    df=df.select_dtypes(include='object')
    return df

#function for label encoding objects
def label_encode(df):
    df_encoded=df.copy()
    cols=df_encoded.columns
    for col in cols:
        df[col]=label_encoder.fit_transform(df[col])
    return df

#function for getting all ints and floats
def get_ints_and_floats(df):
    df_ints=df.select_dtypes(include='int')
    df_float=df.select_dtypes(include='float')
    return concat_dfs(df_ints,df_float)

#function for concattting
def concat_dfs(df_1,df_2):
    df_concated=pd.concat([df_1,df_2],axis=1)
    return df_concated

#function for min max scalling
def min_max_scaler(df):
    df_scaled=scaler.fit_transform(df)
    df_min_max=pd.DataFrame(df_scaled,columns=df.columns)
    return df_min_max

#function for getting y and dropping it from dataframe
def get_y(df):
    y_df=df[['price']]
    df=df.drop('price',axis=1)
    return df,y_df

#function for getting part of df
def get_part_of_df(df):
    df_part = df[['carwidth', 'carheight', 'stroke', 'compressionratio', 'wheelbase','price']]    
    return df_part

#function for looping through each
def each_linear_regression(df,y_df):
    for col in df.columns:
        temp_df = df[[col]]        
        #now we are going to have to split data
        X_train,X_test,y_train,y_test=split_data(temp_df,y_df,0.2)
        #now fit linear model
        lm.fit(X_train,y_train)
        #now make predictions
        predictions=lm.predict(X_test)
        #now calculate MAE
        print(mean_absolute_error(y_test,predictions))

#function for splitting data
def split_data(df,y_df,test_ratio):
    X_train,X_test,y_train,y_test=train_test_split(df,y_df,test_size=test_ratio,random_state=42)
    return X_train,X_test,y_train,y_test


#read file
df=pd.read_csv('CarPrice_Assignment.csv')
df,y_df=get_y(df)
#print(df.info())

#okay now label encode data
df_objects=get_objects(df)

#okay now label encode
df_label_encoded=label_encode(df_objects)
#print(df_label_encoded.info())

#okay onw we have to get ints and floats
df_ints_and_floats=get_ints_and_floats(df)
#print(df_ints_and_floats.info())

#okay now concat df ints and float and label encoded
data_df=concat_dfs(df_label_encoded,df_ints_and_floats)
#print(data_df.info())

#okay now minmax scale them
data_normalized=min_max_scaler(data_df)
print(data_normalized.info())

#see distribution of data
#get part of dataframe
#df_part=get_part_of_df(data_normalized)
#seasborn_plots('pairplot','a','b',df_part)

#okay now we are going to see the normalization of each part
each_linear_regression(data_normalized,y_df)