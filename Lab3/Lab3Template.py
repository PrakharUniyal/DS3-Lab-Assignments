import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(path_to_file):
	#Returns Pandas dataframe containing given csv file.
    return pd.read_csv(path_to_file)
 
def show_box_plot(attribute_name,df):
	#Displays boxplot for given atrribute in dataframe.
    plt.boxplot(df[attribute_name])
    plt.show()
    
def replace_outliers(df):
    print("Replacing Outliers with median value..")
    for i in list(df):
        q1,q2,q3  = df[i].quantile(0.25),df[i].quantile(0.50),df[i].quantile(0.75)
        lb,ub = (q1 - (3/2)*(q3-q1)) , (q3 + (3/2)*(q3-q1))
        print("Change in no. of outliers in",i,":",df[i][df[i]<lb].count()+df[i][df[i]>ub].count(),"--> ",end='')
        
        df[i].mask(df[i]<lb,q2,inplace=True)
        df[i].mask(df[i]>ub,q2,inplace=True)
        
        q1,q2,q3  = df[i].quantile(0.25),df[i].quantile(0.50),df[i].quantile(0.75)
        lb,ub = (q1 - (3/2)*(q3-q1)) , (q3 + (3/2)*(q3-q1))
        print(df[i][df[i]<lb].count()+df[i][df[i]>ub].count())
    return df

def range(df,attribute_name):
    return df[attribute_name].min(),df[attribute_name].max()

def min_max_normalization(df):
    for i in list(df): df[i] = (df[i] - range(df,i)[0])/(range(df,i)[1]- range(df,i)[0])
    return df

def standardize(df):
    for i in list(df): df[i] = (df[i] - df[i].mean())/df[i].std()
    return df

def main():
    path_to_file="landslide_data2_original.csv"
    
    df=read_data(path_to_file)
    
    data = df[["temperature","humidity","rain"]]
    
    for i in list(data): show_box_plot(i,data)
    
    ndata1 = replace_outliers(data.copy())
    
    for i in list(ndata1): show_box_plot(i,ndata1)
    
    print(ndata1.head(),"\n\nInitial Ranges:",[range(ndata1,i) for i in list(ndata1)],"\n")
    
    ndata2 = min_max_normalization(ndata1.copy())
    
    print(ndata2.head(),"\n\nRanges after min-max normalization:",[range(ndata2,i) for i in list(ndata2)],"\n")
    
    ndata3 = ndata2*20
    
    print(ndata3.head(),"\n\nRanges after scaling to range [0,20]:",[range(ndata3,i) for i in list(ndata3)],"\n")
    
    ndata4 = standardize(ndata1.copy())
       
    print("\nBefore standardization:\n",ndata1.describe())
    print("\nAfter standardization:\n",ndata4.describe())
    
    return

if __name__=="__main__":
	main()