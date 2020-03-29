import pandas as pd
import matplotlib.pyplot as plt

def read_data(path_to_file):
	#Returns Pandas dataframe containing given csv file.
    return pd.read_csv(path_to_file)
 
def show_box_plot(attribute_name,df):
	#Displays boxplot for given atrribute in dataframe.
    print(attribute_name,":")
    plt.boxplot(df[attribute_name])
    plt.show()
    
def replace_outliers(df):
    #Replaces outliers(according to boxplot) with median value in each attribute.
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
    print()
    return df

def range(df,attribute_name):
    #Returns the min-max values in the given attribute.
    return df[attribute_name].min(),df[attribute_name].max()

def min_max_normalization(df):
    #Does min-max normalization.[0,1]
    for i in list(df): df[i] = (df[i] - range(df,i)[0])/(range(df,i)[1]- range(df,i)[0])
    return df

def standardize(df):
    #Transforms data into normal distribution.
    return (df - df.mean())/df.std()

def main():
    path_to_file="pima_indians_diabetes_original.csv"
    
    df=read_data(path_to_file)
    
    data = df[["BMI","pres","pedi"]]
    
    for i in list(data): show_box_plot(i,data)
    
    print(data.head(),"\n\nInitial Ranges:",[range(data,i) for i in list(data)],"\n")
    
    ndata1 = replace_outliers(data.copy())
    
    for i in list(ndata1): show_box_plot(i,ndata1)
    
    print(ndata1.head(),"\n\nInitial Ranges:",[range(ndata1,i) for i in list(ndata1)],"\n")
    
    ndata2 = min_max_normalization(ndata1.copy())
    
    print(ndata2.head(),"\n\nRanges after min-max normalization:",[range(ndata2,i) for i in list(ndata2)],"\n")

    ndata3 = ndata2*20

    print(ndata3.head(),"\n\nRanges after scaling to range [0,20]:",[range(ndata3,i) for i in list(ndata3)],"\n")

    ndata4 = standardize(ndata1.copy())

    print("\nBefore standardization:\n",ndata1.describe(),"\n\nAfter standardization:\n",ndata4.describe())
    
    return

if __name__=="__main__":
	main()