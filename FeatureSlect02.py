import pandas as pd
import numpy as np
from time import time

path = "/SASHome/data01/CCU/Trucking/Team/Jun/Code/Comm_Auto/Springleaf/"

trainData = pd.read_csv(path + "train_selected_05.csv")
testData = pd.read_csv(path + "test_selected_05.csv")

num = trainData.dtypes != 'object'
num = num[num].index.tolist()
num.remove('target')
num.remove('ID')
num.remove('Unnamed: 0')
num.remove('Unnamed: 0.1')

def extreme_value(Data):
    for var in num:
        if np.max(Data[var]) == 98:
            Data[var][Data[var] == 98] = -1
            Data[var][Data[var] == 97] = -1
        if np.max(Data[var]) == 97:
            Data[var][Data[var] == 97] = -1
        if np.max(Data[var]) == 96:
            Data[var][Data[var] == 96] = -1
        if np.max(Data[var]) == 94:
            Data[var][Data[var] == 94] = -1 

        if np.max(Data[var]) == 998:
            Data[var][Data[var] == 998] = -1
            Data[var][Data[var] == 997] = -1
            Data[var][Data[var] == 996] = -1
            Data[var][Data[var] == 995] = -1
            Data[var][Data[var] == 994] = -1

        Data[var][Data[var] == 9997] = -1        
        Data[var][Data[var] == 9996] = -1
        Data[var][Data[var] == 9994] = -1
        Data[var][Data[var] == 999999] = -1    
        Data[var][Data[var] == 999994] = -1
    
    return Data

trainData = extreme_value(trainData)
testData = extreme_value(testData)

def jw_1way_cnt(dsn, var1):
     df = dsn[[var1]]
     df['cnt'] = 1
     return df.groupby([var1]).transform(np.sum)
     
def jw_2way_cnt(dsn, var1, var2):
     df = dsn[[var1, var2]]
     df['cnt'] = 1
     return df.groupby([var1, var2]).transform(np.sum)

def jw_3way_cnt(dsn, var1, var2, var3):
     df = dsn[[var1, var2, var3]]
     df['cnt'] = 1
     return df.groupby([var1, var2, var3]).transform(np.sum)

def jw_exp2(dsn, var1, var2, y, split, cred_k, r_k=.3):
    y0 = np.mean(dsn[y][dsn[split] == 0])
    df1 = dsn[[var1, var2, split]]
    df1['y'] = np.array(dsn[y], dtype=float)
    df2 = dsn[[var1, var2]][dsn[split] == 0]
    df2['y'] = np.array(dsn[y][dsn[split] == 0], dtype=float)       
    df2['cnt'] = 1.0
    grouped = df2.groupby([var1, var2]).sum().add_prefix('sum_')
    df1 = pd.merge(df1, grouped, left_on = [var1, var2], right_index = True, how = 'left')
    df1.fillna(0, inplace=True)
    df1['sum_cnt'][df1[split] == 0] = df1['sum_cnt'][df1[split] == 0] - 1
    df1['sum_y'][df1[split] == 0] = df1['sum_y'][df1[split] == 0] - df1['y'][df1[split] == 0]
    df1['exp_y'] = (df1['sum_y'] + y0*cred_k)*1.0 / (df1['sum_cnt'] + cred_k)
    df1['exp_y'][df1['exp_y'].isnull()] = y0
    df1['exp_y'][df1[split] == 0] = df1['exp_y'][df1[split] == 0]*(1+(np.random.uniform(0,1,sum(df1.split == 0))-0.5)*r_k)
    return df1['exp_y']


def create_exp_var1(Data):
    Data['VAR_0001_exp'] = jw_exp2(Data, 'VAR_0001', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_exp'] = jw_exp2(Data, 'VAR_0005', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0169_exp'] = jw_exp2(Data, 'VAR_0169', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0200_exp'] = jw_exp2(Data, 'VAR_0200', 'dummy', 'target', 'split', 500, r_k=.3)
#    Data['VAR_0226_exp'] = jw_exp2(Data, 'VAR_0226', 'dummy', 'target', 'split', 10000, r_k=.3)
#   Data['VAR_0230_exp'] = jw_exp2(Data, 'VAR_0230', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0232_exp'] = jw_exp2(Data, 'VAR_0232', 'dummy', 'target', 'split', 10000, r_k=.3)
#    Data['VAR_0236_exp'] = jw_exp2(Data, 'VAR_0236', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0237_exp'] = jw_exp2(Data, 'VAR_0237', 'dummy', 'target', 'split', 2000, r_k=.3)
    Data['VAR_0274_exp'] = jw_exp2(Data, 'VAR_0274', 'dummy', 'target', 'split', 2000, r_k=.3)
    Data['VAR_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'dummy', 'target', 'split', 8000, r_k=.3)
    Data['VAR_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'dummy', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'dummy', 'target', 'split', 10000, r_k=.3) 
    Data['VAR_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'dummy', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'dummy', 'target', 'split', 500, r_k=.3) 
    Data['VAR_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'dummy', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'dummy', 'target', 'split', 8000, r_k=.3)
    Data['VAR_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'dummy', 'target', 'split', 200, r_k=.3) 
    Data['VAR_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'dummy', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'dummy', 'target', 'split', 50, r_k=.3)

    Data['VAR_0001_0005_exp'] = jw_exp2(Data, 'VAR_0005', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0169_exp'] = jw_exp2(Data, 'VAR_0169', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0200_exp'] = jw_exp2(Data, 'VAR_0200', 'VAR_0001', 'target', 'split', 300, r_k=.3)
#    Data['VAR_0001_0226_exp'] = jw_exp2(Data, 'VAR_0226', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
#    Data['VAR_0001_0230_exp'] = jw_exp2(Data, 'VAR_0230', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0232_exp'] = jw_exp2(Data, 'VAR_0232', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
#    Data['VAR_0001_0236_exp'] = jw_exp2(Data, 'VAR_0236', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0237_exp'] = jw_exp2(Data, 'VAR_0237', 'VAR_0001', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0001_0274_exp'] = jw_exp2(Data, 'VAR_0274', 'VAR_0001', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0001_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0001', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0001_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0001', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0001_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0001', 'target', 'split', 10000, r_k=.3) 
    Data['VAR_0001_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0001', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0001_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0001', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0001_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0001', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0001_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0001', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0001_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0001', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0001_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0001', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0001_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0001', 'target', 'split', 30, r_k=.3)

    Data['VAR_0005_0169_exp'] = jw_exp2(Data, 'VAR_0169', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_0200_exp'] = jw_exp2(Data, 'VAR_0200', 'VAR_0005', 'target', 'split', 300, r_k=.3)
 #   Data['VAR_0005_0226_exp'] = jw_exp2(Data, 'VAR_0226', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
 #   Data['VAR_0005_0230_exp'] = jw_exp2(Data, 'VAR_0230', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_0232_exp'] = jw_exp2(Data, 'VAR_0232', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
 #   Data['VAR_0005_0236_exp'] = jw_exp2(Data, 'VAR_0236', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_0237_exp'] = jw_exp2(Data, 'VAR_0237', 'VAR_0005', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0005_0274_exp'] = jw_exp2(Data, 'VAR_0274', 'VAR_0005', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0005_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0005', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0005_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0005', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0005_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0005', 'target', 'split', 10000, r_k=.3) 
    Data['VAR_0005_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0005', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0005_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0005', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0005_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0005', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0005_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0005', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0005_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0005', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0005_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0005', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0005_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0005', 'target', 'split', 30, r_k=.3)

    Data['VAR_0169_0200_exp'] = jw_exp2(Data, 'VAR_0200', 'VAR_0169', 'target', 'split', 300, r_k=.3)
 #   Data['VAR_0169_0226_exp'] = jw_exp2(Data, 'VAR_0226', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
 #   Data['VAR_0169_0230_exp'] = jw_exp2(Data, 'VAR_0230', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0169_0232_exp'] = jw_exp2(Data, 'VAR_0232', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
  #  Data['VAR_0169_0236_exp'] = jw_exp2(Data, 'VAR_0236', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0169_0237_exp'] = jw_exp2(Data, 'VAR_0237', 'VAR_0169', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0169_0274_exp'] = jw_exp2(Data, 'VAR_0274', 'VAR_0169', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0169_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0169_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0169', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0169_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0169_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0169', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0169_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0169', 'target', 'split', 10000, r_k=.3) 
    Data['VAR_0169_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0169_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0169', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0169_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0169', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0169_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0169', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0169_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0169', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0169_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0169', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0169_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0169', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0169_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0169', 'target', 'split', 30, r_k=.3)

#    Data['VAR_0200_0226_exp'] = jw_exp2(Data, 'VAR_0226', 'VAR_0200', 'target', 'split', 10000, r_k=.3)
#    Data['VAR_0200_0230_exp'] = jw_exp2(Data, 'VAR_0230', 'VAR_0200', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0200_0232_exp'] = jw_exp2(Data, 'VAR_0232', 'VAR_0200', 'target', 'split', 300, r_k=.3)
 #   Data['VAR_0200_0236_exp'] = jw_exp2(Data, 'VAR_0236', 'VAR_0200', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0200_0237_exp'] = jw_exp2(Data, 'VAR_0237', 'VAR_0200', 'target', 'split', 200, r_k=.3)
    Data['VAR_0200_0274_exp'] = jw_exp2(Data, 'VAR_0274', 'VAR_0200', 'target', 'split', 200, r_k=.3)
    Data['VAR_0200_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0200', 'target', 'split', 250, r_k=.3)
    Data['VAR_0200_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0200', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0200_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0200', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0200_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0200', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0200_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0200', 'target', 'split', 300, r_k=.3)
    Data['VAR_0200_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0200', 'target', 'split', 20, r_k=.3)


#    Data['VAR_0232_0236_exp'] = jw_exp2(Data, 'VAR_0236', 'VAR_0232', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0232_0237_exp'] = jw_exp2(Data, 'VAR_0237', 'VAR_0232', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0232_0274_exp'] = jw_exp2(Data, 'VAR_0274', 'VAR_0232', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0232_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'VAR_0232', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0232_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0232', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0232_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0232', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0232_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0232', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0232_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0232', 'target', 'split', 10000, r_k=.3) 
    Data['VAR_0232_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0232', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0232_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0232', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0232_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0232', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0232_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0232', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0232_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0232', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0232_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0232', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0232_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0232', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0232_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0232', 'target', 'split', 30, r_k=.3)

    Data['VAR_0237_0274_exp'] = jw_exp2(Data, 'VAR_0274', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'VAR_0237', 'target', 'split', 1500, r_k=.3)
    Data['VAR_0237_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0237', 'target', 'split', 1000, r_k=.3) 
    Data['VAR_0237_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0237', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0237_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0237', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0237_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0237', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0237_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0237', 'target', 'split', 30, r_k=.3)

    Data['VAR_0274_0283_exp'] = jw_exp2(Data, 'VAR_0283', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0274', 'target', 'split', 1000, r_k=.3) 
    Data['VAR_0274_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0274', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0274_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0274', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0274_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0274', 'target', 'split', 1000, r_k=.3)
    Data['VAR_0274_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0274', 'target', 'split', 30, r_k=.3)

    Data['VAR_0283_0305_exp'] = jw_exp2(Data, 'VAR_0305', 'VAR_0283', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0283_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0283', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0283_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0283', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0283_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0283', 'target', 'split', 10000, r_k=.3) 
    Data['VAR_0283_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0283', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0283_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0283', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0283_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0283', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0283_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0283', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0283_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0283', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0283_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0283', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0283_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0283', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0283_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0283', 'target', 'split', 30, r_k=.3)

    Data['VAR_0305_0325_exp'] = jw_exp2(Data, 'VAR_0325', 'VAR_0305', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0305_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0305', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0305_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0305', 'target', 'split', 5000, r_k=.3) 
    Data['VAR_0305_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0305', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0305_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0305', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0305_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0305', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0305_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0305', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0305_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0305', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0305_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0305', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0305_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0305', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0305_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0305', 'target', 'split', 30, r_k=.3)

    Data['VAR_0325_0342_exp'] = jw_exp2(Data, 'VAR_0342', 'VAR_0325', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0325_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0325', 'target', 'split', 10000, r_k=.3) 
    Data['VAR_0325_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0325', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0325_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0325', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0325_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0325', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0325_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0325', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0325_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0325', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0325_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0325', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0325_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0325', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0325_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0325', 'target', 'split', 30, r_k=.3)

    Data['VAR_0342_0352_exp'] = jw_exp2(Data, 'VAR_0352', 'VAR_0342', 'target', 'split', 3000, r_k=.3) 
    Data['VAR_0342_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0342', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0342_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0342', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0342_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0342', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0342_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0342', 'target', 'split', 2500, r_k=.3)
    Data['VAR_0342_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0342', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0342_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0342', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0342_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0342', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0342_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0342', 'target', 'split', 20, r_k=.3)

    Data['VAR_0352_0353_exp'] = jw_exp2(Data, 'VAR_0353', 'VAR_0352', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0352_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0352', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0352_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0352', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0352_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0352', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0352_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0352', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0352_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0352', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0352_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0352', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0352_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0352', 'target', 'split', 30, r_k=.3)

    Data['VAR_0353_0354_exp'] = jw_exp2(Data, 'VAR_0354', 'VAR_0353', 'target', 'split', 10000, r_k=.3)
    Data['VAR_0353_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0353', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0353_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0353', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0353_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0353', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0353_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0353', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0353_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0353', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0353_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0353', 'target', 'split', 30, r_k=.3)

    Data['VAR_0354_0404_exp'] = jw_exp2(Data, 'VAR_0404', 'VAR_0354', 'target', 'split', 300, r_k=.3) 
    Data['VAR_0354_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0354', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0354_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0354', 'target', 'split', 5000, r_k=.3)
    Data['VAR_0354_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0354', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0354_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0354', 'target', 'split', 2000, r_k=.3)
    Data['VAR_0354_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0354', 'target', 'split', 30, r_k=.3)

    Data['VAR_0404_0466_exp'] = jw_exp2(Data, 'VAR_0466', 'VAR_0404', 'target', 'split', 200, r_k=.3)
    Data['VAR_0404_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0404', 'target', 'split', 250, r_k=.3)
    Data['VAR_0404_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0404', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0404_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0404', 'target', 'split', 200, r_k=.3)
    Data['VAR_0404_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0404', 'target', 'split', 10, r_k=.3)

    Data['VAR_0466_0467_exp'] = jw_exp2(Data, 'VAR_0467', 'VAR_0466', 'target', 'split', 3000, r_k=.3)
    Data['VAR_0466_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0466', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0466_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0466', 'target', 'split', 2500, r_k=.3)
    Data['VAR_0466_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0466', 'target', 'split', 20, r_k=.3)

    Data['VAR_0467_0493_exp'] = jw_exp2(Data, 'VAR_0493', 'VAR_0467', 'target', 'split', 100, r_k=.3) 
    Data['VAR_0467_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0467', 'target', 'split', 2500, r_k=.3)
    Data['VAR_0467_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0467', 'target', 'split', 15, r_k=.3)

    Data['VAR_0493_1934_exp'] = jw_exp2(Data, 'VAR_1934', 'VAR_0493', 'target', 'split', 100, r_k=.3)
    Data['VAR_0493_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_0493', 'target', 'split', 10, r_k=.3)

    Data['VAR_1934_0241_exp'] = jw_exp2(Data, 'VAR_0241', 'VAR_1934', 'target', 'split', 20, r_k=.3)                   
    return Data      

trainData['test'] = 0
testData['test'] = 1
testData['target'] = 0

combineData = pd.concat([trainData, testData])
combineData['dummy'] = 1
combineData['split'] = 1
combineData['split'][9700:] = 0

combineData = create_exp_var1(combineData)

trainData = combineData[combineData['test']==0]
testData = combineData[combineData['test']==1]

if trainData.isnull().values.any():
    trainData = trainData.fillna(-1)

if testData.isnull().values.any():
    testData = testData.fillna(-1)

del_col = ['VAR_0001','VAR_0005','VAR_0169','VAR_0200','VAR_0226','VAR_0230','VAR_0232','VAR_0236','VAR_0237','VAR_0274','VAR_0283','VAR_0305','VAR_0325','VAR_0342','VAR_0352','VAR_0353','VAR_0354','VAR_0404','VAR_0466','VAR_0467','VAR_0493','VAR_1934']
del_col = del_col + ['Unnamed: 0', 'Unnamed: 0.1']

trainData.drop(del_col, axis=1, inplace=True)
testData.drop(del_col, axis=1, inplace=True)

trainData.to_csv(path + "train_selected_09.csv", index=False)
testData.to_csv(path + "test_selected_09.csv", index=False)


