#importing labraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# adjusting how my console view dataframes
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 121)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

# reading datasets
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
VS = pd.read_csv('VariableDescription.csv')
SS=pd.read_csv('SampleSubmission.csv')

# dependent variable
Y=train['target']

#dcleaning dataset 
train.drop(['target'],axis=1 , inplace=True)
drop=(['CTR_CATEGO_X','FAC_MFODEC_F', 'FAC_MNTDCO_C', 'FAC_MNTDCO_F', 'FAC_MNTPRI_C',
       'FAC_MNTPRI_F', 'FAC_MNTTVA_C', 'FAC_MNTTVA_F', 'FJU_CODFJU',
       'RES_ANNIMP', 'SND_MNTAIR_A','SND_MNTAIR_E', 'SND_MNTAIR_I', 'SND_MNTAVA_A', 'SND_MNTAVA_E',
       'SND_MNTAVA_I', 'SND_MNTDRC_A', 'SND_MNTDRC_E', 'SND_MNTDRC_I',
       'SND_MNTPAY_A', 'SND_MNTPAY_E', 'SND_MNTPAY_I', 'SND_MNTPRD_A',
       'SND_MNTPRD_E', 'SND_MNTPRD_I', 'SND_MNTTAX_A', 'SND_MNTTAX_E',
       'SND_MNTTAX_I', 'SND_MNTTVA_A', 'SND_MNTTVA_E', 'SND_MNTTVA_I',
       'TVA_AACHAB','id','EXE_EXERCI','ACT_CODACT','CTR_MATFIS','BCT_CODBUR'])
train.drop(drop, axis=1 , inplace=True)
test.drop(drop, axis=1 , inplace=True)

train.isnull().sum()
test.isnull().sum()
train= train.fillna(0)
test= test.fillna(0)
Y=Y.fillna(method='ffill')


# training model using dicision tree regression model
regressor = RandomForestRegressor(n_estimators=500, random_state=50)
regressor.fit(train,Y)

# making some prediction
y_pred= regressor.predict(test)

# saving prediction results to csv
sub1=pd.DataFrame({
        'id': SS['id'],
        'target': y_pred
        })

sub1.to_csv('sharmainesub10.csv' , index = False)




