import os
import pandas as pd
import numpy as np

basic_features=np.array(['loan_status','loan_amnt','term','emp_length','home_ownership','annual_inc',\
					     'verification_status','purpose','addr_state','dti','delinq_2yrs','mths_since_last_delinq',\
					     'open_acc','pub_rec','total_acc','acc_now_delinq','open_il_24m','inq_last_12m','mths_since_recent_bc_dlq',\
					     'num_accts_ever_120_pd','pct_tl_nvr_dlq','pub_rec_bankruptcies'])
					     
#一.数据集生成
print('--------------------数据生成--------------------\n')
os.chdir(r'C:\Users\HP\Desktop\lending club(美国P2P放款业务中介公司) 贷款数据集') #设置工作路径
dataset = pd.read_csv('LoanStats_2017Q1.csv',encoding = 'GBK',skiprows=1,usecols=basic_features)
for i in range(1,5): #将2017年的数据集进行合并
	data=pd.read_csv('LoanStats_2017Q{}.csv'.format(i),encoding = 'GBK',skiprows=1,usecols=basic_features)
	dataset=pd.concat([dataset,data.iloc[1:,:]],axis = 0)
dataset=dataset.loc[(dataset['loan_status']=='Charged Off')|(dataset['loan_status']=='Fully Paid')] #只选出带标签的样本
dataset=dataset.reset_index(drop=True) #将索引重置
loan_status=dataset['loan_status'] #将目标变量放在第一列,改变loan_status的值，原数据框中的值也会改变，如果不想要改变，加一个.copy()即可
del dataset['loan_status']
dataset.insert(0,'target',loan_status)
dataset.to_csv(r'C:\Users\HP\Desktop\lending club(美国P2P放款业务中介公司) 贷款数据集\loan_data.csv',index=False)
print('数据集生成成功\n')







