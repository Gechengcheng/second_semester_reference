
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
#消除版本引起的参数设置警告
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.float_format', lambda x: '%.5f' % x) #为了直观的显示数字，不采用科学计数法

#设置图标可以显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
sns.set(font="simhei")

#设置宽度显示
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

print('导入数据\n')
#os.chdir(r'C:\Users\HP\Desktop\lending club(美国P2P放款业务中介公司) 贷款数据集')
os.chdir(r'F:\资料\信用卡项目最终代码\lending club(美国P2P放款业务中介公司) 贷款数据集')
final_dataset = pd.read_csv('final_dataset.csv',encoding = 'utf-8')
#print(final_dataset.head())


#七.模型构建（LR,RF,XGboost）
print('\n----------------------模型构建----------------------\n')
#创建验证集
X_var=list(final_dataset.columns)
X_var.remove('target') #去掉'target'列
X = np.array(final_dataset[X_var].values)
Y = final_dataset['target']
validation_size = 0.30 #训练集：测试集=7:3
seed = 100
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) #添加随机数种子使得每次运行的验证集和训练集是不变的
LR=LogisticRegression(solver='newton-cg',multi_class='ovr')  #创建模型
LR.fit(X_train,Y_train) #传入训练数据
print('LR模型构建完成\n')

'''预测测试数据的LR概率值，返回i*j列的数据，i为样本数,j为类别数,ij表示第i个样本是j类的概率；第i个样本的所有类别概率和为1。
LR.predict()，因为输出的是0或1，并不是概率值，不能对后续的roc曲线进行计算'''
Y_pre=LR.predict_proba(X_validation) #Y_pre与测试集的标签相匹配而不是和整个数据集的标签相匹配
Y_pre=np.array(Y_pre[:,1]) #取第二列数据，因为第二列概率为趋于0时分类类别为0，概率趋于1时分类类别为1


#八.模型评估及验证
print('----------------------模型评估及验证----------------------\n')

fpr,tpr,thresholds=roc_curve(Y_validation,Y_pre)  #计算fpr,tpr,thresholds,此处的Y_pre是一个概率值
print(thresholds.shape)
print(fpr)
print(tpr)
auc=roc_auc_score(Y_validation,Y_pre) #计算auc
print('auc值为：{:.4f}'.format(auc))

#画ROC曲线图
fig=plt.figure()
ax1 = plt.subplot2grid((1,11),(0,0),rowspan=1,colspan=5)
ax2 = plt.subplot2grid((1,11),(0,6),rowspan=1,colspan=5)
plt.sca(ax1) #选择子图
plt.xlim(0,1.05)
plt.ylim(0,1.05)
plt.plot(fpr,tpr)
plt.title('$ROC curve$')
plt.plot([0,1],[0,1],'r--')
plt.fill_between(fpr,tpr,color='lightgreen',alpha=0.6)
plt.text(0.04,0.96,'AUC = {:.3f}'.format(auc),fontsize=15)

#计算ks
KS_max=0
best_thr=0
for i in range(len(fpr)):
	if (i==0):
		KS_max=tpr[i]-fpr[i]
		best_thr=thresholds[i]
	elif (tpr[i]-fpr[i]>KS_max):
		KS_max = tpr[i] - fpr[i]
		best_thr = thresholds[i]
		index=i

fpr_best=fpr[index]
tpr_best=tpr[index]

print('最大KS为：{:.3f}'.format(KS_max))
print('最佳阈值为：{:.3f}'.format(best_thr))

#画KS曲线图
plt.sca(ax2) #选择子图
x_pos=[i for i in np.linspace(0,1,11)]
y_pos=[i for i in np.linspace(0,1,11)]
#fpr,tpr
plt.plot(thresholds,fpr,color='b',linestyle=':',linewidth = 3.0, alpha = 0.8)
plt.plot(thresholds,tpr,color='r',linestyle=':',linewidth = 3.0, alpha = 0.8)
plt.title('KS曲线',fontsize=14)
plt.xlim(0,1.0)
plt.ylim(0,1.05)
plt.legend(loc='best',labels=['FPR','TPR'],fontsize=15,framealpha = 0.7)
plt.vlines(best_thr,fpr_best,tpr_best,color='black',linewidth=2,linestyle = 'dashdot')
plt.text(best_thr*1.05,(fpr_best+tpr_best)/2,'KS = {:.3f}'.format(KS_max),fontsize=14)
plt.text(best_thr*1.02,tpr_best,'({0:.2f},{1:.3f})'.format(best_thr,tpr_best),fontsize=11)
plt.text(best_thr*1.02,fpr_best,'({0:.2f},{1:.3f})'.format(best_thr,fpr_best),fontsize=11)

#plt.show()

#模型精确度
print(Y_pre)
for i in range(len(Y_pre)):
	if Y_pre[i] >= 0.5:
		Y_pre[i]=1
	else:
		Y_pre[i]=0
print(Y_pre)
print("\nLR模型的精确度为：{}\n".format(accuracy_score(Y_validation, Y_pre)))
LR.score(X_validation, Y_validation)

#九.概率转化为评分
print('----------------------概率转化为评分----------------------\n')

odds=Y_pre/(1-Y_pre) #相对风险
y=np.log(odds)
base_point=600 #设定基准分(基准分和pdo联合作用后score>0)
PDO=20 #(point-to-double-odds)好坏比每升高一倍，分数升高PDO个单位
score=base_point-(PDO/np.log(2))*y
Y_pre.shape=len(Y_pre),1

score_df = pd.Series(data = score) #构建测试集最终得分组成的DataFrame
print(score_df.head())


#十、构建每个特征的对应评分表
print('----------------------建每个特征的对应评分表----------------------\n')

coef=LR.coef_#各个特征的回归系数
print("各特征的回归系数为：")
coef=np.squeeze(coef) #将coef压缩成1行10列的数组
print(coef)
print()
intercept=LR.intercept_[0] #提取列表中的数字
print("各特征的回归常数为：{0:.3f}\n".format(intercept))

score_dataset=final_dataset.copy()
score_feature=[]
#特征分箱转换分数
C=base_point-(PDO/np.log(2))*intercept #'常数'项
for i in range(len(final_dataset[X_var].columns)):
	var_box=[]
	box_score_feature=[]
	for j in final_dataset[X_var[i]].unique(): #j为第i个特征的各个分箱的Woe编码
		box_score=(PDO/(np.log(2)))*coef[i]*np.round(j,decimals=2)
		final_score=C/10-box_score
		var_box.append(j)
		box_score_feature.append(final_score)
		print('特征{0}的分箱Woe值为{1}对应的信用分数为：'.format(X_var[i],np.round(j,decimals=2)))
		print(np.round(final_score,decimals=4))
	score_feature.append(box_score_feature)
	var_mapping={m:n for m,n in zip(var_box,box_score_feature)}
	score_dataset[X_var[i]]=final_dataset[X_var[i]].map(var_mapping)
print()
#print(score_feature) #列表中第i个子列表的第j个元素为数据集特征中第i个特征对应第j个分箱的信用分数(最终)


#导入辅助数据集，进行特征——Woe编码——信用分数对应转换
assist_dataset = pd.read_csv('A.csv',encoding = 'utf-8')
print(assist_dataset.head()) #卡方分箱后对应数据集
print()
print(final_dataset.head()) #Woe编码后对应数据集
print()
print(score_dataset.head()) #特征分数转换后对应数据集
print()

Feature=[]
chi_box=[]
Woe=[]
Score=[]
for i in range(len(final_dataset[X_var].columns)):
	sol_chi_box=[]
	sol_Woe=[]
	sol_Score=[]
	Feature.append([X_var[i]])
	for m in assist_dataset[X_var[i]].unique():
		sol_chi_box.append(m)	
	chi_box.append(sol_chi_box)
	for n in final_dataset[X_var[i]].unique():
		sol_Woe.append('{0:.3f}'.format(n))
	Woe.append(sol_Woe)				
	for k in score_dataset[X_var[i]].unique():
		sol_Score.append('{0:.2f}'.format(k))
	Score.append(sol_Score)		
	
print('\n特征为：')
print(Feature)
print('\n各特征对应的卡方分箱为：')		
print(chi_box)
print('\n各特征经过分箱后每一箱对应的Woe编码为：')	
print(Woe)	
print('\n各特征经过分箱后每一箱对应的评分为：')
print(Score)

MultiIndex_0 = pd.MultiIndex.from_product([Feature[0],chi_box[0]])
data_0=np.array([Woe[0],Score[0]])	
data_0=data_0.T
columns=['Woe','得分']
point_table=pd.DataFrame(data_0,index = MultiIndex_0,columns = columns)
point_table.sort_values('得分',inplace=True,ascending=False)
#构建多重索引对应评分表
for i in range(1,len(chi_box)):
	MultiIndex = pd.MultiIndex.from_product([Feature[i],chi_box[i]])
	data = np.array([Woe[i],Score[i]])
	data = data.T
	point_table_i = pd.DataFrame(data,index = MultiIndex,columns = columns)
	point_table_i.sort_values('得分',inplace=True,ascending=False)
	point_table = pd.concat([point_table,point_table_i],axis = 0)
print(point_table)
point_table.to_csv(r'C:\Users\HP\Desktop\lending club(美国P2P放款业务中介公司) 贷款数据集\point_table.csv',index=True) #最终得分表



	
#XGBOOST集成建模
import xgboost as xgb
data = pd.read_csv(r'C:\Users\HP\Desktop\lending club(美国P2P放款业务中介公司) 贷款数据集\final_dataset.csv')
X=data[data.columns[1:11]].as_matrix()
Y=data[data.columns[0]].as_matrix()
indices = np.random.permutation(data.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量
X= X[indices]
Y= Y[indices]
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test =model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

#调参,汇总代码，需分开调试，否则速度极慢
#parameters = {
#              'min_child_weight': [2,4,5],
#              'max_delta_step': [0, 0.2, 0.6],
#              'colsample_bytree': [0.6, 0.7, 0.8],
#              'scale_pos_weight': [0.1,0.2, 0.4, 0.6, 0.8, 1],
#              'gamma':[0.0,0.1,0.2]
#}
#xlf = xgb.XGBClassifier(max_depth=15,
#            learning_rate=0.2,
#            n_estimators=150,
#            silent=True,
#            objective='binary:logistic',
#            nthread=-1,
#            gamma=0,
#            min_child_weight=1,
#            max_delta_step=0.2,
#            subsample=1,
#            colsample_bytree=0.7,
#            colsample_bylevel=1,
#            reg_alpha=0,
#            reg_lambda=1,
#            seed=10,
#            missing=None)
##网格调参            
#from sklearn.model_selection import GridSearchCV
#gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)#cv取值3到5，默认为3
#gsearch.fit(X_train,y_train)
#print("Best score: %0.3f" % gsearch.best_score_)
#print("Best parameters set:")
#best_parameters = gsearch.best_estimator_.get_params()
#for param_name in sorted(parameters.keys()):
#    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#model = xgb.XGBClassifier(max_depth=15,#待修改
#            learning_rate=0.2,
#            n_estimators=50,
#            silent=True,
#            objective='binary:logistic',
#            nthread=-1,
#            gamma=0,
#            min_child_weight=1,
#            max_delta_step=0.2,
#            subsample=1,
#            colsample_bytree=0.7,
#            colsample_bylevel=1)
#eval_set = [(X_test, y_test)]
#model.fit(X_train, y_train,eval_metric="logloss", eval_set=eval_set, verbose=False)
#训练集0.7850826605151865
#测试集0.6781137878011276
#特征工程的上限决定，改进方法，优化特征工程.

#最佳模型参数
model = xgb.XGBClassifier(max_depth=15,#待修改
            learning_rate=0.2,
            n_estimators=250,
            silent=True,
            objective='binary:logistic',
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0.2,
            subsample=1,
            colsample_bytree=0.7,
            colsample_bylevel=1)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train,eval_metric="logloss", eval_set=eval_set, verbose=False)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)#model.score(X_test, y_test)两个输出结果一致
print("验证集Accuracy: %.2f%%" % (accuracy * 100.0))
#print(model.score(X_train,y_train))
#print(model.score(X_test,y_test))
#改为True就能可视化loss

y_pred = model.predict(X_test)
Aaccuracy = accuracy_score(y_test, y_pred)#model.score(X_test, y_test)两个输出结果一致
print("验证集Accuracy: %.2f%%" % (accuracy * 100.0))
#print(model.score(X_train,y_train))
#print(model.score(X_test,y_test))

print(classification_report(y_test,y_pred))
confusion_matrix(y_test, y_pred)
fpr,tpr,thresholds=roc_curve(y_test,y_pred)  #计算fpr,tpr,thresholds
auc=roc_auc_score(y_test,y_pred) #计算auc
print('auc值为：{:.4f}'.format(auc))
#from sklearn.model_selection import cross_val_score # 交叉验证所需的函数

#scores_cv = cross_val_score(model, X,y, cv=5)  #cv为迭代次数。
#print(scores_cv.mean())  # 打印输出每次迭代的度量值（准确度）
#print(scores_cv.max())

     
#绘制特征重要性曲线
x=data.columns[1:].tolist()
param={'max_depth':15,#待修改
            'learning_rate':0.2,
            'n_estimators':250,
            'silent':True,
            'objective':'binary:logistic',
            "nthread":-1,
            'gamma':0,
            'min_child_weight':1,
            'max_delta_step':0.2,
            'colsample_bytree':0.7
            }
dic={}
for i,f in zip(data.columns[1:].tolist(),range(10)):
    dic[i]=str('f')+str(f)
print("特征映射：\n",dic)
from xgboost import plot_importance
model.fit(X, Y)
plot_importance(model, title='特征重要性排序', xlabel='得分', ylabel='特征', grid=False)
plt.show()
	
	
	

