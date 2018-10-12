model,feature=ow_train.train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,seasonality_df,process_f01_flag=False,mode='dev')
import matplotlib.pylab as plt
from xgboost import plot_tree
from xgboost import plot_importance
from graphviz import Digraph
import pydot

pd.options.display.max_columns=999
pd.options.display.max_rows=999
pd.options.display.width=160

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

xColumns = scenario['selectedColumns']['features']
yColumns = scenario['selectedColumns']['target']

ceate_feature_map(xColumns)

for rdc_id in list(feature.RDCKey.unique()):
	bst = model[rdc_id]
	importance = bst.get_fscore(fmap='xgb.fmap')
	di = sorted(importance.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
	print 'rdc: %s'%str(rdc_id)
	count = 0
	for i in di:
	   if i[0] in scenario['promo_feature_cols'] and count<=9:
	       count += 1
	       print i[0]
	   elif count > 9:
	       exit
	   else:
	       continue
	#sorted(d.items, key=lambda d:d[1]) 
	f = open('f_score_11922/rdc_'+ str(rdc_id) + '_feature_importance.txt','w')
	for i in di:
	    f.write(str(i))
	    f.write('\n')
	f.close()
'''
top 10 important promotion related features of rdc 3:
	MaxSyntheticDiscountA
	daynumberinpromotion
	PromotionCount
	MaxDiscount
	MinDiscount
	durationinhours
	AvgDiscount
	bd_discount_sgp_wgt
	MinSyntheticDiscountA
	cnt_period/dd_price_weighted
'''
from code.refactor.common import loadSettingsFromYamlFile
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('tmp/11922_2018/at180323_11922_v_oct_feature_20171010.csv',parse_dates=['Date'])
df['month'] = df['Date'].apply(lambda x: x.month)
df['year'] = df['Date'].apply(lambda x: x.year)
monthly_sales = df.groupby(['RDCKey','ProductKey','month','year'])['salesForecast'].sum().reset_index()
monthly_sales.rename(columns={'salesForecast': 'monthly_salesForecast'},inplace=True)
df = pd.merge(df,monthly_sales,how='left',on=['RDCKey','ProductKey','month','year'])

pd.options.display.max_columns=999
pd.options.display.max_rows=999
pd.options.display.width=160
scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
###rdc###
item = 'MaxSyntheticDiscountA'  #['MaxDirectDiscount','MaxBundleDiscount','MaxSyntheticDiscountA','MaxDiscount']
plt.figure(1)
plot_num = 811
count = 0
for rdc_id in [3,4,5,6,9,10,316,772]:
	df_rdc = df[(df.RDCKey==rdc_id)&(df[item].notnull())]
	df_rdc.ProductKey.drop_duplicates().shape
	grouped = df_rdc.groupby('ProductKey')
	scale_list = []
	for sku, group in grouped:
		tmp = group.groupby(item)['salesForecast'].mean().reset_index()
		scalerX = MinMaxScaler(feature_range=(0, 1))
		sales_tmp = scalerX.fit_transform(tmp['salesForecast'].values.reshape(-1,1))
		tmp['scale_sales']=sales_tmp.reshape(-1)
		#fig = tmp[[item,'scale_sales']].set_index(item).plot().get_figure()
		#fig.savefig('./tmp/11922_2018/plot/'+str(sku)+'_rdc_3.pdf')
		scale_list.append(tmp)
	scale_sales_df = pd.concat(scale_list)
	#scale_sales_df[scale_sales_df[item].between(0,1)].shape
	clean_scale_df = scale_sales_df[scale_sales_df[item].between(0,1)]
	final_scale_sales_discount = clean_scale_df.groupby(pd.cut(clean_scale_df[item],np.arange(0, 1, 0.05)))['scale_sales'].mean()
	raw_final_df = pd.DataFrame({item+'_vigor':final_scale_sales_discount.index,'mean_scale_sales':final_scale_sales_discount.values})
	print raw_final_df
	raw_final_df[item+'_vigor'].cat.categories = list(raw_final_df.index)
	
	#raw_final_df.set_index('maxdiscount_vigor').plot(kind='bar')
	#plt.show()
	
	plt.subplot(plot_num + count)
	plt.plot(raw_final_df[item+'_vigor'], raw_final_df['mean_scale_sales'])
	#raw_final_df.set_index('maxdiscount_vigor').plot(color='r',title='rdc_'+str(rdc_id))
	#plt.xticks([])
	plt.grid(True)
	count += 1
plt.show()
plt.close('all')

###scale monthly###
item = 'MaxSyntheticDiscountA'  #['MaxDirectDiscount','MaxBundleDiscount','MaxSyntheticDiscountA','MaxDiscount']
plt.figure(1)
plot_num = 811
count = 0
for rdc_id in [3,4,5,6,9,10,316,772]:
	df_rdc = df[(df.RDCKey==rdc_id)&(df[item].notnull())]
	df_rdc.ProductKey.drop_duplicates().shape
	grouped = df_rdc.groupby('ProductKey')
	scale_list = []
	for sku, group in grouped:
		group['scale_sales'] = group.salesForecast / group.monthly_salesForecast
		tmp = group.groupby(item)['scale_sales'].mean().reset_index()
		#scalerX = MinMaxScaler(feature_range=(0, 1))
		#sales_tmp = scalerX.fit_transform(tmp['salesForecast'].values.reshape(-1,1))
		#tmp['scale_sales']=sales_tmp.reshape(-1)
		#fig = tmp[[item,'scale_sales']].set_index(item).plot().get_figure()
		#fig.savefig('./tmp/11922_2018/plot/'+str(sku)+'_rdc_3.pdf')
		scale_list.append(tmp)
	scale_sales_df = pd.concat(scale_list)
	#scale_sales_df[scale_sales_df[item].between(0,1)].shape
	clean_scale_df = scale_sales_df[scale_sales_df[item].between(0,1)]
	final_scale_sales_discount = clean_scale_df.groupby(pd.cut(clean_scale_df[item],np.arange(0, 1, 0.05)))['scale_sales'].mean()
	raw_final_df = pd.DataFrame({item+'_vigor':final_scale_sales_discount.index,'mean_scale_sales':final_scale_sales_discount.values})
	print raw_final_df
	raw_final_df[item+'_vigor'].cat.categories = list(raw_final_df.index)
	plt.subplot(plot_num + count)
	plt.plot(raw_final_df[item+'_vigor'], raw_final_df['mean_scale_sales'])
	#raw_final_df.set_index('maxdiscount_vigor').plot(color='r',title='rdc_'+str(rdc_id))
	#plt.xticks([])
	plt.grid(True)
	count += 1
plt.show()
plt.close('all')



###0502: mape statistics###
import pandas as pd
df = pd.read_csv('tmp/品类7052.csv')
df.Date = pd.to_datetime(df.Date)
raw = pd.read_csv('tmp/data/0425_result/result_870.csv')
mean_df = pd.read_csv('tmp/data/0425_result/result_870_mean_0502.csv')
no_promo_df = pd.read_csv('tmp/data/0425_result/result_870_no_promo_0504.csv')
list_keys = ['Date','RDCKey','ProductKey']
raw = raw[list_keys + ['salesForecast','ypred']]
mean_df = mean_df[list_keys + ['salesForecast','ypred']]
no_promo_df = no_promo_df[list_keys + ['salesForecast','ypred']]
df1 = raw.groupby(['Date','ProductKey'])['salesForecast','ypred'].sum().reset_index()
df2 = mean_df.groupby(['Date','ProductKey'])['salesForecast','ypred'].sum().reset_index()
df_no_promo = no_promo_df.groupby(['Date','ProductKey'])['salesForecast','ypred'].sum().reset_index()
ts = pd.read_csv('tmp/data/simplified_ts_870.csv')
actual = ts.groupby(['Date','ProductKey'])['salesForecast'].sum().reset_index()
actual[actual.Date.between('2018-02-01','2018-02-10')].shape
pd.options.display.max_columns=999
pd.options.display.max_rows=999
pd.options.display.width=160
actual[actual.ProductKey==244923]
df11 = pd.merge(df1[['Date','ProductKey','ypred']],actual,how='left',on=['Date','ProductKey'])
df22 = df2[['Date','ProductKey','ypred']]
df22.rename(columns={'ypred':'ypred_mean'},inplace=True)
df33 = df_no_promo[['Date','ProductKey','ypred']]
df33.rename(columns={'ypred':'ypred_nopromo'},inplace=True)
df3 = pd.merge(df11, df22, how='left', on=['Date','ProductKey'])
df4 = pd.merge(df3, df33, how='left', on=['Date','ProductKey'])
df5 = df4.groupby('ProductKey').sum().reset_index()
df55 = df5[~(df5.salesForecast==0)]
df55.ypred.sum()
df55.ypred_mean.sum()
df55.ypred_nopromo.sum()
df55[df55.resi_mean<=]0.shape
df55[df55.resi_mean<=0].shape
df55[df55.resi_nopromo<=0].shape
df55.resi_mean.sum()
df55.resi_nopromo.sum()

import numpy as np
np.abs(df4.ypred-df4.salesForecast).sum()/df4.salesForecast.sum()
df.drop(['ypred_regular','ypred'],axis=1,inplace=True)

