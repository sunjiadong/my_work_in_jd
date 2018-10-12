# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
import os.path
from code.refactor.common import (get_setting_from_cfg, get_setting_path, filter_df_by_cate_id,get_future_sales_feature,\
                                  dummy_features, model_predict, get_weekly_df,get_future_condition_sales_feature,gen_train_valid_by_date,\
                                  generate_promotion_future, get_hour, save_object, add_pv,get_lowest_n,generate_cv_train_valid_set,\
                                  generate_default_values_by_dic,fill_col_with_default, week2month,trans_band_to_int,get_future_condition_sales_feature_stage2,\
                                  loadSettingsFromYamlFile,createSeasonalityFeatures,createLevel3Features,createSeasonalityDecomposeFeatures,\
                                 calculateNationalRolling,calculateRolling,calculateLagging,createDateFeatures,splitTimeWindow,calculateSimilarRolling,calculateStockFeatures,\
                                 process_rdc,clean_data,add_cols,get_dd_price,agg_dd_price,get_bundle_feat,agg_bd_price,process_feature,\
                                 agg_feature,prep_data,calc_weighted_price,agg_feature_day,get_column_by_type,object2Float,object2Int)
from code.refactor.fdc_flow import  filter_non_price_fill_it
import pandas as pd
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.options.display.max_columns=999
pd.options.display.max_rows=999
pd.options.display.width=160


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)


# pro_canlender_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/7054_p2.da'

# promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)

# promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
#            'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
#            'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
#            'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
#            'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

# file_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/7054_ts.da'
# ts_df = pd.read_csv(file_path,header=None,sep='\t')

scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)


cate = 7054
if cate in [11922]:
	pred_date = pd.to_datetime('2017-10-10')
	scenario['lookforwardPeriodDays'] = 7
elif cate in [7052, 7054]:
	pred_date = pd.to_datetime('2018-02-01')
	scenario['lookforwardPeriodDays'] = 10
else:
	pred_date = pd.to_datetime('2018-03-10')
	scenario['lookforwardPeriodDays'] = 8
ForecastStartDate = pd.to_datetime(pred_date)
DataStartDate = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDays'])
PredictEndDate = ForecastStartDate + datetime.timedelta(days=(scenario['lookforwardPeriodDays']-1))

actual = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/simplified_ts_7054.csv')
actual.Date = pd.to_datetime(actual.Date)
actual.RDCKey = actual.RDCKey.astype(float)

list_keys = ['Date','RDCKey','ProductKey']

#raw = pd.read_csv('tmp/data/0425_result/result_870.csv')
raw = pd.read_csv('tmp/data/0425_result/result_7054.csv')
raw = raw[list_keys + ['salesForecast','ypred']]
raw.rename(columns={'ypred':'ypred_raw'},inplace=True)
raw.drop('salesForecast',axis=1,inplace=True)

#mean_df = pd.read_csv('tmp/data/0425_result/result_870_mean_0502.csv')
mean_df = pd.read_csv('tmp/data/0425_result/result_7054_mean_0502.csv')
mean_df = mean_df[list_keys + ['salesForecast','ypred']]
mean_df.rename(columns={'ypred':'ypred_mean_promo'},inplace=True)
mean_df.drop('salesForecast',axis=1,inplace=True)

#zero_promo_df = pd.read_csv('tmp/data/0425_result/result_870_zero_promo_0507.csv')
zero_promo_df = pd.read_csv('tmp/data/0425_result/result_7054_zero_promo_0515.csv')
zero_promo_df = zero_promo_df[list_keys + ['salesForecast','ypred']]
zero_promo_df.rename(columns={'ypred':'ypred_zero_promo'},inplace=True)
zero_promo_df.drop('salesForecast',axis=1,inplace=True)

new_df = raw.merge(mean_df,on=list_keys).merge(zero_promo_df,on=list_keys)   #.merge(no_promo_df,on=list_keys)
new_df.Date = pd.to_datetime(new_df.Date)
new_df = pd.merge(new_df, actual[list_keys+['salesForecast']], how='left',on = list_keys)

list_keys = ['Date','RDCKey','ProductKey']
test_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/0425_result/result_7054.csv' #'result_870_with_season_0509.csv'
history_ypred_mean_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/0425_result/history_7054_sales_promo_mean.csv' #'history_870_sales_promo_mean_with_season.csv' 
history_ypred_zero_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/0425_result/history_7054_sales_promo_zero.csv' #'history_870_sales_promo_zero_with_season.csv' 

exclu_promo_features = ['strongmark','flashsale_ind','dd_ind','bundle_ind','bundle_buy199get100_ind','suit_ind','freegift_ind']
update_cols = list(set(scenario['promo_feature_cols'])- set(exclu_promo_features))


raw_train_df = pd.read_csv(history_ypred_mean_path)
for col in update_cols:
    raw_train_df.rename(columns={col: col+'_history_mean'}, inplace=True)
for col in update_cols:
    raw_train_df.rename(columns={col+'_x': col}, inplace=True)
raw_train_df.rename(columns={'ypred':'ypred_mean_promo'}, inplace=True)
train_df_zero = pd.read_csv(history_ypred_zero_path)
train_df_zero.rename(columns={'ypred':'ypred_zero_promo'},inplace=True)
raw_train_df = pd.merge(raw_train_df, train_df_zero[list_keys+['ypred_zero_promo']], how='left', on=list_keys)


raw_test_df = pd.read_csv(test_path)
raw_test_df = raw_test_df[list_keys + scenario['promo_feature_cols'] + ['salesForecast','ypred']]
raw_test_df.Date = pd.to_datetime(raw_test_df.Date)

#col = 'MaxDiscount'
###use mean to fill nan###
used_cols = ['MaxDiscount'] #['MaxBundleDiscount','MaxDirectDiscount','MaxDiscount','MaxSyntheticDiscountA','daynumberinpromotion','PromotionCount']

raw_train_df.Date = pd.to_datetime(raw_train_df.Date)
raw_train_df = raw_train_df[raw_train_df.Date < pred_date]
raw_train_df = raw_train_df[list_keys + scenario['promo_feature_cols'] + ['salesForecast','ypred_mean_promo','ypred_zero_promo']]
bak = raw_train_df.copy()
# train_df = pd.read_csv(train_path)
# train_df = train_df[list_keys + scenario['promo_feature_cols'] + ['salesForecast']]
# train_df.Date = pd.to_datetime(train_df.Date)
# train_df = train_df[train_df.Date<pred_date]
input_df = pd.concat([raw_train_df, raw_test_df])

for col in used_cols:
  #input_df[col] = input_df.groupby(['RDCKey','ProductKey'])[col].transform(lambda x: x.fillna(x.mean()).fillna(0))
  #input_df[col] = input_df.groupby(['RDCKey','ProductKey'])[col].transform(lambda x: x.fillna(method='bfill').fillna(0))
  input_df = input_df[~(input_df[col].isnull())]
  input_df = input_df[input_df[col].between(-1,1)]
  ###use mode to fill nan###
'''
test_df.loc[test_df.MaxDiscount.isnull(), 'MaxDiscount'].shape
def fill_na_col(df, cols=scenario['promo_feature_cols']):
  for col in cols:
    if len(df[df[col].isnull()]) == len(df):
      df[col] = df[col].fillna(0)
    else:
      df[col] = df[col].fillna(df[col].mode().values[0])
  return df
train_df = train_df..groupby(['RDCKey','ProductKey']).apply(fill_na_col)
test_df = test_df.groupby(['RDCKey','ProductKey']).apply(fill_na_col)
'''

#for degree in range(2,20,2):
#print "degree: %d"%degree
final_list = []

for value_type in ['ypred_zero_promo','ypred_mean_promo']:
	final_df = pd.DataFrame()
	a = 1
	grouped = input_df.groupby(['RDCKey','ProductKey'])
	for (rdc, sku), group in grouped:
		if group.Date.min() < ForecastStartDate and group.Date.max() >= ForecastStartDate:
			print a
			a = a + 1
			train_df = group[group.Date < ForecastStartDate]
			test_df = group[group.Date >= ForecastStartDate]
			x_train_df = train_df[used_cols]
			x_test_df = test_df[used_cols]
			y_train = train_df['salesForecast'] - train_df[value_type]
			#y_train = train_df['salesForecast'] - train_df['ypred_mean_promo']
			y_test = test_df['salesForecast']
			
			
			lm = LinearRegression()
			lm.fit(x_train_df, y_train)
			Intercept = lm.intercept_
			RSquare = lm.score(x_train_df, y_train)
			#if(RSquare > 0.5):
			#	print Intercept
			#	print RSquare
			lm_predict_result = lm.predict(x_test_df)
			test_result = pd.DataFrame()
			for col in list_keys+['salesForecast','ypred']:
				test_result[col] = test_df[col]
			test_result['reg_result'] = lm_predict_result
			
			###gaussian###
			'''
			if len(x_train_df[used_cols].drop_duplicates()) == 1:
				test_result = pd.DataFrame()
				for col in list_keys+['salesForecast','ypred']:
					test_result[col] = test_df[col]
				test_result['reg_result'] = y_train.tail().mean()
			else:
				lm = make_pipeline(GaussianFeatures(5), Lasso(alpha=0.1))
				lm.fit(np.array(x_train_df), np.array(y_train))
				Intercept = lm.steps[1][1].intercept_
				RSquare = lm.score(np.array(x_train_df), np.array(y_train))
				lm_predict_result = lm.predict(np.array(x_test_df))
				test_result = pd.DataFrame()
				for col in list_keys+['salesForecast','ypred']:
					test_result[col] = test_df[col]
				test_result['reg_result'] = lm_predict_result
			'''
			###polynomial###
			'''
			pf = PolynomialFeatures(degree=2)
			pModel = LinearRegression()
			pModel.fit(pf.fit_transform(x_train_df), y_train)
			pf_predict_result = pModel.predict(pf.fit_transform(x_test_df))
			test_result = pd.DataFrame()
			for col in list_keys+['salesForecast','ypred']:
				test_result[col] = test_df[col]
			test_result['reg_result'] = pf_predict_result
			'''
			final_df = pd.concat([final_df, test_result])
	final_list.append(final_df)

zero_final = final_list[0]
mean_final = final_list[1]
zero_final.rename(columns={'reg_result':'zero_promo_reg_result'},inplace=True)
mean_final.rename(columns={'reg_result':'mean_promo_reg_result'},inplace=True)
final_df = pd.merge(zero_final,mean_final, on=list_keys+['salesForecast','ypred'])
final_df.drop('salesForecast',axis=1,inplace=True)


# final_result.salesForecast.sum()
# final_sku_result = final_result.groupby('ProductKey')['ypred','reg_result','salesForecast'].sum().reset_index()
# final_sku_result['resi_ypred'] = np.abs(final_sku_result.ypred - final_sku_result.salesForecast)
# final_sku_result['resi_reg'] = np.abs(final_sku_result.reg_result - final_sku_result.salesForecast)
# final_sku_result.resi_ypred.sum() / final_sku_result.salesForecast.sum()
# final_sku_result.resi_reg.sum() / final_sku_result.salesForecast.sum()
# final_rdc_resutl = final_result.groupby(['RDCKey','ProductKey'])['ypred','reg_result','salesForecast'].sum().reset_index()
# for rdc in list(final_rdc_resutl.RDCKey.unique()):
#     this_df = final_rdc_resutl[final_rdc_resutl.RDCKey==rdc]
#     this_df['resi_ypred'] = np.abs(this_df['salesForecast']-this_df['ypred'])
#     this_df['resi_reg'] = np.abs(this_df['salesForecast']-this_df['reg_result'])
#     print "rdc:" + str(rdc)
#     print "mape_ypred  |   mape_reg"
#     print "%f    |   %f"%(this_df['resi_ypred'].sum()/this_df['salesForecast'].sum(),this_df['resi_reg'].sum()/this_df['salesForecast'].sum())

final_df.drop('ypred',axis=1,inplace=True)
new_df_final = new_df.merge(final_df,on=list_keys,how='left')
new_df_final.fillna(0,inplace=True)

"""
ypred_mean + gaussian_result
"""
new_df_final['ypred_mean_promo_new'] = new_df_final['ypred_mean_promo'] + new_df_final['mean_promo_reg_result']
new_df_final['ypred_zero_promo_new'] = new_df_final['ypred_zero_promo'] + new_df_final['zero_promo_reg_result']


new_df_sku = new_df_final.groupby('ProductKey').sum().reset_index()
np.sum(np.abs(new_df_sku.ypred_raw - new_df_sku.salesForecast)) / new_df_sku.salesForecast.sum()
np.sum(np.abs(new_df_sku.ypred_zero_promo_new - new_df_sku.salesForecast)) / new_df_sku.salesForecast.sum()
np.sum(np.abs(new_df_sku.ypred_mean_promo_new - new_df_sku.salesForecast)) / new_df_sku.salesForecast.sum()

non_zero = new_df_sku[~(new_df_sku.salesForecast==0)]
np.sum(np.abs(non_zero.ypred_raw - non_zero.salesForecast)) / non_zero.salesForecast.sum()
np.sum(np.abs(non_zero.ypred_zero_promo_new - non_zero.salesForecast)) / non_zero.salesForecast.sum()
np.sum(np.abs(non_zero.ypred_mean_promo_new - non_zero.salesForecast)) / non_zero.salesForecast.sum()

new_df_sku_rdc = new_df_final.groupby(['RDCKey','ProductKey']).sum().reset_index()
for rdc in list(new_df_sku_rdc.RDCKey.unique()):
	this_df = new_df_sku_rdc[new_df_sku_rdc.RDCKey==rdc]
	print "rdc:" + str(rdc)
	print np.sum(np.abs(this_df.ypred_raw - this_df.salesForecast)) / this_df.salesForecast.sum()
	print np.sum(np.abs(this_df.ypred_zero_promo_new - this_df.salesForecast)) / this_df.salesForecast.sum()
	print np.sum(np.abs(this_df.ypred_mean_promo_new - this_df.salesForecast)) / this_df.salesForecast.sum()
