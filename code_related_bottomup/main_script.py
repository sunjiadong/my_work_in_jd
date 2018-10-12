# -*- coding: utf-8 -*-
import warnings
import os.path
import ow_f01, ow_train, ow_predict, predict_mean
import pickle
from code.refactor.common import loadSettingsFromYamlFile,save_object
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore", category=DeprecationWarning)

path = 'tmp/data/'
result_path = 'tmp/data/test_0705'
suffix = '.da'
item = 'p1'
for_what = ['train', 'predict']
cate = 7054

scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')
holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')

'''
	function to reduce dataframe memory usage
'''
def reduce_df_mem_usage(df):
    # memery now
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of the dataframe is :", start_mem_usg, "MB")
    
    #np.nan will be handled as float
    NAlist = []
    for col in df.columns:
        # filter object type
        if (df[col].dtypes == np.float64):
            df[col] = df[col].astype(np.float32)
            continue
        if (df[col].dtypes != object)&(df[col].dtypes != 'datetime64[ns]'):
            
            print("**************************")
            print("columns: %s"%col)
            print("dtype before: %s"%df[col].dtype)
            
            # if int or not
            isInt = False
            mmax = df[col].max()
            mmin = df[col].min()
            
            # Integer does not support NA, therefore Na needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                #continue
                df[col].fillna(-999, inplace=True) # fill -999
                
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = np.fabs(df[col] - asint)
            result = result.sum()
            if result < 0.01: # absolute error < 0.01,then could be saw as integer
                isInt = True
            
            # make interger / unsigned Integer datatypes
            if isInt:
                if mmin >= 0: # min>=0, then unsigned integer
                    if mmax <= np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif mmax <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif mmax <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mmin > np.iinfo(np.int64).min and mmax < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
            df.replace(-999, np.nan, inplace=True)
            print("dtype after: %s"%df[col].dtype)
            print("********************************")
    print("___MEMORY USAGE AFTER CONVERSION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


p1_used_header=['productkey', 'promotionkey', 'startdatetime', 'enddatetime', 'jdprice', 'syntheticgrossprice', 'promotiondesc', 'promotiondesc_flag', 'promotiontype', 'promotionsubtype',
                'areatypearray', 'tokenflag', 'directdiscount_discount', 'directdiscount_availabilitynumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway',
                'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3',
                'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1',
                'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3',
                'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool',
                'suit_discount', 'directdiscount_saleprice', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent',
                'bundle_subtype6_percent', 'suit_percent', 'allpercentdiscount', 'mainproductkey', 'hierarchylevel3key', 'createdate', 'statuscode', 'dt']

'''
	hyperparameters
'''
cate = 7054
pred_date = pd.to_datetime('2018-02-01')
scenario['lookforwardPeriodDays'] = 10

'''
**1. f01 part to handle p1
output : p1_train_out & p1_predict_out		
'''
p1_path = os.path.join(path, str(cate)+'_'+item+suffix)
period_promo_raw = pd.read_csv(p1_path,sep='\t',header=None)
period_promo_raw.columns=p1_used_header
    
for fw in for_what:
    print "output and save: %s_p1_%s"%(str(cate),fw)
    period_promo_raw_clean = period_promo_raw.copy()
    train_pred_gate = fw   # 'train'
    f01_output = ow_f01.generate_f01_promo(area_rdc_map, period_promo_raw_clean,scenario, train_pred_gate, ForecastStartDate=pred_date)
    f01_output.to_csv(os.path.join(result_path, train_pred_gate+'_'+item+'_'+str(cate)+'.csv'),index=False)
'''
f01 part finish
'''
'''
global dataframes used several times later
'''
seasonality_df = pd.read_csv('tmp/data/870_season.csv', parse_dates=['Date'])
file_path = os.path.join(path, str(cate)+'_'+'ts.da')
ts_df = pd.read_csv(file_path,header=None,sep='\t')

pro_canlender_path = os.path.join(path, str(cate)+'_'+'p2.da')
promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)
promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
				         'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
				         'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
				         'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
				         'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

'''
**2. train model 
output : model & historical feature datas
'''
p1_out_path = os.path.join(result_path, 'train'+'_p1_'+str(cate)+'.csv')
period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])

seasonality_df_train = seasonality_df.copy()
ts_df_train = ts_df.copy()
promoCalendarDf_train = promoCalendarDf.copy()

model,feature=ow_train.train(area_rdc_map,period_promo_raw,promoCalendarDf_train,ts_df_train,scenario,holidays_df,seasonality_df_train,process_f01_flag=False,mode='dev',ForecastStartDate=pred_date)
feature.to_csv(os.path.join(result_path, str(cate)+'/'+str(cate)+'_train_feature.csv'),index=False)
save_object(model, os.path.join(result_path, str(cate)+'/'+str(cate)+'_train_model.pkl'))

'''
**3. predict
output : pred values with future feature datas
'''
p1_out_path = os.path.join(path, 'predict'+'_p1_'+str(cate)+'.csv')
period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])

seasonality_df_test = seasonality_df.copy()
ts_df_test = ts_df.copy()
promoCalendarDf_test = promoCalendarDf.copy()

# model_path = os.path.join(result_path, str(cate)+'/'+str(cate)+'_train_model.pkl')
# with open(model_path,'r') as input:
#     model = pickle.load(input)
q_pred_result,df_fut=ow_predict.predict(area_rdc_map,period_promo_raw,promoCalendarDf_test,ts_df_test,scenario,holidays_df,model,seasonality_df_test,process_f01_flag=False,mode='dev',ForecastStartDate=pred_date)
q_pred_result.to_csv('tmp/data/test_0705/'+'result_'+str(cate)+'.csv',index=False)

'''
**4. predict with promotion feature mean values
output : pred values with future feature datas and mean feature datas
'''
p1_out_path = os.path.join(result_path, 'predict'+'_p1_'+str(cate)+'.csv')
period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])

seasonality_df_mean = seasonality_df.copy()
ts_df_mean = ts_df.copy()
promoCalendarDf_mean = promoCalendarDf.copy()

#train_feature_path = os.path.join(result_path, str(cate) + '/' + str(cate) + '_train_feature.csv')
train_feature_df = feature
q_mean_result,df_fut_mean=predict_mean.predict(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,model,seasonality_df,process_f01_flag=False,mode='dev',ForecastStartDate=pred_date,train_feature=train_feature_df)
q_mean_result.to_csv('tmp/data/shishang/result/'+'result_'+str(cate)+'_mean.csv',index=False)


'''
**5. get actual sales in test dataset
'''
real_ts_df = ts_df.copy()
real_ts_df.columns = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
real_ts_df['Date'] = pd.to_datetime(real_ts_df['Date'])
simplified_ts_df = real_ts_df[real_ts_df.Date.between('2018-02-01','2018-02-10')]
#df2.to_csv('tmp/data/shishang/actual/simplified_ts_' + str(cate) + '.csv',index=False)

'''
**6. predict history datasets and simulate promotions based on lr
'''

'''
	**6.1 predict q_pred_his & q_mean_his of history datasets 
'''
train_feature_df = feature
#model_path = 'tmp/data/shishang/' + str(cate) + '/' + str(cate) + '_train_model.pkl'

df_new, NAlist = reduce_df_mem_usage(train_feature_df)

ForecastStartDate = pd.to_datetime(pred_date)
DataStartDate = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDays'])
PredictEndDate = ForecastStartDate + datetime.timedelta(days=(scenario['lookforwardPeriodDays']-1))

#actual = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/shishang/actual/simplified_ts_' + str(cate) + '.csv')
actual = simplified_ts_df
actual.Date = pd.to_datetime(actual.Date)
actual.RDCKey = actual.RDCKey.astype(float)

list_keys = ['Date','RDCKey','ProductKey']

raw = q_pred_result  ###bottomup forecast
raw = raw[list_keys + ['salesForecast','ypred']]
raw.rename(columns={'ypred':'ypred_raw'},inplace=True)
raw.drop('salesForecast',axis=1,inplace=True)

feat_cols = ['dd_price_weighted','bd_price_weighted','dd_price_weighted_x','bd_price_weighted_x','SyntheticGrossPrice']
mean_df = q_mean_result
mean_df = mean_df[list_keys + feat_cols + ['salesForecast','ypred']]
mean_df.rename(columns={'ypred':'ypred_mean_promo'},inplace=True)
mean_df.drop('salesForecast',axis=1,inplace=True)

new_df = raw.merge(mean_df,on=list_keys)
new_df.Date = pd.to_datetime(new_df.Date)
new_df = pd.merge(new_df, actual[list_keys+['salesForecast']], how='left',on = list_keys)

exclu_promo_features = ['strongmark','flashsale_ind','dd_ind','bundle_ind','bundle_buy199get100_ind','suit_ind','freegift_ind']
update_cols = list(set(scenario['promo_feature_cols'])- set(exclu_promo_features))
need_cols = ['Date','RDCKey','ProductKey','HierarchyLevel3Key'] + update_cols

uses_promo = ['mean','no']
df = df_new #df_new
reg_cols = []#['Holiday','Ind_1111_pre','Ind_1111','Ind_1111_post','Ind_618_pre','Ind_618','Ind_618_post','Ind_1212','Month','DayOfWeek',]

for use_promo in uses_promo:
    if use_promo == 'mean':
        df1 = df[need_cols]
        groupkeys = ['RDCKey','ProductKey','HierarchyLevel3Key']
        promo_feature_cols =  scenario['promo_feature_cols']
        df11 = df1.groupby(groupkeys)[update_cols].mean().reset_index()
        df2 = pd.merge(df,df11[groupkeys + update_cols], how='left',on=groupkeys)

        rename_update_cols = [col+'_y' for col in update_cols]
        for col in update_cols:
            df2.rename(columns={col+'_y': col},inplace=True)
            df2.drop(col+'_x',axis=1,inplace=True)
        grouped = df2.groupby('RDCKey')
    else:
        #histoty bottomup forecast
        grouped = df.groupby('RDCKey')
    result_list = []
    for rdc, history_df in grouped:
        if rdc in model.keys():
            this_model = model[rdc]
        else:
            continue
        ''' predict model '''
        xColumns = scenario['selectedColumns']['features']

        if 'RDCKey' in xColumns:# 删除季节性,RDCKEY
            xColumns.remove('skuDecomposedTrend')
            xColumns.remove('skuDecomposedSeasonal')
            xColumns.remove('level3DecomposedTrend')
            xColumns.remove('level3DecomposedSeasonal')
            xColumns.remove('Curve')
            xColumns.remove('RDCKey')
        
        X_history = history_df[xColumns]

        history_xtest = xgb.DMatrix(X_history.values, missing=np.NaN )
        ypred = this_model.predict(history_xtest)
        history_df['ypred'] =ypred
        history_df['RDCKey'] = rdc

        ''' Tuning result '''
        lanjie = history_df[(history_df.ypred<0)]
        if len(lanjie)>0:
            history_df.ix[lanjie.index,'ypred'] = 0
        result_list.append(history_df)
    final_result = pd.concat(result_list)
    if use_promo == 'no':
        raw_train_df = final_result[list_keys + reg_cols + scenario['promo_feature_cols'] + ['salesForecast','ypred']]
    else:
        #use_promo == 'mean':
        train_df_mean = final_result[list_keys + reg_cols + scenario['promo_feature_cols'] + ['salesForecast','ypred']]
        train_df_mean.rename(columns={'ypred':'ypred_mean_promo'}, inplace=True)

'''
	**6.2 fit history datasets between promotion features and pred sales residual using linear regression 
'''
raw_train_df = pd.merge(raw_train_df, train_df_mean[list_keys+['ypred_mean_promo']], how='left', on=list_keys)
raw_test_df = q_pred_result
raw_test_df = raw_test_df[list_keys + reg_cols + scenario['promo_feature_cols'] + ['salesForecast','ypred']]
raw_test_df.Date = pd.to_datetime(raw_test_df.Date)

used_cols = reg_cols + ['MaxSyntheticDiscountA']  #['MaxBundleDiscount','MaxDirectDiscount','MaxDiscount','MaxSyntheticDiscountA','daynumberinpromotion','PromotionCount']
raw_train_df.Date = pd.to_datetime(raw_train_df.Date)
raw_train_df = raw_train_df[raw_train_df.Date < pred_date]
raw_train_df = raw_train_df[list_keys + reg_cols + scenario['promo_feature_cols'] + ['ypred','ypred_mean_promo']]
input_df = pd.concat([raw_train_df, raw_test_df])

for col in used_cols:
    #input_df[col] = input_df.groupby(['RDCKey','ProductKey'])[col].transform(lambda x: x.fillna(method='bfill').fillna(0))
    #input_df[col] = input_df.groupby(['RDCKey','ProductKey'])[col].transform(lambda x: x.fillna(method='ffill').fillna(0))
    #input_df = input_df[input_df['MaxSyntheticDiscountA'].between(-1,1)]
    
    input_df = input_df[~(input_df[col].isnull())]
    input_df = input_df[input_df[col].between(-1,1)]

value_type = 'ypred_mean_promo'
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
        
        y_train = train_df['ypred'] - train_df[value_type]
        y_test = test_df['salesForecast']
        lm = LinearRegression()
        lm.fit(x_train_df, y_train)
        Intercept = lm.intercept_
        RSquare = lm.score(x_train_df, y_train)
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

mean_final = final_df
mean_final.rename(columns={'reg_result':'mean_promo_reg_result'},inplace=True)
final_df.drop('salesForecast',axis=1,inplace=True)
final_df.drop('ypred',axis=1,inplace=True)
new_df_final = new_df.merge(final_df,on=list_keys,how='left')
new_df_final.fillna(0,inplace=True)
new_df_final['ypred_mean_promo_new'] = new_df_final['ypred_mean_promo'] + new_df_final['mean_promo_reg_result']
new_df_final.to_csv('result_btup_lr/result_ensemble_%s.csv'%str(cate), index=False)

###evaluate
new_df_sku = new_df_final.groupby('ProductKey').sum().reset_index()
print "ensemble pred sum : %f"%(new_df_sku.ypred_mean_promo_new.sum())
print "raw pred sum :      %f"%(new_df_sku.ypred_raw.sum())
print "actual sum:         %f"%(new_df_sku.salesForecast.sum())

print "raw pred residual:      %f"%(np.sum(np.abs(new_df_sku.ypred_raw - new_df_sku.salesForecast)))
print "ensemble pred residual: %f"%(np.sum(np.abs(new_df_sku.ypred_mean_promo_new - new_df_sku.salesForecast)))

print "raw pred mape: %f"%(np.sum(np.abs(new_df_sku.ypred_raw - new_df_sku.salesForecast)) / new_df_sku.salesForecast.sum())
print "ensemble mape: %f"%(np.sum(np.abs(new_df_sku.ypred_mean_promo_new - new_df_sku.salesForecast)) / new_df_sku.salesForecast.sum())