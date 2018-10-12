# -*- coding: utf-8 -*-
import warnings
import os.path
from code.refactor.common import (get_setting_from_cfg, get_setting_path, filter_df_by_cate_id,get_future_sales_feature,\
                                  dummy_features, model_predict, get_weekly_df,get_future_condition_sales_feature,gen_train_valid_by_date,\
                                  generate_promotion_future, get_hour, save_object, add_pv,get_lowest_n,generate_cv_train_valid_set,\
                                  generate_default_values_by_dic,fill_col_with_default, week2month,trans_band_to_int,get_future_condition_sales_feature_stage2,\
                                  loadSettingsFromYamlFile,createSeasonalityFeatures,createLevel3Features,createSeasonalityDecomposeFeatures,\
                                 calculateNationalRolling_predict,calculateRolling_predict,calculateLagging_predict,createDateFeatures,splitTimeWindow,calculateSimilarRolling_predict,calculateStockFeatures,\
                                 process_rdc,clean_data,add_cols,get_dd_price,agg_dd_price,get_bundle_feat,agg_bd_price,process_feature,\
                                 agg_feature,prep_data,calc_weighted_price,agg_feature_day,get_column_by_type,object2Float,object2Int)
from code.refactor.fdc_flow import  filter_non_price_fill_it
import pandas as pd
import xgboost as xgb
import numpy as np
import datetime
import pickle

scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)

#df = pd.read_csv('./tmp/data/0425_result/870_train_feature.csv')
#model_path = 'tmp/data/0425_result/870_train_model.pkl'
df = pd.read_csv('./tmp/data/0425_result/870_train_feature_with_season.csv')
model_path = 'tmp/data/0425_result/870_train_model_with_season.pkl'

with open(model_path,'r') as input:
    model = pickle.load(input)

exclu_promo_features = ['strongmark','flashsale_ind','dd_ind','bundle_ind','bundle_buy199get100_ind','suit_ind','freegift_ind']
update_cols = list(set(scenario['promo_feature_cols'])- set(exclu_promo_features))
need_cols = ['Date','RDCKey','ProductKey','HierarchyLevel3Key'] + update_cols
"""
history mean 
"""
df1 = df[need_cols]
groupkeys = ['RDCKey','ProductKey','HierarchyLevel3Key']
promo_feature_cols =  scenario['promo_feature_cols']
df11 = df1.groupby(groupkeys)[update_cols].mean().reset_index()
df2 = pd.merge(df,df11[groupkeys + update_cols], how='left',on=groupkeys)

pd.options.display.max_columns=999
pd.options.display.max_rows=999
pd.options.display.width=160

rename_update_cols = [col+'_y' for col in update_cols]
for col in update_cols:
    df2.rename(columns={col+'_y': col},inplace=True)
grouped = df2.groupby('RDCKey')
"""
history zero
"""
# for col in update_cols:
#   df[col] = 0
# grouped = df.groupby('RDCKey')

result_list = []
for rdc, history_df in grouped:
    if rdc in model.keys():
        this_model = model[rdc]
    else:
        continue
    ''' predict model '''
    xColumns = scenario['selectedColumns']['features']

    if 'RDCKey' in xColumns:# 删除季节性,RDCKEY
        #xColumns.remove('skuDecomposedTrend')
        #xColumns.remove('skuDecomposedSeasonal')
        #xColumns.remove('level3DecomposedTrend')
        #xColumns.remove('level3DecomposedSeasonal')
        #Columns.remove('Curve')
        xColumns.remove('RDCKey')
        #for col in update_cols: ###sjd_update
        #    xColumns.remove(col)

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
final_result.to_csv('./tmp/data/0425_result/history_7054_sales_promo_mean.csv',index=False)
