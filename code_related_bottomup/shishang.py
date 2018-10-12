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
import xgboost as xgb
import numpy as np
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)
path = 'tmp/data/shishang'
result_path = 'tmp/data/shishang'
cate_list = [2589]
#cate_list = [2584,2589,12029]  #2584
#cate_list = [7052, 7054, 870]
scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
'''online analysis'''
#scenario['use_cart_feature'] = False
#scenario['use_promo_features'] = False
holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')
area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')
import ow_train
for cate in cate_list:
    if cate in [2589]:
        pred_date = pd.to_datetime('2018-04-01')
        scenario['lookforwardPeriodDays'] = 7
        print "123"
    elif cate in [2584,12029]:
        pred_date = pd.to_datetime('2018-05-07')
        scenario['lookforwardPeriodDays'] = 7
    else:
        pred_date = pd.to_datetime('2018-02-01')
        scenario['lookforwardPeriodDays'] = 10
    print pred_date
    pro_canlender_path = os.path.join(path, str(cate)+'/'+str(cate)+'_'+'p2.da')
    promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)
    promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
                 'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
                 'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
                 'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
                 'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

    p1_out_path = os.path.join(path, str(cate)+'/'+'train'+'_p1_'+str(cate)+'.csv')
    period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])

    seasonality_df = pd.read_csv('tmp/data/870_season.csv', parse_dates=['Date'])

    file_path = os.path.join(path, str(cate)+'/'+str(cate)+'_'+'ts.da')
    ts_df = pd.read_csv(file_path,header=None,sep='\t')
    model,feature=ow_train.train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,seasonality_df,process_f01_flag=False,mode='dev',ForecastStartDate=pred_date)
    feature.to_csv(os.path.join(result_path, str(cate)+'/'+str(cate)+'_train_feature.csv'),index=False)
    save_object(model, os.path.join(result_path, str(cate)+'/'+str(cate)+'_train_model.pkl'))

import pickle
scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)

#scenario['lookforwardPeriodDays'] = 1
#scenario['use_cart_feature'] = False
#scenario['use_promo_features'] = False
path = 'tmp/data/shishang'
result_path = 'tmp/data/shishang'
cate_list = [2589]
#cate_list = [7052, 7054, 870]
#cate_list = [2584,2589,12029]  #2584
holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')
area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')
import ow_predict

for cate in cate_list:
    pro_canlender_path = os.path.join(path, str(cate)+'/'+str(cate)+'_'+'p2.da')
    promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)
    promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
	       'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
	       'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
	       'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
	       'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

    p1_out_path = os.path.join(path, str(cate)+'/'+'predict'+'_p1_'+str(cate)+'.csv')
    period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])

    file_path = file_path = os.path.join(path, str(cate)+'/'+str(cate)+'_'+'ts.da')
    ts_df = pd.read_csv(file_path,header=None,sep='\t')

    model_path = os.path.join(result_path, str(cate)+'/'+str(cate)+'_train_model.pkl')
    with open(model_path,'r') as input:
        model = pickle.load(input)

    seasonality_df = pd.read_csv('tmp/data/870_season.csv', parse_dates=['Date'])

    if cate in [2589]:
	pred_date = pd.to_datetime('2018-04-01')
	scenario['lookforwardPeriodDays'] = 7
    elif cate in [2584,12029]:
	pred_date = pd.to_datetime('2018-05-07')
	scenario['lookforwardPeriodDays'] = 7
    else:
	pred_date = pd.to_datetime('2018-02-01')
	scenario['lookforwardPeriodDays'] = 10
    print pred_date
    result,df_fut=ow_predict.predict(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,model,seasonality_df,process_f01_flag=False,mode='dev',ForecastStartDate=pred_date)
  #result.to_csv('tmp/data/0425_result/'+'result_'+str(cate)+'.csv',index=False)
    result.to_csv('tmp/data/shishang/'+'result_'+str(cate)+'.csv',index=False)
