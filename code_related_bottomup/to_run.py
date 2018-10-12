###seasonality:870###
import warnings
import os.path
from code.refactor.common import (get_setting_from_cfg, get_setting_path, filter_df_by_cate_id,get_future_sales_feature,\
                                  dummy_features, model_predict, get_weekly_df,get_future_condition_sales_feature,gen_train_valid_by_date,\
                                  generate_promotion_future, get_hour, save_object, add_pv,get_lowest_n,generate_cv_train_valid_set,\
                                  generate_default_values_by_dic,fill_col_with_default, week2month,trans_band_to_int,get_future_condition_sales_feature_stage2,\
                                  loadSettingsFromYamlFile,createSeasonalityFeatures,createLevel3Features,createSeasonalityDecomposeFeatures,\
                                 calculateNationalRolling,calculateRolling,calculateLagging,createDateFeatures,splitTimeWindow,calculateSimilarRolling,calculateStockFeatures,\
                                 process_rdc,clean_data,add_cols,get_dd_price,agg_dd_price,get_bundle_feat,agg_bd_price,process_feature,\
                                 agg_feature,prep_data,calc_weighted_price,agg_feature_day)
from code.refactor.fdc_flow import  filter_non_price_fill_it
import pandas as pd
import xgboost as xgb
import numpy as np
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)
scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
cate_list = [870]
for cate in cate_list:
  if cate in [11922]:
    pred_date = pd.to_datetime('2017-10-10')
    scenario['lookforwardPeriodDays'] = 7
  if cate in [7052, 7054]:
    pred_date = pd.to_datetime('2018-02-01')
    scenario['lookforwardPeriodDays'] = 10
  else:
    pred_date = pd.to_datetime('2018-03-10')
    scenario['lookforwardPeriodDays'] = 8
scenario['forecastStartDate'] = pred_date
file_path = 'tmp/data/870_ts.da'
ts_df = pd.read_csv(file_path,header=None,sep='\t')
import ow_seasonality
seasonality_df = ow_seasonality.ow_seasonality(scenario,ts_df,ForecastStartDate=pred_date)
seasonality_df.to_csv('tmp/data/870_season.csv',index=False)


###f01###
# -*- coding: utf-8 -*-
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

#p1_path = 'tmp/0411_2018_brand/p1_0411.csv' 

path = 'tmp/data/'
suffix = '.da'
item = 'p1'
for_what = ['train', 'predict']
cate_list = [7054, 7052, 870]
scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')
import ow_f01

for fw in for_what:
  for cate in cate_list:
  	print "output and save: %s_p1_%s"%(str(cate),fw)
  	p1_path = os.path.join(path, str(cate)+'_'+item+suffix)

  	p1_used_header=['productkey', 'promotionkey', 'startdatetime', 'enddatetime', 'jdprice', 'syntheticgrossprice', 'promotiondesc', 'promotiondesc_flag', 'promotiontype', 'promotionsubtype',
  	                'areatypearray', 'tokenflag', 'directdiscount_discount', 'directdiscount_availabilitynumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway',
  	                'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3',
  	                'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1',
  	                'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3',
  	                'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool',
  	                'suit_discount', 'directdiscount_saleprice', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent',
  	                'bundle_subtype6_percent', 'suit_percent', 'allpercentdiscount', 'mainproductkey', 'hierarchylevel3key', 'createdate', 'statuscode', 'dt']
  	period_promo_raw = pd.read_csv(p1_path,sep='\t',header=None)
  	period_promo_raw.columns=p1_used_header

  	period_promo_raw_clean = period_promo_raw
  	train_pred_gate = fw   # 'train'
  	if cate in [11922]:
  	    pred_date = pd.to_datetime('2017-10-10')
  	    scenario['lookforwardPeriodDays'] = 7
  	if cate in [7052, 7054]:
  	  pred_date = pd.to_datetime('2018-02-01')
  	  scenario['lookforwardPeriodDays'] = 10
  	else:
  	  pred_date = pd.to_datetime('2018-02-01')
  	  scenario['lookforwardPeriodDays'] = 10
  	f01_output = ow_f01.generate_f01_promo(area_rdc_map, period_promo_raw_clean,scenario, train_pred_gate, ForecastStartDate=pred_date)
  	f01_output.to_csv(os.path.join(path, train_pred_gate+'_'+item+'_'+str(cate)+'_2018_02_01.csv'),index=False)
#f01_output.to_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/0411_2018_brand/p1_train_out_0412.csv',index=False)
#f01_output.to_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/0411_2018_brand/p1_predict_out_0412.csv',index=False)


###train###
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

path = 'tmp/data/'
result_path = 'tmp/data/0425_result'
cate_list = [870]
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
  if cate in [11922]:
    pred_date = pd.to_datetime('2017-10-10')
    scenario['lookforwardPeriodDays'] = 7
  if cate in [7052, 7054]:
    pred_date = pd.to_datetime('2018-02-01')
    scenario['lookforwardPeriodDays'] = 10
  else:
    pred_date = pd.to_datetime('2018-02-01')
    scenario['lookforwardPeriodDays'] = 10
    #pred_date = pd.to_datetime('2018-03-10')
    #scenario['lookforwardPeriodDays'] = 8

  pro_canlender_path = os.path.join(path, str(cate)+'_'+'p2.da')
  promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)
  promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
             'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
             'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
             'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
             'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

  p1_out_path = os.path.join(path, 'train'+'_p1_'+str(cate)+'_2018_02_01.csv')
  period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])

  sea_path = os.path.join(path, str(cate)+'_season.csv')
  seasonality_df = pd.read_csv(sea_path, parse_dates=['Date'])

  file_path = os.path.join(path, str(cate)+'_'+'ts.da')
  ts_df = pd.read_csv(file_path,header=None,sep='\t')
  model,feature=ow_train.train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,seasonality_df,process_f01_flag=False,mode='dev',ForecastStartDate=pred_date)
  #feature.to_csv(os.path.join(result_path, str(cate)+'_train_feature_with_season_180201_0515.csv'),index=False)
  save_object(model, os.path.join(result_path, str(cate)+'_train_model_with_season_180201_0515'+'.pkl'))

###predict###
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

#scenario['lookforwardPeriodDays'] = 1
#scenario['use_cart_feature'] = False
#scenario['use_promo_features'] = False
path = 'tmp/data/'
result_path = 'tmp/data/0425_result'
cate_list = [7052]
#cate_list = [7052, 7054, 870]

holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')
area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')
import ow_predict
#cate = 7052
for cate in cate_list:
  pro_canlender_path = os.path.join(path, str(cate)+'_'+'p2.da')
  promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)
  promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
           'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
           'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
           'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
           'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

  #p1_out_path = os.path.join(path, 'predict'+'_p1_'+str(cate)+'_2018_02_01.csv')
  p1_out_path = os.path.join(path, 'predict'+'_p1_'+str(cate)+'.csv')
  period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])

  file_path = file_path = os.path.join(path, str(cate)+'_'+'ts.da')
  ts_df = pd.read_csv(file_path,header=None,sep='\t')

  model_path = os.path.join(result_path, str(cate)+'_train_model'+'.pkl')
  with open(model_path,'r') as input:
    model = pickle.load(input)

  sea_path = os.path.join(path, '870_season.csv')
  seasonality_df = pd.read_csv(sea_path, parse_dates=['Date'])

  if cate in [11922]:
    pred_date = pd.to_datetime('2017-10-10')
    scenario['lookforwardPeriodDays'] = 7
  if cate in [7052, 7054]:
    pred_date = pd.to_datetime('2018-02-01')
    scenario['lookforwardPeriodDays'] = 10
  else:
    pred_date = pd.to_datetime('2018-02-01')
    scenario['lookforwardPeriodDays'] = 10
    #pred_date = pd.to_datetime('2018-03-10')
    #scenario['lookforwardPeriodDays'] = 30
  result,df_fut=ow_predict.predict(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,model,seasonality_df,process_f01_flag=False,mode='dev',ForecastStartDate=pred_date)
  #result.to_csv('tmp/data/0425_result/'+'result_'+str(cate)+'.csv',index=False)
  result.to_csv('tmp/data/0425_result/'+'result_'+str(cate)+'_zero_promo_0515.csv',index=False)

###simplify ts_df
for cate in cate_list:
  file_path = file_path = os.path.join(path, str(cate)+'_'+'ts.da')
  ts_df = pd.read_csv(file_path,header=None,sep='\t')
  ts_df.columns = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
  ts_df['Date'] = pd.to_datetime(ts_df['Date'])
  df = ts_df[ts_df.Date.between('2018-01-01','2018-04-01')]
  df.to_csv('tmp/data/simplified_ts_'+str(cate)+'.csv',index=False)


'''promotion imitate
    3343525     8792.0
    4214248     9898.0
    206748     11235.0
    206690     13811.0
    5046166    24451.0
'''
"""
df_fut.to_csv('tmp/11922_2018/at180319_11922_df_fut_20171010.csv',index=False)
skus=[3343525,4214248,206748,206690,5046166]
# discounts=[0,0.05,0.1,0.2,0.3,0.4]
discounts=[0.2,0.3,0.4]
# discounts=[0,0.03,0.05,0.07,0.1,0.14]
for sku in skus:
    print '------------sku:%r-----------' % sku
    for discount in discounts:
        rdc = 6
        # sku = 206748
        print "=======discount ratio : %r " % discount
        # print 'sku %r '% sku
        print 'rdc %r '% rdc
        this_model = model[rdc]
        pred_df = df_fut[(df_fut.RDCKey==rdc)&(df_fut.ProductKey==sku)]
        # pred_df = df_fut[(df_fut.RDCKey==rdc)]
        xColumns = scenario['selectedColumns']['features']
        X_future = pred_df[xColumns]

        # X_future['MaxDiscount'] = discount
        # X_future['MinDiscount'] = discount
        # X_future['dd_discount_sgp_wgt'] = discount
        # X_future['dd_discount_daily_max'] = discount
        # X_future['dd_discount_wgt'] = discount
        X_future['MaxSyntheticDiscountA'] = discount
        # X_future['MinSyntheticDiscountA'] = discount
        # X_future['AvgDiscount'] = discount
        # X_future['MinSyntheticDiscountA'] = discount
        # X_future['bd_discount_sgp_wgt'] = discount
        # X_future['bd_discount_daily_max'] = discount
        # print X_future.head()
        future_xtest = xgb.DMatrix(X_future.values, missing=np.NaN )
        ypred = this_model.predict(future_xtest)
        print 'prediction:'
        print ypred.mean()
"""

###anlysis###
import pandas as pd
import numpy as np
df1 = pd.read_csv('./result_11922_171201_180322.csv')
df2 = pd.read_csv('./result_no_promo_11922_171201_180322.csv')
cols = ['Date','HierarchyLevel3Key','ProductKey','RDCKey','ypred']
df11 = df1[cols]
df22 = df2[cols]

file_path = '/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/11922_predict_ts'
ts1_df = pd.read_csv(file_path,header=None,sep='\t')
ts1_df.columns = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
#ts1_df['Date'] = pd.to_datetime(ts1_df['Date'])
groupkeys = ['Date','HierarchyLevel3Key','ProductKey','RDCKey']

df11 = pd.merge(df11[groupkeys+['ypred']],ts1_df[groupkeys+['salesForecast']],how='left',on=groupkeys)
df22 = pd.merge(df22[groupkeys+['ypred']],ts1_df[groupkeys+['salesForecast']],how='left',on=groupkeys)

df111 = df11.groupby(['HierarchyLevel3Key','ProductKey','RDCKey']).sum().reset_index()
df222 = df22.groupby(['HierarchyLevel3Key','ProductKey','RDCKey']).sum().reset_index()
df111.RDCKey.drop_duplicates()
print "###promo###"
for rdc in list(df111.RDCKey.unique()):
#for rdc in [3,4,5,6,9,10,316,772]:
    this_df = df111[df111.RDCKey==rdc]
    this_df['resi'] = np.abs(this_df['salesForecast']-this_df['ypred'])
    print "rdc:" + str(rdc)
    print this_df['resi'].sum()/this_df['salesForecast'].sum()
print "###no_promo###"
for rdc in list(df222.RDCKey.unique()):
#for rdc in [3,4,5,6,9,10,316,772]:
    this_df = df222[df222.RDCKey==rdc]
    this_df['resi'] = np.abs(this_df['salesForecast']-this_df['ypred'])
    print "rdc:" + str(rdc)
    print this_df['resi'].sum()/this_df['salesForecast'].sum()

20171010_with_promo:
rdc:3
0.584452904216
rdc:4
0.50459152612
rdc:5
0.559656483525
rdc:6
0.459164205967
rdc:9
0.639824705004
rdc:10
0.47356323862
rdc:316
0.444125947242
rdc:772
0.607733665499

20171010_using_cart:
rdc:3
0.61950189572
rdc:4
0.574585135967
rdc:5
0.637494464522
rdc:6
0.481731483181
rdc:9
0.690757370477
rdc:10
0.511707883364
rdc:316
0.489222587133
rdc:772
0.632678776217

2017-10-10_nocart_step_1:
rdc:3
0.531055743355
rdc:4
0.543857142635
rdc:5
0.632023941386
rdc:6
0.440021171613
rdc:9
0.74762898543
rdc:10
0.51308073992
rdc:316
0.607802575399
rdc:772
0.619626582543


2017-10-10_national_cart_step_1:
rdc:3
0.523670564569
rdc:4
0.548895838992
rdc:5
0.649744387616
rdc:6
0.457305383916
rdc:9
0.717220048212
rdc:10
0.500215681583
rdc:316
0.611979516751
rdc:772
0.635744050684


20171010_without_promo:
rdc:3
0.596172276464
rdc:4
0.51702087262
rdc:5
0.573975730338
rdc:6
0.469837136978
rdc:9
0.65000812652
rdc:10
0.485815007441
rdc:316
0.453240161734
rdc:772
0.592190740732

20171010_using_mean:
rdc:3
0.603310014042
rdc:4
0.528695200412
rdc:5
0.584915072441
rdc:6
0.473640776981
rdc:9
0.669727219134
rdc:10
0.491787774208
rdc:316
0.467035768839
rdc:772
0.627159141561



2017-11-01:
rdc:3
0.813376625697
rdc:4
0.771337642255
rdc:5
0.78969468393
rdc:6
0.737614199533
rdc:9
0.762104963024
rdc:10
0.758181406893
rdc:316
0.613564979204
rdc:772
0.714780346287
###no_promo###
rdc:3
0.80284620948
rdc:4
0.776443852968
rdc:5
0.788841780733
rdc:6
0.738396678287
rdc:9
0.764183562942
rdc:10
0.78544575247
rdc:316
0.630668550257
rdc:772
0.724710220022


2017-12-01:

rdc:3
0.515882519052
rdc:4
0.495543628954
rdc:5
0.556828774679
rdc:6
0.328053129009
rdc:9
0.450720512606
rdc:10
0.358104244965
rdc:316
0.427717694057
rdc:772
0.544402105646
###no_promo###
rdc:3
0.504856686704
rdc:4
0.517241088848
rdc:5
0.618680591044
rdc:6
0.337595630943
rdc:9
0.484913934838
rdc:10
0.388452648812
rdc:316
0.459170993944
rdc:772
0.565556015129



###using historical mean in df_fut###
import pandas as pd
df_past = pd.read_csv('./tmp/11922_2018/at180321_11922_v_oct_feature_20171010.csv')
#df_fut = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/at180319_11922_df_fut_20171010.csv')
s1 = set(list(df_past.columns.values))
s2 = set(list(df_fut.columns.values))
s1-s2
scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
exclu_promo_features = ['strongmark','flashsale_ind','dd_ind','bundle_ind','bundle_buy199get100_ind','suit_ind','freegift_ind']
update_cols = list(set(scenario['promo_feature_cols'])- set(exclu_promo_features))
need_cols = ['Date','RDCKey','ProductKey','HierarchyLevel3Key'] + update_cols
df1 = df_past[need_cols]
groupkeys = ['RDCKey','ProductKey','HierarchyLevel3Key']
promo_feature_cols =  scenario['promo_feature_cols']
df11 = df1.groupby(groupkeys)[update_cols].mean().reset_index()
tmp = df_past[(df_past.RDCKey==3)&(df_past.ProductKey==188059)&(df_past.HierarchyLevel3Key==11922)]
tmp.PromotionCount.mean()
tmp.MaxSyntheticDiscountA.mean()

df2 = df_fut[need_cols]
df22 = pd.merge(df2,df11[groupkeys + update_cols], how='left',on=groupkeys)
rename_update_cols = [col+'_y' for col in update_cols]
