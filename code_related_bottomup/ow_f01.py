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

def generate_f01_promo(area_rdc_map, period_promo_raw, scenario, train_pred_gate='train',ForecastStartDate=None):
    if ForecastStartDate is None:
        ForecastStartDate = pd.to_datetime(scenario['forecastStartDate'])

    ForecastStartDate = pd.to_datetime(ForecastStartDate)
    DataStartDate = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDays']) # 2year ago

    PredictEndDate = ForecastStartDate + datetime.timedelta(days=(scenario['lookforwardPeriodDays']-1)) # 10-28
    # DataStartDate_Pred = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDays_predict']) # 1.5 year before
    for col in area_rdc_map.columns:
        area_rdc_map[col] = area_rdc_map[col].astype('int')

    period_promo_raw.replace('null', np.nan, inplace=True)
    period_promo_raw.replace(' null', np.nan, inplace=True)
    period_promo_raw.replace('None', np.nan, inplace=True)
    period_promo_raw.replace(-999, np.nan, inplace=True)
    # to_float_col = ['productkey', 'promo_inperiod_flag', 'mainproductkey', 'hierarchylevel1key', 'hierarchylevel2key', 'hierarchylevel3key', 'statuscode', 'promotionkey', 'jdprice', 'syntheticgrossprice', 'promotiontype', 'promotionsubtype', 'levelmember', 'directdiscount_productkey', 'directdiscount_maxjdprice', 'directdiscount_minjdprice', 'directdiscount_avgjdprice', 'directdiscount_maxscrapedprice', 'directdiscount_minscrapedprice', 'directdiscount_avgscrapedprice', 'directdiscount_discount', 'directdiscount_price', 'directdiscount_limitationflag', 'directdiscount_limitationcode', 'directdiscount_availabilitynumberflag', 'directdiscount_availabilitynumber', 'directdiscount_minimumnumberflag', 'directdiscount_minimumnumber', 'directdiscount_maximumnumberflag', 'directdiscount_maximumnumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway', 'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3', 'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1', 'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3', 'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool', 'suit_discount', 'freegift_productkey', 'freegift_maxscrapedprice', 'freegift_minscrapedprice', 'freegift_avgscrapedprice', 'freegift_maxjdprice', 'freegift_minjdprice', 'freegift_avgjdprice', 'freegift_freegiftvalue', 'freegift_freegiftvalueatjdprice', 'freegift_freegiftvalueatscrapedprice', 'directdiscount_saleprice', 'directdiscount_percent', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent', 'bundle_subtype6_percent', 'suit_percent', 'freegift_percent', 'allpercentdiscount']
    to_float_col = ['allpercentdiscount','bundle_subtype15_giveawayrate1','bundle_subtype15_giveawayrate2','bundle_subtype15_giveawayrate3','bundle_subtype15_percent','bundle_subtype15_thresholdnumber1','bundle_subtype15_thresholdnumber2','bundle_subtype15_thresholdnumber3','bundle_subtype1_giveaway','bundle_subtype1_percent','bundle_subtype1_threshold','bundle_subtype2_giveaway','bundle_subtype2_maximumgiveaway','bundle_subtype2_percent','bundle_subtype2_threshold','bundle_subtype4_giveaway1','bundle_subtype4_giveaway2','bundle_subtype4_giveaway3','bundle_subtype4_percent','bundle_subtype4_threshold1','bundle_subtype4_threshold2','bundle_subtype4_threshold3','bundle_subtype6_freenumber','bundle_subtype6_percent','bundle_subtype6_thresholdnumber','directdiscount_availabilitynumber','directdiscount_discount','directdiscount_saleprice','hierarchylevel3key','jdprice','mainproductkey','productkey','promotionkey','promotionsubtype','promotiontype','statuscode','suit_avgvaluepool','suit_discount','suit_maxvaluepool','suit_minvaluepool','suit_percent','syntheticgrossprice']
    period_promo_raw['enddatetime'] = pd.to_datetime(period_promo_raw['enddatetime'])
    period_promo_raw['startdatetime'] = pd.to_datetime(period_promo_raw['startdatetime'])

    if 'object' in period_promo_raw[to_float_col].dtypes.values:
        object2Float(period_promo_raw,to_float_col)
    if train_pred_gate=='train':

        period_promo_raw = period_promo_raw[(period_promo_raw.enddatetime>DataStartDate)&(period_promo_raw.startdatetime<=ForecastStartDate)]
    else: #predict
        period_promo_raw = period_promo_raw[(period_promo_raw.enddatetime>=ForecastStartDate)&(period_promo_raw.startdatetime<PredictEndDate)]

    period_promo_raw.rename(columns={'directdiscount_availabilitynumber':'dd_availabilitynumber','directdiscount_availabilitynumberflag':'dd_availabilitynumberflag'\
            ,'directdiscount_productkey':'dd_productkey','directdiscount_maxjdprice':'dd_maxjdprice',\
            'directdiscount_minjdprice':'dd_minjdprice','directdiscount_avgjdprice':'dd_avgjdprice',\
            'directdiscount_maxscrapedprice':'dd_maxscrapedprice','directdiscount_minscrapedprice':'dd_minscrapedprice',\
            'directdiscount_avgscrapedprice':'dd_avgscrapedprice','directdiscount_jdpricedate':'dd_jdpricedate',\
            'directdiscount_discount':'dd_discount','directdiscount_price':'dd_price',\
            'directdiscount_limitationflag':'dd_limitationflag','directdiscount_limitationcode':'dd_limitationcode',\
            'directdiscount_availabilitynumberflag':'dd_availabilitynumberflag','directdiscount_availabilitynumber':'dd_availabilitynumber',\
            'directdiscount_minimumnumberflag':'dd_minimumnumberflag','directdiscount_minimumnumber':'dd_minimumnumber',\
            'directdiscount_maximumnumberflag':'dd_maximumnumberflag','directdiscount_maximumnumber':'dd_maximumnumber',\
            'directdiscount_saleprice':'dd_saleprice','directdiscount_percent':'dd_percent',\
            'bundle_subtype1_threshold':'bd_subtype1_threshold',\
            'bundle_subtype1_giveaway':'bd_subtype1_giveaway',\
            'bundle_subtype4_threshold1':'bd_subtype4_threshold1',\
            'bundle_subtype4_threshold2':'bd_subtype4_threshold2',\
            'bundle_subtype4_threshold3':'bd_subtype4_threshold3',\
            'bundle_subtype4_giveaway1':'bd_subtype4_giveaway1',\
            'bundle_subtype4_giveaway2':'bd_subtype4_giveaway2',\
            'bundle_subtype4_giveaway3':'bd_subtype4_giveaway3',\
            'bundle_subtype2_threshold':'bd_subtype2_threshold',\
            'bundle_subtype2_giveaway':'bd_subtype2_giveaway',\
            'bundle_subtype2_maximumgiveaway':'bd_subtype2_maximumgiveaway',\
            'bundle_subtype15_thresholdnumber1':'bd_subtype15_thresholdnumber1',\
            'bundle_subtype15_thresholdnumber2':'bd_subtype15_thresholdnumber2',\
            'bundle_subtype15_thresholdnumber3':'bd_subtype15_thresholdnumber3',\
            'bundle_subtype15_giveawayrate1':'bd_subtype15_giveawayrate1',\
            'bundle_subtype15_giveawayrate2':'bd_subtype15_giveawayrate2',\
            'bundle_subtype15_giveawayrate3':'bd_subtype15_giveawayrate3',\
            'bundle_subtype6_thresholdnumber':'bd_subtype6_thresholdnumber',\
            'bundle_subtype6_freenumber':'bd_subtype6_freenumber',\
            'bundle_subtype1_percent'  :'bd_subtype1_percent',\
            'bundle_subtype4_percent'  :'bd_subtype4_percent',\
            'bundle_subtype2_percent'  :'bd_subtype2_percent',\
            'bundle_subtype15_percent' :'bd_subtype15_percent',\
            'bundle_subtype6_percent'  :'bd_subtype6_percent',\
            'freegift_productkey'                 :'fg_productkey',\
            'freegift_maxscrapedprice'            :'fg_maxscrapedprice',\
            'freegift_minscrapedprice'            :'fg_minscrapedprice',\
            'freegift_avgscrapedprice'            :'fg_avgscrapedprice',\
            'freegift_maxjdprice'                 :'fg_maxjdprice',\
            'freegift_minjdprice'                 :'fg_minjdprice',\
            'freegift_avgjdprice'                 :'fg_avgjdprice',\
            'freegift_freegiftvalue'              :'fg_fgvalue',\
            'freegift_freegiftvalueatjdprice'     :'fg_fgvalueatjdprice',\
            'freegift_freegiftvalueatscrapedprice':'fg_fgvalueatscrapedprice',\
            'freegift_percent'                    :'fg_percent'},inplace=True)

    raw_rdc = process_rdc(period_promo_raw, area_rdc_map)

    raw_clean = clean_data(raw_rdc)
    raw_clean_add = add_cols(raw_clean)


    raw_dd = get_dd_price(raw_clean_add)
    raw_dd_price = agg_dd_price(raw_dd)


    raw_bd = get_bundle_feat(raw_dd_price)
    out_df = agg_bd_price(raw_bd)

    for col in ['areatypearray', 'productdesc', 'mainproductkey', 'bd_subtype1_giveaway', 'bd_subtype4_threshold1',
       'bd_subtype4_giveaway1', 'bd_subtype4_threshold2', 'bd_subtype4_giveaway2', 'bd_subtype4_threshold3',
       'bd_subtype4_giveaway3', 'bd_subtype2_threshold', 'bd_subtype2_giveaway', 'bd_subtype2_maximumgiveaway',
       'bd_subtype15_thresholdnumber1', 'bd_subtype15_giveawayrate1', 'bd_subtype15_thresholdnumber2', 'bd_subtype15_giveawayrate2',
       'bd_subtype15_thresholdnumber3', 'bd_subtype15_giveawayrate3', 'bd_subtype6_thresholdnumber', 'bd_subtype6_freenumber', 'fg_productkey',
       'dd_percent', 'bd_subtype1_percent', 'bd_subtype4_percent', 'bd_subtype2_percent', 'bd_subtype15_percent',
       'bd_subtype6_percent', 'suit_percent', 'fg_percent', 'allpercentdiscount']:
        if col in out_df.columns:
            del out_df[col]

    print("| - Reformat date")

    out_df['start_ts'] = pd.to_datetime(out_df['startdatetime'], format = "%Y-%m-%d %H:%M:%S")
    out_df['end_ts'] = pd.to_datetime(out_df['enddatetime'], format = "%Y-%m-%d %H:%M:%S")
    out_df['Date'] = out_df['start_ts'].dt.date

    ''' end main_process & start main_expand'''

    df = out_df

    df = process_feature(df)
    df_grp, df_promo, df_max = agg_feature(df)
    def re_index(df):
        """
        Auxiliary function to re-index period level table into a day level table

        Args:
        ----------
            df: Pandas DataFrame
                Pandas dataframe containing all features at period level

        Returns:
        ----------
            df: Pandas DataFrame
                Pandas dataframe containing all features with new day level index
        """

        #print("| | - reindex period to day")
        start_dt = df['start_ts'].dt.date.min()
        end_dt = df['end_ts'].dt.date.max()
        dt_range = pd.date_range(start_dt, end_dt)
        df.index = df['Date']
        #print(df.index)
        df = df.reindex(dt_range, method = 'ffill')
        df = df.reset_index()
        return df

    df_grp_new = df_grp.groupby(['productkey', 'start_ts', 'end_ts', 'Date'], as_index = False).apply(re_index)
    df_out = df_grp_new.merge(df_promo, on = ['productkey', 'start_ts', 'end_ts'], how = 'left').\
                        merge(df_max, on = ['productkey', 'start_ts', 'end_ts'], how = 'left')

    ''' main_agg_day
        Aggregate features dataset from period * day level to day level and link
        three auxiliary functions together
        1. prep_data: calculate additional featurse required for the aggregation
        2. calc_weighted_price: calculate the weighted by duration features that
        will be eventually aggregated at day level
        3. agg_feature_day: performs the aggregation
    '''
    df = prep_data(df_out)
    df = calc_weighted_price(df)
    df_out = agg_feature_day(df)

    print 'f01 shape'
    print df_out.shape
    return df_out

if __name__ == "__main__":

    # p1 header : ['productkey','promotionkey','startdatetime','enddatetime','jdprice','syntheticgrossprice','promotiondesc','promotiontype','promotionsubtype','areatypearray','tokenflag','directdiscount_discount','directdiscount_availabilitynumber','bundle_subtype1_threshold','bundle_subtype1_giveaway','bundle_subtype4_threshold1','bundle_subtype4_giveaway1','bundle_subtype4_threshold2','bundle_subtype4_giveaway2','bundle_subtype4_threshold3','bundle_subtype4_giveaway3','bundle_subtype2_threshold','bundle_subtype2_giveaway','bundle_subtype2_maximumgiveaway','bundle_subtype15_thresholdnumber1','bundle_subtype15_giveawayrate1','bundle_subtype15_thresholdnumber2','bundle_subtype15_giveawayrate2','bundle_subtype15_thresholdnumber3','bundle_subtype15_giveawayrate3','bundle_subtype6_thresholdnumber','bundle_subtype6_freenumber','suit_maxvaluepool','suit_minvaluepool','suit_avgvaluepool','suit_discount','directdiscount_saleprice','bundle_subtype1_percent','bundle_subtype4_percent','bundle_subtype2_percent','bundle_subtype15_percent','bundle_subtype6_percent','suit_percent','allpercentdiscount','productdesc','mainproductkey','hierarchylevel3key','createdate','statuscode','dt']
    # p1_used_header=['productkey','promotionkey','startdatetime','enddatetime','jdprice','syntheticgrossprice','promotiondesc','promotiontype','promotionsubtype','areatypearray','tokenflag','directdiscount_discount','directdiscount_availabilitynumber','bundle_subtype1_threshold','bundle_subtype1_giveaway','bundle_subtype4_threshold1','bundle_subtype4_giveaway1','bundle_subtype4_threshold2','bundle_subtype4_giveaway2','bundle_subtype4_threshold3','bundle_subtype4_giveaway3','bundle_subtype2_threshold','bundle_subtype2_giveaway','bundle_subtype2_maximumgiveaway','bundle_subtype15_thresholdnumber1','bundle_subtype15_giveawayrate1','bundle_subtype15_thresholdnumber2','bundle_subtype15_giveawayrate2','bundle_subtype15_thresholdnumber3','bundle_subtype15_giveawayrate3','bundle_subtype6_thresholdnumber','bundle_subtype6_freenumber','suit_maxvaluepool','suit_minvaluepool','suit_avgvaluepool','suit_discount','directdiscount_saleprice','bundle_subtype1_percent','bundle_subtype4_percent','bundle_subtype2_percent','bundle_subtype15_percent','bundle_subtype6_percent','suit_percent','allpercentdiscount','mainproductkey','hierarchylevel3key','createdate','statuscode','dt']

    p1_used_header=['productkey', 'promotionkey', 'startdatetime', 'enddatetime', 'jdprice', 'syntheticgrossprice', 'promotiondesc', 'promotiondesc_flag', 'promotiontype', 'promotionsubtype',
                    'areatypearray', 'tokenflag', 'directdiscount_discount', 'directdiscount_availabilitynumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway',
                    'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3',
                    'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1',
                    'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3',
                    'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool',
                    'suit_discount', 'directdiscount_saleprice', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent',
                    'bundle_subtype6_percent', 'suit_percent', 'allpercentdiscount', 'mainproductkey', 'hierarchylevel3key', 'createdate', 'statuscode', 'dt']
    p1_path ='tmp/11922_2018/11922_train_p1'
    period_promo_raw = pd.read_csv(p1_path,sep='\t',header=None)
    period_promo_raw.columns=p1_used_header

    # period_promo_raw = pd.read_csv('tmp_p1_1211.csv',sep=',',header=None)
    # period_promo_raw.columns=['productkey','promotionkey','startdatetime','enddatetime','jdprice','syntheticgrossprice','promotiondesc','promotiontype','promotionsubtype','areatypearray','tokenflag','directdiscount_discount','directdiscount_availabilitynumber','bundle_subtype1_threshold','bundle_subtype1_giveaway','bundle_subtype4_threshold1','bundle_subtype4_giveaway1','bundle_subtype4_threshold2','bundle_subtype4_giveaway2','bundle_subtype4_threshold3','bundle_subtype4_giveaway3','bundle_subtype2_threshold','bundle_subtype2_giveaway','bundle_subtype2_maximumgiveaway','bundle_subtype15_thresholdnumber1','bundle_subtype15_giveawayrate1','bundle_subtype15_thresholdnumber2','bundle_subtype15_giveawayrate2','bundle_subtype15_thresholdnumber3','bundle_subtype15_giveawayrate3','bundle_subtype6_thresholdnumber','bundle_subtype6_freenumber','suit_maxvaluepool','suit_minvaluepool','suit_avgvaluepool','suit_discount','directdiscount_saleprice','bundle_subtype1_percent','bundle_subtype4_percent','bundle_subtype2_percent','bundle_subtype15_percent','bundle_subtype6_percent','suit_percent','allpercentdiscount','mainproductkey','hierarchylevel3key','createdate','statuscode','dt']
    # exclude_col = ['promotionstartdatetime','promotionenddatetime','jdpricestartdatetime','jdpriceenddatetime','spricestartdatetime','spriceenddatetime','promo_inperiod_flag','createdatetime','levelmember','tokentype','hierarchylevel1key','hierarchylevel2key','directdiscount_availabilitynumberflag','directdiscount_productkey','directdiscount_maxjdprice','directdiscount_minjdprice','directdiscount_avgjdprice','directdiscount_maxscrapedprice','directdiscount_minscrapedprice','directdiscount_avgscrapedprice','directdiscount_jdpricedate','directdiscount_price','directdiscount_limitationflag','directdiscount_limitationcode','directdiscount_minimumnumberflag','directdiscount_minimumnumber','directdiscount_maximumnumberflag','directdiscount_maximumnumber','directdiscount_percent','freegift_productkey','freegift_maxscrapedprice','freegift_minscrapedprice','freegift_avgscrapedprice','freegift_maxjdprice','freegift_minjdprice','freegift_avgjdprice','freegift_freegiftvalue','freegift_freegiftvalueatjdprice','freegift_freegiftvalueatscrapedprice','freegift_percent','productdesc']
    # period_promo_raw_clean = period_promo_raw.drop(exclude_col, axis=1)
    # period_promo_raw_clean = period_promo_raw[p1_used_header]
    period_promo_raw_clean = period_promo_raw
    train_pred_gate='predict'
    scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
    scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
    area_rdc_map = pd.read_csv('tmp/ow_deploy_single/area_rdc_mapping.csv')
    f01_output = generate_f01_promo(area_rdc_map, period_promo_raw_clean,scenario, train_pred_gate)
    f01_output.to_csv('tmp/11922_2018/p1_out_2018_predict.csv',index=False)
if False:

    p1_path = 'tmp/fushi/12015/12015_p1.da' #'tmp/ow_1315/9708_p1out.da'

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
    train_pred_gate='predict'
    scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
    scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
    area_rdc_map = pd.read_csv('tmp/ow_deploy_single/area_rdc_mapping.csv')
    f01_output = generate_f01_promo(area_rdc_map, period_promo_raw_clean,scenario, train_pred_gate)
    f01_output.to_csv('tmp/fushi/12015/p1_out.csv',index=False)
