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

def predict(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,model,seasonality_df=None,process_f01_flag=True,ForecastStartDate=None,mode='product'):
    # process_f01_flag 需要不需要f01去预处理，if yes : period_promo_raw
    #                                         if no  : period_promo_raw=f01输出的结果
    ts_df.columns = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])

    #ts_df = ts_df[ts_df.ProductKey==188059]
    if ts_df is None:
        return None

    if (len(ts_df)<300) | len(set(ts_df[scenario['Sku_col']]))<30:
        return None

    # 一个品类，预测时候最少的天数
    if (ts_df['Date'].max() - ts_df['Date'].min()).days<scenario['min_onstock_day']:
        return None

    '''Configuration '''
    if ForecastStartDate is None:
        ForecastStartDate = pd.to_datetime(scenario['forecastStartDate'])

    ForecastStartDate = pd.to_datetime(ForecastStartDate)

    PredictEndDate = ForecastStartDate + datetime.timedelta(days=(scenario['lookforwardPeriodDays']-1))
    DataStartDate = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDays_predict'])
    #DataStartDate = pd.to_datetime('2016-10-05')##sjd-0328 for cart_test
    ts_df = ts_df[(ts_df['Date']<ForecastStartDate)&(ts_df['Date']>DataStartDate)]
    ts_df.replace('null', np.nan, inplace=True)
    ts_df.replace(-999, np.nan, inplace=True)
    ts_df.replace('None', np.nan, inplace=True)
    ts_to_float_col = ['RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
    if 'object' in ts_df[ts_to_float_col].dtypes.values:
        object2Float(ts_df,ts_to_float_col)

    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    object2Int(holidays_df,['Holiday','Ind_1111_pre','Ind_1111','Ind_1111_post','Ind_618_pre','Ind_618','Ind_618_post','Ind_1212'])

    for col in area_rdc_map.columns:
        area_rdc_map[col] = area_rdc_map[col].astype('int')
    ''' Pre-Process period Calender '''
#    pro_columns = ['ProductKey','Date','HierarchyLevel1Key','HierarchyLevel2Key','HierarchyLevel3Key','PromotionCount','bundlecount','directdiscountcount','freegiftcount','suitcount','MaxDiscount','MinDiscount','AvgDiscount','MaxSyntheticDiscountA','MinSyntheticDiscountA','AvgSyntheticDiscountA','maxhourlyaveragesyntheticdiscounta','minhourlyaveragesyntheticdiscounta','avghourlyaveragesyntheticdiscounta','MaxBundleDiscount','MinBundleDiscount','AvgBundleDiscount','MaxDirectDiscount','MinDirectDiscount','AvgDirectDiscount','MaxFreegiftDiscount','MinFreegiftDiscount','AvgFreegiftDiscount','maxfreegiftdiscount','minfreegiftdiscount','avgfreegiftdiscount','maxjdprice','maxscrapedprice','syntheticpromoflaga','SyntheticGrossPrice','maxdirectdiscountsyntheticdiscounta','maxbundlesyntheticdiscounta','inddirectdiscountday','promotionkey','promotiontype','promotionsubtype','syntheticgrossprice_vb','jdprice','syntheticdiscounta_vb','syntheticpromoflaga_vb','durationinhours','daynumberinpromotion','bundleflag','directdiscountflag','freegiftflag','suitflag','numberproducts','numberhierarchylevel1','numberhierarchylevel2','numberhierarchylevel3','strongmark','stockprice','dt']
#    promoCalendarDf.columns = pro_columns
    promoCalendarDf['Date'] = pd.to_datetime(promoCalendarDf['Date'])
    promoCalendarDf['dt'] = pd.to_datetime(promoCalendarDf['dt'])
    promoCalendarDf.replace('null', np.nan, inplace=True)
    promoCalendarDf.replace(-999, np.nan, inplace=True)
    promoCalendarDf.replace('None', np.nan, inplace=True)
    promoCalendarDf.drop_duplicates(inplace=True)


    # Convert Object -> float
    if 'object' in promoCalendarDf.dtypes.values:
        obj_cols = get_column_by_type(promoCalendarDf,'object')
        object2Float(promoCalendarDf,obj_cols)

    promoCalendarDf = promoCalendarDf[(promoCalendarDf['Date']<PredictEndDate)&(promoCalendarDf['Date']>DataStartDate)]


    if process_f01_flag: # Pre-Process period promo raw 

        period_promo_raw.replace('null', np.nan, inplace=True)
        period_promo_raw.replace(-999, np.nan, inplace=True)
	period_promo_raw.replace(' null', np.nan, inplace=True)
	period_promo_raw.replace('None', np.nan, inplace=True)
        #to_float_col = ['productkey', 'promo_inperiod_flag', 'mainproductkey', 'hierarchylevel1key', 'hierarchylevel2key', 'hierarchylevel3key', 'statuscode', 'promotionkey', 'jdprice', 'syntheticgrossprice', 'promotiontype', 'promotionsubtype', 'levelmember', 'directdiscount_productkey', 'directdiscount_maxjdprice', 'directdiscount_minjdprice', 'directdiscount_avgjdprice', 'directdiscount_maxscrapedprice', 'directdiscount_minscrapedprice', 'directdiscount_avgscrapedprice', 'directdiscount_discount', 'directdiscount_price', 'directdiscount_limitationflag', 'directdiscount_limitationcode', 'directdiscount_availabilitynumberflag', 'directdiscount_availabilitynumber', 'directdiscount_minimumnumberflag', 'directdiscount_minimumnumber', 'directdiscount_maximumnumberflag', 'directdiscount_maximumnumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway', 'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3', 'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1', 'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3', 'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool', 'suit_discount', 'freegift_productkey', 'freegift_maxscrapedprice', 'freegift_minscrapedprice', 'freegift_avgscrapedprice', 'freegift_maxjdprice', 'freegift_minjdprice', 'freegift_avgjdprice', 'freegift_freegiftvalue', 'freegift_freegiftvalueatjdprice', 'freegift_freegiftvalueatscrapedprice', 'directdiscount_saleprice', 'directdiscount_percent', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent', 'bundle_subtype6_percent', 'suit_percent', 'freegift_percent', 'allpercentdiscount']
        to_float_col = ['allpercentdiscount','bundle_subtype15_giveawayrate1','bundle_subtype15_giveawayrate2','bundle_subtype15_giveawayrate3','bundle_subtype15_percent','bundle_subtype15_thresholdnumber1','bundle_subtype15_thresholdnumber2','bundle_subtype15_thresholdnumber3','bundle_subtype1_giveaway','bundle_subtype1_percent','bundle_subtype1_threshold','bundle_subtype2_giveaway','bundle_subtype2_maximumgiveaway','bundle_subtype2_percent','bundle_subtype2_threshold','bundle_subtype4_giveaway1','bundle_subtype4_giveaway2','bundle_subtype4_giveaway3','bundle_subtype4_percent','bundle_subtype4_threshold1','bundle_subtype4_threshold2','bundle_subtype4_threshold3','bundle_subtype6_freenumber','bundle_subtype6_percent','bundle_subtype6_thresholdnumber','directdiscount_availabilitynumber','directdiscount_discount','directdiscount_saleprice','hierarchylevel3key','jdprice','mainproductkey','productkey','promotionkey','promotionsubtype','promotiontype','statuscode','suit_avgvaluepool','suit_discount','suit_maxvaluepool','suit_minvaluepool','suit_percent','syntheticgrossprice']
	period_promo_raw['enddatetime'] = pd.to_datetime(period_promo_raw['enddatetime'])
        period_promo_raw['startdatetime'] = pd.to_datetime(period_promo_raw['startdatetime'])

        if 'object' in period_promo_raw[to_float_col].dtypes.values:
            object2Float(period_promo_raw,to_float_col)

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
        promo_period = df_out
    else:
        # promo_period=pd.read_csv('f01.csv')
        # promo_period['Date'] = pd.to_datetime(promo_period['Date'])
        period_promo_raw.drop_duplicates(inplace=True)
        promo_period = period_promo_raw


    ''' f02 predict'''

    raw_df= ts_df[ts_df['Date']<ForecastStartDate]


    ''' initialization'''
    if 'ind' in raw_df.columns.values:
        del raw_df['ind']
    groupKeys = scenario['selectedColumns']['levelsCoarse'] # ['HierarchyLevel3Key','ProductKey','RDCKey']

    ''' 获取新品的上柜时间 'dt_onstaock_min'. nan if 上柜时间小于ts start_time '''
    min_dt_record = raw_df.groupby(groupKeys)['Date'].min().reset_index(name='dt_min')
    min_dt_onstock_record = raw_df[raw_df.counterState==1].groupby(groupKeys)['Date'].min().reset_index(name='dt_onstock_min')
    merged_min_dt = pd.merge(min_dt_record,min_dt_onstock_record,on=groupKeys)
    valid_merged_min_dt = merged_min_dt[merged_min_dt.dt_min ==merged_min_dt.dt_onstock_min]

    raw_df = pd.merge(raw_df, valid_merged_min_dt[groupKeys+['dt_onstock_min']],how='left',on=groupKeys)
    raw_df.reset_index(drop=True,inplace=True)
    ''' fillZeroRows '''
    raw_df= filter_non_price_fill_it(raw_df, scenario, price_col_ts='priceAfterDiscount')

    '''
        capOutlierInTrain
    '''

    # calculate quantile
    q = scenario['quantile'] # 0.95
    # q = 0.9

    unitVolumeQuantile = raw_df.groupby(groupKeys)[scenario['Target_col']].quantile(q, interpolation='lower')
    quantileDf = pd.DataFrame(unitVolumeQuantile).reset_index()
    quantileDf.rename(columns = {scenario['Target_col']: 'UnitCapVolume'}, inplace=True)

    # join quantile
    raw_df= pd.merge(raw_df, quantileDf, how='left', on=groupKeys)
    raw_df['OrderNonOutlierVolume'] = raw_df[['salesForecast', 'UnitCapVolume']].min(axis=1)

    # drop cap volume
    raw_df.drop(['UnitCapVolume'], axis=1, inplace = True)
    if True:
        f = lambda x: x.shift(1).fillna(method='bfill')
        raw_df[['counterState1']] = raw_df[['counterState','ProductKey','RDCKey']].groupby(['ProductKey','RDCKey']).transform(f)
        raw_df[['vendibility1']] = raw_df[['vendibility','ProductKey','RDCKey']].groupby(['ProductKey','RDCKey']).transform(f)

    df_past = raw_df

    '''  SET predict start date '''
    def return_nearest_date_index():
        index = [i for i in range(1,10) if (ForecastStartDate- pd.DateOffset(days=i)).strftime('%Y-%m-%d') not in scenario['Remove_by_stat_date']]
        return index[0]

    fut_train_end = ForecastStartDate- pd.DateOffset(days=1)
    # make sure train end date not in remove date
    if fut_train_end.strftime('%Y-%m-%d') in scenario['Remove_by_stat_date']:
        i = return_nearest_date_index()

        fut_train_end = ForecastStartDate- pd.DateOffset(days=i)

    ''' concat Future Df by RDC'''
    dfs=[]
    for rdc in list(set(df_past.RDCKey)):

        valid_skus_fut=list(set(df_past[(df_past['Date']==fut_train_end)&(df_past.RDCKey==rdc)]['ProductKey'].values))

        # generate_default_df
        def generate_strutured_null_df(data_start_date, data_end_date, valid_skus, sku_name='ProductKey', date_name='Date'):
            rng = pd.date_range(start=data_start_date, end=data_end_date, freq='D')
            valid_date_span_length = len(rng)
            sku_col = []
            rng_col = []
            for sku_id in valid_skus:
                sku_col.extend(np.repeat(sku_id, valid_date_span_length))
                rng_col.extend(rng)

            dic = {sku_name : sku_col, date_name: rng_col}

            return pd.DataFrame(dic)

        fut_null_df = generate_strutured_null_df(ForecastStartDate, PredictEndDate, valid_skus_fut)  # Filling 0601-0604 cols [stat_date, sku]
        fill_one_cols = {'vendibility1': 1}

        basic_col=['ProductKey','Date','RDCKey','HierarchyLevel1Key','HierarchyLevel2Key','HierarchyLevel3Key'\
                ,'dt_onstock_min','brand_code','salesForecast','reserveState','OrderNonOutlierVolume','vendibility1']
        to_fillna_col=['dt_onstock_min','RDCKey','brand_code','reserveState','HierarchyLevel1Key','HierarchyLevel2Key'\
                ,'HierarchyLevel3Key','vendibility1']

        default_kv = generate_default_values_by_dic(fill_nan_list=to_fillna_col, fill_default_kv=fill_one_cols)

        fut_null_df = fill_col_with_default(fut_null_df, default_kv)

        df_future = df_past[df_past.RDCKey==rdc][basic_col].append(fut_null_df)
        df_future.sort_values(['ProductKey', 'Date'], inplace=True)
        df_future.reset_index(drop=True,inplace=True)
        col = list(df_future.columns.values)
        col.remove('ProductKey')

        df_future[col] = df_future.groupby(['ProductKey']).transform(lambda x: x.fillna(method='ffill'))
        dfs.append(df_future)
    df_with_future=pd.concat(dfs)
    df_with_future.reset_index(drop=True,inplace=True)

    featuresDf = df_with_future
    def createCartFeatures(df):
        df2 = df[['Date','RDCKey','ProductKey','salesForecast']]
        df3 = df2.groupby(['Date','ProductKey'])['salesForecast'].sum().reset_index()
        df3.rename(columns={"salesForecast":"all_sales"},inplace=True)
        df33 = pd.merge(df,df3,how='left',on=['Date','ProductKey'])
        df33['prop_rdc'] = df33.salesForecast / df33.all_sales
        cart = pd.read_csv('./tmp/11922_cart_sjd_20180328151150.csv')
        cart.addtocart_date = pd.to_datetime(cart.addtocart_date)
        del cart['item_third_cate_cd']
        df44 = pd.merge(df33,cart,how='left',left_on=['Date','ProductKey'],right_on=['addtocart_date','item_sku_id'])
        df44['cart_cnt'] = df44.addtocart_cnt + df44.salesForecast
        f = lambda x: x.shift(1).fillna(method='bfill')
        df44[['cart_cnt1']] = df44[['cart_cnt','ProductKey','RDCKey']].groupby(['ProductKey','RDCKey']).transform(f)
        return df44

    if scenario['use_cart_feature']:
        featuresDf = createCartFeatures(featuresDf)


    '''
        seasonality 使用了seasonal_decompose,并且预测了trend和seasonality,注意，至少2year数据才可以
    '''
    if seasonality_df is None:
        if (ts_df.Date.max() - ts_df.Date.min()).days > 730:
            print 'running Seasonality'
            featuresDf = createSeasonalityDecomposeFeatures(featuresDf, ['HierarchyLevel3Key', 'ProductKey'], 'sku', scenario)
            featuresDf = createSeasonalityDecomposeFeatures(featuresDf, ['HierarchyLevel3Key'], 'level3', scenario)

            '''
                createLevel3Features
            '''
            featuresDf = createLevel3Features(featuresDf,scenario)
        else:
            print 'faking Seasonality Due to less 2 year Data'
            featuresDf['skuDecomposedTrend'],featuresDf['skuDecomposedSeasonal'],featuresDf['level3DecomposedTrend'],featuresDf['level3DecomposedSeasonal'],featuresDf['Curve'] = [np.nan,np.nan,np.nan,np.nan,np.nan]
    else:
        if 'OrderNonOutlierVolume' in seasonality_df.columns:
            seasonality_df.drop(['OrderNonOutlierVolume'], axis=1, inplace = True)
        featuresDf = pd.merge(featuresDf,seasonality_df,on=['HierarchyLevel3Key','ProductKey','Date','RDCKey'],how='left')

    '''加入Holiday '''
    featuresDf = pd.merge(featuresDf, holidays_df, on='Date', how='left')

    '''
        createDateFeatures
    '''
    featuresDf = createDateFeatures(featuresDf)

    '''
        rolling part, 28min, too slow!
    '''

    # featuresDf = pd.merge(featuresDf, promoCalendarDf[promoCalendarDf.columns.difference(['HierarchyLevel1Key','HierarchyLevel2Key', 'HierarchyLevel3Key'])],   left_on = ['ProductKey', 'Date'], right_on = ['ProductKey', 'Date'], how='left')
    featuresDf = pd.merge(featuresDf, promoCalendarDf[promoCalendarDf.columns.difference(['HierarchyLevel1Key','HierarchyLevel2Key', 'HierarchyLevel3Key'])].drop_duplicates(),   on = ['ProductKey', 'Date'], how='left')
    if scenario['STOCK_PRICE_FLAG']:
        featuresDf = calculateStockFeatures(featuresDf)

    df_fut = featuresDf[featuresDf.Date>=pd.to_datetime(ForecastStartDate)]
    # df_fut = featuresDf[featuresDf.Date>=pd.to_datetime('2017-10-01')]
    df_past = featuresDf[featuresDf.Date<pd.to_datetime(ForecastStartDate)]
    ####notice: use selected promotion cols' mean values to fill the same cols in df_future

    df_fut = calculateNationalRolling_predict(df_fut,df_past, scenario, scenario['selectedColumns']['target'])
    df_fut = calculateRolling_predict(df_fut,df_past, scenario, scenario['selectedColumns']['target'])
    df_fut = calculateSimilarRolling_predict(df_fut,df_past, scenario, scenario['Target_col'])
    df_fut = calculateLagging_predict(df_fut,df_past, scenario, scenario['Target_col'])

    # promo_period.read_csv('f01.csv',parse_dates=['Date'])
    df_fut = pd.merge(df_fut,  promo_period,on = ['ProductKey', 'Date'], how = 'left')

    ### test sjd_180323 for mean-replacing###

    exclu_promo_features = ['strongmark','flashsale_ind','dd_ind','bundle_ind','bundle_buy199get100_ind','suit_ind','freegift_ind']
    update_cols = list(set(scenario['promo_feature_cols'])- set(exclu_promo_features))
    need_cols = ['Date','RDCKey','ProductKey','HierarchyLevel3Key'] + update_cols
    
    #df_past = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/0425_result/870_train_feature.csv')
    #df1 = df_past[need_cols]
    #groupkeys = ['RDCKey','ProductKey','HierarchyLevel3Key']
    #promo_feature_cols =  scenario['promo_feature_cols']
    #df11 = df1.groupby(groupkeys)[update_cols].mean().reset_index()
    #df_fut = pd.merge(df_fut,df11[groupkeys + update_cols], how='left',on=groupkeys)
    #rename_update_cols = [col+'_y' for col in update_cols]
    #for col in update_cols:
    #    df_fut.rename(columns={col+'_y': col},inplace=True)

    ### test: promo features set to 0  @sjd 2018-05-07
    #for col in update_cols:
    #    df_fut[col] = 0
    
    grouped = df_fut.groupby('RDCKey')
    result_list = []
    for rdc, pred_df in grouped:
        if rdc in model.keys():
            this_model = model[rdc]
        else:
            continue
        ''' predict model '''
        xColumns = scenario['selectedColumns']['features']
	"""
        ### for A/B test of with/without promotion features.
        """
        promo_cols = scenario['promo_feature_cols']
        if not scenario['use_promo_features']:
            for col in promo_cols:
                xColumns.remove(col)
            scenario['use_promo_features'] = True

        if 'RDCKey' in xColumns:# 删除季节性,RDCKEY
            xColumns.remove('skuDecomposedTrend')
            xColumns.remove('skuDecomposedSeasonal')
            xColumns.remove('level3DecomposedTrend')
            xColumns.remove('level3DecomposedSeasonal')
            xColumns.remove('Curve')
            xColumns.remove('RDCKey')
            #for col in update_cols: ###sjd_update
            #    xColumns.remove(col)

        if scenario['use_cart_feature']:
            X_future = pred_df[xColumns + ['cart_cnt1']]
        else:
            X_future = pred_df[xColumns]

        future_xtest = xgb.DMatrix(X_future.values, missing=np.NaN )
        ypred = this_model.predict(future_xtest)
        pred_df['ypred'] =ypred
        pred_df['RDCKey'] = rdc

        ''' Tuning result '''
        lanjie = pred_df[(pred_df.ypred<0)]
        if len(lanjie)>0:
            pred_df.ix[lanjie.index,'ypred'] = 0
        result_list.append(pred_df)
    final_result = pd.concat(result_list)

    if mode=='dev':
        return final_result,df_fut
    else:
        return final_result

if __name__ == "__main__":
    # 服饰品类：9708,9710,9732,9733,9734

    scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
    scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
    '''online analysis'''

    holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')

    area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')

    def func_concat_pdf(li_subs):
        import base64
        import zlib as zl
        import cPickle as cp
        import pandas as pd

        li_pdf = []
        for str_sub in li_subs:
            pdf_sub = cp.loads(zl.decompress(base64.b64decode(str_sub)))
            li_pdf.append(pdf_sub)

        return pd.concat(li_pdf)

    p1_path = '/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/9708_p1out.da'
    p2_path = '/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/9708_p2out.da'
    ts_path = '/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/9708_tsout.da'
    sea_path = "/home/ubuntu/hehongjie/data/11922_train_season"
    def get_compress_file(path):
        f = open(path, "r")
        li_lines = []
        for line in f.readlines():
            li_data = line.split(",")
            li_lines.append(li_data[4])
        pdf = func_concat_pdf(li_lines)
        return pdf

    period_promo_raw = get_compress_file(p1_path)
    promoCalendarDf = get_compress_file(p2_path)
    ts_df = get_compress_file(ts_path)
    seasonality_df= get_compress_file(sea_path).head(1) # 服饰不用季节性，先随便生成一个

    model_path = '/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/model_ow_20180201_rdc_at180320_9708.pkl'
    with open(model_path,'r') as input:
        model = pickle.load(input)


    pred_date = pd.to_datetime('2018-02-01')
    scenario['lookforwardPeriodDays'] = 91
    result=predict(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,model,seasonality_df,process_f01_flag=False,ForecastStartDate=pred_date)
    result.to_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/result_20180201_rdc_at180320_9708.csv',index=False)
    # save_object(model,'tmp/11922_1219/sh_tmp6.pkl')

    ''' end '''
    if False:
        import cPickle as cp
        import zlib
        import base64
        def unzip_base64(str_src, bool_use_wbits=False):
            li_chunks = str_src.split("$")
            li_unzip_chunks = []
            for i in range(0, len(li_chunks)):
                if bool_use_wbits:
                    li_unzip_chunks.append(zlib.decompress(base64.b64decode(li_chunks[i]), zlib.MAX_WBITS | 16))
                else:
                    li_unzip_chunks.append(zlib.decompress(base64.b64decode(li_chunks[i])))
            return "".join(li_unzip_chunks)


        pro_canlender_path = '/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/11922_predict_p2'

        promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)

        promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
                   'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
                   'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
                   'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
                   'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

        p1_out_path ='/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/p1_out_2018.csv'
        period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])


        file_path = '/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/11922_predict_ts'
        ts_df = pd.read_csv(file_path,header=None,sep='\t')

        scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
        scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
        '''online analysis'''

        holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')

        area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')

        seasonality_df = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/ts_11922_sea_from2018.csv',parse_dates=['Date'])

        #model_path = 'tmp/11922_2018/model_ow_v_oct_20171010_rdc_at180321_11922.pkl'
        model_path = 'tmp/11922_2018/model_no_promo_ow_20171010_rdc_at180321_11922.pkl'
        with open(model_path,'r') as input:
            model = pickle.load(input)

        result,df_fut=predict(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,model,seasonality_df,process_f01_flag=False,mode='dev')
        #df_fut.to_csv('tmp/11922_2018/at180321_11922_df_fut_20171010.csv',index=False)
        result.to_csv('tmp/11922_2018/result_no_promo_11922_171010_180321.csv',index=False)

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
