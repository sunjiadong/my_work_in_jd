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

def train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,seasonality_df=None,process_f01_flag=True,ForecastStartDate=None,mode='product'):

    if ForecastStartDate is None:
        ForecastStartDate = pd.to_datetime(scenario['forecastStartDate'])

    ForecastStartDate = pd.to_datetime(ForecastStartDate)

    DataStartDate = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDays'])
    #DataStartDate = pd.to_datetime('2016-10-05')
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    object2Int(holidays_df,['Holiday','Ind_1111_pre','Ind_1111','Ind_1111_post','Ind_618_pre','Ind_618','Ind_618_post','Ind_1212'])

    for col in area_rdc_map.columns:
        area_rdc_map[col] = area_rdc_map[col].astype('int')
    ''' Pre-Process period Calender '''
    promoCalendarDf['Date'] = pd.to_datetime(promoCalendarDf['Date'])
    promoCalendarDf['dt'] = pd.to_datetime(promoCalendarDf['dt'])
    promoCalendarDf.replace('null', np.nan, inplace=True)
    promoCalendarDf.replace('None', np.nan, inplace=True)
    promoCalendarDf.replace(-999, np.nan, inplace=True)
    promoCalendarDf.drop_duplicates(inplace=True)

    # Convert Object -> float
    if 'object' in promoCalendarDf.dtypes.values:
        obj_cols = get_column_by_type(promoCalendarDf,'object')
        object2Float(promoCalendarDf,obj_cols)

    promoCalendarDf = promoCalendarDf[(promoCalendarDf['Date']<ForecastStartDate)&(promoCalendarDf['Date']>DataStartDate)]

    ts_df.columns = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df_pred = ts_df[(ts_df['Date']<ForecastStartDate)&(ts_df['Date']>DataStartDate)]
    ts_df_pred.replace('null', np.nan, inplace=True)
    ts_df_pred.replace(-999, np.nan, inplace=True)
    ts_to_float_col = ['RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
    if 'object' in ts_df_pred[ts_to_float_col].dtypes.values:
        object2Float(ts_df_pred,ts_to_float_col)

    if process_f01_flag: # Pre-Process period promo raw 

        period_promo_raw.replace('null', np.nan, inplace=True)
        period_promo_raw.replace(-999, np.nan, inplace=True)
        period_promo_raw.replace(' null', np.nan, inplace=True)
	period_promo_raw.replace('None', np.nan, inplace=True)
	# to_float_col = ['productkey', 'promo_inperiod_flag', 'mainproductkey', 'hierarchylevel1key', 'hierarchylevel2key', 'hierarchylevel3key', 'statuscode', 'promotionkey', 'jdprice', 'syntheticgrossprice', 'promotiontype', 'promotionsubtype', 'levelmember', 'directdiscount_productkey', 'directdiscount_maxjdprice', 'directdiscount_minjdprice', 'directdiscount_avgjdprice', 'directdiscount_maxscrapedprice', 'directdiscount_minscrapedprice', 'directdiscount_avgscrapedprice', 'directdiscount_discount', 'directdiscount_price', 'directdiscount_limitationflag', 'directdiscount_limitationcode', 'directdiscount_availabilitynumberflag', 'directdiscount_availabilitynumber', 'directdiscount_minimumnumberflag', 'directdiscount_minimumnumber', 'directdiscount_maximumnumberflag', 'directdiscount_maximumnumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway', 'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3', 'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1', 'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3', 'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool', 'suit_discount', 'freegift_productkey', 'freegift_maxscrapedprice', 'freegift_minscrapedprice', 'freegift_avgscrapedprice', 'freegift_maxjdprice', 'freegift_minjdprice', 'freegift_avgjdprice', 'freegift_freegiftvalue', 'freegift_freegiftvalueatjdprice', 'freegift_freegiftvalueatscrapedprice', 'directdiscount_saleprice', 'directdiscount_percent', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent', 'bundle_subtype6_percent', 'suit_percent', 'freegift_percent', 'allpercentdiscount']
        to_float_col = ['allpercentdiscount','bundle_subtype15_giveawayrate1','bundle_subtype15_giveawayrate2','bundle_subtype15_giveawayrate3','bundle_subtype15_percent','bundle_subtype15_thresholdnumber1','bundle_subtype15_thresholdnumber2','bundle_subtype15_thresholdnumber3','bundle_subtype1_giveaway','bundle_subtype1_percent','bundle_subtype1_threshold','bundle_subtype2_giveaway','bundle_subtype2_maximumgiveaway','bundle_subtype2_percent','bundle_subtype2_threshold','bundle_subtype4_giveaway1','bundle_subtype4_giveaway2','bundle_subtype4_giveaway3','bundle_subtype4_percent','bundle_subtype4_threshold1','bundle_subtype4_threshold2','bundle_subtype4_threshold3','bundle_subtype6_freenumber','bundle_subtype6_percent','bundle_subtype6_thresholdnumber','directdiscount_availabilitynumber','directdiscount_discount','directdiscount_saleprice','hierarchylevel3key','jdprice','mainproductkey','productkey','promotionkey','promotionsubtype','promotiontype','statuscode','suit_avgvaluepool','suit_discount','suit_maxvaluepool','suit_minvaluepool','suit_percent','syntheticgrossprice']
        period_promo_raw['enddatetime'] = pd.to_datetime(period_promo_raw['enddatetime'])
        period_promo_raw['startdatetime'] = pd.to_datetime(period_promo_raw['startdatetime'])

        if 'object' in period_promo_raw[to_float_col].dtypes.values:
            object2Float(period_promo_raw,to_float_col)

        period_promo_raw = period_promo_raw[(period_promo_raw.enddatetime>DataStartDate)&(period_promo_raw.startdatetime<=ForecastStartDate)]

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
        period_promo_raw.drop_duplicates(inplace=True)
        promo_period = period_promo_raw

    # promo_period.to_csv('f01.csv',index=False)
    ''' F01 Finish'''

    ''' f02 start '''

    raw_df= ts_df_pred[ts_df_pred['Date']<ForecastStartDate]


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

    featuresDf = raw_df

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
        print "cart feature generated"
        return df44

    if scenario['use_cart_feature']:
        featuresDf = createCartFeatures(featuresDf)

    print "use cart feature?"
    print scenario['use_cart_feature']

    '''
        seasonality
    '''
    # ''' 使用训练数据，过滤了6,11,再使用了monthly/yearly的销量,并没有使用'''
    # # featuresDf = createSeasonalityFeatures(featuresDf, ['HierarchyLevel3Key', 'ProductKey'], 'skuSeasonality', scenario)
    # # featuresDf = createSeasonalityFeatures(featuresDf, ['HierarchyLevel3Key'], 'level3Seasonality', scenario)
    '''使用了seasonal_decompose,并且预测了trend和seasonality,注意，至少2year数据才可以 '''
    if seasonality_df is None:
        if (ts_df_pred.Date.max() - ts_df_pred.Date.min()).days > 730:
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

    featuresDf = pd.merge(featuresDf, promoCalendarDf[promoCalendarDf.columns.difference(['HierarchyLevel1Key','HierarchyLevel2Key', 'HierarchyLevel3Key'])],   left_on = ['ProductKey', 'Date'], right_on = ['ProductKey', 'Date'], how='left')
    if scenario['STOCK_PRICE_FLAG']:
        featuresDf = calculateStockFeatures(featuresDf)


    featuresDf = calculateNationalRolling(featuresDf, scenario, scenario['selectedColumns']['target'])
    featuresDf = calculateRolling(featuresDf, scenario, scenario['selectedColumns']['target'])
    featuresDf = calculateSimilarRolling(featuresDf, scenario, scenario['Target_col'])
    featuresDf = calculateLagging(featuresDf, scenario, scenario['Target_col'])

    featuresDf = pd.merge(featuresDf,  promo_period,on = ['ProductKey', 'Date'], how = 'left')
    print 'featuresDf shape'
    print featuresDf.shape
    # return featuresDf

    ''' Train Models: '''
    def train_model(scenario, df_feature):
        param = {
        'objective': "reg:linear",
        'booster': "gbtree",
        'eta' : 0.3, # 0.06, #0.01,
        'min_child_weight':1.0,
        'gamma':0,
        'seed':10,
        'max_depth' :10, #changed from default of 8
        'subsample' :1.0, # 0.7
        'colsample_bytree': 1.0,# 0.7
        # 'silent':1,
        'nthread':4,
        'num_round':  20
        }
        # param = {
                # 'objective': "reg:linear",
                # 'booster': "gbtree",
                # 'eta' : 0.02, # 0.06, #0.01,
                # 'max_depth' :20, #changed from default of 8
                # 'subsample' :0.7, # 0.7
                # 'colsample_bytree': 0.4,# 0.7
                # # 'silent':1,
                # # 'nthread':4,
                # 'num_round':  300
                # }
        xColumns = scenario['selectedColumns']['features']
        '''
        whether to use promo features
        '''
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

        yColumns = scenario['selectedColumns']['target']

        y = df_feature[yColumns].values
        if scenario['use_cart_feature']:
            X = df_feature[xColumns + ['cart_cnt1']]
        else:
            X = df_feature[xColumns]
        xgmat = xgb.DMatrix(X.values, label=y, missing=np.NaN)
        model = xgb.train(param, xgmat, int(param['num_round']))
        return model

    model_hash = {}
    grouped = featuresDf.groupby('RDCKey')
    for rdc, group in grouped:
        print 'rdc'
        print rdc
        this_model = train_model(scenario, group)
        model_hash[rdc] = this_model
    if mode=='dev':
        return model_hash,featuresDf
    else:
        return model_hash
    ''' Train Models: '''
    # def train_model(scenario, df_feature):
        # param = {
                # 'objective': "reg:linear",
                # 'booster': "gbtree",
                # 'eta' : 0.3, # 0.06, #0.01,
                # 'min_child_weight':1.0,
                # 'gamma':0,
                # 'seed':10,
                # 'max_depth' :10, #changed from default of 8
                # 'subsample' :1.0, # 0.7
                # 'colsample_bytree': 1.0,# 0.7
                # # 'silent':1,
                # # 'nthread':4,
                # 'num_round':  30
                # }
        # xColumns = scenario['selectedColumns']['features']
        # yColumns = scenario['selectedColumns']['target']

        # y = df_feature[yColumns].values

        # X = df_feature[xColumns]
        # xgmat = xgb.DMatrix(X.values, label=y, missing=np.NaN)
        # model = xgb.train(param, xgmat, int(param['num_round']))
        # return model

    # model = train_model(scenario, featuresDf)
    # return model, featuresDf

if __name__ == "__main__":
    # 服饰品类：9708,9710,9732,9733,9734

    scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
    scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
    '''online analysis'''

    # file_path = 'tmp/11922_1219/ts_train_11922'
    # ts_df = pd.read_csv(file_path,header=None,sep='\t')

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


    holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')
    area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')


    #p1_path = '/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/9708_p1out.da'
    #p2_path = '/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/9708_p2out.da'
    #ts_path = '/home/ubuntu/yulong/promotion_offline/tmp/ow_1315/9708_tsout.da'
    #sea_path = "/home/ubuntu/hehongjie/data/11922_train_season"

    p1_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/870_p1out.da'
    p2_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/870_p2out.da'
    ts_path = '/home/ubuntu/sunjiadong/promotion_offline/tmp/data/870_tsout.da'
    seasonality_df = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/ts_11922_sea_from2018.csv',parse_dates=['Date'])

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
    seasonality_df= get_compress_file(sea_path).head(1) # 服饰不用季节性，先随便生成一个,大多数sku都不足两年，不稳定

    train_date = pd.to_datetime('2018-02-01')
    model_hash, train_features=train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,seasonality_df,process_f01_flag=False, mode='dev',ForecastStartDate=train_date)

    save_object(model_hash,'tmp/ow_1315/model_ow_20180321_rdc_phx0416.pkl')
    # train_features.to_csv('tmp/ow_1315/train_fea_20171001_rdc_at180320_9734.csv',index=False,sep='^')
    print 'save_object'

    ''' end '''

    # file_path = 'tmp/ow_deploy_single/owtime2_1590.csv'
    # ts_df = pd.read_csv(file_path,header=None)
    # holidays_df=pd.read_csv('tmp/ow_deploy_single/holidays.csv')
    # holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    # pro_canlender_path = 'tmp/ow_deploy_single/temp_4cate.csv'

    # promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)

    # area_rdc_map = pd.read_csv('tmp/ow_deploy_single/area_rdc_mapping.csv')
    # period_promo_raw = pd.read_csv('tmp/ow_deploy_single/Clean_Detail1590_new.csv')
    # seasonality_df = pd.read_csv('seasonality_ow.csv',parse_dates=['Date'])
    # model=train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df)
    # save_object(model,'model_ow.pkl')

    # '''Jie Ge Test'''
    # file_path = 'tmp/ow_deploy_jiege/ts.csv'
    # ts_df = pd.read_csv(file_path,header=None)
    # holidays_df=pd.read_csv('tmp/ow_deploy_single/holidays.csv')
    # pro_canlender_path = 'tmp/ow_deploy_jiege/p2.csv'

    # promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)

    # area_rdc_map = pd.read_csv('tmp/ow_deploy_single/area_rdc_mapping.csv')
    # # period_promo_raw = pd.read_csv('tmp/ow_deploy_jiege/p1.csv',sep='\t',header=None)
    # # period_promo_raw.columns=['productkey', 'promotionkey', 'startdatetime', 'enddatetime', 'promotionstartdatetime', 'promotionenddatetime', 'jdpricestartdatetime', 'jdpriceenddatetime', 'jdprice', 'spricestartdatetime', 'spriceenddatetime', 'syntheticgrossprice', 'promo_inperiod_flag', 'promotiondesc', 'promotiontype', 'promotionsubtype', 'createdatetime', 'areatypearray', 'levelmember', 'tokenflag', 'tokentype', 'directdiscount_productkey', 'directdiscount_maxjdprice', 'directdiscount_minjdprice', 'directdiscount_avgjdprice', 'directdiscount_maxscrapedprice', 'directdiscount_minscrapedprice', 'directdiscount_avgscrapedprice', 'directdiscount_jdpricedate', 'directdiscount_discount', 'directdiscount_price', 'directdiscount_limitationflag', 'directdiscount_limitationcode', 'directdiscount_availabilitynumberflag', 'directdiscount_availabilitynumber', 'directdiscount_minimumnumberflag', 'directdiscount_minimumnumber', 'directdiscount_maximumnumberflag', 'directdiscount_maximumnumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway', 'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3', 'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1', 'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3', 'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool', 'suit_discount', 'freegift_productkey', 'freegift_maxscrapedprice', 'freegift_minscrapedprice', 'freegift_avgscrapedprice', 'freegift_maxjdprice', 'freegift_minjdprice', 'freegift_avgjdprice', 'freegift_freegiftvalue', 'freegift_freegiftvalueatjdprice', 'freegift_freegiftvalueatscrapedprice', 'directdiscount_saleprice', 'directdiscount_percent', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent', 'bundle_subtype6_percent', 'suit_percent', 'freegift_percent', 'allpercentdiscount', 'productdesc', 'mainproductkey', 'hierarchylevel1key', 'hierarchylevel2key', 'hierarchylevel3key', 'createdate', 'statuscode', 'dt']
    # period_promo_raw = pd.read_csv('tmp/ow_deploy_jiege/p1_out.csv')

    # # promoCalendarDf:bak_pro_columns = ['ProductKey','Date','HierarchyLevel1Key','HierarchyLevel2Key','HierarchyLevel3Key','PromotionCount','bundlecount','directdiscountcount','freegiftcount','suitcount','MaxDiscount','MinDiscount','AvgDiscount','MaxSyntheticDiscountA','MinSyntheticDiscountA','AvgSyntheticDiscountA','maxhourlyaveragesyntheticdiscounta','minhourlyaveragesyntheticdiscounta','avghourlyaveragesyntheticdiscounta','MaxBundleDiscount','MinBundleDiscount','AvgBundleDiscount','MaxDirectDiscount','MinDirectDiscount','AvgDirectDiscount','MaxSuitDiscount','MinSuitDiscount','AvgSuitDiscount','MaxFreegiftDiscount','MinFreegiftDiscount','AvgFreegiftDiscount','maxjdprice','maxscrapedprice','syntheticpromoflaga','SyntheticGrossPrice','maxdirectdiscountsyntheticdiscounta','maxbundlesyntheticdiscounta','inddirectdiscountday','promotionkey','promotiontype','promotionsubtype','syntheticgrossprice_vb','jdprice','syntheticdiscounta_vb','syntheticpromoflaga_vb','durationinhours','daynumberinpromotion','bundleflag','directdiscountflag','freegiftflag','suitflag','numberproducts','numberhierarchylevel1','numberhierarchylevel2','numberhierarchylevel3','strongmark','stockprice','dt']
    # #                 slice_pro_columns=['ProductKey','Date','HierarchyLevel3Key','PromotionCount','bundlecount','MaxDiscount','MinDiscount','AvgDiscount','MaxSyntheticDiscountA','MinSyntheticDiscountA','AvgSyntheticDiscountA','MaxBundleDiscount','MinBundleDiscount','AvgBundleDiscount','MaxDirectDiscount','MinDirectDiscount','AvgDirectDiscount','MaxFreegiftDiscount','MinFreegiftDiscount','AvgFreegiftDiscount','SyntheticGrossPrice','promotionkey','promotiontype','promotionsubtype','syntheticgrossprice_vb','jdprice','syntheticdiscounta_vb','durationinhours','daynumberinpromotion','bundleflag','directdiscountflag','freegiftflag','suitflag','numberproducts','numberhierarchylevel1','numberhierarchylevel2','numberhierarchylevel3','strongmark','stockprice','dt']
    # p2_used_header=['ProductKey','Date','HierarchyLevel3Key','PromotionCount','bundlecount','MaxDiscount','MinDiscount','AvgDiscount','MaxSyntheticDiscountA','MinSyntheticDiscountA','AvgSyntheticDiscountA','MaxBundleDiscount','MinBundleDiscount','AvgBundleDiscount','MaxDirectDiscount','MinDirectDiscount','AvgDirectDiscount','MaxFreegiftDiscount','MinFreegiftDiscount','AvgFreegiftDiscount','SyntheticGrossPrice','promotionkey','promotiontype','promotionsubtype','syntheticgrossprice_vb','jdprice','syntheticdiscounta_vb','durationinhours','daynumberinpromotion','bundleflag','directdiscountflag','freegiftflag','suitflag','numberproducts','numberhierarchylevel1','numberhierarchylevel2','numberhierarchylevel3','strongmark','stockprice','dt']

    # promoCalendarDf.columns= ['ProductKey','Date','HierarchyLevel1Key','HierarchyLevel2Key','HierarchyLevel3Key','PromotionCount','bundlecount','directdiscountcount','freegiftcount','suitcount','MaxDiscount','MinDiscount','AvgDiscount','MaxSyntheticDiscountA','MinSyntheticDiscountA','AvgSyntheticDiscountA','maxhourlyaveragesyntheticdiscounta','minhourlyaveragesyntheticdiscounta','avghourlyaveragesyntheticdiscounta','MaxBundleDiscount','MinBundleDiscount','AvgBundleDiscount','MaxDirectDiscount','MinDirectDiscount','AvgDirectDiscount','MaxSuitDiscount','MinSuitDiscount','AvgSuitDiscount','MaxFreegiftDiscount','MinFreegiftDiscount','AvgFreegiftDiscount','maxjdprice','maxscrapedprice','syntheticpromoflaga','SyntheticGrossPrice','maxdirectdiscountsyntheticdiscounta','maxbundlesyntheticdiscounta','inddirectdiscountday','promotionkey','promotiontype','promotionsubtype','syntheticgrossprice_vb','jdprice','syntheticdiscounta_vb','syntheticpromoflaga_vb','durationinhours','daynumberinpromotion','bundleflag','directdiscountflag','freegiftflag','suitflag','numberproducts','numberhierarchylevel1','numberhierarchylevel2','numberhierarchylevel3','strongmark','stockprice','dt']
    # promoCalendarDf_clean = promoCalendarDf[p2_used_header]

    # model=train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,process_f01_flag=False)

    # pro_canlender_path = 'tmp/11922_1219/p2_train_11922'
    # pro_canlender_path = 'tmp/11922_2018/11922_train_p2'

    # promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)

    # promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
               # 'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
               # 'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
               # 'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
               # 'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']
    # f = open(pro_canlender_path)
    # for line in f:
        # tup_item = eval(line)
        # promoCalendarDf = cp.loads(unzip_base64(tup_item[1]))

    # p1_out_path ='tmp/11922_2018/p1_out_2018.csv'
    # period_promo_raw = pd.read_csv(p1_out_path)

    # seasonality_df = pd.read_csv('tmp/11922_2018/ts_11922_sea_from2018.csv',parse_dates=['Date'])
    # import cPickle as cp
    # import zlib
    # import base64
    # def unzip_base64(str_src, bool_use_wbits=False):
        # li_chunks = str_src.split("$")
        # li_unzip_chunks = []
        # for i in range(0, len(li_chunks)):
            # if bool_use_wbits:
                # li_unzip_chunks.append(zlib.decompress(base64.b64decode(li_chunks[i]), zlib.MAX_WBITS | 16))
            # else:
                # li_unzip_chunks.append(zlib.decompress(base64.b64decode(li_chunks[i])))
        # return "".join(li_unzip_chunks)

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


        pro_canlender_path = '/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/11922_train_p2'

        promoCalendarDf = pd.read_csv(pro_canlender_path,sep='\t',header=None)

        promoCalendarDf.columns=['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',
                   'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',
                   'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',
                   'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',
                   'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']

        p1_out_path ='/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/p1_out_2018.csv'
        period_promo_raw = pd.read_csv(p1_out_path,parse_dates=['Date'])


        file_path = '/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/11922_train_ts'
        ts_df = pd.read_csv(file_path,header=None,sep='\t')

        scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
        scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
        '''online analysis'''

        holidays_df=pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/holidays.csv')

        area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')

        seasonality_df = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/11922_2018/ts_11922_sea_from2018.csv',parse_dates=['Date'])


        model,feature=train(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,seasonality_df,process_f01_flag=False,mode='dev')
        #save_object(model,'tmp/11922_2018/model_ow_v_oct_20171010_rdc_at180321_11922.pkl')
        save_object(model,'tmp/11922_2018/model_no_promo_ow_20171010_rdc_at180321_11922.pkl')

        #feature.to_csv('tmp/11922_2018/at180321_11922_v_oct_feature_20171010.csv',index=False)
        feature.to_csv('tmp/11922_2018/at180321_11922_v_oct_no_promo_feature_20171010.csv',index=False)
