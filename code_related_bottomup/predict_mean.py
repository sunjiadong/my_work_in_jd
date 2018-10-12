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

def predict(area_rdc_map,period_promo_raw,promoCalendarDf,ts_df,scenario,holidays_df,model,seasonality_df=None,process_f01_flag=True,ForecastStartDate=None,mode='product',train_feature=None):
    # process_f01_flag 需要不需要f01去预处理，if yes : period_promo_raw
    #                                         if no  : period_promo_raw=f01输出的结果
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
    
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    object2Int(holidays_df,['Holiday','Ind_1111_pre','Ind_1111','Ind_1111_post','Ind_618_pre','Ind_618','Ind_618_post','Ind_1212'])

    for col in area_rdc_map.columns:
        area_rdc_map[col] = area_rdc_map[col].astype('int')
    ''' Pre-Process period Calender '''
    promoCalendarDf = promoCalendarDf[(promoCalendarDf['Date']<PredictEndDate)&(promoCalendarDf['Date']>DataStartDate)]


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
    #df_past = pd.read_csv(train_path)
    df_past = train_feature
    df1 = df_past[need_cols]
    groupkeys = ['RDCKey','ProductKey','HierarchyLevel3Key']
    promo_feature_cols =  scenario['promo_feature_cols']
    df11 = df1.groupby(groupkeys)[update_cols].mean().reset_index()
    df_fut = pd.merge(df_fut,df11[groupkeys + update_cols], how='left',on=groupkeys)
    rename_update_cols = [col+'_y' for col in update_cols]
    for col in update_cols:
        df_fut.rename(columns={col+'_y': col},inplace=True)

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

