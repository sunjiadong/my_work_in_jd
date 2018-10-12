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
                                 agg_feature,prep_data,calc_weighted_price,agg_feature_day)
from code.refactor.fdc_flow import  filter_non_price_fill_it
import pandas as pd
import xgboost as xgb
import numpy as np
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

def ow_seasonality(scenario,ts_df,ForecastStartDate=None):
    '''
        Configuration:
            1 At least 2 year Data to get seasonality
            2 lookbackPeriodDaysSeasonality:  How many days ues to get seasonality
            3 lookforwardSeasonality: length of output Future seasonality
        Output:
            Sku,level3 Seasonality by Times Series Decompose in Month
            Curve: last year Level3 Sales
    '''

    if ForecastStartDate is None:
        ForecastStartDate = pd.to_datetime(scenario['forecastStartDate'])

    ForecastStartDate = pd.to_datetime(ForecastStartDate)

    DataStartDate = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDaysSeasonality'])
    FutureSeasonaDate = ForecastStartDate + datetime.timedelta(days=scenario['lookforwardSeasonality'])

    ts_df.columns = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    raw_df = ts_df[(ts_df['Date']<ForecastStartDate)&(ts_df['Date']>DataStartDate)]

    if (raw_df.Date.max() - raw_df.Date.min()).days <= 730:
        print 'less 2 year date to get seasonality'
        sea_col = ['Date', 'HierarchyLevel3Key', 'OrderNonOutlierVolume', 'ProductKey',\
       'RDCKey', 'skuDecomposedSeasonal', 'skuDecomposedTrend',\
       'level3DecomposedTrend', 'level3DecomposedSeasonal', 'Curve']
        return pd.DataFrame(columns=sea_col)



    q = scenario['quantile'] # 0.95

    groupKeys = scenario['selectedColumns']['levelsCoarse'] # ['HierarchyLevel3Key','ProductKey','RDCKey']
    unitVolumeQuantile = raw_df.groupby(groupKeys)[scenario['Target_col']].quantile(q, interpolation='lower')
    quantileDf = pd.DataFrame(unitVolumeQuantile).reset_index()
    quantileDf.rename(columns = {scenario['Target_col']: 'UnitCapVolume'}, inplace=True)

    # join quantile
    raw_df= pd.merge(raw_df, quantileDf, how='left', on=groupKeys)
    raw_df['OrderNonOutlierVolume'] = raw_df[['salesForecast', 'UnitCapVolume']].min(axis=1)

    # drop cap volume
    raw_df.drop(['UnitCapVolume'], axis=1, inplace = True)
    targetCol = scenario['selectedColumns']['target']
    df_past = raw_df[groupKeys + ['Date', targetCol]]
    dfs=[]
    # concat future Df by RDC
    for rdc in list(set(df_past.RDCKey)):

        valid_skus_fut=list(set(df_past[(df_past['Date']==df_past.Date.max())&(df_past.RDCKey==rdc)]['ProductKey'].values))

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

        fut_null_df = generate_strutured_null_df(ForecastStartDate, FutureSeasonaDate, valid_skus_fut)  # Filling 0601-0604 cols [stat_date, sku]

        df_future = df_past[df_past.RDCKey==rdc].append(fut_null_df)
        df_future.sort_values(['ProductKey', 'Date'], inplace=True)
        df_future.reset_index(drop=True,inplace=True)
        col = list(df_future.columns.values)
        col.remove('ProductKey')

        df_future[col] = df_future.groupby(['ProductKey']).transform(lambda x: x.fillna(method='ffill'))
        dfs.append(df_future)
    df_future_all =pd.concat(dfs)
    df_future_all.reset_index(drop=True,inplace=True)
    
    ''' 使用训练数据，过滤了6,11,再使用了monthly/yearly的销量,并没有使用'''
    # featuresDf = createSeasonalityFeatures(featuresDf, ['HierarchyLevel3Key', 'ProductKey'], 'skuSeasonality', scenario)
    # featuresDf = createSeasonalityFeatures(featuresDf, ['HierarchyLevel3Key'], 'level3Seasonality', scenario)
    '''使用了seasonal_decompose,并且预测了trend和seasonality '''
    featuresDf = createSeasonalityDecomposeFeatures(df_future_all, ['HierarchyLevel3Key', 'ProductKey'], 'sku', scenario)

    featuresDf = createSeasonalityDecomposeFeatures(featuresDf, ['HierarchyLevel3Key'], 'level3', scenario)

    featuresDf = createLevel3Features(featuresDf,scenario)
    return featuresDf
    # featuresDf.to_csv('seasonality_ow.csv',index=False)

if __name__ == "__main__":
    scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
    scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
    # file_path = 'tmp/ow_deploy_single/owtime2_1590.csv'

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

    def main_ts(path):
        f = open(path, "r")
        li_lines = []
        for line in f.readlines():
            li_data = line.split(",")
            li_lines.append(li_data[4])
        pdf = func_concat_pdf(li_lines)
        return pdf

    #file_path = 'tmp/ow_deploy_jiege/ts.csv'
    #ts_df = pd.read_csv(file_path,header=None)
    file_path = 'tmp/data/7052_ts.da'
    #ts_df = main_ts(file_path)
    ts_df = pd.read_csv(file_path,header=None,sep='\t') 
    seasonality_df = ow_seasonality(scenario,ts_df)
    seasonality_df.to_csv('tmp/data/7052_season.csv',index=False)
