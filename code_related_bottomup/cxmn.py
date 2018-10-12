# -*- coding: utf-8 -*-
import warnings
import os.path
import sys
import pickle
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
import ow_f01, cxmn_train, cxmn_predict, predict_mean
from code.refactor.common import loadSettingsFromYamlFile, save_object, object2Float, get_column_by_type

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('app_sfs_promotion_simulate').enableHiveSupport().getOrCreate()
# sc = SparkContext(conf=SparkConf().setAppName("promotion_simulate_sjd"))
sc = spark.sparkContext
'''
### donot need ow_f01
p1_used_header = ['productkey', 'promotionkey', 'startdatetime', 'enddatetime', 'jdprice', 'syntheticgrossprice', 'promotiondesc', 'promotiondesc_flag', 'promotiontype', 'promotionsubtype',
                'areatypearray', 'tokenflag', 'directdiscount_discount', 'directdiscount_availabilitynumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway',
                'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3',
                'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1',
                'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3',
                'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool',
                'suit_discount', 'directdiscount_saleprice', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent',
                'bundle_subtype6_percent', 'suit_percent', 'allpercentdiscount', 'mainproductkey', 'hierarchylevel3key', 'createdate', 'statuscode', 'dt']
p2_used_header = ['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount', 'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA',\
                  'MinSyntheticDiscountA', 'AvgSyntheticDiscountA', 'MaxBundleDiscount', 'MinBundleDiscount', 'AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',\
                  'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount', 'SyntheticGrossPrice', 'promotionkey', 'promotiontype',\
                  'promotionsubtype', 'syntheticgrossprice_vb', 'jdprice', 'syntheticdiscounta_vb', 'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag',\
                  'freegiftflag', 'suitflag', 'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']
suffix = '.da'
item = 'p1'
for_what = ['train', 'predict']
'''

p1_used_header = ['Date', 'ProductKey', 'dd_price_weighted', 'bd_price_weighted', 'flashsale_ind', 'dd_ind',
                  'bundle_ind','bundle_buy199get100_ind', 'suit_ind', 'freegift_ind', 'cnt_period', 'dd_discount_daily_max', \
                  'dd_discount_sgp_daily_max', 'bd_discount_daily_max', 'suit_discount_new_daily_max',
                  'dd_price_daily_min','bd_price_daily_min', 'jdprice_daily_min', 'sgp_daily_min', 'dd_discount_wgt', 'dd_discount_sgp_wgt', \
                  'bd_discount_wgt', 'bd_discount_sgp_wgt']
p2_used_header = ['ProductKey', 'Date', 'HierarchyLevel3Key', 'PromotionCount', 'bundlecount', 'MaxDiscount',\
                  'MinDiscount', 'AvgDiscount', 'MaxSyntheticDiscountA','MinSyntheticDiscountA', 'AvgSyntheticDiscountA',\
                  'MaxBundleDiscount', 'MinBundleDiscount','AvgBundleDiscount', 'MaxDirectDiscount', 'MinDirectDiscount',\
                  'AvgDirectDiscount', 'MaxFreegiftDiscount', 'MinFreegiftDiscount', 'AvgFreegiftDiscount','SyntheticGrossPrice',\
                  'syntheticgrossprice_vb', 'syntheticdiscounta_vb', 'promotionkey', 'promotiontype', 'promotionsubtype', 'jdprice',\
                  'durationinhours', 'daynumberinpromotion', 'bundleflag', 'directdiscountflag', 'freegiftflag', 'suitflag',\
                  'numberproducts', 'numberhierarchylevel1', 'numberhierarchylevel2', 'numberhierarchylevel3', 'strongmark', 'stockprice', 'dt']
ts_df_used_header = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key',\
                     'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast',\
                     'reserveState', 'stockQuantity', 'utc_flag']
ts_raw_used_header = ['group_id', 'dc_id', 'sku_id', 'cate1_cd', 'cate2_cd', 'cate3_cd', 'brand_cd', 'sku_status_cd',\
                      'utc_dt', 'utc_flag','start_date', 'end_date', 'len', 'y', 'data_key', 'dt', 'biz']

str_p1_table = "dev.dev_sfs_bottomup_p1"
str_p2_table = "dev.dev_sfs_bottomup_p2"
str_ts_table = "dev.dev_sfs_bottomup_ts"
str_query_dt = "ACTIVE"  # "2018-02-01"
str_biz = ["rdc_train", "rdc_predict"]


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


def reduce_df_mem_usage(df):
    # memery now
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of the dataframe is :", start_mem_usg, "MB")
    # np.nan will be handled as float
    NAlist = []
    for col in df.columns:
        # filter object type
        if (df[col].dtypes == np.float64):
            df[col] = df[col].astype(np.float32)
            continue
        if (df[col].dtypes != object) & (df[col].dtypes != 'datetime64[ns]'):
            print("**************************")
            print("columns: %s" % col)
            print("dtype before: %s" % df[col].dtype)
            # if int or not
            isInt = False
            mmax = df[col].max()
            mmin = df[col].min()
            # Integer does not support NA, therefore Na needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                # continue
                df[col].fillna(-999, inplace=True)  # fill -999
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = np.fabs(df[col] - asint)
            result = result.sum()
            if result < 0.01:  # absolute error < 0.01,then could be saw as integer
                isInt = True
            # make interger / unsigned Integer datatypes
            if isInt:
                if mmin >= 0:  # min>=0, then unsigned integer
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
            print("dtype after: %s" % df[col].dtype)
            print("********************************")
    print("___MEMORY USAGE AFTER CONVERSION:___")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return df


def statitics_mape(new_df_final):
    new_df_sku = new_df_final.groupby('ProductKey').sum().reset_index()
    print
    "ensemble pred sum : %f" % (new_df_sku.ypred_mean_promo_new.sum())
    print
    "raw pred sum :      %f" % (new_df_sku.ypred_raw.sum())
    print
    "actual sum:         %f" % (new_df_sku.salesForecast.sum())
    print
    "raw pred residual:      %f" % (np.sum(np.abs(new_df_sku.ypred_raw - new_df_sku.salesForecast)))
    print
    "ensemble pred residual: %f" % (np.sum(np.abs(new_df_sku.ypred_mean_promo_new - new_df_sku.salesForecast)))
    print
    "raw pred mape: %f" % (
    np.sum(np.abs(new_df_sku.ypred_raw - new_df_sku.salesForecast)) / new_df_sku.salesForecast.sum())
    print
    "ensemble mape: %f" % (
    np.sum(np.abs(new_df_sku.ypred_mean_promo_new - new_df_sku.salesForecast)) / new_df_sku.salesForecast.sum())


def date_delta(base_date, days, fmt=None):
    """
    相对于基准日期相差天数的日期
    :param base_date: 基准日期'%Y-%m-%d'格式
    :param days: 相差的天数
    :param fmt: 日期格式，默认为'%Y-%m-%d'格式
    :return: 相对于基准日期相差天数的日期
    """
    if fmt is None:
        fmt = '%Y-%m-%d'
    dt_base = datetime.datetime.strptime(base_date, fmt)
    return (dt_base + datetime.timedelta(days)).strftime(fmt)


def expand_ts(row_ts):
    str_dc_id = row_ts["dc_id"]
    str_sku_id = row_ts["sku_id"]
    str_cate1_cd = row_ts["cate1_cd"]
    str_cate2_cd = row_ts["cate2_cd"]
    str_cate3_cd = row_ts["cate3_cd"]
    str_brand_cd = row_ts["brand_cd"]
    str_utc_flag = row_ts["utc_flag"]
    i_len = row_ts["len"]
    str_end_date = row_ts["end_date"]
    li_y = eval(row_ts["y"].replace("nan", "float('nan')"))
    li_data_key = eval(row_ts["data_key"])
    # 展开时序
    li_ts = []
    for i in range(0, i_len):
        str_date = date_delta(str_end_date, -1 * i)
        dict_row_value = {}
        i_pos = i_len - i - 1
        for j in range(0, len(li_data_key)):
            str_key = li_data_key[j]
            str_key_value = li_y[j][i_pos]
            dict_row_value[str_key] = str_key_value
        li_ts.append([str_date, i, str_dc_id, str_sku_id, str_cate1_cd, str_cate2_cd, str_cate3_cd, str_brand_cd,
                      dict_row_value["sales"],
                      dict_row_value["priceAfterDiscount"], dict_row_value["priceBeforeDiscount"],
                      dict_row_value["vendibility"],
                      dict_row_value["counterState"], dict_row_value["salesForecast"], dict_row_value["reserveState"],
                      dict_row_value["stockQuantity"], str_utc_flag])
    return li_ts


def get_online_history_data(cate3, dt_query='ACTIVE'):
    str_query_dt = dt_query
    str_cate3 = cate3
    for str_ts_biz in str_biz:
        str_sql = """select * from {source_table} where dt = '{dt}' and biz = '{biz}' and cate3_cd = {cate3_cd}""" \
            .format(source_table=str_ts_table, dt=str_query_dt, biz=str_ts_biz, cate3_cd=str_cate3)
        rdd = spark.sql(str_sql).rdd
        rdd_list = rdd.collect()
        raw_df = pd.DataFrame(rdd_list)
        raw_df.columns = ts_raw_used_header
        li_ts = []
        for index, row in raw_df.iterrows():
            li_ts = li_ts + expand_ts(row)
        ts_df = pd.DataFrame(li_ts)
        ts_df.columns = ts_df_used_header
        del raw_df
        if str_ts_biz == 'rdc_train':
            ts_df_train = ts_df
            del ts_df
        else:
            ts_df_predict = ts_df
            del ts_df
    for str_p2_biz in str_biz:
        str_sql = """select * from {source_table} where dt = '{dt}' and biz = '{biz}' and cate3_cd = {cate3_cd}""" \
            .format(source_table=str_p2_table, dt=str_query_dt, biz=str_p2_biz, cate3_cd=str_cate3)
        # str_sql = "select * from dev.dev_sfs_bottomup_p2 where dt = 'ACTIVE' and biz = 'rdc_predict' and cate3_cd = 7054 limit 10"
        rdd = spark.sql(str_sql).rdd
        rdd_list = rdd.collect()
        raw_df = pd.DataFrame(rdd_list)
        del raw_df[0]
        del raw_df[41]
        raw_df.columns = p2_used_header
        raw_df['dt'] = raw_df['Date']
        raw_df.drop_duplicates(inplace=True)
        if str_p2_biz == "rdc_train":
            p2_df_train = raw_df
            del raw_df
        else:
            p2_df_predict = raw_df
            del raw_df
    for str_p1_biz in str_biz:
        str_sql = """select * from {source_table} where dt = '{dt}' and biz = '{biz}' and cate3_cd = {cate3_cd}""" \
            .format(source_table=str_p1_table, dt=str_query_dt, biz=str_p1_biz, cate3_cd=str_cate3)
        rdd = spark.sql(str_sql).rdd
        rdd_list = rdd.collect()
        raw_df = pd.DataFrame(rdd_list)
        del raw_df[0]
        del raw_df[3]
        del raw_df[25]
        del raw_df[26]
        raw_df.columns = p1_used_header
        raw_df.drop_duplicates(inplace=True)
        if str_p1_biz == "rdc_train":
            p1_df_train = raw_df
            del raw_df
        else:
            p1_df_predict = raw_df
            del raw_df
    return p1_df_train, p1_df_predict, p2_df_train, p2_df_predict, ts_df_train, ts_df_predict


def get_offline_history_data():
    # spark = SparkSession(sc)
    # columns = ['tag', 'rowkey', 'date_unit', 'start_date', 'history_len',
    #    'unit_len', 'time_span', 'his_ts_qttys', 'remark', 'source_type',
    #    'date_type']
    # sql = "select * from app.app_sfs_ord_history_time_series where source_type='"+source_type+"' and date_type='"+date_type+"'"
    # rdd = spark.sql(sql).rdd
    # rdd_list=rdd.collect()
    # raw_df=pd.DataFrame(rdd_list,columns=columns)
    # return raw_df

    p1 = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/7054_p1.da', sep='\t', header=None)
    p2 = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/7054_p2.da', sep='\t', header=None)
    ts = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/7054_ts.da', sep='\t', header=None)

    # p1 = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/shishang/12029/12029_p1.da',sep='\t',header=None)
    # p2 = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/shishang/12029/12029_p2.da',sep='\t',header=None)
    # ts = pd.read_csv('/home/ubuntu/sunjiadong/promotion_offline/tmp/data/shishang/12029/12029_ts.da',sep='\t',header=None)


    return p1, p2, ts


def preprocessing(df):
    return False


def main():
    scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
    scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
    area_rdc_map = pd.read_csv('tmp/data/area_rdc_mapping.csv')
    holidays_df = pd.read_csv('tmp/data/holidays.csv')
    seasonality_df = pd.read_csv('tmp/data/870_season.csv', parse_dates=['Date'])

    '''
    hyperparameters
    '''
    # for i in range(1,len(sys.argv)):
    #        if sys.argv[i] == '-cate3':
    #            cate=sys.argv[i+1]
    #        if sys.argv[i] == '-pred_dt':
    #            pred_date = pd.to_datetime(sys.argv[i+1])
    #        if sys.argv[i] == '-forecast_days':
    #           scenario['lookforwardPeriodDays'] = sys.argv[i+1]

    cate = 7054
    pred_date = pd.to_datetime('2018-07-19')
    scenario['lookforwardPeriodDays'] = 10

    # cate = 12029
    # pred_date = pd.to_datetime('2018-05-07')
    # scenario['lookforwardPeriodDays'] = 7

    train_p1_df, predict_p1_df, train_p2_df, predict_p2_df, ts_df_train, ts_df_predict = get_online_history_data(cate)

    # handle p1
    for df in [train_p1_df, predict_p1_df]:
        df['Date'] = pd.to_datetime(df['Date'])
        df['cnt_period'] = df['cnt_period'].astype(int)
        if 'object' in df.dtypes.values:
            obj_cols = get_column_by_type(df, 'object')
            object2Float(df, obj_cols)

    # handle p2
    for df in [train_p2_df, predict_p2_df]:
        df['Date'] = pd.to_datetime(df['Date'])
        df['dt'] = pd.to_datetime(df['dt'])
        df.replace('null', np.nan, inplace=True)
        df.replace('None', np.nan, inplace=True)
        df.replace(-999, np.nan, inplace=True)
        df.drop_duplicates(inplace=True)
        # Convert Object -> float
        if 'object' in df.dtypes.values:
            obj_cols = get_column_by_type(df, 'object')
            object2Float(df, obj_cols)

    def handle_f01(p1_df, for_what, area_rdc_map, pred_date, scenario):
        for fw in for_what:
            print
            "output and save: %s_p1_%s" % (str(cate), fw)
            train_pred_gate = fw  # 'train'
            f01_output = ow_f01.generate_f01_promo(area_rdc_map, p1_df, scenario, train_pred_gate,
                                                   ForecastStartDate=pred_date)
            if train_pred_gate == 'train':
                train_p1_df = f01_output
            else:
                predict_p1_df = f01_output
        return train_p1_df, predict_p1_df
        # train_p1_df,predict_p1_df = handle_f01(p1_df, for_what, area_rdc_map, pred_date, scenario)
        # ts_df.columns = ['Date', 'ind', 'RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key', 'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState', 'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
    # handle ts
    for df in [ts_df_train, ts_df_predict]:
        df['Date'] = pd.to_datetime(df['Date'])
        df.replace('null', np.nan, inplace=True)
        df.replace(-999, np.nan, inplace=True)
        df.replace('None', np.nan, inplace=True)
        ts_to_float_col = ['RDCKey', 'ProductKey', 'HierarchyLevel1Key', 'HierarchyLevel2Key', 'HierarchyLevel3Key',
                           'brand_code', 'sales', 'priceAfterDiscount', 'jd_prc', 'vendibility', 'counterState',
                           'salesForecast', 'reserveState', 'stockQuantity', 'utc_flag']
        if 'object' in df[ts_to_float_col].dtypes.values:
            object2Float(df, ts_to_float_col)

    def train_model(area_rdc_map, train_p1_df, p2_df, ts_df, scenario, holidays_df, seasonality_df, pred_date):
        seasonality_df_train = seasonality_df.copy()
        model, feature = cxmn_train.train(area_rdc_map, train_p1_df, p2_df, ts_df, scenario, holidays_df,
                                          seasonality_df_train, process_f01_flag=False, mode='dev',
                                          ForecastStartDate=pred_date)
        return model, feature

    model, feature = train_model(area_rdc_map, train_p1_df, train_p2_df, ts_df_train, scenario, holidays_df,
                                 seasonality_df, pred_date)

    def predict_q_pred(area_rdc_map, predict_p1_df, p2_df, ts_df, scenario, holidays_df, seasonality_df, pred_date,
                       model):
        seasonality_df_test = seasonality_df.copy()
        q_pred_result, df_fut = cxmn_predict.predict(area_rdc_map, predict_p1_df, p2_df, ts_df, scenario, holidays_df,
                                                     model, seasonality_df_test, process_f01_flag=False, mode='dev',
                                                     ForecastStartDate=pred_date)
        return q_pred_result, df_fut

    q_pred_result, df_fut = predict_q_pred(area_rdc_map, predict_p1_df, predict_p2_df, ts_df_predict, scenario,
                                           holidays_df, seasonality_df, pred_date, model)

    train_feature_df_new = reduce_df_mem_usage(feature)

    def predict_q_mean(area_rdc_map, predict_p1_df, p2_df, ts_df, scenario, holidays_df, seasonality_df, pred_date,
                       model, train_feature_df_new):
        seasonality_df_mean = seasonality_df.copy()
        q_mean_result, df_fut_mean = predict_mean.predict(area_rdc_map, predict_p1_df, p2_df, ts_df, scenario,
                                                          holidays_df, model, seasonality_df_mean,
                                                          process_f01_flag=False, mode='dev',
                                                          ForecastStartDate=pred_date,
                                                          train_feature=train_feature_df_new)
        return q_mean_result, df_fut_mean

    q_mean_result, df_fut_mean = predict_q_mean(area_rdc_map, predict_p1_df, predict_p2_df, ts_df_predict, scenario,
                                                holidays_df, seasonality_df, pred_date, model, train_feature_df_new)

    def get_actual_sales(ts_df, pred_date, scenario):
        simplified_ts_df = ts_df[ts_df.Date.between(pred_date, pd.to_datetime(pred_date) + pd.DateOffset(
            days=scenario['lookforwardPeriodDays'] - 1))]
        return simplified_ts_df

    simplified_ts_df = get_actual_sales(ts_df_predict, pred_date, scenario)

    ForecastStartDate = pd.to_datetime(pred_date)
    DataStartDate = ForecastStartDate - datetime.timedelta(days=scenario['lookbackPeriodDays'])
    PredictEndDate = ForecastStartDate + datetime.timedelta(days=(scenario['lookforwardPeriodDays'] - 1))

    actual = simplified_ts_df
    actual.Date = pd.to_datetime(actual.Date)
    actual.RDCKey = actual.RDCKey.astype(float)

    list_keys = ['Date', 'RDCKey', 'ProductKey']
    feat_cols = ['dd_price_weighted', 'bd_price_weighted', 'dd_price_weighted_x', 'bd_price_weighted_x',
                 'SyntheticGrossPrice']
    exclu_promo_features = ['strongmark', 'flashsale_ind', 'dd_ind', 'bundle_ind', 'bundle_buy199get100_ind',
                            'suit_ind', 'freegift_ind']

    def get_raw_test_df(raw_pred, mean_pred, keys, feat_cols):

        raw = raw_pred  ###bottomup forecast
        raw = raw[keys + ['salesForecast', 'ypred']]
        raw.rename(columns={'ypred': 'ypred_raw'}, inplace=True)
        raw.drop('salesForecast', axis=1, inplace=True)

        mean_df = mean_pred
        mean_df = mean_df[keys + feat_cols + ['salesForecast', 'ypred']]
        mean_df.rename(columns={'ypred': 'ypred_mean_promo'}, inplace=True)
        mean_df.drop('salesForecast', axis=1, inplace=True)

        new_df = raw.merge(mean_df, on=list_keys)
        new_df.Date = pd.to_datetime(new_df.Date)
        new_df = pd.merge(new_df, actual[list_keys + ['salesForecast']], how='left', on=list_keys)

        return new_df

    new_df = get_raw_test_df(q_pred_result, q_mean_result, list_keys, feat_cols)

    update_cols = list(set(scenario['promo_feature_cols']) - set(exclu_promo_features))
    need_cols = ['Date', 'RDCKey', 'ProductKey', 'HierarchyLevel3Key'] + update_cols
    groupkeys = ['RDCKey', 'ProductKey', 'HierarchyLevel3Key']
    reg_cols = []  # ['Holiday','Ind_1111_pre','Ind_1111','Ind_1111_post','Ind_618_pre','Ind_618','Ind_618_post','Ind_1212','Month','DayOfWeek',]

    def predict_history_mean_and_raw(train_feature_df_new, reg_cols, listkeys, keys, model, update_cols, scenario):
        uses_promo = ['mean', 'no']
        df = train_feature_df_new
        for use_promo in uses_promo:
            if use_promo == 'mean':
                df1 = df[need_cols]
                promo_feature_cols = scenario['promo_feature_cols']
                df11 = df1.groupby(keys)[update_cols].mean().reset_index()
                df2 = pd.merge(df, df11[keys + update_cols], how='left', on=keys)

                rename_update_cols = [col + '_y' for col in update_cols]
                for col in update_cols:
                    df2.rename(columns={col + '_y': col}, inplace=True)
                    df2.drop(col + '_x', axis=1, inplace=True)
                grouped = df2.groupby('RDCKey')
            else:
                # histoty bottomup forecast
                grouped = df.groupby('RDCKey')
            result_list = []
            for rdc, history_df in grouped:
                if rdc in model.keys():
                    this_model = model[rdc]
                else:
                    continue
                ''' predict model '''
                xColumns = scenario['selectedColumns']['features']

                if 'RDCKey' in xColumns:  # 删除季节性,RDCKEY
                    xColumns.remove('skuDecomposedTrend')
                    xColumns.remove('skuDecomposedSeasonal')
                    xColumns.remove('level3DecomposedTrend')
                    xColumns.remove('level3DecomposedSeasonal')
                    xColumns.remove('Curve')
                    xColumns.remove('RDCKey')

                X_history = history_df[xColumns]

                history_xtest = xgb.DMatrix(X_history.values, missing=np.NaN)
                ypred = this_model.predict(history_xtest)
                history_df['ypred'] = ypred
                history_df['RDCKey'] = rdc

                ''' Tuning result '''
                lanjie = history_df[(history_df.ypred < 0)]
                if len(lanjie) > 0:
                    history_df.ix[lanjie.index, 'ypred'] = 0
                result_list.append(history_df)
            final_result = pd.concat(result_list)
            if use_promo == 'no':
                raw_train_df = final_result[
                    listkeys + reg_cols + scenario['promo_feature_cols'] + ['salesForecast', 'ypred']]
            else:
                # use_promo == 'mean':
                train_df_mean = final_result[
                    listkeys + reg_cols + scenario['promo_feature_cols'] + ['salesForecast', 'ypred']]
                train_df_mean.rename(columns={'ypred': 'ypred_mean_promo'}, inplace=True)
        return raw_train_df, train_df_mean


    raw_train_df, train_df_mean = predict_history_mean_and_raw(train_feature_df_new, reg_cols, list_keys, groupkeys,
                                                               model, update_cols, scenario)

    raw_train_df = pd.merge(raw_train_df, train_df_mean[list_keys + ['ypred_mean_promo']], how='left', on=list_keys)
    raw_test_df = q_pred_result
    raw_test_df = raw_test_df[list_keys + reg_cols + scenario['promo_feature_cols'] + ['salesForecast', 'ypred']]
    raw_test_df.Date = pd.to_datetime(raw_test_df.Date)

    used_cols = reg_cols + [
        'MaxSyntheticDiscountA']  # ['MaxBundleDiscount','MaxDirectDiscount','MaxDiscount','MaxSyntheticDiscountA','daynumberinpromotion','PromotionCount']
    raw_train_df.Date = pd.to_datetime(raw_train_df.Date)
    raw_train_df = raw_train_df[raw_train_df.Date < pred_date]
    raw_train_df = raw_train_df[list_keys + reg_cols + scenario['promo_feature_cols'] + ['ypred', 'ypred_mean_promo']]

    input_df = pd.concat([raw_train_df, raw_test_df])

    def process_lr_cols(input_df, cols):
        for col in cols:
            # input_df[col] = input_df.groupby(['RDCKey','ProductKey'])[col].transform(lambda x: x.fillna(method='bfill').fillna(0))
            # input_df[col] = input_df.groupby(['RDCKey','ProductKey'])[col].transform(lambda x: x.fillna(method='ffill').fillna(0))
            # input_df = input_df[input_df['MaxSyntheticDiscountA'].between(-1,1)]

            input_df = input_df[~(input_df[col].isnull())]
            input_df = input_df[input_df[col].between(-1, 1)]
        return input_df

    input_df = process_lr_cols(input_df, used_cols)

    def lr_promo_simulate(input_df, start_dt, used_cols, listkeys):
        value_type = 'ypred_mean_promo'
        final_df = pd.DataFrame()
        a = 1
        model_hash = {}
        grouped = input_df.groupby(['RDCKey', 'ProductKey'])
        for (rdc, sku), group in grouped:
            if not model_hash.has_key(rdc):
                model_hash[rdc] = {}
            if group.Date.min() < start_dt and group.Date.max() >= start_dt:
                print a
                a = a + 1
                train_df = group[group.Date < start_dt]
                test_df = group[group.Date >= start_dt]
                x_train_df = train_df[used_cols]
                x_test_df = test_df[used_cols]
                y_train = train_df['ypred'] - train_df[value_type]
                y_test = test_df['salesForecast']
                lm = LinearRegression()
                lm.fit(x_train_df, y_train)
                model_hash[rdc][sku] = lm
                Intercept = lm.intercept_
                RSquare = lm.score(x_train_df, y_train)
                lm_predict_result = lm.predict(x_test_df)
                test_result = pd.DataFrame()
                for col in listkeys + ['salesForecast', 'ypred']:
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
        return final_df, model_hash

    final_df, model_lr = lr_promo_simulate(input_df, ForecastStartDate, used_cols, list_keys)
    save_object(model_lr, 'model_save_lr/model_lr_' + str(cate) + '.pkl')

    mean_final = final_df
    mean_final.rename(columns={'reg_result': 'mean_promo_reg_result'}, inplace=True)
    final_df.drop('salesForecast', axis=1, inplace=True)
    final_df.drop('ypred', axis=1, inplace=True)
    new_df_final = new_df.merge(final_df, on=list_keys, how='left')
    new_df_final.fillna(0, inplace=True)
    new_df_final['ypred_mean_promo_new'] = new_df_final['ypred_mean_promo'] + new_df_final['mean_promo_reg_result']

    statitics_mape(new_df_final)

if __name__ == "__main__":
    main()
