#!/usr/bin/env python
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

import zlib
import base64
import pandas as pd

p1_df_names = ['productkey', 'promotionkey', 'startdatetime', 'enddatetime', 'jdprice', 'syntheticgrossprice', 'promotiondesc', 'promotiondesc_flag', 'promotiontype', 'promotionsubtype',
               'areatypearray', 'tokenflag', 'directdiscount_discount', 'directdiscount_availabilitynumber', 'bundle_subtype1_threshold', 'bundle_subtype1_giveaway',
               'bundle_subtype4_threshold1', 'bundle_subtype4_giveaway1', 'bundle_subtype4_threshold2', 'bundle_subtype4_giveaway2', 'bundle_subtype4_threshold3',
               'bundle_subtype4_giveaway3', 'bundle_subtype2_threshold', 'bundle_subtype2_giveaway', 'bundle_subtype2_maximumgiveaway', 'bundle_subtype15_thresholdnumber1',
               'bundle_subtype15_giveawayrate1', 'bundle_subtype15_thresholdnumber2', 'bundle_subtype15_giveawayrate2', 'bundle_subtype15_thresholdnumber3',
               'bundle_subtype15_giveawayrate3', 'bundle_subtype6_thresholdnumber', 'bundle_subtype6_freenumber', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool',
               'suit_discount', 'directdiscount_saleprice', 'bundle_subtype1_percent', 'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent',
               'bundle_subtype6_percent', 'suit_percent', 'allpercentdiscount', 'mainproductkey', 'hierarchylevel3key', 'createdate', 'statuscode', 'dt']
p1_df_datetime_cols = ["startdatetime", "enddatetime", "dt"]
p1_df_numeric_cols = ['jdprice', 'promotiondesc_flag', 'promotionsubtype', 'directdiscount_availabilitynumber', 'bundle_subtype15_giveawayrate1', 'bundle_subtype15_giveawayrate2',
                     'bundle_subtype15_giveawayrate3', 'suit_maxvaluepool', 'suit_minvaluepool', 'suit_avgvaluepool', 'directdiscount_saleprice', 'bundle_subtype1_percent',
                     'bundle_subtype4_percent', 'bundle_subtype2_percent', 'bundle_subtype15_percent', 'bundle_subtype6_percent', 'suit_percent', 'allpercentdiscount']

fr = open("./tmp/13765_2018/p_13770_p1.da")
line = fr.readline()
v = line.split(",")[1]
str_data = v
li_df_cols = p1_df_names
li_df_numeric_cols = p1_df_numeric_cols
li_df_datetime_cols = p1_df_datetime_cols

li_data = str_data.split("@")

li_result = []
for str_data_item in li_data:
    str_data_zip = base64.b64decode(str_data_item)
    str_data_unzip = zlib.decompress(str_data_zip, zlib.MAX_WBITS | 16)
    li_row = str_data_unzip.split("\t")
    # 处理null值
    for i in range(0, len(li_row)):
        if li_row[i] in ['NULL', 'null', 'None', 'none', 'NaN', 'nan', '']:
            if li_df_cols[i] in li_df_numeric_cols:
                li_row[i] = -999
            else:
                li_row[i] = 'null'
    li_result.append(li_row)
pdf = pd.DataFrame(li_result, columns=li_df_cols, dtype=float)
# 处理日期值
for str_col in li_df_datetime_cols:
    pdf[str_col] = pd.to_datetime(pdf[str_col])

period_promo_raw_clean = pdf
train_pred_gate='train'
scenarioSettingsPath = 'code/refactor/ow_scenario.yaml'
scenario = loadSettingsFromYamlFile(scenarioSettingsPath)
area_rdc_map = pd.read_csv('/home/ubuntu/yulong/promotion_offline/tmp/ow_deploy_single/area_rdc_mapping.csv')
import ow_f01
f01_output = ow_f01.generate_f01_promo(area_rdc_map, period_promo_raw_clean,scenario, train_pred_gate)
f01_output.to_csv('tmp/fushi/12015/p1_out.csv',index=False)


