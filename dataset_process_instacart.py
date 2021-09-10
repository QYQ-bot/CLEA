import pandas as pd
import gc
import random
import re
import json
import time
import numpy as np  # linear algebra
from datetime import date
import math

import tarfile
import os


def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)


def load_data():
    # train = pd.read_csv('./instacart_2017_05_01/order_products__train.csv',
    #                 dtype={
    #                         'order_id': np.int32,
    #                         'product_id': np.uint16,
    #                         'add_to_cart_order': np.int16,
    #                         'reordered': np.int8})
    prior = pd.read_csv('./instacart_2017_05_01/order_products__prior.csv',
                        dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    orders = pd.read_csv('./instacart_2017_05_01/orders.csv',
                         dtype={
                             'order_id': np.int32,
                             'user_id': np.int64,
                             'eval_set': 'str',
                             'order_number': np.int16,
                             'order_dow': np.int8,
                             'order_hour_of_day': np.int8,
                             'days_since_prior_order': np.float32})
    # products = pd.read_csv('./instacart_2017_05_01/products.csv')
    # aisles = pd.read_csv('./instacart_2017_05_01/aisles.csv')
    return prior, orders
    # return train, orders, products,aisles,prior


def merge_data(prior, orders):
    ratio = 0.1
    # random.seed(11)
    test_orders = orders[orders.eval_set == 'test']
    test_userid_list = list(set(test_orders['user_id'].tolist()))
    # test_userid_list = random.sample(test_userid_list,math.ceil(ratio*len(test_userid_list)))
    test_userid_list = test_userid_list[:math.ceil(ratio * len(test_userid_list))]

    orders = orders[orders.user_id.isin(test_userid_list)]

    orders = orders[orders.eval_set == 'prior']
    mt = pd.merge(orders, prior, on=['order_id', 'order_id'])

    return mt


def get_data_list(mt_df):
    lines = []
    all = mt_df.shape[0]
    for idx, row in mt_df.iterrows():
        if idx % 10000 == 0:
            print('{}/{}'.format(idx, all))
        userid = row['user_id']
        itemid = row['product_id']
        orderid = row['order_number']
        line = [userid, itemid, orderid]
        lines.append(line)
    return lines


def get_should_dele_data(data_list, k):
    user_dict = {}
    item_dict = {}
    user_date_tran_dict = {}  # user:{data:[item1,item2]}

    for row in data_list:
        userid = row[0]
        itemid = row[1]
        dateid_int = int(row[2])

        if userid not in user_date_tran_dict.keys():
            user_date_tran_dict[userid] = {}
        if dateid_int not in user_date_tran_dict[userid].keys():
            user_date_tran_dict[userid][dateid_int] = []
        user_date_tran_dict[userid][dateid_int].append(itemid)

        if userid not in user_dict.keys():
            user_dict[userid] = []
        if itemid not in item_dict.keys():
            item_dict[itemid] = []
        if itemid not in user_dict[userid]:
            user_dict[userid].append(itemid)
        if userid not in item_dict[itemid]:
            item_dict[itemid].append(userid)

    should_dele_user = []
    should_dele_item = []
    for userid in user_dict.keys():
        if len(user_dict[userid]) < k:
            should_dele_user.append(userid)
        elif len(list(user_date_tran_dict[userid].keys())) < 2:
            should_dele_user.append(userid)
    for itemid in item_dict.keys():
        if len(item_dict[itemid]) < k:
            should_dele_item.append(itemid)

    all_user_num = len(list(user_dict.keys()))
    all_item_num = len(list(item_dict.keys()))
    return should_dele_user, should_dele_item, all_user_num, all_item_num


def get_remain_data_list(data_list, k):
    should_dele_user, should_dele_item, all_user_num, all_item_num = get_should_dele_data(data_list, k)
    print('should_dele_user_num:{}  should_dele_item_num:{}  all_user_num:{}  all_item_num:{}  tran_num:{}'.format(
        len(should_dele_user),
        len(should_dele_item),
        all_user_num,
        all_item_num,
        len(data_list)))
    dele_num = len(should_dele_user) + len(should_dele_item)
    if len(should_dele_user) + len(should_dele_item) == 0:
        return data_list, dele_num
    remain_data_list = []
    for row_idx, row in enumerate(data_list):
        if row_idx % 1000 == 0:
            print(row_idx)
        userid = row[0]
        itemid = row[1]
        flag = 0
        if (userid in should_dele_user) | (itemid in should_dele_item):
            flag = 1
        if flag == 0:
            remain_data_list.append(row)
    return remain_data_list, dele_num


def get_ID_user_date_tran_dict(data_list):
    user_ID_dict = {}
    user_count = 0
    item_ID_dict = {}
    item_count = 0

    user_date_tran_dict = {}  # user:{data:[item1,item2]}
    for row in data_list:
        userid = row[0]
        itemid = row[1]
        dateid_int = int(row[2])

        if userid not in user_ID_dict.keys():
            user_ID_dict[userid] = user_count
            user_count += 1
        if itemid not in item_ID_dict.keys():
            item_ID_dict[itemid] = item_count
            item_count += 1

        if userid not in user_date_tran_dict.keys():
            user_date_tran_dict[userid] = {}
        if dateid_int not in user_date_tran_dict[userid].keys():
            user_date_tran_dict[userid][dateid_int] = []
        user_date_tran_dict[userid][dateid_int].append(itemid)

    user_order_date_tran_dict = {}  # user:{ [ b1 b2 b3      ]}
    for userid in user_date_tran_dict.keys():
        user_order_date_tran_dict[user_ID_dict[userid]] = []
        dateid_list = list(user_date_tran_dict[userid].keys())
        index = sorted(range(len(dateid_list)), key=lambda k: dateid_list[k])
        for idx in index:
            date_items = [item_ID_dict[itd] for itd in user_date_tran_dict[userid][dateid_list[idx]]]

            user_order_date_tran_dict[user_ID_dict[userid]].append(date_items)
    return user_order_date_tran_dict, user_ID_dict, item_ID_dict


def all_process(data_list, k):
    remain_data_list, dele_num = get_remain_data_list(data_list, k)
    while dele_num > 0:
        remain_data_list, dele_num = get_remain_data_list(remain_data_list, k)
    user_order_date_tran_dict, user_ID_dict, item_ID_dict = get_ID_user_date_tran_dict(remain_data_list)
    return user_order_date_tran_dict, user_ID_dict, item_ID_dict


def get_satistic(encoded_user_basket_tag_dict, user_dict, item_dict):
    avg_basket_size = 0
    basket_num = 0
    max_basket_length = 0
    max_basket_num = 0
    for userid in encoded_user_basket_tag_dict.keys():
        basket_seq = encoded_user_basket_tag_dict[userid]
        basket_num += len(basket_seq)
        if max_basket_num < len(basket_seq):
            max_basket_num = len(basket_seq)
        for basket in basket_seq:
            avg_basket_size += len(basket)
            if max_basket_length < len(basket):
                max_basket_length = len(basket)
    avg_user_basket_num = basket_num / len(user_dict.keys())
    avg_basket_size = avg_basket_size / basket_num
    users_num = len(user_dict.keys())
    items_num = len(item_dict.keys())
    return users_num, items_num, basket_num, avg_basket_size, avg_user_basket_num, max_basket_length, max_basket_num


if __name__ == "__main__":
    f = open('./propcessed_data/user_date_tran_dict_v4.txt', 'w')
    f_i = open('./propcessed_data/item_dict_v4.txt', 'w')
    f_u = open('./propcessed_data/user_dict_v4.txt', 'w')

    #######解压部分####################
    untar("./raw_data/instacart_online_grocery_shopping_2017_05_01.tar.gz", ".")

    prior, orders = load_data()
    mt = merge_data(prior, orders)

    del (mt['order_id'])
    del (mt['add_to_cart_order'])
    del (mt['reordered'])
    del (mt['eval_set'])
    del (mt['order_dow'])
    del (mt['order_hour_of_day'])
    del (mt['days_since_prior_order'])

    # mt.to_csv('./propcessed_data/user_date_tran.csv')
    data_list = get_data_list(mt)

    encoded_user_basket_tag_dict, user_dict, item_dict = all_process(data_list,
                                                                     10)  # userid:[[tag_basket_1],[tag_basket_2],[tag_basket_3]]
    users_num, items_num, basket_num, avg_basket_size, avg_user_basket_num, max_basket_length, max_basket_num = \
        get_satistic(encoded_user_basket_tag_dict, user_dict, item_dict)

    print(
        'users_num:{}  items_num:{}  baskets_num:{}  avg_basket_size:{}  avg_user_basket_num:{}  max_basket_length:{} max_basket_num:{}'.format(
            users_num,
            items_num,
            basket_num,
            avg_basket_size,
            avg_user_basket_num,
            max_basket_length, max_basket_num))
    f.write(str(encoded_user_basket_tag_dict))
    f.close()
    f_i.write(str(item_dict))
    f_i.close()
    f_u.write(str(user_dict))
    f_u.close()

    print('*')

    '''
    ratio=0.1 K=10  user_date_tran_dict_v4
    users_num:6886  items_num:8222  baskets_num:112503  avg_basket_size:9.204465658693547  avg_user_basket_num:16.337932036015104  max_basket_length:67 max_basket_num:99
    '''
