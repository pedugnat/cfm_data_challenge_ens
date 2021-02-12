from scipy import stats
from collections import defaultdict

import scipy
import gc
import os
import re
import sys
import pip
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns

from typing import List, Tuple

pd.set_option("display.max_columns", 500)

NB_VENUES = 6
MICROSEC = int(1e6)
MIN_PER_HOUR = 60
SEC_PER_MIN = 60
TARGET = "source_id"
FEATURES_LIST = list()

subprocess.run("cat /etc/build_date".split())
subprocess.run("cat /etc/git_commit".split())
pip.main(['install', '--upgrade', "scipy", "-q"])

print(scipy.__version__)


def read_data_files(base_path: str, train_path: str, test_path: str, labels_path: str) -> Tuple[pd.DataFrame]:
    # Reading files
    train_data = pd.read_hdf(os.path.join(
        base_path, train_path), 'data').set_index("ID")
    test_data = pd.read_hdf(os.path.join(
        base_path, test_path),  'data').set_index("ID")

    train_labels = pd.read_csv(os.path.join(base_path, labels_path), dtype={
                               TARGET: np.int8}).set_index("ID")

    # Type casting
    types_conversion = {"stock_id": np.int16, "day_id": np.int16}
    types_conversion.update({(i, TARGET): np.int16 for i in range(10)})

    train_data = train_data.astype(types_conversion)

    # Concat x_train and y_train
    train_data = pd.concat([train_data, train_labels], axis=1)

    # Get rid of useless stocks
    train_stocks = set(train_data["stock_id"])
    test_stocks = set(test_data["stock_id"])

    useless_stocks = list(train_stocks.difference(test_stocks))
    train_data = train_data[~train_data["stock_id"].isin(useless_stocks)]

    del train_labels
    gc.collect()

    return train_data, test_data


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columns are named by tuples
    Replace tuples by Python style column names
    """
    OB_columns = ["bid", "bid1", "bid_size", "bid_size1",
                  "ask", "ask1", "ask_size", "ask_size1",
                  "ts_last_update"]
    new_col_names = list()

    for c in df.columns:
        if type(c) == tuple:
            is_OB_col = bool(c[1] in OB_columns)
            new_name = "_".join(
                ["OB" if is_OB_col else "trade", str(c[1]), str(c[0])])
            new_col_names.append(new_name)

        else:
            new_col_names.append(c)

    df.columns = new_col_names

    print("Columns renamed")

    return df


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def get_quarter_from_time(time_series: int) -> int:
    return int(2 * time_series / (MICROSEC * MIN_PER_HOUR * SEC_PER_MIN))


def get_minute_from_time(time_series: int) -> int:
    return int(time_series % (MICROSEC * MIN_PER_HOUR * SEC_PER_MIN) / (MICROSEC * SEC_PER_MIN))


def normalize_trades_time(df: pd.DataFrame) -> pd.DataFrame:
    """Get time of trades relative to the time of the latest trade
    (trade_tod_0 is always the most recent trade)
    as absolute time is not very meaningful
    """
    trade_time_columns = [
        c for c in df.columns if ("tod_" in c) and ("_0" not in c)]
    df[trade_time_columns] = df["trade_tod_0"].values.reshape(
        -1, 1) - df[trade_time_columns].values

    print("Trade times normalized")

    return df


def get_venues_frequency_last_trades(df: pd.DataFrame, k: int, nb_venues: int = NB_VENUES) -> pd.DataFrame:
    """ Count the number of times each venue was traded on
    for the last 10 trades available
    """
    source_cols = [c for c in df.columns if "_source_id" in c]

    for i in range(NB_VENUES):
        df[f"nb_{i}_{k}past_venues"] = np.count_nonzero(
            df[source_cols].values[:, :k] == i, axis=1).astype(np.int8)

    print(f"{k} last venues frequency computed")

    return df


def normalize_OB_updates(df: pd.DataFrame) -> pd.DataFrame:
    """ Computes the last updates of the order books
    relative to the one update most recently
    as absolute times are not very meaningful
    """
    OB_updates_cols = [c for c in df.columns if 'OB_ts_last_update_' in c]
    df[OB_updates_cols] = df[OB_updates_cols].max(
        axis=1).values.reshape(-1, 1) - df[OB_updates_cols]

    print("Order book normalized")

    return df


def compute_total_book_size(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the size of the book by adding the size of the 
    two first levels provided
    and compute the total size available, by summing bid and ask
    sizes
    """
    for i in range(NB_VENUES):
        df[f"ask_size_{i}"] = df[f"OB_ask_size_{i}"] + df[f"OB_ask_size1_{i}"]
        df[f"bid_size_{i}"] = df[f"OB_bid_size_{i}"] + df[f"OB_bid_size1_{i}"]

        df[f"total_size_{i}"] = df[f"ask_size_{i}"] + df[f"bid_size_{i}"]

    print("Total book size computed")

    return df


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """ Bid prices are negative relative to the mid-price, 
    so take the opposite in order to normalise towards ask
    """
    bid_price_cols = [c for c in df if re.match(r"OB_bid1?_\d{1}", c)]

    df[bid_price_cols] = - df[bid_price_cols]

    print("Bid prices normalized")

    return df


def rank_venues_by_price(df: pd.DataFrame, feats_to_rank: List[str] = [r"OB_ask_[0-5]", r"OB_bid_[0-5]", r"OB_ts", r"weighted_ask_", r"weighted_bid_"], rank_method="average") -> pd.DataFrame:
    """ Rank prices offered on each side by highest to lowest
    (non linear feature)
    """

    for col in feats_to_rank:
        OB_cols = [c for c in df.columns if re.match(col, c)]
        OB_ranked_cols = [c + "_rank" for c in OB_cols]
        assert len(OB_cols) == 6

        df[OB_ranked_cols] = df[OB_cols].rank(
            method=rank_method, axis=1, na_option="bottom")

    print("Venues ranked by price")

    return df


def make_stats_on_lasttrades(df: pd.DataFrame) -> pd.DataFrame:
    """ General descriptive stats on 10 last trades price and qty
    """
    # Price
    price_cols = [c for c in df.columns if "trade_price_" in c]
    df_prices = df[price_cols]
    df["num_lasttrades_ask_side"] = (df_prices > 0).sum(axis=1)
    df["mean_lasttrades_price"] = df_prices.mean(axis=1)
    df["std_lasttrades_price"] = df_prices.std(axis=1)
    df["min_lasttrades_price"] = df_prices.min(axis=1)
    df["max_lasttrades_price"] = df_prices.max(axis=1)

    # Quantity
    quantity_cols = [c for c in df.columns if "trade_qty_" in c]
    df_qty = df[quantity_cols]
    df["mean_lasttrades_qty"] = df_qty.mean(axis=1)
    df["std_lasttrades_qty"] = df_qty.std(axis=1)
    df["min_lasttrades_qty"] = df_qty.min(axis=1)
    df["max_lasttrades_qty"] = df_qty.max(axis=1)

    # Time
    time_cols = [c for c in df.columns if re.match("trade_tod_[1-9]", c)]
    df["std_lasttrades_time"] = df[time_cols].std(axis=1)

    print("Features on last trades computed")

    return df


def compare_subs(path_1: str, path_2: str) -> str:
    first_sub = pd.read_csv(path_1)
    second_sub = pd.read_csv(path_2)

    merged_sub = first_sub.merge(second_sub, on="ID")
    merged_sub["is_equal"] = merged_sub["source_id_x"] == merged_sub["source_id_y"]

    return f"Subs agreement = {merged_sub.is_equal.mean():.2%}"


def encode_OB_features(df: pd.DataFrame, pattern_cols_to_encode="OB_(bid|ask)1?_[0-5]", agg_names=["mean", "med", "std"]) -> pd.DataFrame:
    """ Group by stock and day and encode the mean, med and std of 
    OB-related features
    """
    agg_functions = [np.mean, np.median, np.std]
    stockday_cols = ["stock_id", "day_id"]
    OB_price_cols = [c for c in df.columns if re.match(
        pattern_cols_to_encode, c)]

    base_encoding = df.groupby(stockday_cols)[OB_price_cols].agg(agg_functions)
    base_encoding = base_encoding.reset_index()

    for aggreg in agg_names:
        encoding = base_encoding[[
            c for c in base_encoding.columns if aggreg in c[1]]]
        encoding.columns = ["_".join(c) + "_encoded" for c in encoding.columns]
        encoding = pd.concat(
            [encoding, base_encoding["stock_id"], base_encoding["day_id"]], axis=1)

        df = df.reset_index()
        df = df.merge(encoding, on=stockday_cols)
        df = df.set_index("ID")

    print(f"{len(OB_price_cols)} order book features encoded with pattern {pattern_cols_to_encode}")

    return df


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def get_weighted_qty_price(df: pd.DataFrame, level_coef: float = 0.5) -> pd.DataFrame:
    """Create some sort of indicator combining price and quantity for level 0 and 1
    to indicate whether the venue is attractive
    """
    max_price_books = int(1e5)
    for i in range(NB_VENUES):
        for side in ["bid", "ask"]:
            sq_inverse_price = (
                df[f"OB_{side}_{i}"].fillna(max_price_books) + 1.5).apply(lambda x: 1/x**2)
            sq_inverse_price_1 = (
                df[f"OB_{side}1_{i}"].fillna(max_price_books) + 1.5).apply(lambda x: 1/x**2)

            weighted_0 = sq_inverse_price * df[f"OB_{side}_size_{i}"].fillna(0)
            weighted_1 = sq_inverse_price_1 * \
                df[f"OB_{side}_size_{i}"].fillna(0)

            df[f"weighted_{side}_OB_price_{i}"] = weighted_0 + \
                level_coef * weighted_1

    print("Weighted quantity price feature computed")

    return df


def biggest_best_trade_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Get the source, price, time and qty of 
    the biggest trade (qty) and the best-price trade (price)
    """

    df_trades_qty = df[[
        c for c in df.columns if re.match(r"trade_qty_\d{1}", c)]]
    df["biggest_trade"] = np.argmax(df_trades_qty.values, axis=1)
    biggest_trade_index = pd.get_dummies(
        df["biggest_trade"].values).values.astype(bool)

    df_trades_price = df[[c for c in df.columns if re.match(
        r"trade_price_\d{1}", c)]].apply(abs)
    df["bestprice_trade"] = np.argmax(df_trades_price.values, axis=1)
    bestprice_trade_index = pd.get_dummies(
        df["bestprice_trade"].values).values.astype(bool)

    for feat in ["source_id", "price", "tod", "qty"]:
        trade_feats_values = df[[c for c in df.columns if re.match(
            fr"trade_{feat}_\d{{1}}", c)]].values
        df[f"{feat}_biggest_trade"] = trade_feats_values[biggest_trade_index]
        df[f"{feat}_bestprice_trade"] = trade_feats_values[bestprice_trade_index]

    print("Best trade features computed")

    return df


def visualize_feature_against_target(feature_name: str, df: pd.DataFrame, target=TARGET, axis_display=0) -> pd.DataFrame:
    cm = sns.light_palette("green", as_cmap=True)

    return (df[[feature_name, TARGET]]
            .groupby(TARGET)[feature_name]
            .apply(lambda x: x.value_counts() / x.shape[0])
            .to_frame()
            .unstack()
            .style
            .format("{:.1%}")
            .background_gradient(cmap=cm, axis=axis_display))


def make_relative_price_features(df: pd.DataFrame, patterns_column: List[str]) -> pd.DataFrame:
    """ Get the price relative of each venue relative to the best price 
    across all venues
    """
    for price_pattern in patterns_column:

        price_cols = [c for c in df.columns if re.match(price_pattern, c)]
        normalized_price_cols = ['normalized_' + c for c in price_cols]

        assert len(price_cols) == 6

        for norm_col in normalized_price_cols:
            df[norm_col] = 0    # create empty columns

        best_price_values = np.nanmin(
            (df[price_cols].values + 1.5), axis=1)[:, None]
        df[normalized_price_cols] = (
            df[price_cols].values + 1.5) / best_price_values

    print("Prices normalize vs. best done!")

    return df


def frequency_encode(variables_to_encode: List[str], df_train: pd.DataFrame, dfs_val: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str]]:
    encodings_dict = dict()
    new_features = list()

    for variable in variables_to_encode:
        share_per_venue = (df_train
                           .groupby(variable)[TARGET]
                           .apply(lambda x: x.value_counts() / x.shape[0])
                           .to_frame()
                           .unstack())

        FE_feats = [f"venue_FE_{variable}_{i}" for i in range(NB_VENUES)]
        share_per_venue.columns = FE_feats

        if FE_feats[0] not in FEATURES_LIST:
            FEATURES_LIST.extend(FE_feats)
            new_features.extend(FE_feats)

        encodings_dict[variable] = share_per_venue

    return encodings_dict, new_features


def make_ratios_price_size(df: pd.DataFrame, base_patterns=[r"OB_ask1?_", r"OB_bid1?_", r"OB_ask_size1?_", r"OB_bid_size1?_"]) -> pd.DataFrame:
    for base_pat in base_patterns:
        for i in range(NB_VENUES):
            pair_columns = [
                c for c in df.columns if re.match(f"{base_pat}{i}", c)]
            # if no size available (nan for qty), fill with 0
            if "size" in base_pat:
                pair_cols_values = df[pair_columns].fillna(0).values + 1.5
            # price column      # if no price available (nan for price), fill with huge price
            else:
                max_price_books = int(1e5)
                pair_cols_values = df[pair_columns].fillna(
                    max_price_books).values + 1.5

            df[f"ratio_01_{base_pat[3:]}_{i}"] = pair_cols_values[:,
                                                                  1] / pair_cols_values[:, 0]

    print("Ratios created!")
    return df


def make_lag_feature(df: pd.DataFrame) -> pd.DataFrame:
    sorted_data = df.sort_values(by=["stock_id", "day_id"])
    lagged_features = sorted_data.shift(1)
    lagged_features.columns = ["lagged_" + c for c in lagged_features.columns]
    df = pd.concat([sorted_data, lagged_features], axis=1)

    df["change_stock_day"] = (sorted_data["stock_id"] != sorted_data["stock_id"].shift(
        1)) | (sorted_data["day_id"] != sorted_data["day_id"].shift(1))
    lagged_cols = [c for c in df.columns if "lagged" in c]
    lagged_naned_data = df[lagged_cols].values
    lagged_naned_data[df["change_stock_day"], :] = np.nan

    df = df.drop("change_stock_day", axis=1)

    return df


def get_weighted_OB_share(df: pd.DataFrame, k_best_prices: int = 3) -> pd.DataFrame:
    """Divide each feat by the sum of the k best features of the row
    """
    column_sets = [r"^OB_ask_[0-5]$", r"^OB_bid_[0-5]$", r"^OB_ts_last_update_[0-5]$",
                   # r"^weighted_ask_OB_price_[0-5]$", r"^weighted_bid_OB_price_[0-5]$",  do with max
                   ]

    for col in column_sets:
        OB_cols = [c for c in df.columns if re.match(col, c)]
        assert len(OB_cols) == 6

        df_OB_cols = df[OB_cols].fillna(100_000).values

        partitioned_prices = np.partition(df_OB_cols, k_best_prices)[
            :, :k_best_prices]
        sum_kbest_prices = np.sum(partitioned_prices, axis=1).reshape(-1, 1)

        norm_cols_names = [c + "_sum_norm" for c in OB_cols]
        for col_to_create in norm_cols_names:
            df[col_to_create] = 0       # create empty columns

        df[norm_cols_names] = df_OB_cols / sum_kbest_prices

    print(f"OB features normalised by sum of {k_best_prices} best")

    return df


def smoothed_target_encoding(count_sources: pd.Series, global_count: pd.Series, reg_term: int = 100):
    """Simple formula to smooth target encoding when cardinal of the group is small
    """
    categ_card = count_sources.shape[0]
    raw_TE = count_sources.value_counts() / categ_card
    smoothing = categ_card / (categ_card + reg_term)
    final_TE = (smoothing * raw_TE).add((1 - smoothing) *
                                        global_count, fill_value=0)
#    final_TE = (1 - smoothing) * raw_TE + smoothing * global_count

    return final_TE


best_features = ['OB_ask_4_rank',
                 'OB_bid_4_rank',
                 'nb_3_5past_venues',
                 'normalized_OB_ask_2',
                 'OB_bid_5_rank',
                 'normalized_OB_bid_2',
                 'OB_bid_3',
                 'OB_ask_2_rank',
                 'OB_ask_5_rank',
                 'OB_ask_3',
                 'OB_ask_1_rank',
                 'OB_bid_1_rank',
                 'OB_bid1_3',
                 'OB_bid_2_rank',
                 'venue_FE_stock_id_2',
                 'normalized_OB_ask_3',
                 'normalized_OB_bid_3',
                 'trade_price_0',
                 'venue_FE_stock_id_4',
                 'nb_1_10past_venues',
                 'OB_ts_last_update_3_rank',
                 'nb_4_10past_venues',
                 'normalized_OB_bid_1',
                 'nb_3_10past_venues',
                 'nb_1_5past_venues']


def make_lag_feature(df: pd.DataFrame, feats_to_lag: List[str] = best_features) -> pd.DataFrame:
    """"Double features by giving the previous line for stock and day as a feature
    Doesn't improve the overall score but seems to be uncorrelated with "normal" one
    May be used in a final blend
    """
    sorted_data = df.sort_values(by=["stock_id", "day_id"])
    lagged_features = sorted_data.shift(1)
    lagged_features = lagged_features[[
        f for f in feats_to_lag if f in lagged_features.columns]]
    lagged_features.columns = ["lagged_" + c for c in lagged_features.columns]
    df = pd.concat([sorted_data, lagged_features], axis=1)

    df["change_stock_day"] = (sorted_data["stock_id"] != sorted_data["stock_id"].shift(
        1)) | (sorted_data["day_id"] != sorted_data["day_id"].shift(1))
    lagged_cols = [c for c in df.columns if "lagged" in c]
    lagged_naned_data = df[lagged_cols].values
    lagged_naned_data[df["change_stock_day"], :] = np.nan

    df = df.drop("change_stock_day", axis=1)

    return df


def get_weighted_trade(df: pd.DataFrame, k_trades: int = 3) -> pd.DataFrame:
    for j in range(k_trades):
        square_inv_price = (
            np.abs(df[f"trade_price_{j}"]) + 1).apply(lambda x: 1 / x ** 2)
        df[f"weighted_{j}_trade"] = square_inv_price * \
            df[f"trade_qty_{j}"].values

        df[f"inv_weighted_{j}_trade"] = df[f"trade_price_{j}"] * \
            df[f"trade_qty_{j}"].apply(lambda x: 1 / x)

    print("Weighted trades computation done!")

    return df


def is_lower_ticksize(df: pd.DataFrame) -> pd.DataFrame:
    tick_df = df[[c for c in df.columns if "trade_price" in c]] % 0.5
    odd_tick_size = (tick_df[f"trade_price_{i}"] > 0 for i in range(10))
    df["is_lower_ticksize"] = sum(odd_tick_size).astype(bool)

    print("Lower tick size done!")

    return df


def get_weighted_count(x: np.array, weight_vector: np.array, window_size: int, smoothing: int) -> float:
    n_ = x.shape[0]
    mid = window_size // 2
    lower_bound = mid - n_ // 2
    # to avoid leaks, exlude the current point from the count
    weight_vector[mid - smoothing: mid + smoothing] = 0
    weight_vect_resize = weight_vector[lower_bound:lower_bound + n_]

    weighted_count = np.sum(x @ weight_vect_resize) * window_size / n_

    return weighted_count


def rolling_time_embedding(train_df: pd.DataFrame, test_df: pd.DataFrame, weighting: str = "exp", window_size: int = 1000, smoothing: int = 10) -> pd.DataFrame:
    # prepare combined data from test and train
    time_cols = ["trade_tod_0", "source_id"]
    test_df["source_id"] = np.nan
    mixed_data = (pd.concat([train_df[time_cols], test_df[time_cols]])
                    .sort_values(by="trade_tod_0")
                  )

    if weighting == "square":
        weight_vector = 1 - np.linspace(-1, 1, window_size) ** 2
    elif weighting == "exp":
        weight_vector = (stats
                         .norm(loc=0, scale=.4)
                         .pdf(np.linspace(-1, 1, window_size)))

    rolling_vc = (pd.get_dummies(mixed_data["source_id"])
                    .rolling(window_size, center=True, min_periods=1)
                    .apply(lambda x: get_weighted_count(x, weight_vector, window_size, smoothing), raw=True))
    # return rolling_vc
    rol_val = rolling_vc.values
    rolling_vc[rolling_vc.columns] = rol_val / \
        rol_val.sum(axis=1).reshape(-1, 1)  # normalization step
    rolling_vc = rolling_vc.rolling(
        smoothing, min_periods=0, center=True).mean()  # last smoothing step
    rolling_vc.columns = [f"time_emb_{c}" for c in rolling_vc.columns]

    n_train = train_df.shape[0]
    train_df = pd.concat([train_df, rolling_vc], axis=1)[:n_train]

    n_test = test_df.shape[0]
    test_df = pd.concat([test_df, rolling_vc], axis=1)[-n_test:]
    test_df = test_df.drop("source_id", axis=1)

    print("Rolling time embedding done!")

    return train_df, test_df


def get_neighbors_trades(df: pd.DataFrame, neighbor_size: List[int]) -> pd.DataFrame:
    assert all(n % 2 == 1 for n in neighbor_size)    # needs to be odd

    stockday_columns = ["stock_id", "day_id"]
    source_cols = [c for c in df.columns if "trade_source_id_" in c]
    counts_per_venue = defaultdict(list)

    for k, df_gp in df.groupby(stockday_columns):
        for n_size in neighbor_size:

            mask = np.ones(n_size, dtype=bool)
            mask[n_size // 2] = 0    # don't take the current line into account

            rolled_df = df_gp.rolling(n_size, min_periods=n_size, center=True)[
                source_cols]
            for i in range(NB_VENUES):
                rolled_counted_df = (rolled_df
                                     .apply(lambda x: np.count_nonzero(x[mask] == i), raw=True)
                                     .sum(axis=1, min_count=1))
                rolled_counted_df.name = f"{n_size}neighb_trade_source_{i}"
                counts_per_venue[f"{n_size}venue_{i}"].append(
                    rolled_counted_df)

    nsize_neigh_trades = list()
    for n_size in neighbor_size:
        concat_count_cols = [
            pd.concat(counts_per_venue[f"{n_size}venue_{i}"]) for i in range(NB_VENUES)]
        neighb_trade = pd.concat(concat_count_cols, axis=1)
        nsize_neigh_trades.append(neighb_trade)

    neighb_trade_sources = pd.concat(
        nsize_neigh_trades, axis=1).astype(np.float32)

    df = df.merge(neighb_trade_sources, left_index=True, right_index=True)

    print(f"Added {'-'.join(neighbor_size)}-rows neighbors trades!")

    return df


def trades_last_n_sec(df: pd.DataFrame, limit_time: int) -> pd.DataFrame:
    """
    limit_time: int: time slots for which trades will be counted
    in microseconds
    so the function returns a count of the venues of the last trades
    within the limit_time time
    """

    time_cols = [c for c in df.columns if "trade_tod" in c]
    source_cols = [c for c in df.columns if "trade_source" in c]
    array_source = df[source_cols].values

    mask = (df[time_cols] > limit_time).values[:, 1:]
    array_source[:, 1:][mask] = -1

    seconds_venues = list()
    for venue in range(NB_VENUES):
        df[f"last_trades_{venue}_{limit_time / int(1e3):.0f}"] = np.count_nonzero(
            array_source == venue, axis=1)

    return df
