import pandas as pd
import numpy as np
import datetime
import os
import glob
import gc

from joblib import Parallel, delayed


from pytrend import Compustat_CRSP_Data


## Sklearn Transformers
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion

from pytrend import SignatureTransformer


pd.options.mode.chained_assignment = None

if not os.path.exists("features/"):
    os.mkdir("features/")


### Financials
if not os.path.exists("data/temp_financials/"):
    os.mkdir("data/temp_financials/")

### Ravenpack Sentiment
if not os.path.exists("data/temp_ravenpack/"):
    os.mkdir("data/temp_ravenpack/")


### Comparing different methods of price transforms

### Stats
if not os.path.exists("data/temp_stats/"):
    os.mkdir("data/temp_stats/")


### Signature
if not os.path.exists("data/temp_signature/"):
    os.mkdir("data/temp_signature/")


### Catch22
if not os.path.exists("data/temp_catch22/"):
    os.mkdir("data/temp_catch22/")


industry_list = pd.read_csv("../../data/Compustat_industry_code_2021.csv")
industry_list = industry_list[industry_list["gictype"] == "GSECTOR"]


numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv").dropna(
    subset=["hgsubind"]
)
numerai_targets = pd.read_parquet("data/numerai_signals_target_2021.parquet")


def add_industry_labels(CRSP_single_stock, sample_id, sample_id_type="permno"):
    ## Mapped History
    mapped_history = numerai_signals_metadata[
        numerai_signals_metadata[sample_id_type] == sample_id
    ]
    mapped_history["map_start"] = pd.to_datetime(mapped_history["map_start"])
    mapped_history["map_end"] = pd.to_datetime(mapped_history["map_end"])
    mapped_history["crsp_start"] = pd.to_datetime(mapped_history["crsp_start"])
    mapped_history["crsp_end"] = pd.to_datetime(mapped_history["crsp_end"])
    CRSP_single_stock["bloomberg_ticker"] = None
    CRSP_single_stock["group_subindustry"] = None
    for i, row in mapped_history.iterrows():

        if row["map_start"] == datetime.datetime(year=2007, month=4, day=14):
            valid_start = row["crsp_start"]
            valid_end = min(row["crsp_end"], row["map_end"])
        else:
            valid_start = max(row["crsp_start"], row["map_start"])
            valid_end = min(row["crsp_end"], row["map_end"])
        if valid_end > valid_start:
            CRSP_single_stock.loc[valid_start:valid_end, "group_subindustry"] = row[
                "hgsubind"
            ]
            CRSP_single_stock.loc[valid_start:valid_end, "bloomberg_ticker"] = row[
                "bloomberg_ticker"
            ]
    CRSP_single_stock.dropna(
        subset=["bloomberg_ticker", "group_subindustry"], inplace=True
    )
    if CRSP_single_stock.shape[0] > 0:
        ## Derive Group Labels
        CRSP_single_stock["group_subindustry"] = CRSP_single_stock[
            "group_subindustry"
        ].astype(int)
        CRSP_single_stock["group_industry"] = (
            CRSP_single_stock["group_subindustry"] // 100
        )
        CRSP_single_stock["group_sector"] = (
            CRSP_single_stock["group_subindustry"] // 1000000
        )
        ## Downsample to Friday
        shift = pd.to_datetime(CRSP_single_stock.index).dayofweek[0]
        subsampled = (
            CRSP_single_stock.fillna(method="pad")
            .resample("D")
            .fillna(method="pad", limit=31)[11 - shift :: 7]
        )
        subsampled["friday_date"] = subsampled.index.strftime("%Y%m%d").astype(int)
        subsampled["era"] = subsampled.index
        output = subsampled.set_index(["friday_date", "bloomberg_ticker"])
        outputmerged = output.merge(
            numerai_targets[
                [
                    "target_4d",
                    "target_20d",
                ]
            ],
            how="inner",
            left_index=True,
            right_index=True,
        )
        return outputmerged.dropna(
            subset=[
                "target_4d",
                "target_20d",
            ]
        )
    else:
        return pd.DataFrame()


## Feature Transformation per era


def transform_era(df, feature_cols, group_labels=None, keep_original=False):
    transformed_features = list()
    if group_labels is not None:
        for group in group_labels:
            group_features = list()
            for i, df_group in df.groupby(group):
                df_group_ranked = df_group[feature_cols].rank(pct=True, axis=0) - 0.5
                df_group_ranked.fillna(0, inplace=True)
                df_group_ranked = df_group_ranked * 5
                df_group_ranked.columns = [
                    "{}_{}_ranked".format(x, group) for x in feature_cols
                ]
                group_features.append(pd.concat([df_group_ranked], axis=1))
            group_features_df = pd.concat(group_features, axis=0)
            transformed_features.append(group_features_df)
    ## On All Data
    df_ranked = df[feature_cols].rank(pct=True, axis=0) - 0.5
    df_ranked.fillna(0, inplace=True)
    df_ranked = df_ranked * 5
    df_ranked.columns = ["{}_ranked".format(x) for x in feature_cols]
    transformed_features.append(df_ranked)
    if keep_original:
        transformed_features.append(df[feature_cols])
    transformed_df = pd.concat(transformed_features, axis=1)
    return transformed_df


### Process Financial Ratios from Open Source AP  (2000 to 2021)
if True:
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    financial_ratios = pd.read_parquet("data/numerai_financials_2021.parquet")
    financial_ratios["rawdatadate"] = pd.to_datetime(
        financial_ratios["yyyymm"], format="%Y%m"
    )

    for sample_id in numerai_signals_metadata["permno"].unique():
        print(sample_id)
        single_stock = financial_ratios[
            financial_ratios["permno"] == sample_id
        ].sort_values("rawdatadate")
        if single_stock.shape[0] > 0:
            ## Data calculated at the end of month can be used for the following month
            single_stock["datadate"] = single_stock["rawdatadate"].shift(-1)
            single_stock.drop(
                [
                    "rawdatadate",
                    "permno",
                    "yyyymm",
                ],
                axis=1,
                inplace=True,
            )
            single_stock.dropna(subset=["datadate"], inplace=True)
            single_stock_daily = (
                single_stock.set_index("datadate").resample("D").asfreq()
            )
            single_stock_daily = single_stock_daily.add_prefix("feature_")
            ans = add_industry_labels(single_stock_daily.copy(), sample_id)
            if ans.shape[0] > 0:
                output = ans[ans["era"] <= "2021-12-31"]
                output.to_parquet(
                    f"data/temp_financials/financials_{sample_id}.parquet"
                )

    del financial_ratios
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_financials/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_financials.parquet"
    )

    ## Financials
    raw_features = pd.read_parquet(
        "data/numerai_signals_features_financials.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        df_new = df[~df.index.duplicated(keep=False)]
        print(i)
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [
            col for col in raw_features.columns if col.startswith("group_sector")
        ]
        normalised.append(
            transform_era(df_new, feature_cols=feature_cols, group_labels=group_labels)
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_financials_normalised.parquet"
    )


### Ravenpack
def read_ravenpack_equities(
    ra_folder="../../data/Ravenpack", startyear=2000, endyear=2021, rp_entity_ids=None
):
    ravenpacks = list()
    for year in range(startyear, endyear + 1):
        for month in range(1, 13):
            filename = f"{ra_folder}/Ravenpack_equities_{year}_{month}.parquet"
            ravenpack = pd.read_parquet(filename)
            ravenpack = ravenpack[ravenpack["rp_entity_id"].isin(rp_entity_ids)]
            print(f"Reading Ravepack Equities {year} {month}")
            drop_cols = [
                "headline",
                "rpa_time_utc",
                "timestamp_utc",
                "rp_story_id",
                "product_key",
                "provider_id",
                "provider_story_id",
                "rp_story_event_index",
                "rp_story_event_count",
                "news_type",
                "rp_source_id",
                "source_name",
                "rp_position_id",
                "position_name",
            ]
            ravenpack_small = ravenpack.drop(drop_cols, axis=1)
            ## Filter important events
            ravenpack_important = ravenpack_small[
                (ravenpack_small["event_relevance"] >= 100)
                & (ravenpack_small["event_similarity_days"] >= 1)
                & (ravenpack_small["event_sentiment_score"] != 0)
            ]
            ## Summarise data by event similar keys
            ravenpacks.append(ravenpack_important)
    return pd.concat(ravenpacks, axis=0)


## Get a daily summary of sentiment based on events in the last trading week,month,

import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class RavenpackSentimentTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, reference_categories_file="features/ravenpack_category.csv"):
        self.top_categories = (
            pd.read_csv(reference_categories_file, index_col=0).head(200).index
        )

    def transform(self, X):

        output_cols = list()

        X["rpa_date_utc"] = pd.to_datetime(X["rpa_date_utc"])
        daily_newssentiment = (
            X.groupby("rpa_date_utc")[["event_sentiment_score"]]
            .mean()
            .resample("B")
            .mean()
            .fillna(0)
        )
        dailyX = pd.DataFrame(index=daily_newssentiment.index)

        ## Event Sentiment
        for lookback in [
            1,
            21,
            63,
            252,
        ]:
            output_col = f"rp_EventSentiment_{lookback}"
            dailyX[output_col] = daily_newssentiment.rolling(lookback).mean()
            output_cols.append(output_col)

        ## Count by Category
        for category in self.top_categories:
            X_category = X[X["category"] == category]
            daily_categorysentiment = (
                X_category.groupby("rpa_date_utc")[["event_sentiment_score"]]
                .mean()
                .resample("B")
                .mean()
                .fillna(0)
            )
            ## Event Sentiment
            for lookback in [
                252,
            ]:
                output_col = f"rp_Sentiment_{category}_{lookback}"
                dailyX[output_col] = daily_categorysentiment.rolling(lookback).mean()
                output_cols.append(output_col)

        output = dailyX[output_cols]
        return output.add_prefix("feature_").fillna(0)


### Process Ravenpack
if True:
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    rp_entity_ids = numerai_signals_metadata["rp_entity_id"].unique()
    ravenpack = read_ravenpack_equities(rp_entity_ids=rp_entity_ids)
    for sample_id in rp_entity_ids:
        print(sample_id)
        sample = ravenpack[ravenpack["rp_entity_id"] == sample_id]
        if sample.shape[0] > 0:
            transformer = RavenpackSentimentTransformer()
            output = transformer.transform(sample)
            ans = add_industry_labels(
                output.copy(),
                sample_id,
                sample_id_type="rp_entity_id",
            )
            if ans.shape[0] > 0:
                ans.to_parquet(f"data/temp_ravenpack/ravenpack_{sample_id}.parquet")

    del ravenpack
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_ravenpack/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_ravenpack.parquet"
    )

    del ans_list
    gc.collect()

    raw_features = pd.read_parquet(
        "data/numerai_signals_features_ravenpack.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [
            col for col in raw_features.columns if col.startswith("group_sector")
        ]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_ravenpack_normalised.parquet"
    )

##### Compare Different Transformation applied on price data (OHLC)

## Calculate Signatures
## Transform a single asset into signatures and zscores


class CompustatSignatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        no_channels,
        signature_level,
    ):
        self.no_channels = no_channels
        self.signature_level = signature_level
        self.signaturetransformer_fast = SignatureTransformer(
            no_channels, 21, signature_level
        )
        self.signaturetransformer_mid = SignatureTransformer(
            no_channels, 63, signature_level
        )
        self.signaturetransformer_slow = SignatureTransformer(
            no_channels, 252, signature_level
        )

    def transform(self, X):
        X["average_price"] = (
            (
                X["adjusted_open"]
                + X["adjusted_close"]
                + X["adjusted_high"]
                + X["adjusted_low"]
            )
            / 4
        ).astype(float)
        X["average_price"] = np.log(X["average_price"])
        selected_cols = ["average_price"]
        for smooth in [5, 21]:
            X[f"average_price_{smooth}"] = X["average_price"].rolling(smooth).mean()
            selected_cols.append(f"average_price_{smooth}")
        signatures_fast = self.signaturetransformer_fast.transform(
            X[selected_cols].astype(float).dropna()
        )
        signatures_mid = self.signaturetransformer_mid.transform(
            X[selected_cols].astype(float).dropna()
        )
        signatures_slow = self.signaturetransformer_slow.transform(
            X[selected_cols].astype(float).dropna()
        )
        output = pd.concat(
            [
                signatures_fast,
                signatures_mid,
                signatures_slow,
            ],
            axis=1,
        )
        return output.dropna().add_prefix("feature_")


if False:
    CRSP_data = Compustat_CRSP_Data(
        debug=True,
        startyear=2000,
        endyear=2021,
        sectors=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        market="Numerai",
        use_option_volume=False,
        use_vol_surface=False,
        use_fundamentals=False,
        quantile=1,
    )
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    permnos = numerai_signals_metadata["permno"].unique()

    def joblib_signature(CRSP_data, sample_id):
        try:
            CRSP_single_stock = CRSP_data.xs(sample_id, level=1, axis=1).dropna(
                subset=["market_cap"]
            )
            transformer = CompustatSignatureTransformer(3, 4)
            output = transformer.transform(
                CRSP_single_stock[
                    [
                        "adjusted_open",
                        "adjusted_close",
                        "adjusted_high",
                        "adjusted_low",
                    ]
                ]
                .dropna()
                .astype(float)
            )
            ans = add_industry_labels(output.copy(), sample_id)
            if ans.shape[0] > 0:
                ans.to_parquet(f"data/temp_signature/signature_{sample_id}.parquet")
            return None
        except:
            print(f"Missing Data {sample_id}")
            return None

    ## Transform for each stock
    ##results = Parallel(n_jobs=5)(delayed(joblib_signature)(CRSP_data,sample_id) for sample_id in permnos)
    for sample_id in permnos:
        joblib_signature(CRSP_data, sample_id)

    del CRSP_data
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_signature/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_signature.parquet"
    )

    del ans_list
    gc.collect()

    raw_features = pd.read_parquet(
        "data/numerai_signals_features_signature.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [
            col for col in raw_features.columns if col.startswith("group_sector")
        ]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_signature_normalised.parquet"
    )


## Calculate Catch22
## Transform a single asset into signatures and zscores

import pycatch22


class CompustatCatch22Transformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def transform(self, X):

        if X.shape[0] > 252:
            X["average_price"] = (
                (
                    X["adjusted_open"]
                    + X["adjusted_close"]
                    + X["adjusted_high"]
                    + X["adjusted_low"]
                )
                / 4
            ).astype(float)
            X["average_price"] = np.log(X["average_price"])

            catch22_short = list()
            catch22_mid = list()
            catch22_long = list()

            for df in X["average_price"].rolling(21):
                if df.shape[0] >= 21:
                    temp = pycatch22.catch22_all(df.values)
                    col_names = temp["names"]
                    catch22_short.append(temp["values"])
                else:
                    catch22_short.append([np.nan for i in range(22)])
            catch22_short_df = pd.DataFrame(catch22_short, columns=col_names)
            catch22_short_df = catch22_short_df.add_suffix("_short")
            catch22_short_df.index = X.index

            for df in X["average_price"].rolling(63):
                if df.shape[0] >= 63:
                    temp = pycatch22.catch22_all(df.values)
                    col_names = temp["names"]
                    catch22_mid.append(temp["values"])
                else:
                    catch22_mid.append([np.nan for i in range(22)])
            catch22_mid_df = pd.DataFrame(catch22_mid, columns=col_names)
            catch22_mid_df = catch22_mid_df.add_suffix("_mid")
            catch22_mid_df.index = X.index

            for df in X["average_price"].rolling(252):
                if df.shape[0] >= 252:
                    temp = pycatch22.catch22_all(df.values)
                    col_names = temp["names"]
                    catch22_long.append(temp["values"])
                else:
                    catch22_long.append([np.nan for i in range(22)])
            catch22_long_df = pd.DataFrame(catch22_long, columns=col_names)
            catch22_long_df = catch22_long_df.add_suffix("_long")
            catch22_long_df.index = X.index

            output = pd.concat(
                [
                    catch22_short_df,
                    catch22_mid_df,
                    catch22_long_df,
                ],
                axis=1,
            )
            return output.dropna().add_prefix("feature_")
        else:
            return pd.DataFrame()


if True:
    CRSP_data = Compustat_CRSP_Data(
        debug=True,
        startyear=2000,
        endyear=2021,
        sectors=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        market="Numerai",
        use_option_volume=False,
        use_vol_surface=False,
        use_fundamentals=False,
        quantile=1,
    )
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    permnos = numerai_signals_metadata["permno"].unique()

    def joblib_catch22(CRSP_data, sample_id):
        CRSP_single_stock = CRSP_data.xs(sample_id, level=1, axis=1).dropna(
            subset=["market_cap"]
        )
        transformer = CompustatCatch22Transformer()
        output = transformer.transform(
            CRSP_single_stock[
                [
                    "adjusted_open",
                    "adjusted_close",
                    "adjusted_high",
                    "adjusted_low",
                ]
            ]
            .dropna()
            .astype(float)
        )
        ans = add_industry_labels(output.copy(), sample_id)
        if ans.shape[0] > 0:
            ans.to_parquet(f"data/temp_catch22/catch22_{sample_id}.parquet")
        return None

    for sample_id in permnos:
        try:
            joblib_catch22(CRSP_data, sample_id)
        except:
            print("Missing ID")

    del CRSP_data
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_catch22/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_catch22.parquet"
    )

    del ans_list
    gc.collect()

    raw_features = pd.read_parquet(
        "data/numerai_signals_features_catch22.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [
            col for col in raw_features.columns if col.startswith("group_sector")
        ]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_catch22_normalised.parquet"
    )

### Stats
class CompustatStatsTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def transform(self, X):

        X["average_price"] = (
            (
                X["adjusted_open"]
                + X["adjusted_close"]
                + X["adjusted_high"]
                + X["adjusted_low"]
            )
            / 4
        ).astype(float)

        log_returns = np.log(X["average_price"]) - np.log(X["average_price"].shift(1))

        output_cols = list()

        ## Momentum
        for lookback in [21, 63, 252]:
            output_col = f"momentum_{lookback}"
            X[output_col] = log_returns.rolling(lookback).sum()
            output_cols.append(output_col)

        ## Volatility
        for lookback in [21, 63, 252]:
            output_col = f"volatility_{lookback}"
            X[output_col] = log_returns.rolling(lookback).std() * np.sqrt(
                252 / lookback
            )
            output_cols.append(output_col)

        ## Skewness
        for lookback in [21, 63, 252]:
            output_col = f"skewness_{lookback}"
            X[output_col] = log_returns.rolling(lookback).skew()
            output_cols.append(output_col)

        ## Kurtosis
        for lookback in [21, 63, 252]:
            output_col = f"kurtosis_{lookback}"
            X[output_col] = log_returns.rolling(lookback).kurt()
            output_cols.append(output_col)

        output = X[output_cols]

        return output.dropna().add_prefix("feature_")


if True:
    CRSP_data = Compustat_CRSP_Data(
        debug=True,
        startyear=2000,
        endyear=2021,
        sectors=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        market="Numerai",
        use_option_volume=False,
        use_vol_surface=False,
        use_fundamentals=False,
        quantile=1,
    )
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    permnos = numerai_signals_metadata["permno"].unique()

    def joblib_stats(CRSP_data, sample_id):
        CRSP_single_stock = CRSP_data.xs(sample_id, level=1, axis=1).dropna(
            subset=["market_cap"]
        )
        transformer = CompustatStatsTransformer()
        output = transformer.transform(
            CRSP_single_stock[
                [
                    "adjusted_open",
                    "adjusted_close",
                    "adjusted_high",
                    "adjusted_low",
                ]
            ]
            .dropna()
            .astype(float)
        )
        ans = add_industry_labels(output.copy(), sample_id)
        if ans.shape[0] > 0:
            ans.to_parquet(f"data/temp_stats/stats_{sample_id}.parquet")
        return None

    for sample_id in permnos:
        try:
            joblib_stats(CRSP_data, sample_id)
        except:
            print("Missing ID")

    del CRSP_data
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_stats/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_stats.parquet"
    )

    del ans_list
    gc.collect()

    raw_features = pd.read_parquet(
        "data/numerai_signals_features_stats.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [
            col for col in raw_features.columns if col.startswith("group_sector")
        ]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_stats_normalised.parquet"
    )


## Calculate Basic Factors
class CompustatBasicTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def transform(self, X):
        ## Price Based Factors
        log_returns = np.log(1 + X["return"].astype(float))
        market_cap = X["market_cap"].astype(float)

        output_cols = list()

        ## Momentum
        for lookback in [21, 63, 126, 252]:
            output_col = f"momentum_{lookback}"
            X[output_col] = log_returns.rolling(lookback).sum()
            output_cols.append(output_col)

        ## Volatility
        for lookback in [63, 126, 252]:
            output_col = f"volatility_{lookback}"
            X[output_col] = log_returns.rolling(lookback).std() * np.sqrt(
                252 / lookback
            )
            output_cols.append(output_col)

        ## Z-scores
        for lookback in [40, 80, 120, 160, 200]:
            output_col = f"zscore_{lookback}"
            X[output_col] = (
                market_cap - market_cap.rolling(lookback).mean()
            ) / market_cap.rolling(lookback).std()
            output_cols.append(output_col)

        ## Sharpe
        for lookback in [21, 63, 126, 252]:
            output_col = f"sharpe_{lookback}"
            X[output_col] = (
                log_returns.rolling(lookback).mean() / log_returns.rolling(252).std()
            )
            output_cols.append(output_col)

        ## Skewness
        for lookback in [63, 252]:
            output_col = f"skewness_{lookback}"
            X[output_col] = log_returns.rolling(lookback).skew()
            output_cols.append(output_col)

        ## Liquidity,
        for lookback in [63, 252]:
            output_col = f"liquidity_{lookback}"
            X[output_col] = (
                X["dollar_volume"].rolling(lookback).mean() / X["market_cap"]
            )
            output_cols.append(output_col)

        ## Short Interest
        for lookback in [21, 63, 252]:
            output_col = f"shortint_{lookback}"
            X[output_col] = (
                (X["shortint"].fillna(method="pad") * X["close"] / X["market_cap"])
                .rolling(lookback)
                .mean()
            )
            output_cols.append(output_col)

        output = X[output_cols]

        return output.dropna().add_prefix("feature_")


if False:
    CRSP_sample = Compustat_CRSP_Data(
        debug=True,
        startyear=2000,
        endyear=2021,
        sectors=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        market="Numerai",
        use_option_volume=False,
        use_vol_surface=False,
        use_fundamentals=False,
        quantile=1,
    )
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    permnos = numerai_signals_metadata["permno"].unique()
    ## Transform for each stock
    for sample_id in permnos:
        print(sample_id)
        try:
            CRSP_single_stock = CRSP_sample.xs(sample_id, level=1, axis=1).dropna(
                subset=["market_cap"]
            )
            transformer = CompustatBasicTransformer()
            output = transformer.transform(CRSP_single_stock.copy())
            ans = add_industry_labels(output.copy(), sample_id)
            if ans.shape[0] > 0:
                ans.to_parquet(f"data/temp_basic/basic_{sample_id}.parquet")
        except:
            print(f"Stock with no data {sample_id}")

    del CRSP_sample
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_basic/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_basic.parquet"
    )

    del ans_list
    gc.collect()

    raw_features = pd.read_parquet(
        "data/numerai_signals_features_basic.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [
            col for col in raw_features.columns if col.startswith("group_sector")
        ]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_basic_normalised.parquet"
    )
