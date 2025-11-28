import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

HOLIDAY_PRIORITY = {
    'Holiday': 2,
    'Bridge': 4,
    'Transfer': 5,
    'Additional': 6,
    'Work Day': 1,
    'Event': 3
}

def load_and_clean_auxiliary_data():
    with zipfile.ZipFile('Origin_Data.zip', 'r') as zip_ref:
        train = pd.read_csv(zip_ref.open('train.csv'), parse_dates=["date"])
        test = pd.read_csv(zip_ref.open('test.csv'), parse_dates=["date"])
        stores = pd.read_csv(zip_ref.open('stores.csv'))
        holidays = pd.read_csv(zip_ref.open('holidays_events.csv'), parse_dates=["date"])
        oil = pd.read_csv(zip_ref.open('oil.csv'), parse_dates=["date"])


    holidays = holidays[holidays["transferred"] == False].copy()
    holidays["priority"] = holidays["type"].map(HOLIDAY_PRIORITY)
    idx = holidays.groupby("date")["priority"].idxmax()
    holidays = holidays.loc[idx].reset_index(drop=True)
    holidays = holidays.drop("transferred", axis=1)
    
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    
    return train, test, stores, holidays, oil


def merge_auxiliary_data(df, stores, holidays, oil):
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays[["date", "type"]], on="date", how="left")

    df = df.rename(columns={
        'type_x': 'store_type',
        'type_y': 'day_type'
    })

    df['day_type'] = df['day_type'].fillna('Work Day')
    
    return df


def extract_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["Day"] = df["date"].dt.day
    df["Week_day"] = df["date"].dt.strftime("%A")
    df['Weekend'] = df["Week_day"].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
    return df


def handle_train_specific_features(train_df, test_df):
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    train_df["onpromotion"] = np.log1p(train_df["onpromotion"])
    test_df["onpromotion"] = np.log1p(test_df["onpromotion"])
    
    return train_df, test_df


def encode_and_scale_data(train_df, test_df):
    le = LabelEncoder()
    train_df["family"] = le.fit_transform(train_df["family"])
    test_df["family"] = le.transform(test_df["family"])

    cat_cols = ['city', 'state', 'store_type', 'Week_day', 'day_type', 'Weekend']
    train_encoded = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

    sales = train_encoded["sales"]
    test_id = test_encoded["id"]
    train_dates = train_encoded["date"]
    test_dates = test_encoded["date"]

    train_encoded = train_encoded.drop(["sales", "date"], axis=1)
    test_encoded = test_encoded.drop(["id", "date"], axis=1)

    train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

    train_encoded["sales"] = sales
    test_encoded["id"] = test_id

    scaler = StandardScaler()
    numerical_cols = ['store_nbr', 'cluster', 'dcoilwtico', 'Year', 'Month', 'Day']
    train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])
    test_encoded[numerical_cols] = scaler.transform(test_encoded[numerical_cols])

    train_encoded.index = train_dates
    test_encoded.index = test_dates
    train_encoded.index.name = 'date'
    test_encoded.index.name = 'date'

    return train_encoded, test_encoded


def full_preprocessing_pipeline(data_path="."):
    train_raw, test_raw, stores, holidays_cleaned, oil_cleaned = load_and_clean_auxiliary_data()

    train_merged = merge_auxiliary_data(train_raw, stores, holidays_cleaned, oil_cleaned)
    test_merged = merge_auxiliary_data(test_raw, stores, holidays_cleaned, oil_cleaned)

    train_fe = extract_time_features(train_merged)
    test_fe = extract_time_features(test_merged)

    train_cleaned, test_cleaned = handle_train_specific_features(train_fe, test_fe)

    df_final, test_final = encode_and_scale_data(train_cleaned, test_cleaned)

    return train_cleaned, df_final, test_final


def save_data(train_df, df_final, test_final, output_path="."):
    train_df.to_csv(f"{output_path}/train_cleaned.csv", index=False)
    df_final.to_csv(f"{output_path}/train_encoded.csv", index=True)
    test_final.to_csv(f"{output_path}/test_encoded.csv", index=True)
