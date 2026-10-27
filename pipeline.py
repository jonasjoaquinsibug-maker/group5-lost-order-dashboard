"""Reproduces the Group 5 notebook's cleaning and feature steps from the raw CSV."""
import pandas as pd
import numpy as np

DATE_SPECS = [
    (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
    (r'^\d{2}-\d{2}-\d{4}$', '%m-%d-%Y'),
    (r'^\d{2}/\d{2}/\d{4}$', '%d/%m/%Y'),
    (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),
    (r'^\d{1,2} [A-Za-z]{3} \d{4}$', '%d %b %Y'),
]
PAY = {'PAYPAL': 'PayPal', 'NETBANKING': 'Net Banking', 'UPI': 'UPI', 'DEBITCARD': 'Debit Card',
       'CREDITCARD': 'Credit Card', 'COD': 'Cash on Delivery', 'CASHONDELIVERY': 'Cash on Delivery'}
CTRY = {'UK': 'United Kingdom', 'UNITEDKINGDOM': 'United Kingdom', 'USA': 'United States',
        'US': 'United States', 'UNITEDSTATES': 'United States', 'UAE': 'United Arab Emirates',
        'DE': 'Germany', 'GERMANY': 'Germany', 'IN': 'India', 'INDIA': 'India',
        'CA': 'Canada', 'CANADA': 'Canada', 'AU': 'Australia', 'AUSTRALIA': 'Australia'}


def clean(path):
    df = pd.read_csv(path)
    raw = df.copy()
    s = df['Order_Date'].astype(str)
    parsed = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    for pat, fmt in DATE_SPECS:
        m = s.str.match(pat)
        parsed[m] = pd.to_datetime(s[m], format=fmt, errors='coerce')
    df['Order_Date'] = parsed
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df['Quantity'] > 0].reset_index(drop=True)
    df['Payment_Method'] = df['Payment_Method'].map(
        lambda v: PAY.get(str(v).upper().replace('_', ' ').replace('.', '').replace(' ', '')))
    df['Country'] = df['Country'].map(lambda v: CTRY.get(str(v).upper().replace('.', '').replace(' ', '')))
    df['Discount_Percent'] = df['Discount_Percent'].fillna(0.0)
    df['Shipping_City'] = df['Shipping_City'].fillna('Unknown')
    df['Rating_Available'] = df['Customer_Rating'].notna()
    df['Gross_Revenue_USD'] = (df['Quantity'] * df['Unit_Price_USD']).round(2)
    df['Net_Revenue_USD'] = (df['Gross_Revenue_USD'] * (1 - df['Discount_Percent'] / 100)).round(2)
    df['Discount_Amount_USD'] = (df['Gross_Revenue_USD'] * df['Discount_Percent'] / 100).round(2)
    df['Order_Year'] = df['Order_Date'].dt.year
    df['Order_Month'] = df['Order_Date'].dt.to_period('M').astype(str)
    df['Order_Quarter'] = 'Q' + df['Order_Date'].dt.quarter.astype(str)
    df['Order_DayOfWeek'] = df['Order_Date'].dt.day_name()
    df['Is_Weekend'] = df['Order_Date'].dt.dayofweek >= 5
    df['Is_Lost'] = df['Order_Status'].isin(['Cancelled', 'Returned']).astype(int)
    return raw, df


if __name__ == '__main__':
    raw, df = clean('data/ecommerce_retail_transactions_raw.csv')
    print('raw', raw.shape, 'clean', df.shape)
    print('net', round(df.Net_Revenue_USD.sum(), 2), 'gross', round(df.Gross_Revenue_USD.sum(), 2),
          'aov', round(df.Net_Revenue_USD.mean(), 2), 'median', df.Net_Revenue_USD.median())
    print(df.Order_Status.value_counts().to_dict())
    print('unmapped pay/country', df.Payment_Method.isna().sum(), df.Country.isna().sum(), 'date NaT', df.Order_Date.isna().sum())
    print('rating mean delivered', round(df[df.Order_Status == 'Delivered'].Customer_Rating.mean(), 3))
    print('discount', round(df.Discount_Amount_USD.sum(), 2))
