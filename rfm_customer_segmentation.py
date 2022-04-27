import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import datetime as dt
import xlrd
import missingno as msno
import plotly.express as px



def get_dataset_clean():
    df_ = pd.read_excel('online_retail_II.xlsx', sheet_name="Year 2010-2011")
    df = df_.copy()

    # Değişkenler:

    # InvoiceNo – Fatura Numarası Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.\n
    # StockCode – Ürün kodu Her bir ürün için eşsiz numara.\n
    # Description – Ürün ismi\n
    # Quantity – Ürün adedi Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.\n
    # InvoiceDate – Fatura tarihi\n
    # UnitPrice – Fatura fiyatı (Sterlin)\n
    # CustomerID – Eşsiz müşteri numarası\n
    # Country – Ülke ismi\n

    # df.head()
    # df.shape
    # df.dtypes
    # df.describe().T
    # df.isnull().sum()
    # msno.matrix(df)
    # plt.show()
    df.dropna(subset=["Customer ID"], inplace=True)
    # df.isnull().sum()
    # df["Description"].nunique()
    # df["Description"].value_counts()
    
    # issued_number = df.loc[df["Invoice"].str.startswith('C', na = False)].shape[0]
    # not_issued_number = df.loc[~df['Invoice'].str.startswith('C', na = False)].shape[0]
    # issued_number / not_issued_number  = 0.0174   -> Issue Rate 0.01
    df = df.loc[~df["Invoice"].str.startswith('C', na=False)]
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    return df


def dataframe_to_rfm(dataframe):

    today_date = pd.to_datetime('2011-12-11')

    rfm = dataframe.groupby("Customer ID").agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm.loc[rfm['monetary'] > 0]  # monetary value always bigger than 0

    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(
        method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm['rf_score'] = rfm["recency_score"].astype(
        str) + rfm['frequency_score'].astype(str)

    seg_map = {     # segmentation map
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalist',
        r'5[4-5]': 'champions'}

    rfm['segment'] = rfm['rf_score'].replace(seg_map, regex=True)

    # rfm[rfm["segment"] == "need_attention"].head()

    return rfm
    new_df = pd.DataFrame()
    new_df["new_customer_id"] = rfm[rfm["segment"] == "need_attention"].index
    new_df.head()


if __name__ == "__main__":
    df = get_dataset_clean()

    countries = st.sidebar.multiselect(
        "Choose countries", ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"], [
            "United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        st.header("Customer Segmentation with RFM :dollar: :bar_chart:")
        data = df[df["Country"].isin(countries)]
        data["StockCode"] = data["StockCode"].astype(str)

        with st.expander("Cleaned Dataframe"):
            st.dataframe(data)
        rfm = dataframe_to_rfm(data)
        with st.expander("Rfm Dataframe"):
            st.dataframe(rfm)

        df["Country"].value_counts().head(6)
        # draw_bar("total_gain_loss_percent")
        # """def draw_bar(y_val: str) -> None:
          #   fig = px.bar(data, y=y_val, x="symbol", **COMMON_ARGS)
            # fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
            # chart(fig)"""

        st.subheader("Total monetary value of each Country")

        country_monetaries = df["Country"].value_counts().head(6)
        fig = px.pie(country_monetaries, values=country_monetaries.values, names = country_monetaries.index)
        st.plotly_chart(fig)
        st.text("RFM Scores by Segment")
        st.dataframe(rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"]))

        best_customers = rfm.sort_values("monetary", ascending = False).head(10).index
        # best_customers
        best_seller_products = df.loc[df["Customer ID"].isin(best_customers)].groupby("Description").agg({'TotalPrice': 'sum'}).sort_values(by = "TotalPrice",
                                                                                                             ascending = False).head()
        st.subheader('Best Seller Products based on top 10 customers')
        
        st.dataframe(best_seller_products)


        col1, col2 = st.columns(2)
        # col1.metric("Temperature", "70 °F", "1.2 °F")
        # col2.metric("Wind", "9 mph", "-8%")
        col1.metric(label="Need Attention", value="100$", delta="-25%", delta_color="inverse")
        col2.metric(label="Potential Loyalist", value="100$", delta="-10%", delta_color="inverse")



        def write_segment(rfm, segment):
            new_df = pd.DataFrame()
            new_df["new_customer_id"] = rfm[rfm["segment"] == segment].index
            
            st.download_button(label = f"Download {segment} as csv", data = new_df.to_csv(), file_name = f'{segment}_segment_customers.csv', mime = 'text/csv', key = segment)
        
        st.subheader("Download Segment Customer Data")
     
        write_segment(rfm, "champions")
        write_segment(rfm, "loyal_customers")
        write_segment(rfm, "potential_loyalist")
        write_segment(rfm, "need_attention")
        # rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
