import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from sklearn.metrics import silhouette_score
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import time


def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True)
load_css()


st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main{
background-color:#F8F9FA;
}
.metric{
padding:15px;
border-radius:12px;
background:#FFFFFF;
box-shadow:2px 2px 10px rgba(0,0,0,0.2);
}
h1{
color:#0E1117;}
</style>""",unsafe_allow_html=True)

st.markdown(""""
<h1 class='title'>
🛒 Shopper Spectrum Dashboard
</h1>
""",unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png",width=100)

st.sidebar.title("📊 Analytics")
st.sidebar.markdown("---")
with st.spinner("🔄 Loading Please Wait..."):
    time.sleep(2)

page=st.sidebar.radio(
"Navigation",
[
"🏠 Home",
"📈 Dashboard",
"📊 EDA",
"👥 Segmentation",
"🛍 Recommendation"
]
)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/online_retail.csv",encoding="latin1")
    return df
df = load_data()

# Data Cleaning..
df = df.drop_duplicates()
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalAmount"] = (df["Quantity"] *df["UnitPrice"])

# Sidebar Statistics
st.sidebar.markdown("---")
st.sidebar.metric("Customers",df.CustomerID.nunique())
st.sidebar.metric("Products",df.Description.nunique())
st.sidebar.metric("Countries",df.Country.nunique())

# ========================
# PRODUCT RECOMMENDATION
# ========================

pivot = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        fill_value=0
    )
pivot.head()

# Product Matrix
product_matrix = pivot.T
product_matrix.head()

# Calculate Similarity Matrix
similarity = cosine_similarity(product_matrix)

# Similarity DataFrame
similarity_df = pd.DataFrame(
similarity,
index=product_matrix.index,
columns=product_matrix.index
)
similarity_df.head()

# Recommendation Function
def recommend_product(product_name):
    if product_name not in similarity_df.index:
        return None
    recommendations = similarity_df[product_name]\
        .sort_values(ascending=False)\
        .iloc[1:6]
    return recommendations

# =============
# HOME PAGE 
# =============

if page=="🏠 Home":
    st.markdown("""
        <div class="home-container">
        <div class="home-title">
        🛒 Shopper Spectrum
        </div>
        <div class="home-subtitle">
        <b>AI-Powered Retail Analytics Dashboard</b>
        <br>
        Analyze Sales • Understand Customers • Recommend Products
        </div>
        <div class="feature-card">
        <div class="feature-title">
        📈 Sales Analysis
        </div>
        <div class="feature-desc">
        ✔ Monthly Sales Trends
        <br>
        ✔ Revenue Analysis
        <br>
        ✔ Top Selling Products
        </div>
        </div>
        <div class="feature-card">
        <div class="feature-title">
        👥 Customer Analytics
        </div>
        <div class="feature-desc">
        ✔ Customer Purchase Behaviour
        <br>
        ✔ Country Wise Analysis
        <br>
        ✔ Customer Insights
        </div>
        </div>
        <div class="feature-card">
        <div class="feature-title">
        📦 Product Analytics
        </div>
        <div class="feature-desc">
        ✔ Best Selling Products
        <br>
        ✔ Revenue by Product
        <br>
        ✔ Product Performance
        </div>
        </div>
        <div class="feature-card">
        <div class="feature-title">
        🎯 Customer Segmentation
        </div>
        <div class="feature-desc">
        ✔ RFM Analysis
        <br>
        ✔ K-Means Clustering
        <br>
        ✔ High Value Customer Detection
        </div>
        </div>
        <div class="feature-card">
        <div class="feature-title">
        🤖 Product Recommendation System
        </div>
        <div class="feature-desc">
        ✔ Collaborative Filtering
        <br>
        ✔ Cosine Similarity
        <br>
        ✔ Top 5 Product Recommendations
        </div>
        </div>
        <div class="footer">
        🚀 Developed using <b>Python • Streamlit • Plotly • Scikit-Learn • Pandas</b>
        </div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="developer-card">
        <div class="developer-title">
        👨‍💻 Developed By
        </div>
        <div class="developer-name">
        ✨ SAHIL KHAN ✨
        </div>
        <div class="developer-role">
        Data Analyst | Python Developer | Machine Learning Enthusiast
        </div>
        <div class="developer-tech">
        🐍 Python &nbsp;&nbsp;|&nbsp;&nbsp;
        📊 Streamlit &nbsp;&nbsp;|&nbsp;&nbsp;
        📈 Plotly &nbsp;&nbsp;|&nbsp;&nbsp;
        🤖 Scikit-Learn &nbsp;&nbsp;|&nbsp;&nbsp;
        🗄️ SQL
        </div>
        </div>""", unsafe_allow_html=True)


# ==============
# Dashboard Page
# ==============

elif page=="📈 Dashboard":
    st.header("Business KPIs")
    sales=df.TotalAmount.sum()
    orders=df.InvoiceNo.nunique()
    customers=df.CustomerID.nunique()
    products=df.Description.nunique()
    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.markdown(
        f"""
        <div class="kpi-card">
        <h3>💰 Sales</h3>
        <h2>₹{sales:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
        )
    with c2:
        st.markdown(
        f"""
        <div class="kpi-card">
        <h3>📦 Orders</h3>
        <h2>{orders}</h2>
        </div>
        """,
        unsafe_allow_html=True
        )
    with c3:
        st.markdown(
        f"""
        <div class="kpi-card">
        <h3>👥 Customers</h3>
        <h2>{customers}</h2>
        </div>
        """,
        unsafe_allow_html=True
        )
    with c4:
        st.markdown(
        f"""
        <div class="kpi-card">
        <h3>🛒 Products</h3>
        <h2>{products}</h2>
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown("---")

    # Global Filters
    st.subheader("🔎 Dashboard Filters")
    col1,col2,col3=st.columns(3)
    country=col1.selectbox("Country",["All"]+sorted(df.Country.unique()))
    year=col2.selectbox("Year",["All"]+sorted(df.InvoiceDate.dt.year.unique()))
    month=col3.selectbox("Month",["All"]+list(range(1,13)))

    # Apply Filter
    filtered=df.copy()
    if country!="All":
        filtered=filtered[filtered.Country==country]
    if year!="All":
        filtered=filtered[filtered.InvoiceDate.dt.year==year]
    if month!="All":
        filtered=filtered[filtered.InvoiceDate.dt.month==month]
    
    # Dashboard Layout
    left,right=st.columns([2,1])

    with left:
        st.subheader("Monthly Sales")
        # monthly=filtered.groupby(filtered.InvoiceDate.dt.to_period("M"))
        # ["TotalAmount"].sum().reset_index()
        monthly=filtered.groupby(filtered["InvoiceDate"].dt.to_period("M")
        )["TotalAmount"].sum().reset_index()
        monthly.InvoiceDate=monthly.InvoiceDate.astype(str)
        fig=px.line(
            monthly,
            x="InvoiceDate",
            y="TotalAmount",
            markers=True
        )
        st.plotly_chart(fig,use_container_width=True)

    with right:
        st.subheader("Top Countries")
        top=filtered.groupby("Country")["TotalAmount"].sum().nlargest(5)
        fig=px.pie(values=top.values,names=top.index)
        st.plotly_chart(fig,use_container_width=True)

    # Tabs
    tab1,tab2,tab3=st.tabs(["Sales","Customer","Product"])
    
    with tab1:
        st.subheader("Sales Analysis")
        fig=px.histogram(filtered,x="TotalAmount")
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        top_customer=filtered.groupby("CustomerID")["TotalAmount"].sum().nlargest(10)
        fig=px.bar(top_customer)
        st.plotly_chart(fig,use_container_width=True)

    with tab3:
        product=filtered.groupby("Description")["Quantity"].sum().nlargest(10)
        fig=px.bar(product)
        st.plotly_chart(fig,use_container_width=True)

    # Download Report
    csv=filtered.to_csv(index=False)
    st.download_button("⬇ Download Report",csv,"Report.csv")

    # Dashboard Matrics
    st.metric("Average Order Value",f"₹{filtered.TotalAmount.mean():,.2f}")
    st.metric("Average Quantity",round(filtered.Quantity.mean(),2))

    # Progress Bar
    st.progress(100)

    st.header("Dataset")
    st.write(df.head())
    st.write("Rows :",df.shape[0])
    st.write("Columns :",df.shape[1])
    st.write(df.dtypes)

    # Fotter
    st.markdown("---")
    st.markdown(
    """
    <center>
    Made with ❤️ using Streamlit
    </center>
    """,unsafe_allow_html=True)


# =========
# EDA Page
# =========

elif page=="📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("---")

    # Dataset Filter
    col1,col2=st.columns(2)
    countries=["All"]+sorted(df["Country"].unique().tolist())
    country=col1.selectbox("Select Country",countries)
    if country!="All":
        data=df[df["Country"]==country]
    else:
        data=df.copy()

    # Dataset Summary
    st.subheader("Dataset Summary")
    c1,c2,c3,c4=st.columns(4)

    c1.metric("Rows",len(data))
    c2.metric("Customers",data["CustomerID"].nunique())
    c3.metric("Products",data["Description"].nunique())
    c4.metric("Countries",data["Country"].nunique())

    # Sales Trend
    st.subheader("📈 Monthly Sales")

    monthly=data.groupby(data["InvoiceDate"].dt.to_period("M"))["TotalAmount"].sum().reset_index()
    monthly["InvoiceDate"]=monthly["InvoiceDate"].astype(str)

    fig=px.line(monthly,x="InvoiceDate",y="TotalAmount",markers=True,title="Monthly Sales Trend")
    st.plotly_chart(fig,use_container_width=True)

    # Daily Sales
    st.subheader("📅 Daily Sales")

    daily=data.groupby(data["InvoiceDate"].dt.date)["TotalAmount"].sum().reset_index()
    fig=px.line(
    daily,
    x="InvoiceDate",
    y="TotalAmount",
    title="Daily Sales"
    )
    st.plotly_chart(fig,use_container_width=True)

    # Top Countries
    st.subheader("🌍 Top Countries")

    country_sales=data.groupby("Country")["TotalAmount"]\
    .sum()\
    .sort_values(ascending=False)\
    .head(10)\
    .reset_index()

    fig=px.bar(country_sales,x="Country",y="TotalAmount",text_auto=".2s",color="TotalAmount")
    st.plotly_chart(fig,use_container_width=True)

    # Top Selling Products
    st.subheader("🏆 Top Selling Products")

    top_products=data.groupby("Description")["Quantity"]\
    .sum()\
    .sort_values(ascending=False)\
    .head(15)\
    .reset_index()

    fig=px.bar(top_products,x="Quantity",y="Description",orientation="h",color="Quantity")
    st.plotly_chart(fig,use_container_width=True)

    # Top Customers
    st.subheader("👥 Top Customers")

    customer_sales=data.groupby("CustomerID")["TotalAmount"]\
    .sum()\
    .sort_values(ascending=False)\
    .head(15)\
    .reset_index()

    fig=px.bar(customer_sales,x="CustomerID",y="TotalAmount",color="TotalAmount")
    st.plotly_chart(fig,use_container_width=True)

    # Quantity Distribution
    st.subheader("📦 Quantity Distribution")

    fig=px.histogram(
    data,
    x="Quantity",
    nbins=60
    )
    st.plotly_chart(fig,use_container_width=True)

    # Unit Price Distribution
    st.subheader("💲 Price Distribution")

    fig=px.histogram(
    data,
    x="UnitPrice",
    nbins=60
    )
    st.plotly_chart(fig,use_container_width=True)

    #Total Amount Distribution
    st.subheader("💰 Transaction Value")

    fig=px.histogram(
    data,
    x="TotalAmount",
    nbins=70
    )
    st.plotly_chart(fig,use_container_width=True)

    # Boxplot
    st.subheader("📦 Quantity Boxplot")

    fig=px.box(
    data,
    y="Quantity"
    )
    st.plotly_chart(fig,use_container_width=True)

    # Scatter Plot
    st.subheader("Quantity vs Price")

    fig=px.scatter(
    data.sample(min(len(data), 5000), random_state=42),
    x="Quantity",
    y="UnitPrice",
    color="Country"
    )
    st.plotly_chart(fig,use_container_width=True)

    # Heatmap
    st.subheader("Correlation Heatmap")

    corr=data[["Quantity","UnitPrice","TotalAmount"]].corr()
    fig,ax=plt.subplots(figsize=(6,5))
    sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    ax=ax
    )
    st.pyplot(fig)

    # pie chart
    st.subheader("Sales Share")

    pie=data.groupby("Country")["TotalAmount"]\
    .sum()\
    .sort_values(ascending=False)\
    .head(5)

    fig=px.pie(values=pie.values,names=pie.index)
    st.plotly_chart(fig,use_container_width=True)

    # Line chart
    st.subheader("Orders Per Month")

    orders=data.groupby(data["InvoiceDate"].dt.to_period("M"))["InvoiceNo"]\
    .nunique()\
    .reset_index()

    orders["InvoiceDate"]=orders["InvoiceDate"].astype(str)

    fig=px.line(orders,x="InvoiceDate",y="InvoiceNo",markers=True)
    st.plotly_chart(fig,use_container_width=True)

    # Country Table
    st.subheader("Country Wise Report")

    country_table=data.groupby("Country").agg({
    "InvoiceNo":"nunique",
    "CustomerID":"nunique",
    "TotalAmount":"sum"
    })
    st.dataframe(country_table)

    # Download Dataset
    csv=data.to_csv(index=False)
    st.download_button("Download Filtered Dataset",csv,"filtered_dataset.csv","text/csv")

# ==========================
# Customer Segmentation Page
# ==========================

elif page=="👥 Segmentation":
    st.header("Customer Segmentation")

    # ==============================
    # RFM FEATURE ENGINEERING
    # ==============================

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum"
    })
    rfm.columns = ["Recency","Frequency","Monetary"]
    rfm.reset_index(inplace=True)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    # Elbow Method
    inertia=[]
    for k in range(2,11):
        model=KMeans(n_clusters=k,random_state=42,n_init=10)
        model.fit(rfm_scaled)
        inertia.append(model.inertia_)

    # Elbow Graph
    fig=px.line(
        x=list(range(2,11)),
        y=inertia,
        markers=True,
        labels={"x":"Clusters","y":"Inertia"},title="Elbow Method")
    st.plotly_chart(fig,use_container_width=True)

    # Silhouette Score
    scores=[]
    for k in range(2,11):
        model=KMeans(n_clusters=k,random_state=42,n_init=10)
        labels=model.fit_predict(rfm_scaled)
        score=silhouette_score(rfm_scaled,labels)
        scores.append(score)

    # Silhouette Plot
    fig=px.line(
        x=list(range(2,11)),
        y=scores,
        markers=True,
        title="Silhouette Score"
    )
    st.plotly_chart(fig,use_container_width=True)

    # Train Final Model
    kmeans=KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    rfm["Cluster"]=kmeans.fit_predict(rfm_scaled)

    # Cluster Statistics
    cluster_summary=rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    st.dataframe(cluster_summary)

    # Cluster Labels
    cluster_labels = {
        0: "Occasional",
        1: "At Risk",
        2: "High Value",
        3: "Regular"
    }
    rfm["Segment"]=rfm["Cluster"].map(cluster_labels)

    # Cluster Count
    fig=px.bar(rfm["Segment"].value_counts(),title="Customer Segments")
    st.plotly_chart(fig,use_container_width=True)

    # Scatter Plot
    fig=px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color="Segment",
        hover_data=["CustomerID"]
    )
    st.plotly_chart(fig,use_container_width=True)

    # Bubble Chart
    fig=px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        size="Frequency",
        color="Segment",
        hover_data=["CustomerID"]
    )
    st.plotly_chart(fig,use_container_width=True)

    # Save Model

    joblib.dump(kmeans,"models/kmeans_model.pkl")
    joblib.dump(scaler,"models/scaler.pkl")

    # Save Similarity Model
    joblib.dump(similarity_df,"models/product_similarity.pkl",compress=7)

    # Save Product List
    joblib.dump(product_matrix.index.tolist(),"models/product_names.pkl")

    # User Input
    recency=st.number_input("Recency",1,1000,30)
    frequency=st.number_input("Frequency",1,500,5)
    monetary=st.number_input("Monetary",0.0,500000.0,5000.0)

    # Predict Button
    segment = None
    if st.button("Predict Customer Segment"):
        user=pd.DataFrame({
            "Recency":[recency],
            "Frequency":[frequency],
            "Monetary":[monetary]
        })
        scaled=scaler.transform(user)
        cluster=kmeans.predict(scaled)[0]
        segment=cluster_labels.get(cluster,"Unknown")
        st.success(f"Predicted Segment : {segment}")

    # Display Customer Profile
    st.info(f"""
        Recency : {recency}
        Frequency : {frequency}
        Monetary : ₹{monetary:,.2f}
        """)
    
    # Segment Cards
    if segment=="High Value":
        st.success("""
                   
        ⭐ Premium Customer
                   
        Offer Exclusive Rewards
                   
        """)

    elif segment=="Regular":
        st.info("""

        👍 Loyal Customer

        Offer Membership Benefits

        """)

    elif segment=="Occasional":
        st.warning("""

        🛍 Encourage Repeat Purchases

        """)

    else:
        st.error("""

        ⚠ Customer is At Risk

        Launch Retention Campaign

        """)

    # Download RFM Table
    csv=rfm.to_csv(index=False)
    st.download_button("Download RFM Table",csv,"rfm.csv")

# ====================
#  Recommendation Page
#  ===================

elif page=="🛍 Recommendation":
    st.header("🛍 Product Recommendation System")
    st.markdown("---")    
    product = st.selectbox("Select Product",sorted(product_matrix.index))

    recommendations = None
    if st.button("Recommend Products"):
        recommendations = recommend_product(product)

    if recommendations is None:
        st.error("Product Not Found")
    else:
        st.success(f"Top Recommendations for {product}")

    recommend_df = pd.DataFrame({
                "Recommended Products": recommendations.index,
                "Similarity": recommendations.values
            })
    st.dataframe(recommend_df)

    # Card View
    for item,score in recommendations.items():
        st.markdown(f"""### 📦 {item}Recommended Product---""")
        st.progress(float(score))
        st.write(f"{item}")
        st.caption(f"Similarity : {score:.2f}")

    csv = recommend_df.to_csv(index=False)
    st.download_button("Download Recommendation",csv,"recommendation.csv")

    # Show Total Products
    st.metric("Total Products",len(product_matrix.index))

    # Show Similar Products
    st.subheader("Most Similar Products")
    recommendations = recommend_product(product)
    st.table(recommendations.reset_index())

    # Product Frequency
    product_sales = df.groupby("Description")["Quantity"].sum()
    st.metric("Units Sold",int(product_sales[product]))

    # Product Revenue
    revenue = df.groupby("Description")["TotalAmount"].sum()
    st.metric("Revenue",f"₹{revenue[product]:,.2f}")

    # Product Trend
    trend = df[df["Description"]==product]
    trend = trend.groupby(trend["InvoiceDate"].dt.to_period("M"))["Quantity"].sum().reset_index()
    trend["InvoiceDate"] = trend["InvoiceDate"].astype(str)
    fig = px.line(trend,x="InvoiceDate",y="Quantity",markers=True)
    st.plotly_chart(fig,use_container_width=True)

    # Product Details
    details = df[df["Description"]==product]
    st.write(details.head())





