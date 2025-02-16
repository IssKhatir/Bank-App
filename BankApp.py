import streamlit as st 
import numpy as np
import pandas as pd
import os
os.system("pip install matplotlib")
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import squarify
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from plotly.offline import iplot
from plotly import tools
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy import sparse
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier 

with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 

st.sidebar.markdown("<h1 style='color: darkblue; font-size: 40px; font-weight: bold;'>Bank Marketing Campaign</h1>", unsafe_allow_html=True)


pages = [ "Introduction", "Data Exploration", "Analysis","Conclusion" ]
icons = ["📝", "📊", "🔍", "🎯"]

page = st.sidebar.radio(" ", pages, format_func=lambda page: f"{icons[pages.index(page)]} {page}")

if page == pages[0] :
    st.write(" # Introduction to Marketing ")
    st.write("Simply put, marketing is managing profitable relationships, by attracting new customers by superior value and keeping current customers by delivering satisfaction. Marketing must be understood in the sense of satisfying customer needs. Marketing can be defined as the process by which companies create value for customers and build strong customer relationships to capture value from customers in return. A five-step model of the marketing process will provide the structure of this chapter.")
    st.write (" **1- Understanding the marketplace and customer needs** ")
    st.write (" **2- Designing a customer-driven marketing strategy** ")
    st.write (" **3- Constructing an integrated marketing plan** ")
    st.write (" **4- Building customer relationships** ")
    st.write (" **5- Capturing customer value** ")
    st.write ("Cited from Kotler and Armstrong (2010).")
    st.image("marketing.png")

    st.write("Marketing campaigns prioritize understanding and fulfilling customer needs and ensuring their satisfaction. However, the success of a marketing campaign hinges on various factors. It's crucial to consider specific variables when strategizing and executing a marketing campaign.")
    st.write("The ""4 Ps"" is a marketing concept that stands for the four elements of a marketing mix, originally proposed by E. Jerome McCarthy in the 1960s. These elements are:")
    st.write("**1- Product:** This refers to the actual product or service being offered to customers. It involves decisions regarding product design, features, quality, branding, packaging, and other aspects related to the product itself.")
    st.write("**2- Price:** This represents the pricing strategy for the product or service. It involves decisions about setting the price level, discounts, pricing models, and payment terms, among others.")
    st.write("**3- Place:** Also known as distribution, this refers to the channels and methods used to make the product or service available to customers. It involves decisions about distribution channels, logistics, inventory management, and physical or digital presence.")
    st.write("**4- Promotion:** This involves the activities used to communicate and promote the product or service to the target market. It includes advertising, sales promotions, public relations, direct marketing, and other promotional activities.")
    st.write("The 4 Ps framework provides a structured approach for marketers to analyze and develop marketing strategies by considering these four key elements.")

    st.write("The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit.")

    st.write("## Dataset Information")
    st.write("The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. ")
    st.write("There are four datasets: ")
    st.write("bank.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).")

    st.write("## I. Bank client data")
    st.write("**1 - age:** (numeric)")
    st.write("**2 - job:** type of job (categorical: 'admin.','bluecollar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')")
    st.write("**3 - marital:** marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)")
    st.write("**4 - education:** (categorical: primary, secondary, tertiary and unknown)")
    st.write("**5 - default:** has credit in default? (categorical: 'no','yes','unknown')")
    st.write("**6 - housing:** has housing loan? (categorical: 'no','yes','unknown')")
    st.write("**7 - loan:** has personal loan? (categorical: 'no','yes','unknown')")
    st.write("**8 - balance:** Balance of the individual.")

    st.write("## II. Related with the last contact of the current campaign")
    st.write("**8 - contact:** contact communication type (categorical: 'cellular','telephone')")
    st.write("**9 - month:** last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')")
    st.write("**10 - day:** last contact day of the week (categorical: 'mon','tue','wed','thu','fri')")
    st.write("**11 - duration:** last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.")

    st.write("## III. other attributes")
    st.write("**12 - campaign:** number of contacts performed during this campaign and for this client (numeric, includes last contact)")
    st.write("**13 - pdays:** number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)")
    st.write("**14 - previous:** number of contacts performed before this campaign and for this client (numeric)")
    st.write("**15 - poutcome:** outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')")
    st.write("#### Output variable (desired target):")
    st.write("**16 - deposit:** has the client subscribed a term deposit? (binary: 'yes','no')")

    st.write("# **Understanding Term Deposits**")
    st.write("A term deposit is a financial product provided by banks or other financial institutions, offering a fixed interest rate (typically higher than standard deposit accounts). It involves depositing a sum of money for a specified period, with the principal amount returned to the depositor upon maturity.")
 
elif page == pages[1] :

    st.write("# **Data exploration**")
    df =pd.read_csv('bank.csv', sep = ",")
    st.dataframe(df.head())

    st.write("## I - Structuring the data ")

    st.write("### I.1. Overall Analysis of the Data")

    st.write("#### I.1.1 Summary :")

    st.dataframe(df.describe())
    st.write("- Mean Age is aproximately 41 years old. (Minimum: 18 years old and Maximum: 95 years old.)")
    st.write("- The mean balance is 1,528. However, the Standard Deviation (std) is a high number so we can understand through this that the balance is heavily distributed across the dataset.")
    st.write("- As the data information said it will be better to drop the duration column since duration is highly correlated in whether a potential client will buy a term deposit. Also, duration is obtained after the call is made to the potential client so if the target client has never received calls this feature is not that useful. The reason why duration is highly correlated with opening a term deposit is because the more the bank talks to a target client the higher the probability the target client will open a term deposit since a higher duration means a higher interest (commitment) from the potential client.")

    st.write("#### I.1.2 Check for missing values in the DataFrame : ")

    st.write("Thankfully, there are no missing values in the dataset. However, if there were any, we would need to address them by filling them with either the median, mean, or mode. I typically opt for the median, but since there are no missing values in this case, we can proceed without the need to fill any. This simplifies our task significantly!")
    missing_count = df.isnull().sum()
    st.dataframe(missing_count )
    
    st.write("#### I.1.3 Information on Term Suscriptions : ")

    f, ax = plt.subplots(1,2, figsize=(16,8))
    colors = ["#FA5858", "#64FE2E"]
    labels ="Did not Open Term Suscriptions", "Opened Term Suscriptions"

    plt.suptitle('Information on Term Suscriptions', fontsize=20)

    df["deposit"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                          labels=labels, fontsize=12, startangle=25)
    ax[0].set_ylabel('% of Condition of Loans', fontsize=14)
    palette = ["#64FE2E", "#FA5858"]
    sns.barplot(x="education", y="balance", hue="deposit", data=df, palette=palette, estimator=lambda x: len(x) / len(df) * 100)
    ax[1].set(ylabel="(%)")
    ax[1].set_xticklabels(df["education"].unique(), rotation=0, rotation_mode="anchor")
    st.pyplot(f)

    st.write("#### I.1.4 The distribution of the numerical data : ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.style.use('seaborn-v0_8-darkgrid')
    df.hist(bins=20, figsize=(14,10), color='#E14906')
    st.pyplot(fig)

    st.write("#### I.1.5 Comptage des valeurs de la colonne 'deposit' : ")

    st.dataframe(df['deposit'].value_counts())

    st.write("#### I.1.6 Balance Amount Across Term Subscriptions : ")

    fig1 = plt.figure(figsize=(20,20)) 
    ax1 = fig1.add_subplot(221)
    g = sns.boxplot(x="default", y="balance", hue="deposit",data=df, palette="muted",
                    ax=ax1)
    g.set_title("Amount of Balance by Term Suscriptions")
    st.pyplot(fig1)

    st.write("#### I.1.7 Type of Work by Term Suscriptions : ")

    fig2 = plt.figure(figsize=(20,20))
    ax2 = fig2.add_subplot(222)
    g1 = sns.boxplot(x="job", y="balance", hue="deposit",
                 data=df, palette="RdBu", ax=ax2)
    g1.set_xticklabels(df["job"].unique(), rotation=90, rotation_mode="anchor")
    g1.set_title("Type of Work by Term Suscriptions")
    st.pyplot(fig2)

    st.write("#### I.1.8 Distribution of Balance by Education : ")
    fig3 = plt.figure(figsize=(20,20))
    ax3 = fig3.add_subplot(212)
    g2 = sns.violinplot(data=df, x="education", y="balance", hue="deposit", palette="RdBu_r", ax=ax3)
    g2.set_title("Distribution of Balance by Education")
    st.pyplot(fig3)

    st.write("### I.2 Analysis by Occupation")
    st.write("- **Number of Occupations :** Management is the occupation that is more prevalent in this dataset. ")
    st.write("- **Age by Occupation :** As expected, the retired are the ones who have the highest median age while student are the lowest.")
    st.write("- **Balance by Occupation :** Management and Retirees are the ones who have the highest balance in their accounts.")

    # Drop the Job Occupations that are "Unknown"
    df = df.drop(df.loc[df["job"] == "unknown"].index)
    lst = [df]
    for col in lst:
        col.loc[col["job"] == "admin.", "job"] = "management"
    
    df = df.drop(df.loc[df["balance"] == 0].index)
    st.write ("### I.2.1  A hierarchical visualization of rectangular data")
    x = 0
    y = 0
    width = 100
    height = 100

    job_names = df['job'].value_counts().index
    values = df['job'].value_counts().tolist()

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    colors = ['rgb(200, 255, 144)','rgb(135, 206, 235)',
          'rgb(235, 164, 135)','rgb(220, 208, 255)',
          'rgb(253, 253, 150)','rgb(255, 127, 80)', 
         'rgb(218, 156, 133)', 'rgb(245, 92, 76)',
         'rgb(252,64,68)', 'rgb(154,123,91)']

    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append(
            dict(type = 'rect',x0 = r['x'],y0 = r['y'],x1 = r['x'] + r['dx'],y1 = r['y'] + r['dy'],
            line = dict(width=2),
            fillcolor = colors[counter]
        )
    )
        annotations.append(
            dict(x = r['x']+(r['dx']/2),y = r['y']+(r['dy']/2),text = values[counter],
            showarrow = False
        )
    )
        counter = counter + 1
        if counter >= len(colors):
            counter = 0
    
    # For hover text
    trace0 = go.Scatter(
        x = [ r['x']+(r['dx']/2) for r in rects],
        y = [ r['y']+(r['dy']/2) for r in rects],
        text = [ str(v) for v in job_names],
        mode='text')

    layout = dict(
        title='Number of Occupations <br> <i>(From our Sample Population)</i>',
        height=700, 
        width=700,
        xaxis=dict(showgrid=False,zeroline=False),
        yaxis=dict(showgrid=False,zeroline=False),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest')

    # With hovertext
    figure = dict(data=[trace0], layout=layout)
    # Display the Plotly figure in Streamlit
    st.plotly_chart(figure,  filename='squarify-treemap')

    st.write(" ### I.2.2 Which profession typically exhibited higher account balances?")

    suscribed_df = df.loc[df["deposit"] == "yes"]
    occupations = df["job"].unique().tolist()

    # Get the balances by jobs
    management = suscribed_df["age"].loc[suscribed_df["job"] == "management"].values
    technician = suscribed_df["age"].loc[suscribed_df["job"] == "technician"].values
    services = suscribed_df["age"].loc[suscribed_df["job"] == "services"].values
    retired = suscribed_df["age"].loc[suscribed_df["job"] == "retired"].values
    blue_collar = suscribed_df["age"].loc[suscribed_df["job"] == "blue-collar"].values
    unemployed = suscribed_df["age"].loc[suscribed_df["job"] == "unemployed"].values
    entrepreneur = suscribed_df["age"].loc[suscribed_df["job"] == "entrepreneur"].values
    housemaid = suscribed_df["age"].loc[suscribed_df["job"] == "housemaid"].values
    self_employed = suscribed_df["age"].loc[suscribed_df["job"] == "self-employed"].values
    student = suscribed_df["age"].loc[suscribed_df["job"] == "student"].values


    ages = [management, technician, services, retired, blue_collar, unemployed, 
         entrepreneur, housemaid, self_employed, student]

    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)',
          'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 
          'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(229, 126, 56, 0.5)', 'rgba(229, 56, 56, 0.5)',
         'rgba(174, 229, 56, 0.5)', 'rgba(229, 56, 56, 0.5)']

    traces = []

    for xd, yd, cls in zip(occupations, ages, colors):
        traces.append(go.Box(y=yd,name=xd,boxpoints='all',jitter=0.5,whiskerwidth=0.2,
            fillcolor=cls,marker=dict(size=2,),
            line=dict(width=1)))

    layout = go.Layout(title='Distribution of Ages by Occupation',yaxis=dict(autorange=True,
        showgrid=True,zeroline=True,dtick=5,gridcolor='rgb(255, 255, 255)',gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),margin=dict(l=40,r=30,b=80,t=100),
        paper_bgcolor='rgb(224,255,246)',
        plot_bgcolor='rgb(251,251,251)',
        showlegend=False)

    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig)

    st.write("### I.2.3 Balance Distribution")
    
    df["balance_status"] = np.nan
    lst = [df]

    for col in lst:
        col.loc[col["balance"] < 0, "balance_status"] = "negative"
        col.loc[(col["balance"] >= 0) & (col["balance"] <= 30000), "balance_status"] = "low"
        col.loc[(col["balance"] > 30000) & (col["balance"] <= 40000), "balance_status"] = "middle"
        col.loc[col["balance"] > 40000, "balance_status"] = "high"
    
    # balance by balance_status
    negative = df["balance"].loc[df["balance_status"] == "negative"].values.tolist()
    low = df["balance"].loc[df["balance_status"] == "low"].values.tolist()
    middle = df["balance"].loc[df["balance_status"] == "middle"].values.tolist()
    high = df["balance"].loc[df["balance_status"] == "high"].values.tolist()


    # Get the average by occupation in each balance category
    job_balance = df.groupby(['job', 'balance_status'])['balance'].mean()

    trace1 = go.Barpolar(r=[-199.0, -392.0, -209.0, -247.0, -233.0, -270.0, -271.0, 0, -276.0, -134.5],
                         text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
                               "services", "student", "technician", "unemployed"],name='Negative Balance',
                               marker=dict(color='rgb(246, 46, 46)'))
    
    trace2 = go.Barpolar(r=[319.5, 283.0, 212.0, 313.0, 409.0, 274.5, 308.5, 253.0, 316.0, 330.0],
                         text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
                                "services", "student", "technician", "unemployed"],name='Low Balance',
                                marker=dict(color='rgb(246, 97, 46)'))
    
    trace3 = go.Barpolar(r=[2128.5, 2686.0, 2290.0, 2366.0, 2579.0, 2293.5, 2005.5, 2488.0, 2362.0, 1976.0],
                         text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
                               "services", "student", "technician", "unemployed"],name='Middle Balance',
                               marker=dict(color='rgb(246, 179, 46)'))
    
    trace4 = go.Barpolar(r=[14247.5, 20138.5, 12278.5, 12956.0, 20723.0, 12159.0, 12223.0, 13107.0, 12063.0, 15107.5],
                         text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
                               "services", "student", "technician", "unemployed"],name='High Balance',
                               marker=dict(color='rgb(46, 246, 78)'))
    
    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(title='Mean Balance in Account<br> <i> by Job Occupation</i>',font=dict(
        size=12),legend=dict(font=dict(size=16)),polar=dict(radialaxis=dict(ticksuffix='%'),
                                                            angularaxis=dict(direction="clockwise")))

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,  filename='polar-area-chart')

    st.write("### I.2.4 Relationship Status")

    st.write("In this analysis, we didn't discover any noteworthy findings except that most divorced individuals have low financial resources. This observation isn't surprising, considering the division of financial assets during divorce proceedings. However, as no additional insights have emerged, we'll now move on to clustering marital status with educational status. This approach may reveal other distinct groups within the sample population.")
    st.write(" Comptage des valeurs de la colonne 'marital' :")

    st.dataframe(df['marital'].value_counts())

    vals = df['marital'].value_counts().tolist()
    labels = ['married', 'divorced', 'single']
    data = [go.Bar(x=labels,y=vals,marker=dict(color="#FE9A2E"))]
    layout = go.Layout(title="Count by Marital Status")
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,  filename='basic-bar')
    
    st.write("#### Distribution of Balances by Marital status")

    single = df['balance'].loc[df['marital'] == 'single'].values
    married = df['balance'].loc[df['marital'] == 'married'].values
    divorced = df['balance'].loc[df['marital'] == 'divorced'].values


    single_dist = go.Histogram(x=single,histnorm='density', name='single',marker=dict(color='#6E6E6E'))
    married_dist = go.Histogram(x=married,histnorm='density', name='married',marker=dict(color='#2E9AFE'))
    divorced_dist = go.Histogram(x=divorced,histnorm='density', name='divorced',marker=dict(color='#FA5858'))


    fig = tools.make_subplots(rows=3, print_grid=False)

    fig.append_trace(single_dist, 1, 1)
    fig.append_trace(married_dist, 2, 1)
    fig.append_trace(divorced_dist, 3, 1)


    fig['layout'].update(showlegend=False, title="Price Distributions by Marital Status",
                         height=1000, width=800)

    st.plotly_chart(fig,  filename='custom-sized-subplot-with-subplot-titles')

    st.write("Observe the significantly lower account balances among individuals who are divorced")

    fig = ff.create_facet_grid(df,x='duration',y='balance',color_name='marital',
                               show_boxes=False,marker={'size': 10, 'opacity': 1.0},
                               colormap={'single': 'rgb(165, 242, 242)', 'married': 'rgb(253, 174, 216)', 'divorced': 'rgba(201, 109, 59, 0.82)'})
    st.plotly_chart(fig,  filename='facet - custom colormap')

    st.write("#### Verify clients with significant balances")
    fig = ff.create_facet_grid(df,y='balance',facet_row='marital',facet_col='deposit',
    trace_type='box')
    st.plotly_chart(fig,filename='facet - box traces')

    st.write("### I.3 Clustering Marital Status and Education")
    st.write("**Marital Status:** As previously discussed, divorce can greatly influence an individual's balance.")
    st.write("**Education:** The level of education similarly influences the prospective balance significantly.")
    st.write("**Loans:** The presence of prior loans significantly affects the prospective balance.")
    df = df.drop(df.loc[df["education"] == "unknown"].index)
    df['education'].unique()
    df['marital/education'] = np.nan
    lst = [df]

    for col in lst:
        col.loc[(col['marital'] == 'single') & (df['education'] == 'primary'), 'marital/education'] = 'single/primary'
        col.loc[(col['marital'] == 'married') & (df['education'] == 'primary'), 'marital/education'] = 'married/primary'
        col.loc[(col['marital'] == 'divorced') & (df['education'] == 'primary'), 'marital/education'] = 'divorced/primary'
        col.loc[(col['marital'] == 'single') & (df['education'] == 'secondary'), 'marital/education'] = 'single/secondary'
        col.loc[(col['marital'] == 'married') & (df['education'] == 'secondary'), 'marital/education'] = 'married/secondary'
        col.loc[(col['marital'] == 'divorced') & (df['education'] == 'secondary'), 'marital/education'] = 'divorced/secondary'
        col.loc[(col['marital'] == 'single') & (df['education'] == 'tertiary'), 'marital/education'] = 'single/tertiary'
        col.loc[(col['marital'] == 'married') & (df['education'] == 'tertiary'), 'marital/education'] = 'married/tertiary'
        col.loc[(col['marital'] == 'divorced') & (df['education'] == 'tertiary'), 'marital/education'] = 'divorced/tertiary'
    
    st.write("#### Marital_Education_Combinations")
    st.write("Create combinations of marital status and education level")
    st.dataframe(df.head(10))

    st.write("#### Marital_Education_Balance_Distribution")
    st.write("Visualizes the distribution of balances based on marital status and education level")
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="marital/education", hue="marital/education", aspect=12, palette=pal)

    g.map(sns.kdeplot, "balance", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "balance", clip_on=False, color="w", lw=1, bw=0)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    st.pyplot()

    education_groups = df.groupby(['marital/education'], as_index=False)['balance'].median()
    fig = plt.figure(figsize=(12,8))
    sns.barplot(x="balance", y="marital/education", data=education_groups,label="Total", palette="RdBu")
    plt.title('Median Balance by Educational/Marital Group', fontsize=16)
    st.write(fig)

    st.write("#### Loan Distribution Across Marital/Education Groups")

    loan_balance = df.groupby(['marital/education', 'loan'], as_index=False)['balance'].median()

    no_loan = loan_balance['balance'].loc[loan_balance['loan'] == 'no'].values
    has_loan = loan_balance['balance'].loc[loan_balance['loan'] == 'yes'].values

    labels = loan_balance['marital/education'].unique().tolist()

    trace0 = go.Scatter(x=no_loan,y=labels,mode='markers',name='No Loan',marker=dict(color='rgb(175,238,238)',line=dict(color='rgb(0,139,139)',width=1),symbol='circle',size=16,))
    trace1 = go.Scatter(x=has_loan,y=labels,mode='markers',name='Has a Previous Loan', marker=dict(color='rgb(250,128,114)',line=dict(color='rgb(178,34,34)',width=1,),symbol='circle',size=16))
    
    data = [trace0, trace1]
    layout = go.Layout(title="The Impact of Loans to Married/Educational Clusters",
                       xaxis=dict(showgrid=False,showline=True,linecolor='rgb(102, 102, 102)',
                                  titlefont=dict(color='rgb(204, 204, 204)'),
                                  tickfont=dict(color='rgb(102, 102, 102)'),
                                  showticklabels=False,dtick=10,ticks='outside',tickcolor='rgb(102, 102, 102)'),
                        margin=dict(l=140,r=40,b=50,t=80),
                        legend=dict(font=dict(size=10),yanchor='middle',xanchor='right'),
                        width=1000,
                        height=800,
                        paper_bgcolor='rgb(255,250,250)',
                        plot_bgcolor='rgb(255,255,255)',
                        hovermode='closest'
                        )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,filename='lowest-oecd-votes-cast')
    
    st.write("#### Marital/Education Pairwise Relationship Visualization")
    sns.set(style="ticks")
    sns.pairplot(df, hue="marital/education", palette="Set1")
    st.pyplot()

    st.write("#### Balances Distribution by Job and Deposit Status")

    fig = plt.figure(figsize=(12,8))
    sns.violinplot(x="balance", y="job", hue="deposit", palette="RdBu_r",data=df);
    plt.title("Job Distribution of Balances by Deposit Status", fontsize=16)
    st.pyplot()

    st.write("### I.4 Campaign Duration")
    st.write("**Campaign Duration:** We've noticed a strong correlation between the duration and term deposits, indicating that longer campaign durations tend to result in a higher likelihood of clients opening term deposits.")
    st.write("**Mean Campaign Duration:** The mean duration of the campaign is 374.76. Now, let's investigate whether clients who were above this average duration were more inclined to open a term deposit.")
    st.write("**Duration Status Analysis:** Individuals exceeding the average duration were significantly more inclined to open term deposits. Within this group, 78% opened term deposit accounts, compared to only 32% among those below the average duration. This suggests that targeting individuals above the average duration could be a strategic approach.")
    st.write("#### Remove the variables ""marital/education"" and ""balance status."" ")
    df.drop(['marital/education', 'balance_status'], axis=1, inplace=True)
    st.dataframe(df.head())

    st.write("#### Correlation Analysis of Duration and Term Deposits")
    fig = plt.figure(figsize=(12,8))
    df['deposit'] = LabelEncoder().fit_transform(df['deposit'])

    # Separate both dataframes into 
    numeric_df = df.select_dtypes(exclude="object")

    # categorical_df = df.select_dtypes(include="object")
    corr_numeric = numeric_df.corr()

    sns.heatmap(corr_numeric, cbar=True, cmap="RdBu_r")
    plt.title("Correlation Matrix", fontsize=16)
    st.pyplot()

    st.write("#### The Impact of Duration in Opening a Term Deposit")

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_style('whitegrid')
    avg_duration = df['duration'].mean()

    lst = [df]
    df["duration_status"] = np.nan

    for col in lst:
        col.loc[col["duration"] < avg_duration, "duration_status"] = "below_average"
        col.loc[col["duration"] > avg_duration, "duration_status"] = "above_average"
    
    pct_term = pd.crosstab(df['duration_status'], df['deposit']).apply(lambda r: round(r/r.sum(), 2) * 100, axis=1)

    ax = pct_term.plot(kind='bar', stacked=False, cmap='RdBu')
    plt.title("The Impact of Duration \n in Opening a Term Deposit", fontsize=18)
    plt.xlabel("Duration Status", fontsize=18);
    plt.ylabel("Percentage (%)", fontsize=18)

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))
    st.pyplot()

elif page == pages[2] :
    st.write( " # Classification Model")
    df = pd.read_csv('bank.csv', sep = ",")

    dep = df['deposit']
    df.drop(labels=['deposit'], axis=1,inplace=True)
    df.insert(0, 'deposit', dep)
    st.dataframe(df.head())
    st.dataframe(df["housing"].value_counts()/len(df))
    st.write("The results indicate that housing has a negative correlation of -20% with term deposits. Upon examining the distribution, it's observed that 52% of the dataset consists of individuals without housing loans (denoted as ""no""), while 47% have housing loans (denoted as ""yes"").")
    st.dataframe(df["loan"].value_counts()/len(df))
    st.write("The results show that 86.92% of the dataset consists of individuals without personal loans (denoted as ""no""), while only 13.08% have personal loans (denoted as ""yes"").")    
    
    st.write("## I. Stratified Sampling")
    st.write("**Stratified Sampling** is a critical concept often overlooked during model development, be it for regression or classification tasks. It's essential to implement cross-validation to prevent overfitting, but we also need to ensure that features with the most significant influence on our target label (whether a potential client will open a term deposit or not) are evenly distributed. This means that each category or subgroup within these influential features should be represented proportionally in the training and testing datasets.")
    st.write("**Personal Loans:** For example, the presence of a personal loan is a significant determinant of whether a potential client will open a term deposit. To validate its considerable impact, you can refer to the correlation matrix above, where it exhibits a -11% correlation with deposit opening. Before implementing stratified sampling in our training and testing data, what precautions should we consider?")
    st.write("1) We need to examine the distribution of our data.")
    st.write("2) Upon observing that the loan column comprises 87% ""no"" (indicating no personal loans) and 13% ""yes"" (indicating presence of personal loans).")
    st.write("3) We aim to ensure that both our training and test sets maintain the same proportion of 87% ""no"" and 13% ""yes"" for the loan column.")
    # Here we split the data into training and test sets and implement a stratified shuffle split.
    stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_set, test_set in stratified.split(df, df["loan"]):
        stratified_train = df.iloc[train_set]
        stratified_test = df.iloc[test_set]
    st.write("- Proportion of Loan Categories in Stratified Training Data")
    st.dataframe(stratified_train["loan"].value_counts()/len(df))
    st.write("- Proportion of Loan Categories in Stratified Testing Data")
    st.dataframe(stratified_test["loan"].value_counts()/len(df))

    st.write("#### Separate the labels and the features")
    train_data = stratified_train # Make a copy of the stratified training set.
    test_data = stratified_test

    st.write("- Dimensions of Training Data")
    train_data.shape

    st.write("- Dimensions of Testing Data")
    test_data.shape

    st.write("Display Deposit Distribution in Training Data")
    st.write(train_data['deposit'].value_counts())

    st.write("#### Constructing Pipelines")
    class CategoricalEncoder(BaseEstimator, TransformerMixin):

        def __init__(self, encoding='onehot', categories='auto', dtype=np.float64 ,handle_unknown='error'):
            self.encoding = encoding
            self.categories = categories
            self.dtype = dtype
            self.handle_unknown = handle_unknown

        def fit(self, X, y=None):
            if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
                template = ("encoding should be either 'onehot', 'onehot-dense' "
                            "or 'ordinal', got %s")
                raise ValueError(template % self.handle_unknown)

            if self.handle_unknown not in ['error', 'ignore']:
                template = ("handle_unknown should be either 'error' or "
                           "'ignore', got %s")
                raise ValueError(template % self.handle_unknown)

            if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
                raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

            X = check_array(X, dtype=np.object_, accept_sparse='csc', copy=True)
            n_samples, n_features = X.shape

            self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

            for i in range(n_features):
                le = self._label_encoders_[i]
                Xi = X[:, i]
                if self.categories == 'auto':
                    le.fit(Xi)
                else:
                    valid_mask = np.in1d(Xi, self.categories[i])

                    if not np.all(valid_mask):
                        if self.handle_unknown == 'error':
                            diff = np.unique(Xi[~valid_mask])
                            msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                            raise ValueError(msg) 
                    
                    le.classes_ = np.array(np.sort(self.categories[i]))

            self.categories_ = [le.classes_ for le in self._label_encoders_]

            return self

        def transform(self, X):
            X = check_array(X, accept_sparse='csc', dtype=np.object_, copy=True)
            n_samples, n_features = X.shape
            X_int = np.zeros_like(X, dtype=np.int64)
            X_mask = np.ones_like(X, dtype=np.bool_)

            for i in range(n_features):
                valid_mask = np.in1d(X[:, i], self.categories_[i])

                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(X[~valid_mask, i])
                        msg = ("Found unknown categories {0} in column {1}"
                                 " during transform".format(diff, i))
                        raise ValueError(msg)
                    else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                        X_mask[:, i] = valid_mask
                        X[:, i][~valid_mask] = self.categories_[i][0]
                X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

            if self.encoding == 'ordinal':
                return X_int.astype(self.dtype, copy=False)

            mask = X_mask.ravel()
            n_values = [cats.shape[0] for cats in self.categories_]
            n_values = np.array([0] + n_values)
            indices = np.cumsum(n_values)

            column_indices = (X_int + indices[:-1]).ravel()[mask]
            row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),n_features)[mask]
            data = np.ones(n_samples * n_features)[mask]

            out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
            if self.encoding == 'onehot-dense':
                return out.toarray()
            else:
                return out
    # A class to select numerical or categorical columns 
    # since Scikit-Learn doesn't handle DataFrames yet
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X[self.attribute_names]
    
    train_data.info()

    # Assuming df is your DataFrame
    df['job'] = df['job'].astype('category').cat.codes
    df['marital'] = df['marital'].astype('category').cat.codes
    df['education'] = df['education'].astype('category').cat.codes
    df['default'] = df['default'].astype('category').cat.codes
    df['housing'] = df['housing'].astype('category').cat.codes
    df['loan'] = df['loan'].astype('category').cat.codes
    df['contact'] = df['contact'].astype('category').cat.codes
    df['month'] = df['month'].astype('category').cat.codes
    df['poutcome'] = df['poutcome'].astype('category').cat.codes

    # Making pipelines
    numerical_pipeline = Pipeline([("select_numeric", DataFrameSelector(["age", "balance", "day", "campaign", "pdays", "previous","duration"])),
                                   ("std_scaler", StandardScaler())])

    categorical_pipeline = Pipeline([("select_cat", DataFrameSelector(["job", "education", "marital", "default", "housing", "loan", "contact", "month",
                                     "poutcome"])),("cat_encoder", CategoricalEncoder(encoding='onehot-dense'))])

    preprocess_pipeline = FeatureUnion(transformer_list=[("numerical_pipeline", numerical_pipeline),
                                                         ("categorical_pipeline", categorical_pipeline)])
    
    X_train = preprocess_pipeline.fit_transform(train_data)
    st.dataframe(X_train)

    y_train = train_data['deposit']
    y_test = test_data['deposit']

    encode = LabelEncoder()
    y_train = encode.fit_transform(y_train)
    y_test = encode.fit_transform(y_test)
    y_train_yes = (y_train == 1)

    some_instance = X_train[1250]

    st.write("## Models")
    dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=18),
    "Neural Net": MLPClassifier(alpha=1),
    "Naive Bayes": GaussianNB()
    }

    #  Thanks to Ahspinar for the function. 
    no_classifiers = len(dict_classifiers.keys())

    def batch_classify(X_train, Y_train, verbose = True):
        df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers,3)), columns = ['classifier', 'train_score', 'training_time'])
        count = 0
        for key, classifier in dict_classifiers.items():
            t_start = time.perf_counter()
            classifier.fit(X_train, Y_train)
            t_end = time.perf_counter()
            t_diff = t_end - t_start
            train_score = classifier.score(X_train, Y_train)
            df_results.loc[count,'classifier'] = key
            df_results.loc[count,'train_score'] = train_score
            df_results.loc[count,'training_time'] = t_diff
            if verbose:
                print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
            count+=1
        return df_results
    df_results = batch_classify(X_train, y_train)
    st.dataframe(df_results.sort_values(by='train_score', ascending=False))

    st.write("## Avoiding Overfitting")
    st.write("**Overfitting** occurs when a modeling algorithm incorporates random noise during the fitting process rather than focusing on the underlying pattern in the data. This phenomenon becomes evident when the model performs exceptionally well on the training set but poorly on unseen data, such as a test set. Overfitting is likely to happen when the model captures the noise in the data instead of the true underlying pattern. In our context, we aim for the model to accurately generalize the overall pattern of the data, enabling it to correctly classify whether a potential client will subscribe to a term deposit or not. In the provided examples, it's probable that the Decision Tree Classifier and Random Forest classifiers are overfitting, as indicated by their nearly perfect accuracy scores of 100% and 99%, respectively.")
    st.write("To mitigate overfitting, employing cross-validation is a robust approach. This technique involves partitioning the training dataset into multiple subsets. For example, if we split it into three parts, two-thirds (66%) of the data are used for training, and the remaining one-third (33%) is reserved for testing. This process is repeated multiple times, iterating through different training and test sets. The overarching goal of cross-validation is to capture the general pattern of the data by systematically varying the training and test data.")
    
    # Logistic Regression
    log_reg = LogisticRegression()
    log_scores = cross_val_score(log_reg, X_train, y_train, cv=3)
    log_reg_mean = log_scores.mean()

    # SVC
    svc_clf = SVC()
    svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=3)
    svc_mean = svc_scores.mean()

    # KNearestNeighbors
    knn_clf = KNeighborsClassifier()
    knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=3)
    knn_mean = knn_scores.mean()

    # Decision Tree
    tree_clf = tree.DecisionTreeClassifier()
    tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=3)
    tree_mean = tree_scores.mean()

    # Gradient Boosting Classifier
    grad_clf = GradientBoostingClassifier()
    grad_scores = cross_val_score(grad_clf, X_train, y_train, cv=3)
    grad_mean = grad_scores.mean()

    # Random Forest Classifier
    rand_clf = RandomForestClassifier(n_estimators=18)
    rand_scores = cross_val_score(rand_clf, X_train, y_train, cv=3)
    rand_mean = rand_scores.mean()

    # NeuralNet Classifier
    neural_clf = MLPClassifier(alpha=1)
    neural_scores = cross_val_score(neural_clf, X_train, y_train, cv=3)
    neural_mean = neural_scores.mean()

    # Naives Bayes
    nav_clf = GaussianNB()
    nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=3)
    nav_mean = neural_scores.mean()

    # Create a Dataframe with the results.
    d = {'Classifiers': ['Logistic Reg.', 'SVC', 'KNN', 'Dec Tree', 'Grad B CLF', 'Rand FC', 'Neural Classifier', 'Naives Bayes'], 
    'Crossval Mean Scores': [log_reg_mean, svc_mean, knn_mean, tree_mean, grad_mean, rand_mean, neural_mean, nav_mean]}

    result_df = pd.DataFrame(data=d)

    # All our models perform well but I will go with GradientBoosting.
    result_df = result_df.sort_values(by=['Crossval Mean Scores'], ascending=False)
    st.dataframe(result_df)

    st.write("## Confusion Matrix")
    st.image("matrix.png")
    st.write(" **Insights of a Confusion Matrix:** The confusion matrix provides valuable insights into the performance of our model, particularly in classifying potential clients likely to subscribe to a term deposit. It consists of four key terms: True Positives, False Positives, True Negatives, and False Negatives. These terms help us understand how accurately our model is identifying positive and negative cases. ")
    st.write("**Positive/Negative:** Positive and negative refer to the types of classes or labels in our dataset, typically represented as [""No"", ""Yes""]. True and false indicate whether a data point has been correctly or incorrectly classified by the model. ")
    st.write("**True Negatives (Top-Left Square):** True Negatives, found in the top-left square of the confusion matrix, represent the count of correctly classified instances belonging to the ""No"" class or potential clients who are not willing to subscribe to a term deposit.")
    st.write("**False Negatives (Top-Right Square):** False Positives, located in the top-right square of the confusion matrix, represent the count of incorrectly classified instances belonging to the ""No"" class or potential clients who are not willing to subscribe to a term deposit but have been wrongly predicted as willing to do so.")
    st.write("**False Positives (Bottom-Left Square):** False Negatives, situated in the bottom-left square of the confusion matrix, represent the count of incorrectly classified instances belonging to the ""Yes"" class or potential clients who are willing to subscribe to a term deposit but have been wrongly predicted as not willing to do so.")
    st.write("**True Positives (Bottom-Right Square):** True Positives, located in the bottom-right square of the confusion matrix, represent the count of correctly classified instances belonging to the ""Yes"" class or potential clients who are willing to subscribe to a term deposit.")
    y_train_pred = cross_val_predict(grad_clf, X_train, y_train, cv=3)
    grad_clf.fit(X_train, y_train)
    st.write("Gradient Boost Classifier accuracy is : ", "%2.2f" % accuracy_score(y_train, y_train_pred))

    conf_matrix = confusion_matrix(y_train, y_train_pred)
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.title("Confusion Matrix", fontsize=20)
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels("")
    ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)
    st.pyplot(f)

    st.write("## Precision and Recall ")
    st.write(" **Recall**, also known as sensitivity or true positive rate, measures the proportion of actual positive cases (i.e., ""Yes"" labels) that our model correctly identifies. In other words, it indicates how many of the ""Yes"" labels in the dataset our model successfully detects.")
    st.write("**Precision** quantifies the accuracy of positive predictions made by the model. Specifically, it measures the proportion of correctly predicted ""Yes"" labels among all instances predicted as ""Yes"" by the model. In simpler terms, precision indicates how confident the model is when it predicts that the actual label is ""Yes"".")
    st.write("**Recall-Precision Tradeoff:** As precision increases, recall tends to decrease, and vice versa. For example, if we raise precision from 30% to 60%, the model becomes more selective in its predictions, focusing on instances it believes to be 60% likely to be a ""Yes"". However, this increased selectivity may cause the model to miss instances where it's less certain but still correct. For instance, if the model encounters an instance with a 58% likelihood of being a ""Yes"", it might classify it as a ""No"", even though it is actually a ""Yes"" (i.e., the potential client did subscribe to a term deposit). Consequently, higher precision often leads to a decrease in recall, as the model becomes more conservative in its predictions and may overlook instances that are actually positive.")

    # The model is only retaining 60% of clients that agree to suscribe a term deposit.
    st.write("Precision Score: ", precision_score(y_train, y_train_pred))
    # The classifier only detects 60% of potential clients that will suscribe to a term deposit.
    st.write("Recall Score: ",  recall_score(y_train, y_train_pred))

    st.write("F1 score : ", f1_score(y_train, y_train_pred))

    y_scores = grad_clf.decision_function([some_instance])

    # Increasing the threshold decreases the recall.
    threshold = 0
    y_some_digit_pred = (y_scores > threshold)

    y_scores = cross_val_predict(grad_clf, X_train, y_train, cv=3, method="decision_function")
    neural_y_scores = cross_val_predict(neural_clf, X_train, y_train, cv=3, method="predict_proba")
    naives_y_scores = cross_val_predict(nav_clf, X_train, y_train, cv=3, method="predict_proba")

    if y_scores.ndim == 2:
        y_scores = y_scores[:, 1]

    if neural_y_scores.ndim == 2:
        neural_y_scores = neural_y_scores[:, 1]
    
    if naives_y_scores.ndim == 2:
        naives_y_scores = naives_y_scores[:, 1]

    precisions, recalls, threshold = precision_recall_curve(y_train, y_scores)

    st.write( "### Plot_precision_recall_tradeoff")
    def precision_recall_curve(precisions, recalls, thresholds):
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(thresholds, precisions[:-1], "r--", label="Precisions")
        plt.plot(thresholds, recalls[:-1], "#424242", label="Recalls")
        plt.title("Precision and Recall \n Tradeoff", fontsize=18)
        plt.ylabel("Level of Precision and Recall", fontsize=16)
        plt.xlabel("Thresholds", fontsize=16)
        plt.legend(loc="best", fontsize=14)
        plt.xlim([-2, 4.7])
        plt.ylim([0, 1])
        plt.axvline(x=0.13, linewidth=3, color="#0B3861")
        plt.annotate('Best Precision and \n Recall Balance \n is at 0.13 \n threshold ', xy=(0.13, 0.83), xytext=(55, -40),
                     textcoords="offset points",arrowprops=dict(facecolor='black', shrink=0.05),fontsize=12, color='k')
        st.pyplot(fig)

    precision_recall_curve(precisions, recalls, threshold)

    st.write("## ROC Curve (Receiver Operating Characteristic)")

    st.write("The **ROC curve** provides insight into the classifier's ability to distinguish between term deposit subscriptions (True Positives) and non-term deposit subscriptions. It plots the False Positive Rate (Specificity) on the **X-axis** against the True Positive Rate (Sensitivity) on the **Y-axis**. As the curve shifts, the classification threshold changes, yielding varying values. The closer the curve is to the top-left corner, the better the model is at accurately separating both classes.")
    
    # Gradient Boosting Classifier
    # Neural Classifier
    # Naives Bayes Classifier
    grd_fpr, grd_tpr, thresold = roc_curve(y_train, y_scores)
    neu_fpr, neu_tpr, neu_threshold = roc_curve(y_train, neural_y_scores)
    nav_fpr, nav_tpr, nav_threshold = roc_curve(y_train, naives_y_scores)

    st.write("### plot_roc_curve_gbclassifier")

    def graph_roc_curve(false_positive_rate, true_positive_rate, label=None):
        plt.figure(figsize=(10,6))
        plt.title('ROC Curve \n Gradient Boosting Classifier', fontsize=18)
        plt.plot(false_positive_rate, true_positive_rate, label=label)
        plt.plot([0, 1], [0, 1], '#0C8EE0')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.annotate('ROC Score of 91.73% \n (Not the best score)', xy=(0.25, 0.9), xytext=(0.4, 0.85),
                     arrowprops=dict(facecolor='#F75118', shrink=0.05))
        plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                     arrowprops=dict(facecolor='#F75118', shrink=0.05))
        st.pyplot()
        
    graph_roc_curve(grd_fpr, grd_tpr, threshold)

    st.write("**Gradient Boost Classifier Score:**", roc_auc_score(y_train, y_scores))

    st.write("**Neural Classifier Score:** " , roc_auc_score(y_train, neural_y_scores))

    st.write("**Naives Bayes Classifier:**", roc_auc_score(y_train, naives_y_scores))

    st.write("### Plot_roc_curve top3 classifiers")
    def graph_roc_curve_multiple(grd_fpr, grd_tpr, neu_fpr, neu_tpr, nav_fpr, nav_tpr):
        plt.figure(figsize=(8,6))
        plt.title('ROC Curve \n Top 3 Classifiers', fontsize=18)
        plt.plot(grd_fpr, grd_tpr, label='Gradient Boosting Classifier (Score = 91.72%)')
        plt.plot(neu_fpr, neu_tpr, label='Neural Classifier (Score = 91.54%)')
        plt.plot(nav_fpr, nav_tpr, label='Naives Bayes Classifier (Score = 80.33%)')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                    arrowprops=dict(facecolor='#6E726D', shrink=0.05))
        plt.legend()
        st.pyplot()
    graph_roc_curve_multiple(grd_fpr, grd_tpr, neu_fpr, neu_tpr, nav_fpr, nav_tpr)

    st.write("## What factors impact the likelihood of subscribing to a term deposit?")
     
    st.write("### Decision Tree Classifier:")
    st.write("The three key factors that significantly influence our classifier are the duration of the conversation between the sales representative and the potential client, the number of contacts made to the potential client within the same marketing campaign, and the month of the year.")

    plt.style.use('seaborn-v0_8-darkgrid')
    # Let's create new splittings like before but now we modified the data so we need to do it one more time.
    # Create train and test splits

    target_name = 'deposit'
    X = df.drop('deposit', axis=1)
    st.dataframe(X)

    label=df[target_name]
    st.dataframe(label)
    
    st.write("#### Splitting Data into Training and Testing Sets")
    X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=0.2, random_state=42, stratify=label)

    st.write("Given 'X_train' as your feature dataset containing categorical variables, we split the data into training and testing sets using a test size of 20%, a random state of 42, and stratifying by the labels. Then, we encode the categorical variables in 'X_train' using one-hot encoding.")

    # Assuming 'X_train' is your feature dataset containing categorical variables
    X_train_encoded = pd.get_dummies(X_train)     
    st.dataframe(X_train_encoded)
    
    st.write("#### Decision Tree Feature Importance Analysis") 
    # Build a classification task using 3 informative features
    tree_model = DecisionTreeClassifier(class_weight='balanced',min_weight_fraction_leaf = 0.01)

    tree = tree_model.fit(X_train_encoded, y_train)
    importances = tree_model.feature_importances_
    feature_names = df.drop('deposit', axis=1).columns
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    st.write("#### **Feature ranking:**")
    for f in range(X_train.shape[1]) :
        st.write( "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        #st.dataframe( ["%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])])
    
    st.write("#### Visualizing Feature Importance with Decision Tree Classifier")

    def feature_importance_graph(indices, importances, feature_names):
        plt.figure(figsize=(12, 6))
        plt.title("Determining Feature importances \n with DecisionTreeClassifier", fontsize=18)
        plt.barh(range(len(indices)), importances[indices], color='#31B173',  align="center")
    
        # Ensure that indices are within the bounds of feature_names
        valid_indices = [idx for idx in indices if idx < len(feature_names)]
    
        plt.yticks(range(len(valid_indices)), [feature_names[idx] for idx in valid_indices], rotation='horizontal', fontsize=15)
        plt.ylim([-1, len(valid_indices)])
        plt.axhline(y=1.85, xmin=0.21, xmax=0.952, color='k', linewidth=3, linestyle='--')
        st.pyplot()
        
        #plt.text(0.30, 2.8, '46% Difference between \n duration and contacts', color='k', fontsize=15)
    feature_importance_graph(indices, importances, feature_names)

    st.write("### GradientBoosting Classifier ")
    st.write("The Gradient Boosting classifier stands out as the optimal model for predicting whether a potential client will subscribe to a term deposit, achieving an impressive accuracy rate of 84%.")

    voting_clf = VotingClassifier( estimators=[('gbc', grad_clf), ('nav', nav_clf), ('neural', neural_clf)],voting='soft')
    voting_clf.fit(X_train_encoded , y_train)

    X_test_encoded = pd.get_dummies(X_test)

    # Now, you can proceed with prediction
    for clf in (grad_clf, nav_clf, neural_clf, voting_clf):
        clf.fit(X_train_encoded, y_train)
        predict = clf.predict(X_test_encoded)
        #st.write(clf.__class__.__name__)
        st.write(clf.__class__.__name__, accuracy_score(y_test, predict))

elif page == pages[3] :

    st.write("# What steps should the bank take into consideration?")
    st.write("## Strategies for the upcoming marketing campaign")

    st.write("**1) Optimal Months for Marketing Campaign:** Our analysis revealed that May witnessed the highest level of marketing activity, yet potential clients were least likely to accept term deposit offers during this period (lowest effective rate: -34.49%). Therefore, for the upcoming marketing campaign, it would be prudent for the bank to concentrate its efforts on March, September, October, and December. December, despite having the lowest marketing activity, should also be considered, as there may be underlying factors influencing client behavior during this period.")

    st.write("**2) Leveraging Seasonal Trends:** Our analysis indicates that potential clients showed a preference for subscribing to term deposits during the fall and winter seasons. Therefore, the next marketing campaign should strategically target its activities during these seasons to capitalize on this trend.")

    st.write("**3) Limiting Campaign Calls:** Implementing a policy restricting the number of calls to each potential client to a maximum of three can help conserve time and effort while also preventing potential client fatigue. It's crucial to remember that excessive calls to the same potential client can increase the likelihood of rejection for opening a term deposit.")

    st.write("**4) Targeting Age Segments:** To optimize the next marketing campaign, the bank should focus on two specific age brackets: individuals aged 20 or younger and those aged 60 or older. Our analysis indicates that the youngest category exhibited a 60% likelihood of subscribing to a term deposit, while the eldest category showed a 76% likelihood. By directing campaign efforts towards these age segments, the bank can potentially boost the number of term deposit subscriptions.")

    st.write("**5) Occupational Influence:** It comes as no surprise that students and retired individuals are among the most inclined to subscribe to a term deposit. Retired individuals often opt for term deposits to earn interest on their savings, as they are less likely to require immediate access to their funds. Term deposits function as short-term loans, where the individual, particularly retirees, agrees not to withdraw the deposited amount until a predetermined date. Subsequently, they receive both their initial capital and the accumulated interest. Given their tendency to be more conservative with their finances, retired individuals are inclined to invest their cash by lending it to financial institutions. Similarly, students are another demographic group known to subscribe to term deposits.")

    st.write("**6) Impact of House Loans and Balances:** Our analysis reveals that potential clients with low or no balances are more prone to having house loans compared to those in the average and high balance categories. Having a house loan implies that the individual has financial commitments to repay, leaving limited available funds for subscribing to a term deposit account. Conversely, potential clients with average and high balances are less likely to have house loans, making them more inclined to open a term deposit. Therefore, the upcoming marketing campaign should prioritize individuals with average and high balances to enhance the probability of term deposit subscriptions.")

    st.write("**7) Implementing a Questionnaire during Calls:** Given that the duration of the call positively correlates with the likelihood of a potential client opening a term deposit, introducing an engaging questionnaire during calls could potentially prolong the conversation. While this strategy does not guarantee term deposit subscriptions, it enhances engagement with potential clients, thereby increasing the likelihood of subscription. Implementing such a strategy incurs minimal risk while potentially improving the effectiveness of the bank's next marketing campaign.")

    st.write("**8) Focusing on High Duration Individuals:** Targeting individuals with a duration above the average threshold of 375 presents a lucrative opportunity, as this group exhibits a significantly higher likelihood of opening a term deposit account. With a probability of 78%, this target demographic shows considerable potential for term deposit subscriptions. By concentrating efforts on this high-duration group, the success rate of the next marketing campaign is poised to achieve remarkable success.")

    st.write("By integrating these strategies and refining the target audience for the next campaign, the bank can anticipate a higher level of effectiveness compared to the current one.")

    





































