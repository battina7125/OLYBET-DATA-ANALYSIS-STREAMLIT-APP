import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the data
df = pd.read_excel('Daily Summary by Country_test data.xlsx')
# Strip any whitespace from the column names
df.columns = df.columns.str.strip()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title of the app
st.title('OlyBet Data Analysis Overview')
# Descriptive Statistics
st.header("Descriptive Statistics")
descriptive_stats = df.describe()
st.write(descriptive_stats)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Correlation Analysis
st.header("Correlation Analysis")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True, ax=ax)
ax.set_title('Correlation Matrix of Key Variables', fontsize=16)
st.pyplot(fig)


# Dropdown for Insights and Recommendations
st.subheader("Insights and Recommendations")

correlation_options = [
    "Select an option",
    "Total Revenue Correlations",
    "Total Sign Ups Correlations",
    "Total Customers Correlations",
    "Total Deposit Amount Correlations",
    "Sports Revenue vs. Casino Revenue Correlations"
]

selected_correlation = st.selectbox("Select a correlation analysis to view insights and recommendations:", correlation_options)

if selected_correlation == "Select an option":
    st.info("Please select a correlation analysis from the dropdown to view insights and recommendations.")
else:
    if selected_correlation == "Total Revenue Correlations":
        revenue_signups_corr = correlation_matrix.loc['Total Revenue', 'Total Sign Ups']
        revenue_customers_corr = correlation_matrix.loc['Total Revenue', 'Total Customers']
        revenue_deposits_corr = correlation_matrix.loc['Total Revenue', 'Total Deposit Amount']
        revenue_sports_corr = correlation_matrix.loc['Total Revenue', 'Total Sports Revenue']
        revenue_casino_corr = correlation_matrix.loc['Total Revenue', 'Total Casino Revenue']

        st.write(f"#### Total Revenue vs Total Sign Ups: Correlation = {revenue_signups_corr:.2f}")
        st.write("  - **Insight**: The correlation between Total Revenue and Total Sign Ups is relatively low. This suggests that simply increasing sign-ups may not directly lead to a proportional increase in revenue.")
        st.write("  - **Recommendation**: Focus on converting sign-ups into paying customers by enhancing the onboarding process and offering early incentives.")

        st.write(f"#### Total Revenue vs Total Customers: Correlation = {revenue_customers_corr:.2f}")
        st.write("  - **Insight**: There is a moderate positive correlation between Total Revenue and Total Customers, indicating that a higher customer base tends to increase revenue.")
        st.write("  - **Recommendation**: Implement strategies to retain and engage existing customers to drive higher revenue. Personalized offers could be beneficial.")

        st.write(f"#### Total Revenue vs Total Deposit Amount: Correlation = {revenue_deposits_corr:.2f}")
        st.write("  - **Insight**: The strong correlation between Total Revenue and Total Deposit Amount suggests that deposits are a significant driver of revenue.")
        st.write("  - **Recommendation**: Encourage higher deposits through promotional offers and loyalty programs. Ensure a smooth and secure deposit process to foster customer trust.")

        st.write(f"#### Total Revenue vs Total Sports Revenue: Correlation = {revenue_sports_corr:.2f}")
        st.write("  - **Insight**: A moderate correlation between Total Revenue and Total Sports Revenue indicates that sports revenue contributes substantially to the overall revenue.")
        st.write("  - **Recommendation**: Continue to invest in sports betting features and marketing to maximize this revenue stream.")

        st.write(f"#### Total Revenue vs Total Casino Revenue: Correlation = {revenue_casino_corr:.2f}")
        st.write("  - **Insight**: The high correlation between Total Revenue and Total Casino Revenue indicates that the casino segment is a major contributor to overall revenue.")
        st.write("  - **Recommendation**: Explore opportunities to expand casino offerings and target promotions specifically to casino users to further boost revenue.")

    elif selected_correlation == "Total Sign Ups Correlations":
        signups_customers_corr = correlation_matrix.loc['Total Sign Ups', 'Total Customers']
        signups_deposits_corr = correlation_matrix.loc['Total Sign Ups', 'Total Deposit Amount']

        st.write(f"#### Total Sign-Ups vs Total Customers: Correlation = {signups_customers_corr:.2f}")
        st.write("  - **Insight**: The correlation between Total Sign-Ups and Total Customers is moderate, indicating that sign-ups often convert into customers.")
        st.write("  - **Recommendation**: Improve the sign-up experience and follow up with targeted engagement to convert new sign-ups into active customers.")

        st.write(f"#### Total Sign-Ups vs Total Deposit Amount: Correlation = {signups_deposits_corr:.2f}")
        st.write("  - **Insight**: The correlation suggests that new sign-ups also tend to make deposits, although the relationship isn't extremely strong.")
        st.write("  - **Recommendation**: Consider offering deposit bonuses or initial deposit match offers to incentivize first-time deposits from new sign-ups.")

    elif selected_correlation == "Total Customers Correlations":
        customers_deposits_corr = correlation_matrix.loc['Total Customers', 'Total Deposit Amount']
        customers_sports_corr = correlation_matrix.loc['Total Customers', 'Total Sports Customers']
        customers_casino_corr = correlation_matrix.loc['Total Customers', 'Total Casino Customers']

        st.write(f"#### Total Customers vs Total Deposit Amount: Correlation = {customers_deposits_corr:.2f}")
        st.write("  - **Insight**: The strong correlation indicates that a larger customer base directly contributes to higher deposit amounts.")
        st.write("  - **Recommendation**: Engage existing customers regularly to encourage repeat deposits. Tailored marketing campaigns could help maintain and grow this customer base.")

        st.write(f"#### Total Customers vs Total Sports Customers: Correlation = {customers_sports_corr:.2f}")
        st.write("  - **Insight**: A strong positive correlation shows that as the customer base grows, so does the number of sports customers.")
        st.write("  - **Recommendation**: Continue to target sports enthusiasts through dedicated campaigns and events to sustain this growth.")

        st.write(f"#### Total Customers vs Total Casino Customers: Correlation = {customers_casino_corr:.2f}")
        st.write("  - **Insight**: The very high correlation indicates that most customers participate in casino activities.")
        st.write("  - **Recommendation**: Strengthen the casino offerings and consider loyalty programs specific to casino users to retain and attract more customers.")

    elif selected_correlation == "Total Deposit Amount Correlations":
        deposits_sports_corr = correlation_matrix.loc['Total Deposit Amount', 'Total Sports Revenue']
        deposits_casino_corr = correlation_matrix.loc['Total Deposit Amount', 'Total Casino Revenue']

        st.write(f"#### Total Deposit Amount vs Total Sports Revenue: Correlation = {deposits_sports_corr:.2f}")
        st.write("  - **Insight**: The correlation between deposits and sports revenue is relatively low, suggesting that not all deposits are directed towards sports betting.")
        st.write("  - **Recommendation**: Offer special deposit bonuses for sports betting to encourage more deposits in this area.")

        st.write(f"#### Total Deposit Amount vs Total Casino Revenue: Correlation = {deposits_casino_corr:.2f}")
        st.write("  - **Insight**: The moderate correlation suggests that a significant portion of deposits is used in the casino.")
        st.write("  - **Recommendation**: Continue to enhance the casino experience and promote it as a key area for deposit utilization.")

    elif selected_correlation == "Sports Revenue vs. Casino Revenue Correlations":
        sports_casino_corr = correlation_matrix.loc['Total Sports Revenue', 'Total Casino Revenue']

        st.write(f"#### Total Sports Revenue vs Total Casino Revenue: Correlation = {sports_casino_corr:.2f}")
        st.write("  - **Insight**: The low correlation between sports and casino revenue suggests that these two revenue streams operate relatively independently of each other.")
        st.write("  - **Recommendation**: This indicates an opportunity to cross-promote between these segments. Consider offers that encourage customers to engage with both sports betting and casino activities.")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sidebar: Analysis type selector
st.sidebar.header("Select Analysis")
analysis_options = [
    "Select an analysis",
    "Time Series Analysis",
    "Total Customers by Sports and casino for each country",
    "Total Revenue by Sports and Revenue for each country",
    "Revenue Breakdown By country",
    "ARPU (Average Revenue Per User) by country"
]

selected_analysis = st.sidebar.selectbox("Select an analysis type:", analysis_options)

if selected_analysis == "Time Series Analysis":
    st.sidebar.subheader("Select Time Series Analysis Type")
    time_series_options = [
        "Total Revenue Over Time",
        "Total Customers Over Time",
        "Total Deposit Amount Over Time",
        "Sports vs Casino Revenue Over Time",
        "Total Sign-Ups Over Time"
    ]
    selected_time_series = st.sidebar.selectbox("Select a time series analysis:", time_series_options)

# Display the rest of the sidebar options only if an analysis is selected
if selected_analysis != "Select an analysis":
    st.sidebar.subheader("Filter Data")
    start_date = st.sidebar.date_input("Select Start Date", value=pd.to_datetime("2016-08-01"))
    end_date = st.sidebar.date_input("Select End Date", value=pd.to_datetime("2017-01-31"))
    selected_countries = st.sidebar.multiselect("Select Countries", df['Country'].unique(), default=["Germany", "France", "Spain", "Portugal", "Rest of World"])
    
    # This is where we define selected_product to avoid NameError
    selected_product = st.sidebar.selectbox("Select Product Type", ["All", "Sports", "Casino"])

    # Filter the data based on the selected options
    filtered_df = df[(df['DateTime'] >= pd.to_datetime(start_date)) & 
                     (df['DateTime'] <= pd.to_datetime(end_date)) & 
                     (df['Country'].isin(selected_countries))]

    # Apply product filter dynamically
    if selected_product == "Sports":
        filtered_df = filtered_df[['DateTime', 'Country', 'Total Sports Customers', 'Total Sports Revenue', 'Total Sign Ups', 'Total Customers', 'Total Deposit Amount']]
        filtered_df.rename(columns={'Total Sports Customers': 'Total Customers', 'Total Sports Revenue': 'Total Revenue'}, inplace=True)
    elif selected_product == "Casino":
        filtered_df = filtered_df[['DateTime', 'Country', 'Total Casino Customers', 'Total Casino Revenue', 'Total Sign Ups', 'Total Customers', 'Total Deposit Amount']]
        filtered_df.rename(columns={'Total Casino Customers': 'Total Customers', 'Total Casino Revenue': 'Total Revenue'}, inplace=True)
    else:
        filtered_df = filtered_df[['DateTime', 'Country', 'Total Customers', 'Total Revenue', 'Total Sports Customers', 'Total Sports Revenue', 'Total Casino Customers', 'Total Casino Revenue','Total Sign Ups','Total Deposit Amount']]

    # Perform the selected analysis
    if selected_analysis == "Time Series Analysis":
        if selected_time_series == "Total Revenue Over Time":
            st.title("Total Revenue Over Time")
            # Plotting total revenue over time
            revenue_over_time = filtered_df.groupby('DateTime')['Total Revenue'].sum().reset_index()
            fig = px.line(revenue_over_time, x='DateTime', y='Total Revenue', title="Total Revenue Over Time")
            st.plotly_chart(fig)
            # Dynamic Insights and Recommendations
            revenue_trend = "increasing" if revenue_over_time['Total Revenue'].iloc[-1] > revenue_over_time['Total Revenue'].iloc[0] else "decreasing"
            highest_revenue_month = revenue_over_time.loc[revenue_over_time['Total Revenue'].idxmax(), 'DateTime'].strftime('%B %Y')
            highest_revenue_value = revenue_over_time['Total Revenue'].max()
            revenue_std_dev = revenue_over_time['Total Revenue'].std()

            st.subheader("Insights")
            st.write(f"- The total revenue from **{start_date}** to **{end_date}** shows a **{revenue_trend}** trend.")
            st.write(f"- The highest revenue during this period was in **{highest_revenue_month}**, amounting to **€{highest_revenue_value:,.2f}**.")
            st.write(f"- The revenue volatility (standard deviation) during this period is **€{revenue_std_dev:,.2f}**.")

            st.subheader("Recommendations")
            if revenue_trend == "decreasing":
                st.write("- **Investigate Revenue Decline:** Investigate the causes of the revenue decline and explore strategies to boost sales.")
            st.write("- **Leverage Peak Strategies:** Focus on identifying and replicating successful strategies that led to peak revenue periods.")
            st.write("- **Stabilize Revenue Streams:** The high volatility suggests inconsistency. Focus on stabilizing revenue by enhancing core revenue streams and reducing dependency on highly volatile sources.")
     #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
        elif selected_time_series == "Total Customers Over Time":
            st.title("Total Customers Over Time")
            
            # Plotting total customers over time
            customers_over_time = filtered_df.groupby(['DateTime', 'Country'])['Total Customers'].sum().reset_index()
            fig = px.line(customers_over_time, x='DateTime', y='Total Customers', color='Country')
            st.plotly_chart(fig)
            
            # Dynamic Insights and Recommendations
            # Calculate overall trend
            overall_customer_trend = "increasing" if customers_over_time.groupby('DateTime')['Total Customers'].sum().iloc[-1] > customers_over_time.groupby('DateTime')['Total Customers'].sum().iloc[0] else "decreasing"
            
            # Calculate the highest customer month and value across all countries
            highest_customer_index = customers_over_time['Total Customers'].idxmax()
            highest_customer_month = customers_over_time.loc[highest_customer_index, 'DateTime'].strftime('%B %Y')
            highest_customer_value = customers_over_time.loc[highest_customer_index, 'Total Customers']

            st.subheader("Insights")
            st.write(f"- The total number of customers from **{start_date}** to **{end_date}** shows a **{overall_customer_trend}** trend.")
            st.write(f"- The highest number of customers during this period was in **{highest_customer_month}**, with **{highest_customer_value:,}** customers.")

            st.subheader("Recommendations")
            if overall_customer_trend == "decreasing":
                st.write("- **Investigate Customer Drop:** Investigate the reasons for the decrease in customers and focus on customer retention strategies.")
            else:
                st.write("- **Sustain Growth Momentum:** With an increasing trend, it's important to sustain the momentum by continuously engaging the customer base.")
            
            st.write("- **Boost Customer Acquisition:** Consider strategies to improve customer acquisition, particularly in months where the customer base was low.")
            st.write("- **Enhance Customer Experience:** Focus on maintaining and growing customer satisfaction, which can stabilize and even grow the customer base over time.")
            st.write(f"- **Leverage Peak Periods:** Identify and leverage the factors that led to the highest customer acquisition in **{highest_customer_month}** to replicate similar success in other periods.")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif selected_time_series == "Total Deposit Amount Over Time":

            st.title("Total Deposit Amount Over Time")
            # Plotting total deposit amount over time
            deposits_over_time = filtered_df.groupby('DateTime')['Total Deposit Amount'].sum().reset_index()
            fig = px.line(deposits_over_time, x='DateTime', y='Total Deposit Amount', title="Total Deposit Amount Over Time")
            st.plotly_chart(fig)
            
            # Dynamic Insights and Recommendations
            deposit_trend = "increasing" if deposits_over_time['Total Deposit Amount'].iloc[-1] > deposits_over_time['Total Deposit Amount'].iloc[0] else "decreasing"
            highest_deposit_month = deposits_over_time.loc[deposits_over_time['Total Deposit Amount'].idxmax(), 'DateTime'].strftime('%B %Y')
            highest_deposit_value = deposits_over_time['Total Deposit Amount'].max()

            st.subheader("Insights")
            st.write(f"**Insight:** The total deposit amount from **{start_date}** to **{end_date}** shows a **{deposit_trend}** trend.")
            st.write(f"**Insight:** The highest deposit amount during this period was in **{highest_deposit_month}**, totaling **€{highest_deposit_value:,.2f}**.")

            st.subheader("Recommendations")
            if deposit_trend == "decreasing":
                st.write("- **Investigate Deposit Decline:** Investigate why deposit amounts are declining and consider strategies to encourage higher deposits.")
            st.write("- **Encourage Higher Deposits:** Consider promotions or incentives that encourage higher deposit amounts.")
            st.write("- **Leverage Peak Deposit Strategies:** Analyze strategies used during peak deposit months and apply them to boost deposits during low periods.")
#------------------------------------------------------------------------------------------------------------------------------
        elif selected_time_series == "Sports vs Casino Revenue Over Time":
            st.title("Sports vs Casino Revenue Over Time")

            # Plotting sports vs casino revenue over time
            sports_vs_casino = filtered_df.groupby('DateTime').agg({
                'Total Sports Revenue': 'sum',
                'Total Casino Revenue': 'sum'
            }).reset_index()
            fig = px.line(sports_vs_casino, x='DateTime', y=['Total Sports Revenue', 'Total Casino Revenue'], title="Sports vs Casino Revenue Over Time")
            st.plotly_chart(fig)
            # Dynamic Insights and Recommendations
            # Calculate overall trends
            sports_revenue_trend = "increasing" if sports_vs_casino['Total Sports Revenue'].iloc[-1] > sports_vs_casino['Total Sports Revenue'].iloc[0] else "decreasing"
            casino_revenue_trend = "increasing" if sports_vs_casino['Total Casino Revenue'].iloc[-1] > sports_vs_casino['Total Casino Revenue'].iloc[0] else "decreasing"

            # Calculate the highest revenue month and value for sports and casino
            highest_sports_index = sports_vs_casino['Total Sports Revenue'].idxmax()
            highest_sports_month = sports_vs_casino.loc[highest_sports_index, 'DateTime'].strftime('%B %Y')
            highest_sports_value = sports_vs_casino.loc[highest_sports_index, 'Total Sports Revenue']

            highest_casino_index = sports_vs_casino['Total Casino Revenue'].idxmax()
            highest_casino_month = sports_vs_casino.loc[highest_casino_index, 'DateTime'].strftime('%B %Y')
            highest_casino_value = sports_vs_casino.loc[highest_casino_index, 'Total Casino Revenue']

            st.subheader("Insights")
            st.write(f"- **Sports Revenue Trend:** The sports revenue from **{start_date}** to **{end_date}** shows a **{sports_revenue_trend}** trend.")
            st.write(f"- **Sports Revenue Peak:** The highest sports revenue during this period was in **{highest_sports_month}**, amounting to **€{highest_sports_value:,.2f}**.")
            st.write(f"- **Casino Revenue Trend:** The casino revenue from **{start_date}** to **{end_date}** shows a **{casino_revenue_trend}** trend.")
            st.write(f"- **Casino Revenue Peak:** The highest casino revenue during this period was in **{highest_casino_month}**, amounting to **€{highest_casino_value:,.2f}**.")

            st.subheader("Recommendations")
            if sports_revenue_trend == "decreasing":
                st.write("- **Investigate Sports Revenue Decline:** Investigate the reasons for the decrease in sports revenue and focus on enhancing sports-related promotions.")
            else:
                st.write("- **Sustain Sports Revenue Growth:** With an increasing trend, it's important to sustain the momentum by continuously engaging sports customers.")

            if casino_revenue_trend == "decreasing":
                st.write("- **Investigate Casino Revenue Decline:** Investigate the reasons for the decrease in casino revenue and focus on enhancing casino-related promotions.")
            else:
                st.write("- **Sustain Casino Revenue Growth:** With an increasing trend, it's important to sustain the momentum by continuously engaging casino customers.")
            
            st.write("- **Leverage Peak Revenue Periods:** Identify and leverage the factors that led to the highest revenue months in **sports** ({highest_sports_month}) and **casino** ({highest_casino_month}) to replicate similar success in other periods.")
            st.write("- **Cross-Promotional Strategies:** Consider cross-promotional activities between sports and casino to boost revenue across both segments, particularly during periods of lower performance.")
#------------------------------------------------------------------------------------------------------------------------------------------------------------
       
                       
        elif selected_time_series == "Total Sign-Ups Over Time":
            st.title("Total Sign-Ups Over Time")
            # Plotting total sign-ups over time
            signups_over_time = filtered_df.groupby('DateTime')['Total Sign Ups'].sum().reset_index()
            fig = px.line(signups_over_time, x='DateTime', y='Total Sign Ups', title="Total Sign-Ups Over Time")
            st.plotly_chart(fig)
            
            # Dynamic Insights and Recommendations
            signups_trend = "increasing" if signups_over_time['Total Sign Ups'].iloc[-1] > signups_over_time['Total Sign Ups'].iloc[0] else "decreasing"
            highest_signups_index = signups_over_time['Total Sign Ups'].idxmax()
            highest_signups_month = signups_over_time.loc[highest_signups_index, 'DateTime'].strftime('%B %Y')
            highest_signups_value = signups_over_time.loc[highest_signups_index, 'Total Sign Ups']
            signups_std_dev = signups_over_time['Total Sign Ups'].std()

            st.subheader("Insights")
            st.write(f"- The total sign-ups from **{start_date}** to **{end_date}** shows a **{signups_trend}** trend.")
            st.write(f"- The highest number of sign-ups during this period was in **{highest_signups_month}**, with **{highest_signups_value:,}** sign-ups.")
            st.write(f"- The volatility (standard deviation) of sign-ups during this period is **{signups_std_dev:,.2f}**.")

            st.subheader("Recommendations")
            
             # Recommendations based on trend
            if signups_trend == "decreasing":
                st.write("- **Investigate Sign-Ups Decline:** Investigate why sign-ups are declining and consider strategies to boost new customer sign-ups.")
            else:
                st.write("- **Sustain Sign-Ups Growth:** With an increasing trend, it’s essential to sustain the growth momentum. Continue to invest in successful marketing channels and customer acquisition strategies.")
                st.write(f"- **Capitalize on Successful Campaigns:** Identify and scale up the marketing campaigns or strategies that have worked best, especially around the time of the peak in **{highest_signups_month}**.")

            # Recommendations based on volatility
            if signups_std_dev > (0.1 * signups_over_time['Total Sign Ups'].mean()):  # Example threshold
                st.write("- **Stabilize Sign-Up Rates:** The high volatility suggests inconsistency in sign-up rates. Focus on stabilizing sign-ups by ensuring consistent marketing efforts and addressing any external factors causing fluctuations.")
            else:
                st.write("- **Explore New Opportunities:** The low volatility indicates stable sign-up rates. Consider experimenting with new acquisition channels or markets to expand the user base further.")
            
            st.write(f"- **Leverage Peak Sign-Up Strategies:** Analyze and replicate the strategies that led to the peak in sign-ups during **{highest_signups_month}** in other periods where sign-ups were lower.")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif selected_analysis == "Total Customers by Sports and casino for each country":
        st.title("Total Customers by Sports and Casino for Each Country")
        # Plotting total customers by sports and casino for each country
        customers_by_product = filtered_df.groupby('Country').agg({
            'Total Sports Customers': 'sum',
            'Total Casino Customers': 'sum'
        }).reset_index()
        customers_by_product['Total Customers'] = customers_by_product['Total Sports Customers'] + customers_by_product['Total Casino Customers']
        fig = px.bar(customers_by_product, x='Country', y=['Total Sports Customers', 'Total Casino Customers'], barmode='group', title="Total Customers by Sports and Casino for Each Country")
        st.plotly_chart(fig)
         # Dynamic Insights and Recommendations
        st.subheader("Insights")
        
        # Country with highest total customers
        highest_country_index = customers_by_product['Total Customers'].idxmax()
        highest_country = customers_by_product.loc[highest_country_index, 'Country']
        highest_sports_customers = customers_by_product.loc[highest_country_index, 'Total Sports Customers']
        highest_casino_customers = customers_by_product.loc[highest_country_index, 'Total Casino Customers']
        highest_total_customers = customers_by_product.loc[highest_country_index, 'Total Customers']

        st.write(f"- **Country with Most Customers:** **{highest_country}** has the highest total customers with **{highest_sports_customers:,}** sports customers and **{highest_casino_customers:,}** casino customers.")

        # Additional Insights per country
        for index, row in customers_by_product.iterrows():
            country = row['Country']
            sports_customers = row['Total Sports Customers']
            casino_customers = row['Total Casino Customers']
            total_customers = row['Total Customers']
            st.write(f"- **{country} Insights:** **{country}** has **{sports_customers:,}** sports customers and **{casino_customers:,}** casino customers, totaling **{total_customers:,}** customers.")
        st.subheader("Recommendations")
        
        # Recommendation based on highest performing country
        st.write(f"- **Focus on High-Performing Countries:** Continue to engage and grow the customer base in **{highest_country}**, where both sports and casino customers are high.")

        # Identify countries with unbalanced engagement (e.g., more sports than casino customers)
        for index, row in customers_by_product.iterrows():
            country = row['Country']
            sports_customers = row['Total Sports Customers']
            casino_customers = row['Total Casino Customers']
            
            if sports_customers > 1.5 * casino_customers:
                st.write(f"- **Balance Engagement in {country}:** Consider increasing casino promotions in **{country}** to balance the engagement between sports and casino customers.")
            elif casino_customers > 1.5 * sports_customers:
                st.write(f"- **Balance Engagement in {country}:** Consider increasing sports promotions in **{country}** to balance the engagement between sports and casino customers.")
        
        # General recommendations
        st.write("- **Tailored Marketing Strategies:** Implement tailored marketing campaigns in countries where either sports or casino customers are lower to balance the engagement across both segments.")
        st.write("- **Leverage High Engagement:** In countries with a higher number of sports or casino customers, leverage cross-promotional strategies to increase the customer base in the other segment.")
        st.write(f"- **Expand Successful Strategies:** Analyze and replicate the successful strategies used in **{highest_country}** in other countries to increase overall customer engagement.")
 #---------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif selected_analysis == "Total Revenue by Sports and Revenue for each country":
        st.title("Total Revenue by Sports and Casino for Each Country")
        # Plotting total revenue by sports and casino for each country
        revenue_by_product = filtered_df.groupby('Country').agg({
            'Total Sports Revenue': 'sum',
            'Total Casino Revenue': 'sum'
        }).reset_index()
        fig = px.bar(revenue_by_product, x='Country', y=['Total Sports Revenue', 'Total Casino Revenue'], barmode='group', title="Total Revenue by Sports and Casino for Each Country")
        st.plotly_chart(fig)

            # Dynamic Insights and Recommendations
        st.subheader("Insights")
        
        # Country with highest total revenue
        revenue_by_product['Total Revenue'] = revenue_by_product['Total Sports Revenue'] + revenue_by_product['Total Casino Revenue']
        highest_revenue_country_index = revenue_by_product['Total Revenue'].idxmax()
        highest_revenue_country = revenue_by_product.loc[highest_revenue_country_index, 'Country']
        highest_sports_revenue = revenue_by_product.loc[highest_revenue_country_index, 'Total Sports Revenue']
        highest_casino_revenue = revenue_by_product.loc[highest_revenue_country_index, 'Total Casino Revenue']
        highest_total_revenue = revenue_by_product.loc[highest_revenue_country_index, 'Total Revenue']
        st.write(f"- **Country with Most Revenue:** **{highest_revenue_country}** generated the highest total revenue, with **€{highest_sports_revenue:,.2f}** from sports and **€{highest_casino_revenue:,.2f}** from casino, totaling **€{highest_total_revenue:,.2f}**.")
        # Additional Insights per country
        for index, row in revenue_by_product.iterrows():
            country = row['Country']
            sports_revenue = row['Total Sports Revenue']
            casino_revenue = row['Total Casino Revenue']
            total_revenue = row['Total Revenue']
            st.write(f"- **{country} Insights:** **{country}** generated **€{sports_revenue:,.2f}** from sports and **€{casino_revenue:,.2f}** from casino, totaling **€{total_revenue:,.2f}** in revenue.")
        st.subheader("Recommendations")
    
        # Recommendation based on highest performing country
        st.write(f"- **Focus on High-Performing Countries:** Continue to invest in and grow the revenue streams in **{highest_revenue_country}**, where both sports and casino revenue are strong.")

        # Identify countries with unbalanced revenue (e.g., more sports than casino revenue)
        for index, row in revenue_by_product.iterrows():
            country = row['Country']
            sports_revenue = row['Total Sports Revenue']
            casino_revenue = row['Total Casino Revenue']
            
            if sports_revenue > 1.5 * casino_revenue:
                st.write(f"- **Balance Revenue Streams in {country}:** Consider increasing casino promotions in **{country}** to balance the revenue between sports and casino.")
            elif casino_revenue > 1.5 * sports_revenue:
                st.write(f"- **Balance Revenue Streams in {country}:** Consider increasing sports promotions in **{country}** to balance the revenue between sports and casino.")
        
        # General recommendations
        st.write("- **Tailored Marketing Strategies:** Implement tailored marketing campaigns in countries where either sports or casino revenue is lower to balance the revenue streams across both segments.")
        st.write("- **Leverage High Engagement:** In countries with higher sports or casino revenue, leverage cross-promotional strategies to increase revenue in the other segment.")
        st.write(f"- **Expand Successful Strategies:** Analyze and replicate the successful strategies used in **{highest_revenue_country}** in other countries to increase overall revenue.")
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif selected_analysis == "Revenue Breakdown By country":
        st.title("Revenue Breakdown By Country")
        # Plotting revenue breakdown by country
        revenue_breakdown = filtered_df.groupby('Country').agg({'Total Revenue': 'sum'}).reset_index()
        fig = px.pie(revenue_breakdown, names='Country', values='Total Revenue', title="Revenue Breakdown By Country")
        st.plotly_chart(fig)
        # Dynamic Insights and Recommendations
        st.subheader("Insights")
        
        # Country with the highest revenue contribution
        highest_revenue_country_index = revenue_breakdown['Total Revenue'].idxmax()
        highest_revenue_country = revenue_breakdown.loc[highest_revenue_country_index, 'Country']
        highest_total_revenue = revenue_breakdown.loc[highest_revenue_country_index, 'Total Revenue']
        total_revenue_sum = revenue_breakdown['Total Revenue'].sum()
        highest_revenue_percentage = (highest_total_revenue / total_revenue_sum) * 100
        
        st.write(f"- **Country with Highest Revenue Contribution:** **{highest_revenue_country}** contributes the most to the total revenue, generating **€{highest_total_revenue:,.2f}**, which is approximately **{highest_revenue_percentage:.2f}%** of the total revenue.")

        # Insights on revenue distribution
        st.write(f"- **Revenue Distribution:** The revenue distribution across countries shows that **{highest_revenue_country}** is a key market, with other countries contributing smaller shares.")
        
        st.subheader("Recommendations")
        
        # Recommendation based on highest revenue contribution
        st.write(f"- **Focus on Key Markets:** Given that **{highest_revenue_country}** contributes significantly to the total revenue, it is crucial to continue investing in and growing this market.")
        
        # Identifying underperforming countries
        for index, row in revenue_breakdown.iterrows():
            country = row['Country']
            country_revenue = row['Total Revenue']
            country_revenue_percentage = (country_revenue / total_revenue_sum) * 100
            
            if country_revenue_percentage < 5:  # Example threshold for underperformance
                st.write(f"- **Revitalize {country}:** With **{country_revenue_percentage:.2f}%** of total revenue, consider developing targeted strategies to increase market share in **{country}**.")
        
        # General recommendations
        st.write("- **Diversify Revenue Streams:** While it's important to maintain strong performance in key markets, consider strategies to diversify revenue streams across other countries.")
        st.write(f"- **Leverage Successful Strategies:** Analyze the factors contributing to the success in **{highest_revenue_country}** and explore the potential to apply similar strategies in other markets.")
        st.write("- **Monitor Market Trends:** Regularly monitor revenue trends across all countries to identify emerging markets and adjust strategies accordingly.")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif selected_analysis == "ARPU (Average Revenue Per User) by country":
        st.title("Average Revenue Per User (ARPU) by Country")
        # Plotting ARPU by country
        arpu_by_country = filtered_df.groupby('Country').agg({'Total Revenue': 'sum', 'Total Customers': 'sum'}).reset_index()
        arpu_by_country['ARPU'] = arpu_by_country['Total Revenue'] / arpu_by_country['Total Customers']
        fig = px.bar(arpu_by_country, x='Country', y='ARPU', title="Average Revenue Per User (ARPU) by Country")
        st.plotly_chart(fig)

        # Dynamic Insights and Recommendations
        st.subheader("Insights")
        
        # Country with highest ARPU
        highest_arpu_index = arpu_by_country['ARPU'].idxmax()
        highest_arpu_country = arpu_by_country.loc[highest_arpu_index, 'Country']
        highest_arpu_value = arpu_by_country.loc[highest_arpu_index, 'ARPU']
        
        st.write(f"- **Country with Highest ARPU:** **{highest_arpu_country}** has the highest ARPU, amounting to **€{highest_arpu_value:,.2f}** per user.")
        
        # Insights on ARPU distribution
        st.write("- **ARPU Distribution:** The ARPU varies significantly across different countries, highlighting disparities in revenue generation per user.")
        
        # Identifying countries with low ARPU
        for index, row in arpu_by_country.iterrows():
            country = row['Country']
            arpu_value = row['ARPU']
            
            if arpu_value < arpu_by_country['ARPU'].mean():  # Example threshold: countries below average ARPU
                st.write(f"- **{country} ARPU Analysis:** **{country}** has an ARPU of **€{arpu_value:,.2f}**, which is below the average. Consider strategies to increase revenue per user in this country.")
        
        st.subheader("Recommendations")
        
        # Recommendations based on highest ARPU
        st.write(f"- **Focus on High ARPU Countries:** Given that **{highest_arpu_country}** has the highest ARPU, continue to explore and replicate the strategies that have led to this success. Consider implementing premium offerings or personalized services that appeal to high-value customers in this market.")
        
        # Recommendations for low ARPU countries
        for index, row in arpu_by_country.iterrows():
            country = row['Country']
            arpu_value = row['ARPU']
            
            if arpu_value < arpu_by_country['ARPU'].mean():
                st.write(f"- **Boost ARPU in {country}:** Implement targeted campaigns to increase ARPU in **{country}**. This could include upselling existing customers, offering exclusive promotions, or enhancing the customer experience to encourage higher spending.")
        
        # General recommendations
        st.write("- **ARPU Optimization Strategies:** Focus on optimizing ARPU across all countries by analyzing customer behavior and tailoring offerings to maximize revenue per user. Consider segmentation to identify high-value customer segments and deliver personalized experiences.")
        st.write("- **Cross-Sell and Upsell Opportunities:** Leverage cross-sell and upsell opportunities, particularly in countries with lower ARPU, to enhance revenue generation per user.")
        st.write("- **Monitor and Adjust Strategies:** Regularly monitor ARPU trends across different countries and adjust marketing and product strategies to improve revenue performance.")
        
    
#-----------------------------------------------------------------------------------------------------------------------------------------------

