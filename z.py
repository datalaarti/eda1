import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
st.set_page_config(layout="wide")
df = pd.read_csv("exams.csv")
df['Decade'] = pd.cut(df['Year'], bins=[1999, 2009, 2019, 2029], labels=['2000s', '2010s', '2020s'])

st.title("📊 UPSC Data Visualization Dashboard")

menu = st.sidebar.selectbox("Select Section", [
    "📚 Overall Trends",
    "📈 Subject-wise Visualization",
    "🧠 Advanced Analysis"
])

if menu == "📚 Overall Trends":
    st.subheader("Overall Subject Trends")
    option = st.radio("Choose a Visualization", [
        "Trend of Questions by Subject (Line Plot)",
        "Total Questions by Subject (Bar Plot)",
        "Year-wise Subject-wise Question Distribution (Stacked Bar)",
        "Correlation Heatmap"
    ])

    if option == "Trend of Questions by Subject (Line Plot)":
        plt.figure(figsize=(14, 7))
        for subject in df.columns[1:-1]:
            plt.plot(df["Year"], df[subject], label=subject, marker="o")
        plt.title("📈 Trend of Questions by Subject (2000–2023)")
        plt.xlabel("Year")
        plt.ylabel("Number of Questions")
        plt.legend()
        st.pyplot(plt)

    elif option == "Total Questions by Subject (Bar Plot)":
        total_by_subject = df.drop(columns=['Year', 'Decade']).sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=total_by_subject.values, y=total_by_subject.index, palette="viridis", ax=ax)
        ax.set_title("🧮 Total Questions by Subject (2000–2023)")
        st.pyplot(fig)

    elif option == "Year-wise Subject-wise Question Distribution (Stacked Bar)":
        df_sorted = df.sort_values("Year")
        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = pd.Series([0] * len(df_sorted))
        for subject in df.columns[1:-1]:
            ax.bar(df_sorted["Year"], df_sorted[subject], bottom=bottom, label=subject)
            bottom += df_sorted[subject]
        ax.set_title("📊 Year-wise Subject-wise Question Distribution")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Questions")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        st.pyplot(fig)

    elif option == "Correlation Heatmap":
        corr_matrix = df.drop(columns=["Year", "Decade"]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("📌 Correlation Between Subjects")
        st.pyplot(fig)

elif menu == "📈 Subject-wise Visualization":
    st.subheader("Subject-wise Visualizations")
    subject = st.selectbox("Select a subject", df.columns[1:-1])
    chart_type = st.selectbox("Choose a graph type", [
        "Line Plot", "Scatter Plot", "Histogram", "Density Plot (KDE)", "Box and Whisker Plot",
        "Interactive Line Chart", "Candlestick Chart", "Treemap", "Sunburst Chart",
        "Sparkline", "Stem-and-Leaf Plot"
    ])

    if chart_type == "Line Plot":
        fig, ax = plt.subplots()
        ax.plot(df["Year"], df[subject], marker="o")
        ax.set_title(f"Line Plot of {subject}")
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        fig, ax = plt.subplots()
        ax.scatter(df["Year"], df[subject])
        ax.set_title(f"Scatter Plot of {subject}")
        st.pyplot(fig)

    elif chart_type == "Histogram":
        fig, ax = plt.subplots()
        ax.hist(df[subject], bins=10, color="skyblue")
        ax.set_title(f"Histogram of {subject}")
        st.pyplot(fig)

    elif chart_type == "Density Plot (KDE)":
        fig, ax = plt.subplots()
        sns.kdeplot(df[subject], fill=True, ax=ax)
        ax.set_title(f"Density Plot of {subject}")
        st.pyplot(fig)

    elif chart_type == "Box and Whisker Plot":
        fig, ax = plt.subplots()
        sns.boxplot(y=df[subject], ax=ax)
        ax.set_title(f"Box and Whisker Plot of {subject}")
        st.pyplot(fig)

    elif chart_type == "Interactive Line Chart":
        fig = px.line(df, x="Year", y=subject, title=f"Interactive Line Chart of {subject}")
        st.plotly_chart(fig)

    elif chart_type == "Candlestick Chart":
        fig = go.Figure(data=[go.Candlestick(
            x=df["Year"],
            open=df[subject],
            high=df[subject] + 5,
            low=df[subject] - 5,
            close=df[subject]
        )])
        fig.update_layout(title=f"Candlestick Chart of {subject}")
        st.plotly_chart(fig)

    elif chart_type == "Treemap":
        fig = px.treemap(df, path=["Year"], values=subject, title=f"Treemap of {subject} Questions by Year")
        st.plotly_chart(fig)

    elif chart_type == "Sunburst Chart":
        fig = px.sunburst(df, path=["Year"], values=subject, title=f"Sunburst Chart of {subject}")
        st.plotly_chart(fig)

    elif chart_type == "Sparkline":
        fig, ax = plt.subplots()
        ax.plot(df["Year"], df[subject], linestyle="-", linewidth=2)
        ax.axis("off")
        st.pyplot(fig)

    elif chart_type == "Stem-and-Leaf Plot":
        try:
            from stemgraphic import stem_graphic
            fig, ax = stem_graphic(df[subject], scale=10)
            st.pyplot(fig)
        except ImportError:
            st.warning("⚠ 'stemgraphic' is not installed. Install it with pip install stemgraphic.")

elif menu == "🧠 Advanced Analysis":
    st.subheader("Advanced Analysis")
    analysis = st.selectbox("Select Analysis Type", [
        "Top Gaining & Declining Subjects",
        "Average Questions per Subject per Decade",
        "Year-on-Year Change (Heatmap)",
        "High-Yield Subject Strategy (Stats)",
        "Clustering Subjects Based on Trends"
    ])

    if analysis == "Top Gaining & Declining Subjects":
        start_year = df[df["Year"] == 2000].iloc[0]
        end_year = df[df["Year"] == 2023].iloc[0]
        subject_changes = (end_year[1:-1] - start_year[1:-1]).sort_values()

        st.write("🔺 Top Gaining Subjects")
        st.dataframe(subject_changes.sort_values(ascending=False).head(3))

        st.write("🔻 Top Declining Subjects")
        st.dataframe(subject_changes.sort_values().head(3))

    elif analysis == "Average Questions per Subject per Decade":
        decade_avg = df.groupby('Decade').mean(numeric_only=True).T
        fig, ax = plt.subplots(figsize=(12, 6))
        decade_avg.plot(kind="bar", ax=ax)
        ax.set_ylabel("Avg Number of Questions")
        ax.set_title("📊 Average Questions per Subject per Decade")
        st.pyplot(fig)

    elif analysis == "Year-on-Year Change (Heatmap)":
        df_diff = df.sort_values("Year").set_index("Year").drop(columns="Decade").diff().dropna()
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(df_diff.T, cmap="coolwarm", center=0, annot=True, fmt=".0f", ax=ax)
        ax.set_title("🔁 Year-on-Year Change in Subject Questions")
        st.pyplot(fig)

    elif analysis == "High-Yield Subject Strategy (Stats)":
        subject_stats = df.drop(columns=["Year", "Decade"]).agg(["mean", "max", "min", "std"]).T
        subject_stats["CV"] = subject_stats["std"] / subject_stats["mean"]
        st.write("🎯 High-Yield Subjects (Based on Averages and Stability)")
        st.dataframe(subject_stats.sort_values("mean", ascending=False))

    elif analysis == "Clustering Subjects Based on Trends":
        subject_data = df.drop(columns=["Year", "Decade"]).T
        scaler = StandardScaler()
        scaled_subjects = scaler.fit_transform(subject_data)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_subjects)
        subject_clusters = pd.DataFrame({
            "Subject": subject_data.index,
            "Cluster Group": clusters
        })
        st.write("🧠 Subject Clusters Based on Trends")
        st.dataframe(subject_clusters.sort_values("Cluster Group"))
