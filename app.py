import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Holiday Package Prediction", page_icon="🧳", layout="wide")

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "Travel_Cleaned.csv"
MODEL_PATH = BASE / "models" / "random_forest_model.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_and_preprocessor():
    df = load_data()
    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    # same split as in the training notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", OneHotEncoder(drop="first"), cat_cols),
            ("StandardScaler", StandardScaler(), num_cols),
        ]
    )
    preprocessor.fit(X_train)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    test_probs = model.predict_proba(preprocessor.transform(X_test))[:, 1]
    return model, preprocessor, y_test, test_probs


df = load_data()
model, preprocessor, y_test, test_probs = load_model_and_preprocessor()
feature_cols = df.drop("ProdTaken", axis=1).columns

# ---------- sidebar ----------
st.sidebar.title("Settings")
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Customers with a probability above this value are marked as likely buyers. "
    "Lower it to catch more buyers (higher recall) at the cost of more false alarms.",
)
st.sidebar.caption("Model: Random Forest (1000 trees)")
st.sidebar.caption("Data: Trips & Travel holiday package dataset")

test_pred = (test_probs >= threshold).astype(int)

# ---------- header ----------
st.title("🧳 Holiday Package Purchase Prediction")
st.write("Predict which customers are likely to buy a travel package, so the sales team can focus on the right leads.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Data Insights", "Model Performance", "Predict"]
)

# ---------- tab 1: overview ----------
with tab1:
    st.subheader("The problem")
    st.write(
        "Trips & Travel is launching a new Wellness Tourism package. Until now customers were "
        "contacted at random, which made marketing expensive. Only about 18% of the customers "
        "bought a package. The goal is to predict who is likely to buy, so that calls are "
        "made to the most promising customers first."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in data", f"{len(df):,}")
    c2.metric("Bought a package", f"{df['ProdTaken'].mean() * 100:.1f}%")
    c3.metric("ROC-AUC (test)", f"{roc_auc_score(y_test, test_probs):.3f}")
    c4.metric("Recall (test)", f"{recall_score(y_test, test_pred):.3f}")

    st.subheader("A look at the data")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("Target column: ProdTaken (1 = bought the package, 0 = did not).")

# ---------- tab 2: data insights ----------
with tab2:
    st.subheader("Who buys the package?")

    left, right = st.columns(2)

    with left:
        cat_choice = st.selectbox(
            "Purchase rate by",
            ["ProductPitched", "Designation", "Occupation", "MaritalStatus",
             "TypeofContact", "Passport", "CityTier"],
        )
        rate = df.groupby(cat_choice)["ProdTaken"].mean() * 100
        st.bar_chart(rate)
        st.caption("Percentage of customers who bought, for each group.")

    with right:
        num_choice = st.selectbox(
            "Distribution of", ["MonthlyIncome", "Age", "DurationOfPitch", "NumberOfTrips"]
        )
        upper = df[num_choice].quantile(0.99)  # ignore extreme outliers in the plot
        fig, ax = plt.subplots(figsize=(6, 4))
        for value, label in [(0, "Did not buy"), (1, "Bought")]:
            subset = df[df["ProdTaken"] == value][num_choice]
            ax.hist(subset, bins=30, range=(df[num_choice].min(), upper), alpha=0.6, label=label)
        ax.set_xlabel(num_choice)
        ax.set_ylabel("Customers")
        ax.legend()
        st.pyplot(fig)
        st.caption("Top 1% of values are cut off so the chart stays readable.")

# ---------- tab 3: model performance ----------
with tab3:
    st.subheader("How well does the model do?")
    st.write(f"All numbers below are on the **test set** (20% of the data) at a threshold of **{threshold:.2f}**.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{roc_auc_score(y_test, test_probs):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, test_pred):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, test_pred):.3f}")
    m4.metric("F1 (buyers)", f"{f1_score(y_test, test_pred):.3f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confusion matrix**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, test_pred, display_labels=["No", "Yes"], cmap="Blues", ax=ax1
        )
        st.pyplot(fig1)

    with col_b:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_test, test_probs):.2f})")
        ax2.plot([0, 1], [0, 1], "r--")
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    st.markdown("**Top 10 most important features**")
    names = [n.split("__")[-1] for n in preprocessor.get_feature_names_out()]
    importances = pd.Series(model.feature_importances_, index=names).sort_values().tail(10)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.barh(importances.index, importances.values)
    ax3.set_xlabel("Importance")
    st.pyplot(fig3)

    st.info(
        "Note: the model is very precise but misses some real buyers (recall is lower than precision "
        "at the default threshold). Moving the threshold slider in the sidebar shows this trade-off."
    )

# ---------- tab 4: predict ----------
with tab4:
    st.subheader("Check a single customer")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=36)
            gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
            marital = st.selectbox("Marital status", sorted(df["MaritalStatus"].unique()))
            occupation = st.selectbox("Occupation", sorted(df["Occupation"].unique()))
            designation = st.selectbox("Designation", sorted(df["Designation"].unique()))
            income = st.number_input(
                "Monthly income", min_value=1000.0, value=float(df["MonthlyIncome"].median()), step=500.0
            )

        with c2:
            city_tier = st.selectbox("City tier", sorted(df["CityTier"].unique()))
            contact = st.selectbox("Type of contact", sorted(df["TypeofContact"].unique()))
            product = st.selectbox("Product pitched", sorted(df["ProductPitched"].unique()))
            star = st.selectbox(
                "Preferred property star",
                sorted(df["PreferredPropertyStar"].unique()),
                format_func=lambda x: str(int(x)),
            )
            trips = st.number_input("Trips per year", min_value=0, max_value=25, value=3)
            total_visiting = st.number_input("Total people visiting", min_value=1, max_value=10, value=3)

        with c3:
            pitch_duration = st.number_input("Pitch duration (minutes)", min_value=1, max_value=130, value=15)
            followups = st.number_input("Number of follow-ups", min_value=1, max_value=6, value=4)
            satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)
            passport = st.radio("Has passport?", ["Yes", "No"], horizontal=True)
            own_car = st.radio("Owns a car?", ["Yes", "No"], horizontal=True)

        submitted = st.form_submit_button("Predict")

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "Age": age,
                    "TypeofContact": contact,
                    "CityTier": city_tier,
                    "DurationOfPitch": pitch_duration,
                    "Occupation": occupation,
                    "Gender": gender,
                    "NumberOfFollowups": followups,
                    "ProductPitched": product,
                    "PreferredPropertyStar": star,
                    "MaritalStatus": marital,
                    "NumberOfTrips": trips,
                    "Passport": 1 if passport == "Yes" else 0,
                    "PitchSatisfactionScore": satisfaction,
                    "OwnCar": 1 if own_car == "Yes" else 0,
                    "Designation": designation,
                    "MonthlyIncome": income,
                    "TotalVisiting": total_visiting,
                }
            ]
        )[feature_cols]

        prob = model.predict_proba(preprocessor.transform(customer))[0, 1]

        st.metric("Probability of buying", f"{prob * 100:.1f}%")
        st.progress(float(prob))
        if prob >= threshold:
            st.success("Likely buyer. Worth prioritising for a call.")
        else:
            st.warning("Unlikely to buy at the current threshold.")

    # ---------- batch scoring ----------
    st.divider()
    st.subheader("Score many customers from a CSV")
    st.caption("The file must have the same columns as the cleaned dataset (without ProdTaken).")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in batch.columns]

        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif batch[feature_cols].isnull().any().any():
            st.error("The file has empty values. Please fill or remove them and upload again.")
        else:
            try:
                batch_probs = model.predict_proba(preprocessor.transform(batch[feature_cols]))[:, 1]
                result = batch.copy()
                result["BuyProbability"] = batch_probs.round(3)
                result["LikelyBuyer"] = (batch_probs >= threshold).astype(int)
                result = result.sort_values("BuyProbability", ascending=False)

                st.write(f"{int(result['LikelyBuyer'].sum())} of {len(result)} customers are likely buyers.")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download results",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="scored_customers.csv",
                    mime="text/csv",
                )
            except ValueError:
                st.error("Could not score the file. Check that category values match the training data.")