import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Customer Churn Risk")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(BASE / "churn_ann.keras")


@st.cache_data
def load_preprocess():
    return json.loads((BASE / "churn_preprocess.json").read_text())


model = load_model()
prep = load_preprocess()
COLUMNS = prep["columns"]
YES_NO = ["No", "Yes"]
EXTRA_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

PRESETS = {
    "Enter manually": None,
    "High-risk example": dict(tenure=2, monthly=85.0, contract="Month-to-month", internet="Fiber optic", payment="Electronic check",
                              paperless="Yes", senior=0, partner="No", dependents="No", multiple="No", extras="No"),
    "Low-risk example": dict(tenure=60, monthly=55.0, contract="Two year", internet="DSL", payment="Credit card (automatic)",
                             paperless="No", senior=0, partner="Yes", dependents="Yes", multiple="No", extras="Yes"),
}


def to_features(c):
    """Turn the answers of one customer into the 30 model inputs, in the training column order."""
    row = dict.fromkeys(COLUMNS, 0.0)
    row["SeniorCitizen"] = float(c["senior"])
    row["tenure"] = float(c["tenure"])
    row["MonthlyCharges"] = float(c["monthly"])
    row["TotalCharges"] = float(c["tenure"]) * float(c["monthly"])

    def on(name):
        if name in row:
            row[name] = 1.0

    if c["gender"] == "Male":
        on("gender_Male")
    for key, name in [("partner", "Partner_Yes"), ("dependents", "Dependents_Yes"), ("phone", "PhoneService_Yes"), ("paperless", "PaperlessBilling_Yes")]:
        if c[key] == "Yes":
            on(name)
    on(f"MultipleLines_{c['multiple']}")
    on(f"InternetService_{c['internet']}")
    for service in EXTRA_SERVICES:
        on(f"{service}_{c[service]}")
    on(f"Contract_{c['contract']}")
    on(f"PaymentMethod_{c['payment']}")

    x = pd.DataFrame([row])[COLUMNS]
    for col, mean, scale in zip(prep["num"], prep["mean"], prep["scale"]):
        x[col] = (x[col] - mean) / scale
    return x


def churn_probability(customers):
    x = pd.concat([to_features(c) for c in customers]).values.astype("float32")
    return model.predict(x, verbose=0).ravel()


st.title("Customer Churn Risk")
st.write(
    "A neural network (ANN) trained on 7,043 telecom customers estimates the probability that a customer "
    "will leave the company. Fill in the customer's details or load an example."
)

choice = st.selectbox("Load an example (optional)", list(PRESETS.keys()))
d = PRESETS[choice] or dict(tenure=12, monthly=70.0, contract="Month-to-month", internet="DSL", payment="Electronic check",
                            paperless="Yes", senior=0, partner="No", dependents="No", multiple="No", extras="No")


def pick(label, options, value):
    return st.selectbox(label, options, index=options.index(value))


col1, col2 = st.columns(2)
with col1:
    st.subheader("Account")
    contract = pick("Contract", ["Month-to-month", "One year", "Two year"], d["contract"])
    tenure = st.slider("Tenure (months with the company)", 0, 72, d["tenure"])
    monthly = st.slider("Monthly charges ($)", 18.0, 120.0, float(d["monthly"]), 0.5)
    payment = pick("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], d["payment"])
    paperless = pick("Paperless billing", YES_NO, d["paperless"])
with col2:
    st.subheader("Services")
    internet = pick("Internet service", ["DSL", "Fiber optic", "No"], d["internet"])
    phone = pick("Phone service", ["Yes", "No"], "Yes")
    multiple = "No phone service" if phone == "No" else pick("Multiple lines", YES_NO, d["multiple"])
    if internet == "No":
        extras = {s: "No internet service" for s in EXTRA_SERVICES}
        st.caption("No internet service, so the online services are not available.")
    else:
        all_extras = st.checkbox("Has online security, backup, device protection, tech support, and streaming", value=d["extras"] == "Yes")
        extras = {s: "Yes" if all_extras else "No" for s in EXTRA_SERVICES}

st.subheader("Customer")
c1, c2, c3, c4 = st.columns(4)
gender = c1.selectbox("Gender", ["Female", "Male"])
senior = c2.selectbox("Senior citizen", [0, 1], index=d["senior"], format_func=lambda v: "Yes" if v else "No")
partner = c3.selectbox("Partner", YES_NO, index=YES_NO.index(d["partner"]))
dependents = c4.selectbox("Dependents", YES_NO, index=YES_NO.index(d["dependents"]))

customer = dict(gender=gender, senior=senior, partner=partner, dependents=dependents, tenure=tenure, phone=phone,
                multiple=multiple, internet=internet, paperless=paperless, payment=payment, monthly=monthly,
                contract=contract, **extras)

if st.button("Estimate churn risk"):
    p = float(churn_probability([customer])[0])
    level = "High" if p >= 0.50 else "Medium" if p >= 0.30 else "Low"
    box = {"High": st.error, "Medium": st.warning, "Low": st.success}[level]
    box(f"**{level} risk**: {p:.0%} probability of leaving")
    st.progress(min(p, 1.0))
    st.caption(
        "Risk levels: Low below 30%, Medium 30-50%, High 50% or more. On the test data, flagging every customer at "
        "30% or more caught about 78% of the customers who left. The average churn rate in the data is 26.5%."
    )

    st.write("**What if this customer had a different contract?**")
    options = ["Month-to-month", "One year", "Two year"]
    probs = churn_probability([{**customer, "contract": o} for o in options])
    st.dataframe(pd.DataFrame({"contract": options, "churn probability": [f"{v:.0%}" for v in probs]}), hide_index=True)
    st.caption(
        "Be careful with unusual combinations: for example, almost no customer in the data has a one-year or two-year "
        "contract and a tenure of only a few months, so the model has seen very few such cases and the result is not reliable."
    )

st.caption(
    "Trained on the IBM Telco Customer Churn sample data (AUC 0.84 on the test set, about the same as Logistic Regression). "
    "TotalCharges is estimated as tenure x monthly charges. The prediction shows who is likely to leave, not why."
)

