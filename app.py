import os
import warnings

# Suprime warnings de Streamlit em notebooks
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection      import train_test_split
from sklearn.preprocessing        import StandardScaler
from sklearn.linear_model         import LogisticRegression
from sklearn.feature_selection    import RFE
from sklearn.metrics              import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling       import SMOTE
import statsmodels.api            as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuração do Streamlit
st.set_page_config(page_title="Hotel Booking Cancellation Dashboard", layout="wide")
sns.set(style="whitegrid")
RANDOM_STATE = 42
TEST_SIZE    = 0.3

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['reservation_status_date'])

@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['adr'] >= 0].copy()
    df['agent'].fillna(0, inplace=True)
    df['company'].fillna(0, inplace=True)
    top_countries = df['country'].value_counts().nlargest(10).index
    df['country'] = df['country'].where(df['country'].isin(top_countries), 'Other')
    return df

# Carrega e limpa
data_path = (
    'sample_data/hotel_bookings.csv'
    if os.path.exists('sample_data/hotel_bookings.csv')
    else 'hotel_bookings.csv'
)
df = load_data(data_path)
df = clean_data(df)

# Sidebar – Filtros
st.sidebar.header("Filtros")
lead_min, lead_max = st.sidebar.slider(
    "Lead time (dias)",
    int(df.lead_time.min()), int(df.lead_time.max()),
    (int(df.lead_time.quantile(0.1)), int(df.lead_time.quantile(0.9)))
)
segments = st.sidebar.multiselect(
    "Market Segment",
    options=df.market_segment.unique(),
    default=list(df.market_segment.unique())
)

# Sidebar – Seleção de variáveis
st.sidebar.header("Variáveis para Modelagem")
all_features = [
    'lead_time', 'adr', 'total_of_special_requests',
    'hotel', 'arrival_date_month', 'market_segment',
    'customer_type', 'deposit_type', 'country'
]
selected_manual = st.sidebar.multiselect(
    "Seleção Manual",
    all_features,
    default=all_features
)

# Sidebar – SMOTE e RFE
use_smote = st.sidebar.checkbox("Aplicar SMOTE", value=True)
n_rfe = st.sidebar.slider("Nº de features (RFE)", 5, len(all_features), 8)

# Filtra dados
df_f = df[
    (df.lead_time.between(lead_min, lead_max)) &
    (df.market_segment.isin(segments))
]

# Prepara X e y
cont_vars = ['lead_time', 'adr', 'total_of_special_requests']
cat_vars  = [v for v in selected_manual if v not in cont_vars]
df_feat   = df_f[cont_vars + cat_vars].copy()
df_enc    = pd.get_dummies(df_feat, columns=cat_vars, drop_first=True)

X = df_enc.copy()
y = df_f['is_canceled']

# Escalonamento
scaler = StandardScaler()
X[cont_vars] = scaler.fit_transform(X[cont_vars])

# Split e SMOTE
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE,
    stratify=y, random_state=RANDOM_STATE
)
if use_smote:
    X_tr, y_tr = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)

# RFE
base_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
rfe     = RFE(base_lr, n_features_to_select=n_rfe)
rfe.fit(X_tr, y_tr)
selected_rfe = list(X_tr.columns[rfe.support_])

# Treino final
X_tr_sel = X_tr[selected_rfe]
X_te_sel = X_te[selected_rfe]
model_final = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model_final.fit(X_tr_sel, y_tr)

# Previsões
y_pred_prob = model_final.predict_proba(X_te_sel)[:,1]
y_pred      = model_final.predict(X_te_sel)

# Interpretação de coeficientes
coef_df = pd.DataFrame({
    'feature': ['Intercept'] + selected_rfe,
    'coef':    [model_final.intercept_[0]] + list(model_final.coef_[0])
})
coef_df['odds_ratio']   = np.exp(coef_df['coef'])
coef_df['interpretation'] = coef_df.apply(
    lambda r: f"Um aumento unitário em {r.feature} multiplica odds por {r.odds_ratio:.2f}x",
    axis=1
)

# Métricas
auc       = roc_auc_score(y_te, y_pred_prob)
precision = precision_score(y_te, y_pred)
recall    = recall_score(y_te, y_pred)
f1        = f1_score(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_pred_prob)

# Layout do Dashboard
st.title("📊 Hotel Booking Cancellation Dashboard")

st.subheader("Amostra de Dados")
st.dataframe(df_f.head())

st.subheader("Interpretação Automática de Coeficientes")
st.table(coef_df[['feature','odds_ratio','interpretation']])

st.subheader("Métricas de Avaliação")
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC",       f"{auc:.2f}")
c2.metric("Precision", f"{precision:.2f}")
c3.metric("Recall",    f"{recall:.2f}")
c4.metric("F1-score",  f"{f1:.2f}")

st.subheader("Curva ROC")
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax_roc.plot([0,1],[0,1],'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
st.pyplot(fig_roc)

st.subheader("Curva Logística")
feat_choice = st.selectbox("Variável para Curva", selected_rfe)
if feat_choice in cont_vars:
    grid = np.linspace(X_te_sel[feat_choice].min(), X_te_sel[feat_choice].max(), 100)
    idx  = cont_vars.index(feat_choice)
    mean_i, scale_i = scaler.mean_[idx], scaler.scale_[idx]
    scaled = (grid - mean_i) / scale_i
    base_row = X_te_sel.mean().values
    Xg = pd.DataFrame([base_row]*len(grid), columns=selected_rfe)
    Xg[feat_choice] = scaled
else:
    grid = [0,1]
    base_row = X_te_sel.mean().values
    Xg = pd.DataFrame([base_row]*2, columns=selected_rfe)
    Xg[feat_choice] = grid

probs = model_final.predict_proba(Xg)[:,1]
fig_log, ax_log = plt.subplots()
ax_log.plot(grid, probs, marker='o')
ax_log.set_xlabel(feat_choice)
ax_log.set_ylabel("P(cancelamento)")
st.pyplot(fig_log)
