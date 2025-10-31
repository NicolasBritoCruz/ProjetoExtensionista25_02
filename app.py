import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Análise de Emoções em Atendimentos", layout="wide")
st.title("📊 Análise de Emoções dos Atendimentos: JCSI")

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    df = pd.read_csv("emocao_clientes_todos.csv")
    return df

try:
    df = carregar_dados()
    st.success("Dados carregados com sucesso!")
except FileNotFoundError:
    st.error("❌ Arquivo 'emocao_clientes_todos.csv' não encontrado. Verifique se está na mesma pasta do app.")
    st.stop()

# --- FILTRO NA SIDEBAR ---
st.sidebar.header("🔍 Filtro por Funcionário")

# Filtro por funcionário
funcionarios = st.sidebar.multiselect(
    "Selecione o(s) funcionário(s):",
    options=df["id_funcionario"].unique(),
    default=df["id_funcionario"].unique()
)

# Aplica o filtro
df_filtrado = df[df["id_funcionario"].isin(funcionarios)]

st.markdown("---")

# --- VISÃO GERAL ---
st.subheader("📈 Visão Geral dos Atendimentos Filtrados")
col1, col2, col3 = st.columns(3)

col1.metric("Total de Mensagens", len(df_filtrado))
col2.metric("Funcionários Analisados", df_filtrado["id_funcionario"].nunique())
col3.metric("Emoções Únicas", df_filtrado["emocao_pt"].nunique())

st.markdown("---")

# --- GRÁFICO 1: Distribuição Geral de Emoções ---
st.subheader("🎭 Distribuição de Emoções (Geral)")

contagem = df_filtrado["emocao_pt"].value_counts()
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.pie(
    contagem,
    labels=contagem.index,
    autopct="%1.1f%%",
    startangle=90,
)
ax1.axis("equal")
st.pyplot(fig1)

st.markdown("---")

# --- GRÁFICO 2: Emoções por Funcionário ---
st.subheader("👤 Emoções por Funcionário")

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(data=df_filtrado, x="id_funcionario", hue="emocao_pt", ax=ax2)
ax2.set_xlabel("Funcionário")
ax2.set_ylabel("Quantidade de mensagens")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig2)

st.markdown("---")

# --- TABELA DE EXEMPLOS DE MENSAGENS ---
st.subheader("💬 Exemplos de Mensagens")

amostra = df_filtrado.sample(min(5, len(df_filtrado))) if len(df_filtrado) > 0 else pd.DataFrame()
st.dataframe(
    amostra[["mensagem", "emocao_pt", "confianca", "id_funcionario"]],
    hide_index=True,
    use_container_width=True
)

st.markdown("---")
st.caption("Desenvolvido para análise emocional de atendimentos - usando PySentimiento + Streamlit")
