import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

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
funcionarios = st.sidebar.multiselect(
    "Selecione o(s) funcionário(s):",
    options=df["id_funcionario"].unique(),
    default=df["id_funcionario"].unique()
)
df_filtrado = df[df["id_funcionario"].isin(funcionarios)]

st.markdown("---")

# --- GRÁFICO 1: Distribuição Geral de Emoções ---
st.subheader("🎭 Distribuição de Emoções (Geral)")
contagem = df_filtrado["emocao_pt"].value_counts()
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.pie(contagem, labels=None, startangle=90)
ax1.legend(contagem.index, title="Emoções", loc="center left", bbox_to_anchor=(1, 0.5))
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

# --- GRÁFICO 3: Velocímetro de Eficiência por Funcionário ---
st.subheader("⚡ Velocímetro de Eficiência por Funcionário")
for func in df_filtrado["id_funcionario"].unique():
    df_func = df_filtrado[df_filtrado["id_funcionario"] == func]
    total = len(df_func)
    resolvidos = len(df_func[df_func["estado_servico"] == "concluido"])
    andamento = len(df_func[df_func["estado_servico"] == "andamento"])
    eficiencia = (resolvidos + andamento) / total if total > 0 else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=eficiencia*100,
        title={'text': f"{func}"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
    ))
    st.plotly_chart(fig)

st.markdown("---")

# --- GRÁFICO 4: Barras de Satisfação Média ---
st.subheader("📊 Satisfação Média por Funcionário")
media_confianca = df_filtrado.groupby("id_funcionario")["confianca"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="id_funcionario", y="confianca", data=media_confianca, ax=ax, palette="turbo")
ax.set_ylabel("Satisfação Média")
ax.set_xlabel("Funcionário")
st.pyplot(fig)

st.markdown("---")

# --- GRÁFICO 5: Sentimento ao Longo do Período ---
st.subheader("📈 Sentimento ao Longo do Período")
df_filtrado['data'] = pd.to_datetime(df_filtrado['data'])
media_sentimento = df_filtrado.groupby('data')['confianca'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(media_sentimento['data'], media_sentimento['confianca'], marker='o', color="red")
ax.set_ylabel("Sentimento Médio")
ax.set_xlabel("Data")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

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
