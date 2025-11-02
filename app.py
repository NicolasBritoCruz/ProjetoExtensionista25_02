import os
import json
import pandas as pd
from tqdm import tqdm
from pysentimiento import create_analyzer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from datetime import datetime # Precisamos disso de volta

# --- PARTE 1: CONFIGURAÇÃO GERAL (Inalterada) ---
# ... (todo o seu EMOCOES_MAP, traducoes, etc. permanecem aqui) ...
PASTA_JSON = "atendimento/"
ARQUIVO_CSV_SAIDA = "emocao_clientes_todos.csv"
PASTA_GRAFICOS = "graficos" 

EMOCOES_MAP = {
    # Emoções primárias e mais comuns
    "joy":          {"pt": "alegria",   "cor": "#2ECC71"},
    "sadness":      {"pt": "tristeza",  "cor": "#3498DB"},
    "anger":        {"pt": "raiva",     "cor": "#E74C3C"},
    "fear":         {"pt": "medo",      "cor": "#F1C40F"},
    "surprise":     {"pt": "surpresa",  "cor": "#9B59B6"},
    "disgust":      {"pt": "desgosto",  "cor": "#795548"},
    "neutral":      {"pt": "neutro",    "cor": "#95A5A6"},
    # Emoções secundárias
    "admiration":   {"pt": "admiração", "cor": "#1ABC9C"},
    "amusement":    {"pt": "diversão",  "cor": "#F39C12"},
    "approval":     {"pt": "aprovação", "cor": "#27AE60"},
    "caring":       {"pt": "carinho",   "cor": "#E84393"},
    "confusion":    {"pt": "confusão",  "cor": "#546E7A"},
    "curiosity":    {"pt": "curiosidade","cor": "#00BCD4"},
    "desire":       {"pt": "desejo",    "cor": "#D81B60"},
    "disappointment": {"pt": "decepção","cor": "#AAB7B8"},
    "disapproval":  {"pt": "desaprovação","cor": "#B71C1C"},
    "excitement":   {"pt": "excitação", "cor": "#FF7043"},
    "gratitude":    {"pt": "gratidão",  "cor": "#8E44AD"},
    "love":         {"pt": "amor",      "cor": "#EC407A"},
    "optimism":     {"pt": "otimismo",  "cor": "#81C784"},
    "pride":        {"pt": "orgulho",   "cor": "#5C6BC0"},
    "realization":  {"pt": "percepção", "cor": "#4DD0E1"},
    "relief":       {"pt": "alívio",    "cor": "#AED581"},
    "remorse":      {"pt": "remorso",   "cor": "#BDBDBD"},
    "others":       {"pt": "outros",    "cor": "#9E9E9E"} # Fallback
}
traducao_emocoes = {en: v["pt"] for en, v in EMOCOES_MAP.items()}
cores_emocoes = {v["pt"]: v["cor"] for en, v in EMOCOES_MAP.items()}
EMOCOES_POSITIVAS = {"joy", "admiration", "amusement", "approval", "caring", 
                     "excitement", "gratitude", "love", "optimism", "pride", "relief"}
EMOCOES_NEGATIVAS = {"sadness", "anger", "fear", "disgust", 
                     "disappointment", "disapproval", "remorse"}

# --- PARTE 2: FUNÇÕES DE PROCESSAMENTO E CARREGAMENTO ---

@st.cache_resource
def carregar_modelo_emocao():
    print("Carregando modelo de ANÁLISE DE EMOÇÕES...")
    analyzer = create_analyzer(task="emotion", lang="pt")
    print("Modelo de EMOÇÕES carregado com sucesso.")
    return analyzer

@st.cache_resource
def carregar_modelo_sentimento():
    print("Carregando modelo de ANÁLISE DE SENTIMENTO...")
    analyzer = create_analyzer(task="sentiment", lang="pt")
    print("Modelo de SENTIMENTO carregado com sucesso.")
    return analyzer

def analisar_texto(emotion_analyzer, sentiment_analyzer, texto):
    """
    Recebe um texto e retorna um dicionário com a análise de emoção,
    validada pela análise de sentimento.
    Versão robusta que assume 'neutro' em caso de falha ou ausência de emoção.
    """
    texto_padronizado = texto.strip().lower()
    if not texto_padronizado: 
        return None

    # --- Passo 1: Analisar Emoção (com fallback) ---
    
    # MODIFICADO: Começamos assumindo 'neutro'
    label_en_emocao = "neutral"
    score_emocao = 0.99 # Damos uma confiança alta para o 'neutro'
    label_pt_emocao = "neutro"

    try:
        analise_emocao = emotion_analyzer.predict(texto_padronizado)
        
        # Se o modelo retornar um output, nós o usamos
        if analise_emocao.output: 
            label_en_emocao = analise_emocao.output[0]
            score_emocao = analise_emocao.probas[label_en_emocao]
            label_pt_emocao = traducao_emocoes.get(label_en_emocao, label_en_emocao)
        
        # Se analise_emocao.output for vazio (ex: "bom dia!"), 
        # o código simplesmente ignora o 'if' e mantém os valores de 'neutro'
            
    except Exception as e:
        print(f"Erro no modelo de emoção: {e}. Assumindo 'neutro'.")
        # Se o modelo falhar, também mantemos 'neutro' em vez de retornar None

    # --- Passo 2: Analisar Sentimento ---
    try:
        analise_sentimento = sentiment_analyzer.predict(texto_padronizado)
        label_sentimento = analise_sentimento.output 
    except Exception as e:
        print(f"Erro no modelo de sentimento: {e}")
        label_sentimento = "NEU" 

    # --- Passo 3: Lógica de Validação Cruzada ---
    
    # A validação só acontece se a emoção NÃO for neutra
    if label_en_emocao != "neutral":
        if label_sentimento == "NEG" and label_en_emocao in EMOCOES_POSITIVAS:
            return {"emocao_en": "anger", "emocao_pt": "raiva", "confianca": 0.50, "observacao": "Contradição: Sentimento NEG / Emoção POS"}
        if label_sentimento == "POS" and label_en_emocao in EMOCOES_NEGATIVAS:
            return {"emocao_en": "joy", "emocao_pt": "alegria", "confianca": 0.50, "observacao": "Contradição: Sentimento POS / Emoção NEG"}

    # Se não houve contradição (ou se a emoção for 'neutro'), retorna a análise
    return {
        "emocao_en": label_en_emocao,
        "emocao_pt": label_pt_emocao,
        "confianca": round(float(score_emocao), 4),
        "observacao": None # (ou a observação de contradição, se houver)
    }

### MODIFICADO: Função 'processar_arquivos_json' agora extrai o id_serviço ###
def processar_arquivos_json(analyzer_emotion, analyzer_sentiment, arquivos_para_processar):
    """
    Processa UMA LISTA ESPECÍFICA de arquivos JSON.
    """
    resultados = []
    
    if not arquivos_para_processar:
        return []

    progress_bar = st.progress(0, "Processando arquivos de atendimento...")
    for i, arquivo in enumerate(arquivos_para_processar):
        caminho = os.path.join(PASTA_JSON, arquivo)
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)

            for entrada in dados:
                # Nós só analisamos as mensagens do cliente
                if entrada.get("autor") == "cliente":
                    texto = entrada.get("mensagem", "").strip()
                    if texto:
                        resultado_analise = analisar_texto(analyzer_emotion, analyzer_sentiment, texto)
                        if resultado_analise:
                            resultados.append({
                                "arquivo": arquivo,
                                "id_cliente": entrada.get("id_cliente"),
                                "id_funcionario": entrada.get("id_funcionario"),
                                "id_serviço": entrada.get("id_serviço"), # <-- NOVO CAMPO
                                "mensagem": texto,
                                **resultado_analise, 
                                "estado_servico": entrada.get("estado_servico"),
                                "data": entrada.get("data"),
                                "hora": entrada.get("hora")
                            })
        except Exception as e:
            st.error(f"Erro ao processar o arquivo '{arquivo}': {e}")
        
        # Atualiza a barra de progresso
        progresso_atual = (i + 1) / len(arquivos_para_processar)
        progress_bar.progress(progresso_atual, f"Processando: {arquivo}")
    
    progress_bar.empty()
    return resultados

### MODIFICADO: 'carregar_dados_csv' agora inclui 'id_serviço' ###
def carregar_dados_csv():
    colunas_esperadas = [
        "arquivo", "id_cliente", "id_funcionario", "id_serviço", # <-- NOVO CAMPO
        "mensagem", "emocao_en", "emocao_pt", "confianca", 
        "estado_servico", "data", "hora", "observacao"
    ]
    try:
        df = pd.read_csv(ARQUIVO_CSV_SAIDA)
        
        # Garante que colunas novas existam se o CSV for antigo
        if "observacao" not in df.columns:
            df["observacao"] = None
        if "id_serviço" not in df.columns:
            df["id_serviço"] = None # Adiciona a coluna se ela não existir
            
    except FileNotFoundError:
        st.info(f"Arquivo '{ARQUIVO_CSV_SAIDA}' não encontrado. Criando um novo.")
        df = pd.DataFrame(columns=colunas_esperadas)
    
    return df

def salvar_dados_csv(df):
    try:
        df.to_csv(ARQUIVO_CSV_SAIDA, index=False, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"Falha ao salvar o arquivo CSV: {e}")


# --- PARTE 3: INICIALIZAÇÃO DA APLICAÇÃO STREAMLIT ---

st.set_page_config(page_title="Análise de Emoções em Atendimentos", layout="wide")
analyzer_emotion = carregar_modelo_emocao()
analyzer_sentiment = carregar_modelo_sentimento()
if 'df' not in st.session_state:
    st.session_state.df = carregar_dados_csv()
st.title("📊 Análise de Emoções dos Atendimentos: JCSI")


# --- PARTE 4: SIDEBAR INTERATIVA (Formulário + Processamento) ---

### NOVO: Formulário de Adição Rápida ###
st.sidebar.header("⚡ Adicionar Atendimento Rápido")
st.sidebar.caption("Adiciona uma *única mensagem de cliente* ao sistema.")

with st.sidebar.form("novo_atendimento_form", clear_on_submit=True):
    # Pega funcionários existentes para o selectbox
    funcionarios_existentes = ["-"] # Default
    if 'df' in st.session_state and not st.session_state.df.empty:
         funcionarios_existentes = list(st.session_state.df["id_funcionario"].unique())
         
    novo_id_funcionario = st.selectbox("ID do Funcionário*", options=funcionarios_existentes)
    novo_id_cliente = st.text_input("ID do Cliente*")
    novo_id_servico = st.text_input("ID do Serviço/Atendimento*") # <-- CAMPO-CHAVE
    
    novo_estado = st.selectbox("Estado do Serviço", ["pendente", "andamento", "concluido"])
    nova_mensagem = st.text_area("Mensagem do Cliente*")
    
    submit_button = st.form_submit_button("Analisar e Adicionar Mensagem")

if submit_button:
    # Validação simples
    if not all([novo_id_funcionario, novo_id_cliente, novo_id_servico, nova_mensagem]):
        st.sidebar.error("Por favor, preencha todos os campos com *.")
    else:
        with st.spinner("Analisando emoção..."):
            
            resultado_analise = analisar_texto(
                analyzer_emotion, 
                analyzer_sentiment, 
                nova_mensagem
            )
            
            if resultado_analise:
                agora = datetime.now()
                
                novo_registro = {
                    "arquivo": "adicionado_via_app", # Fonte "dinâmica"
                    "id_cliente": novo_id_cliente,
                    "id_funcionario": novo_id_funcionario,
                    "id_serviço": novo_id_servico, # <-- CAMPO-CHAVE
                    "mensagem": nova_mensagem,
                    **resultado_analise, 
                    "estado_servico": novo_estado,
                    "data": agora.strftime('%Y-%m-%d'),
                    "hora": agora.strftime('%H:%M:%S')
                }
                
                df_novo_registro = pd.DataFrame([novo_registro])
                st.session_state.df = pd.concat([st.session_state.df, df_novo_registro], ignore_index=True)
                
                salvar_dados_csv(st.session_state.df)
                
                st.sidebar.success("Atendimento adicionado com sucesso!")
                st.rerun() 
            else:
                st.sidebar.error("Não foi possível analisar a emoção da mensagem.")

st.sidebar.divider()

### Lógica de Processamento em Lote (Inteligente) ###
st.sidebar.header("🔄 Processamento em Lote")
st.sidebar.write("Busca por arquivos .json na pasta `atendimento/`.")
force_reanalysis = st.sidebar.checkbox("Forçar re-análise de TODOS os arquivos")
st.sidebar.caption("Marque esta caixa se você atualizou a lógica de análise e quer corrigir os dados antigos.")

if st.sidebar.button("Iniciar Processamento em Lote"):
    
    if not os.path.isdir(PASTA_JSON):
        st.error(f"ERRO: A pasta '{PASTA_JSON}' não foi encontrada.")
        st.stop()
        
    todos_arquivos_na_pasta = [f for f in os.listdir(PASTA_JSON) if f.endswith(".json")]
    if not todos_arquivos_na_pasta:
        st.info("Nenhum arquivo .json encontrado na pasta 'atendimento/'.")
        st.stop()

    arquivos_para_processar = []
    df_antigo = st.session_state.df.copy() 

    if force_reanalysis:
        st.sidebar.warning("Forçando re-análise de todos os arquivos...")
        arquivos_para_processar = todos_arquivos_na_pasta
        # Limpa o DataFrame antigo (apenas as linhas vindas de JSONs) para que seja substituído
        # Mantém os adicionados via app
        df_antigo = df_antigo[df_antigo['arquivo'] == 'adicionado_via_app'] 
        
    else:
        # Lógica Normal: Processar apenas os novos
        arquivos_processados = set(df_antigo['arquivo'].unique())
        arquivos_para_processar = [f for f in todos_arquivos_na_pasta if f not in arquivos_processados]

    if not arquivos_para_processar:
        st.info("Nenhum arquivo NOVO para processar.")
        st.stop()
    
    st.sidebar.info(f"Processando {len(arquivos_para_processar)} arquivo(s)...")
    
    novos_resultados = processar_arquivos_json(
        analyzer_emotion, 
        analyzer_sentiment,
        arquivos_para_processar
    )
    
    if novos_resultados:
        df_novos = pd.DataFrame(novos_resultados)
        st.session_state.df = pd.concat([df_antigo, df_novos], ignore_index=True)
        salvar_dados_csv(st.session_state.df)
        st.success(f"Sucesso! {len(df_novos)} registros foram processados e adicionados.")
        st.rerun()
    else:
        st.info("Processamento concluído, mas nenhum dado de cliente foi extraído dos novos arquivos.")


st.sidebar.divider()

# --- PARTE 5: DASHBOARD ---
st.sidebar.header("🔍 Filtro por Funcionário")
if st.session_state.df.empty:
    st.sidebar.warning("Nenhum dado para filtrar.")
    funcionarios_selecionados = []
else:
    opcoes_funcionarios = sorted(st.session_state.df["id_funcionario"].unique())
    funcionarios_selecionados = st.sidebar.multiselect(
        "Selecione o(s) funcionário(s):",
        options=opcoes_funcionarios,
        default=opcoes_funcionarios
    )

# ... (código do df_filtrado) ...
if not funcionarios_selecionados and not st.session_state.df.empty:
    st.warning("Por favor, selecione pelo menos um funcionário no filtro da sidebar.")
    df_filtrado = pd.DataFrame(columns=st.session_state.df.columns)
elif st.session_state.df.empty:
     df_filtrado = st.session_state.df.copy()
else:
    df_filtrado = st.session_state.df[st.session_state.df["id_funcionario"].isin(funcionarios_selecionados)]

st.markdown("---")

if df_filtrado.empty:
    st.header("Sem dados para exibir. Processe arquivos JSON ou adicione um atendimento.")
else:
    # --- GRÁFICOS (Inalterados) ---
    # ... (GRÁFICO 1, 2, 3, 4, 5) ...
    st.subheader("🎭 Distribuição de Emoções (Geral)")
    contagem = df_filtrado["emocao_pt"].value_counts()
    cores_mapeadas = [cores_emocoes.get(emocao, "#B0BEC5") for emocao in contagem.index]
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.pie(contagem, labels=None, startangle=90, colors=cores_mapeadas)
    ax1.legend(contagem.index, title="Emoções", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax1.axis("equal")
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("👤 Emoções por Funcionário")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cores_grafico = [cores_emocoes.get(emocao, "#B0BEC5") for emocao in df_filtrado['emocao_pt'].unique()]
    sns.countplot(data=df_filtrado, x="id_funcionario", hue="emocao_pt", ax=ax2, palette=cores_grafico)
    ax2.set_xlabel("Funcionário")
    ax2.set_ylabel("Quantidade de mensagens")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("⚡ Velocímetro de Eficiência por Funcionário")
    cols = st.columns(3) 
    idx = 0
    for func in sorted(df_filtrado["id_funcionario"].unique()):
        df_func = df_filtrado[df_filtrado["id_funcionario"] == func]
        total = len(df_func)
        resolvidos = len(df_func[df_func["estado_servico"] == "concluido"])
        eficiencia = (resolvidos / total) * 100 if total > 0 else 0

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eficiencia,
            title={'text': f"{func}"},
            gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                   'bar': {'color': "green"},
                   'steps': [
                       {'range': [0, 50], 'color': 'lightgray'},
                       {'range': [50, 80], 'color': 'gray'}],
                   'threshold': {
                       'line': {'color': "red", 'width': 4},
                       'thickness': 0.75,
                       'value': 90}}
        ))
        fig.update_layout(height=250)
        
        col = cols[idx % len(cols)]
        with col:
            st.plotly_chart(fig, use_container_width=True)
        idx += 1

    st.markdown("---")
    st.subheader("📊 Confiança Média da Emoção por Funcionário")
    media_confianca = df_filtrado.groupby("id_funcionario")["confianca"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="id_funcionario", y="confianca", data=media_confianca, ax=ax, palette="turbo")
    ax.set_ylabel("Confiança Média da Emoção")
    ax.set_xlabel("Funcionário")
    ax.set_ylim(0, 1) 
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📈 Confiança da Emoção ao Longo do Período")
    df_temp = df_filtrado.copy()
    try:
        df_temp['data'] = pd.to_datetime(df_temp['data'])
        media_sentimento = df_temp.groupby('data')['confianca'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(media_sentimento['data'], media_sentimento['confianca'], marker='o', color="red")
        ax.set_ylabel("Confiança Média")
        ax.set_xlabel("Data")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Não foi possível gerar gráfico de linha temporal (verifique os formatos de data): {e}")

    st.markdown("---")

    ### MODIFICADO: Tabela agora inclui 'id_serviço' e 'arquivo' para contexto ###
    st.subheader("💬 Exemplos de Mensagens (do Filtro Atual)")
    amostra = df_filtrado.sample(min(10, len(df_filtrado))) if len(df_filtrado) > 0 else pd.DataFrame()
    
    st.dataframe(
        amostra[["data", "hora", "id_funcionario", "id_serviço", "mensagem", "emocao_pt", "confianca", "observacao", "arquivo"]],
        hide_index=True,
        use_container_width=True
    )

st.markdown("---")
st.caption("Desenvolvido para análise emocional de atendimentos - usando PySentimiento + Streamlit")