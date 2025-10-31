import os
import json
import pandas as pd
from tqdm import tqdm
from pysentimiento import create_analyzer
import matplotlib.pyplot as plt
import numpy as np

# --- PARTE 1: CONFIGURAÇÃO GERAL ---

# Pasta onde estão os seus arquivos JSON de atendimentos
PASTA_JSON = "atendimento/"

# Dicionário completo para tradução e cores das emoções.
# A chave é o termo em inglês retornado pelo modelo.
EMOCOES_MAP = {
    # Emoções primárias e mais comuns
    "joy":          {"pt": "alegria",   "cor": "#2ECC71"},
    "sadness":      {"pt": "tristeza",  "cor": "#3498DB"},
    "anger":        {"pt": "raiva",     "cor": "#E74C3C"},
    "fear":         {"pt": "medo",      "cor": "#F1C40F"},
    "surprise":     {"pt": "surpresa",  "cor": "#9B59B6"},
    "disgust":      {"pt": "desgosto",  "cor": "#795548"},
    "neutral":      {"pt": "neutro",    "cor": "#95A5A6"},
    
    # Emoções secundárias que o modelo também pode retornar
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

# Cria dicionários específicos para facilitar o uso no código
traducao_emocoes = {en: v["pt"] for en, v in EMOCOES_MAP.items()}
cores_emocoes = {v["pt"]: v["cor"] for en, v in EMOCOES_MAP.items()}


# --- PARTE 2: ANÁLISE DE SENTIMENTOS ---

print("Carregando modelo de análise de emoções (isso pode levar um momento)...")
analyzer = create_analyzer(task="emotion", lang="pt")
print("Modelo carregado com sucesso.")

resultados = []

# Verifica se a pasta de origem dos dados existe
if not os.path.isdir(PASTA_JSON):
    print(f"❌ ERRO: A pasta '{PASTA_JSON}' não foi encontrada. Verifique o caminho e tente novamente.")
else:
    arquivos_json = [f for f in os.listdir(PASTA_JSON) if f.endswith(".json")]
    
    if not arquivos_json:
        print(f"⚠️ AVISO: Nenhum arquivo .json foi encontrado na pasta '{PASTA_JSON}'.")
    else:
        for arquivo in tqdm(arquivos_json, desc="Processando arquivos de atendimento"):
            caminho = os.path.join(PASTA_JSON, arquivo)
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    dados = json.load(f)

                for entrada in dados:
                    if entrada.get("autor") == "cliente":
                        texto = entrada.get("mensagem", "").strip()

                        if texto:
                            analise = analyzer.predict(texto)
                            
                            if analise.output:
                                label_en = analise.output[0]
                                score = analise.probas[label_en]
                                
                                # Usa o dicionário para traduzir, se não encontrar, mantém o original em inglês
                                label_pt = traducao_emocoes.get(label_en, label_en)

                                resultados.append({
                                    "arquivo": arquivo,
                                    "id_cliente": entrada.get("id_cliente"),
                                    "id_funcionario": entrada.get("id_funcionario"),
                                    "mensagem": texto,
                                    "emocao_en": label_en,
                                    "emocao_pt": label_pt,
                                    "confianca": round(float(score), 4),
                                    "estado_servico": entrada.get("estado_servico")
                                })
            except Exception as e:
                print(f"❌ Erro inesperado ao processar o arquivo '{arquivo}': {e}")


# --- PARTE 3: GERAÇÃO DE ARQUIVOS (CSV E GRÁFICOS) ---

if resultados:
    # Cria o DataFrame principal com todos os dados analisados
    df = pd.DataFrame(resultados)
    
    # Salva o arquivo CSV consolidado
    caminho_csv = "emocao_clientes_todos.csv"
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Análise concluída! Resultados consolidados salvos em '{caminho_csv}'")

    # Cria a pasta para os gráficos, se ela não existir
    PASTA_GRAFICOS = "graficos"
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    
    print("\nGerando gráficos de pizza por atendimento (estilo simples)...")
    for arquivo, grupo in tqdm(df.groupby("arquivo"), desc="Criando gráficos"):
        contagem = grupo["emocao_pt"].value_counts()
        
        # Mapeia as cores para cada emoção no gráfico
        # Fallback para cinza claro (#B0BEC5) se uma emoção não tiver cor mapeada
        cores_mapeadas = [cores_emocoes.get(emocao, "#B0BEC5") for emocao in contagem.index]
        
        plt.figure(figsize=(8, 8))
        
        # --- LÓGICA DO GRÁFICO DE PIZZA SIMPLES ---
        # Removido 'shadow=True' e 'explode'
        plt.pie(
            contagem,
            labels=contagem.index,
            colors=cores_mapeadas,
            autopct="%1.0f%%", # Simplificado para porcentagem inteira (ex: 13% em vez de 12.5%)
            startangle=90,     # Ângulo inicial para o primeiro slice
            # pctdistance=0.85 # Removido para deixar a distância padrão da porcentagem
            # textprops={'fontsize': 12, 'color': 'black'} # Opcional: para ajustar a fonte dos rótulos
        )
        
        plt.title(f"Distribuição de Emoções\nAtendimento: {arquivo}", fontsize=14, weight='bold')
        plt.axis('equal')  # Garante que o gráfico seja um círculo perfeito
        plt.tight_layout()

        # Salva o gráfico em um arquivo de imagem
        nome_arquivo_grafico = os.path.splitext(arquivo)[0]
        plt.savefig(f"{PASTA_GRAFICOS}/{nome_arquivo_grafico}_emocoes.png", dpi=120, bbox_inches='tight')
        plt.close() # Libera a memória da figura

    print(f"✅ Gráficos gerados com sucesso! Verifique a pasta '{PASTA_GRAFICOS}/'")

    # --- PARTE 4: VERIFICAÇÃO DE EMOÇÕES NÃO MAPEADAS (permanece inalterada) ---
    
    emocoes_unicas_en = df['emocao_en'].unique()
    emocoes_nao_mapeadas = [e for e in emocoes_unicas_en if e not in EMOCOES_MAP]

    if emocoes_nao_mapeadas:
        print("\n" + "="*60)
        print("🚨 ATENÇÃO: As seguintes emoções foram detectadas nos seus dados,")
        print("   mas ainda não foram adicionadas ao dicionário EMOCOES_MAP:")
        for emocao in emocoes_nao_mapeadas:
            print(f"   - {emocao}")
        print("\n   Para traduzir e colorir corretamente, adicione-as ao dicionário no topo do script.")
        print("="*60)
    else:
        print("\n✅ Verificação final: Todas as emoções detectadas já estão mapeadas no dicionário.")

else:
    print("\nℹ️ Análise concluída, mas nenhum dado de cliente foi encontrado para processar. Nenhum arquivo foi gerado.")
