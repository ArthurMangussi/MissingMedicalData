import pandas as pd
import numpy as np

datasets = ["inbreast", "mias", "vindr-reduzido"]
MD_MECHANISM = ["MCAR", "MNAR", "SQUARE"]
MISSING_RATE = [0.05, 0.10, 0.20, 0.3, 0.4, 0.5]
imputers = ["knn", "mc", "mice", "vaewl","mae-vit-gan", "mae-vit"]

results = []
for dataset in datasets:
  for md in MD_MECHANISM:
    for imputer in imputers:
      for mr in MISSING_RATE:
        if md == "SQUARE":
            mr = 0.05
        else:
            mr = mr
            df = pd.read_csv(f"{dataset}_{imputer}_{md}_{mr}_results.csv")
            df["DATASET"] = dataset
            df["ALGORITHS"] = imputer
            df["MD_MECHANISM"] = md
            df["MISSING_RATE"] = f"mr{mr}"
            results.append(df)

df_results = pd.concat(results).rename(columns={"Unnamed: 0":"fold"})
df_results["MISSING_RATE"] = df_results["MISSING_RATE"].map({"mr0.05": "5%",
                                             "mr0.1":"10%",
                                             "mr0.2":"20%",
                                              "mr0.3":"30%",
                                                             "mr0.4":"40%",
                                                             "mr0.5":"50%",})

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados (tratando o cabeçalho complexo manualmente)
df_raw = pd.read_excel('classificação.xlsx')

metrics = ['ACC', 'AUC_ROC', 'F1']
datasets = ['INBreast', 'MIAS', 'VinDR-mammo']
records = []

# 2. Extrair o Baseline (que está na linha de índice 2 do CSV original)
baseline_row = df_raw.iloc[2]
for i in range(2, 11):
    metric = metrics[(i-2) // 3]
    dataset = datasets[(i-2) % 3]
    val_str = str(baseline_row[i])
    if '±' in val_str:
        mean_val = float(val_str.split('±')[0].strip())
        records.append({
            'Missing Rate': 0.0,
            'Method': 'BASELINE',
            'Metric': metric,
            'Dataset': dataset,
            'Value': mean_val
        })

# 3. Extrair os dados de Inpainting (começam na linha de índice 4)
current_mr = None
for idx in range(4, len(df_raw)):
    row = df_raw.iloc[idx]

    # Atualiza o Missing Rate atual quando encontra um novo valor na coluna 0
    mr_cell = str(row[0])
    if mr_cell != 'nan' and mr_cell != 'Missing Rate':
        try:
            current_mr = float(mr_cell)
        except ValueError:
            pass

    method = str(row[1])
    if method == 'nan' or method == 'Inpainting Methods':
        continue

    for i in range(2, 11):
        metric = metrics[(i-2) // 3]
        dataset = datasets[(i-2) % 3]
        val_str = str(row[i])
        if '±' in val_str:
            try:
                mean_val = float(val_str.split('±')[0].strip())
                records.append({
                    'Missing Rate': current_mr,
                    'Method': method,
                    'Metric': metric,
                    'Dataset': dataset,
                    'Value': mean_val
                })
            except ValueError:
                continue

# Criar DataFrame limpo
df_all = pd.DataFrame(records)

# 4. Agrupar por Missing Rate (calculando a média entre os diferentes métodos de inpainting)
df_summary = df_all.groupby(['Missing Rate', 'Metric', 'Dataset'])['Value'].mean().reset_index()

# 5. Gerar Visualização
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, metric in enumerate(metrics):
    ax = axes[i]
    data_metric = df_summary[df_summary['Metric'] == metric]

    if metric == "ACC":
      baseline_value_inbreast = 0.7781
      baseline_value_mias = 0.6959
      baseline_value_vindr = 0.5981
    elif metric == "AUC-ROC":
      baseline_value_inbreast = 0.7602
      baseline_value_mias = 0.7098
      baseline_value_vindr = 0.6675
    else:
      baseline_value_inbreast = 0.5705
      baseline_value_mias = 0.4973
      baseline_value_vindr = 0.5842

    # Criar gráfico de barras com Missing Rate no eixo X
    sns.barplot(data=data_metric, x='Missing Rate', y='Value', hue='Dataset', ax=ax)
    ax.axhline(y=baseline_value_inbreast, color='red', linestyle='--', label='Baseline INBreast')
    ax.axhline(y=baseline_value_mias, color='black', linestyle='--', label='Baseline Mias')
    ax.axhline(y=baseline_value_vindr, color='purple', linestyle='--', label='Baseline VinDr-mammo')

    ax.set_title(f'Classification Metric: {metric}', fontsize=14)
    ax.set_ylabel('Average Metric', fontsize=12)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylim(0, 0.8)
    ax.legend(title='Dataset', bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("classification.png", transparent=True, dpi=300,  bbox_inches='tight')

import pandas as pd
import numpy as np

# O nome do arquivo CSV carregado é 'results.csv'
FILE_PATH = "results.csv"

def format_mean_std(series_mean, series_std, decimals=4):
    """
    Formata as séries de Média e Desvio Padrão como uma única string: "Média ± Desvio Padrão".
    Trata casos onde o desvio padrão é NaN (ex: apenas 1 observação) como 0.
    """
    # Tratar NaN no desvio padrão como 0.0
    series_std = series_std.fillna(0.0)

    # Formato: "Média.dddd ± DesvioPadrão.dddd"
    formatted_results = (
        series_mean.round(decimals).astype(str) +
        " ± " +
        series_std.round(decimals).astype(str)
    )
    return formatted_results

def create_pivot_table(file_path):
    """
    Carrega os dados, calcula Média ± Desvio Padrão e cria uma tabela pivotada
    com as colunas aninhadas (Dataset -> Mecanismo -> Métrica).
    """
    try:
        # 1. Carregar o arquivo CSV
        df = pd.read_csv(file_path)
        print(f"Arquivo '{file_path}' carregado. Total de linhas: {len(df)}")

        # 2. Colunas de métricas e de agrupamento
        metrics_cols = ['ACC', 'F1', 'AUC_ROC']
        grouping_cols = ['DATASET', 'ALGORITHS','MISSING_RATE']

        # 3. Converter colunas de métricas para tipo numérico
        for col in metrics_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Agrupar e calcular Média (mean) e Desvio Padrão (std)
        aggregated_results = df.groupby(grouping_cols)[metrics_cols].agg(['mean', 'std'])

        # 5. Resetar o índice e achatar o MultiIndex das colunas
        aggregated_results = aggregated_results.reset_index()
        aggregated_results.columns = [
            '_'.join(col).strip() if col[1] else col[0]
            for col in aggregated_results.columns.values
        ]

        # 6. Formatar as colunas de Média e Desvio Padrão juntas
        formatted_df = aggregated_results[grouping_cols].copy()

        for metric in metrics_cols:
            formatted_df[metric] = format_mean_std(
                aggregated_results[f'{metric}_mean'],
                aggregated_results[f'{metric}_std']
            )

        # 7. Criar a Tabela Pivotada
        # Indices (linhas): MISSING_RATE e ALGORITHS (métodos de imputação)
        # Colunas (cabeçalhos aninhados): DATASET, MD_MECHANISM (mecanismo), Métrica (MSE, PSNR, SSIM)

        pivot_table_df = formatted_df.pivot_table(
            index=['MISSING_RATE', 'ALGORITHS'],
            columns=['DATASET'],
            values=metrics_cols,
            # Mantém a ordem das colunas de métricas (MSE, PSNR, SSIM)
            aggfunc=lambda x: x
        )

        # 8. Limpar e reorganizar as colunas para a ordem desejada (Métrica aninhada em Mecanismo)
        # O pivot_table inverte as colunas, então precisamos reordená-las:
        # (DATASET, MD_MECHANISM, Métrica)

        # O nível 0 é DATASET, o nível 1 é MD_MECHANISM, o nível 2 é Métrica
        # Reordenamos o nível 2 (Métricas) para vir primeiro, seguido por Dataset e Mecanismo.

        # Trocamos os níveis para que as Métricas (MSE, PSNR, SSIM) venham antes no nome da coluna.
        # No entanto, para a visualização, o ideal é: Dataset -> Mecanismo -> Métrica

        # Apenas renomeamos o índice e mantemos a estrutura padrão para exportação fácil
        pivot_table_df.index.names = ['Taxa de Missing', 'Método de Imputação']

        print("\nTabela pivotada criada com sucesso.")

        return pivot_table_df

    except FileNotFoundError:
        print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return pd.DataFrame()

# --- Execução Principal ---
pivot_table_metrics_df = create_pivot_table(FILE_PATH)

if not pivot_table_metrics_df.empty:
    # 9. Salvar os resultados em um novo arquivo CSV.
    # Salvamos o cabeçalho com o MultiIndex, que é interpretado por programas como Excel/Google Sheets.
    output_file = 'pivot_table_summary.csv'
    pivot_table_metrics_df.to_csv(output_file)
    print(f"\nResultados salvos com sucesso em '{output_file}'.")

    print("\nPrévia da Tabela Pivotada (as colunas estão aninhadas):")
    # Usar to_string() para exibir o MultiIndex formatado no console
    print(pivot_table_metrics_df.head().to_string())