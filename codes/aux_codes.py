import pandas as pd

datasets = ["inbreast", "mias", "vindr-reduzido"]
MD_MECHANISM = ["MNAR-SQUARES"]
imputers = ["knn", "mc", "dip", "vaewl","mae-vit-gan", "mae-vit", "diffusion"]

results = []
for dataset in datasets:
  for md in MD_MECHANISM:
    for imputer in imputers:
        df = pd.read_csv(f"/home/gpu-10-2025/Área de trabalho/MissingMedicalData/results/{imputer}/{dataset}_{imputer}_{md}_results.csv")
        df["DATASET"] = dataset
        df["ALGORITHS"] = imputer
        df["MD_MECHANISM"] = md
        results.append(df)

df_results = pd.concat(results).rename(columns={"Unnamed: 0":"fold"})
df_results.to_csv("results.csv", index=False)


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
        metrics_cols = ['PSNR', 'MAE', 'SSIM']
        grouping_cols = ['DATASET', 'ALGORITHS']

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
            index=['ALGORITHS'],
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
        pivot_table_df.index.names = ['Método de Imputação']

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