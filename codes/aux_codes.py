import pandas as pd

datasets = ["breakhist"]  # List of datasets to process
MD_MECHANISMS = ["MCAR", "MAR", "MNAR"]  # processadas separadamente, uma tabela por mecanismo
imputers = ["knn", "mae-vit", "dip", "mat", "harp"]

def format_mean_std(series_mean, series_std, decimals=3):
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
# Roda cada mecanismo (MCAR, MAR, MNAR) separadamente: um results_breakhist_{md}.csv
# e uma tabela pivotada (CSV + LaTeX) próprios para cada um, em vez de misturar tudo
# numa unica tabela.
RESULTS_DIR = "/home/gpu-10-2025/Área de trabalho/MissingMedicalData/results"

pivot_tables_by_mechanism = {}

for md in MD_MECHANISMS:
    results = []
    for dataset in datasets:
        for imputer in imputers:
            df = pd.read_csv(
                f"{RESULTS_DIR}/{imputer}/{dataset}_{imputer}_{md}_results.csv"
            )
            df["DATASET"] = dataset
            df["ALGORITHS"] = imputer
            df["MD_MECHANISM"] = md
            results.append(df)

    df_results = pd.concat(results).rename(columns={"Unnamed: 0": "fold"})
    results_csv = f"{RESULTS_DIR}/results_breakhist_{md}.csv"
    df_results.to_csv(results_csv, index=False)

    pivot_table_metrics_df = create_pivot_table(results_csv)

    if not pivot_table_metrics_df.empty:
        # Salva o CSV (cabeçalho com o MultiIndex, interpretado por Excel/Google Sheets)
        # e a versão LaTeX da mesma tabela pivotada.
        csv_out = f"{RESULTS_DIR}/pivot_table_summary_breakhist_{md}.csv"
        tex_out = f"{RESULTS_DIR}/pivot_table_summary_breakhist_{md}.tex"

        pivot_table_metrics_df.to_csv(csv_out)
        pivot_table_metrics_df.to_latex(
            tex_out,
            multicolumn=True,
            multicolumn_format="c",
            caption=f"Qualidade de imputação (MAE, PSNR, SSIM) no BreaKHis -- mecanismo {md}.",
            label=f"tab:breakhist_{md.lower()}",
        )

        print(f"\n=== {md} ===")
        print(f"Resultados salvos em '{csv_out}' e '{tex_out}'.")
        print("\nPrévia da Tabela Pivotada (as colunas estão aninhadas):")
        print(pivot_table_metrics_df.to_string())

        # DATASET so tem o valor "breakhist" aqui -- descartamos esse nivel
        # antes de juntar os 3 mecanismos numa unica tabela abaixo.
        pivot_tables_by_mechanism[md] = pivot_table_metrics_df.droplevel(
            "DATASET", axis=1
        )

# --- Tabela unica com os 3 mecanismos lado a lado ---
if len(pivot_tables_by_mechanism) == len(MD_MECHANISMS):
    combined_table = pd.concat(
        [pivot_tables_by_mechanism[md] for md in MD_MECHANISMS],
        axis=1,
        keys=MD_MECHANISMS,
    )
    combined_table.columns.names = ["Mechanism", "Metric"]

    combined_csv = f"{RESULTS_DIR}/pivot_table_summary_breakhist_all_mechanisms.csv"
    combined_tex = f"{RESULTS_DIR}/pivot_table_summary_breakhist_all_mechanisms.tex"

    combined_table.to_csv(combined_csv)
    combined_table.to_latex(
        combined_tex,
        multicolumn=True,
        multicolumn_format="c",
        caption=(
            "Imputation quality (MAE, PSNR, SSIM; mean $\\pm$ std over 5 folds) "
            "on BreaKHis, across missingness mechanisms (MCAR, MAR, MNAR)."
        ),
        label="tab:breakhist_all_mechanisms",
    )

    print("\n=== Combined (MCAR + MAR + MNAR) ===")
    print(f"Resultados salvos em '{combined_csv}' e '{combined_tex}'.")
    print("\nPrévia da Tabela Combinada:")
    print(combined_table.to_string())