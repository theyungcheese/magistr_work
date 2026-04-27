"""
Фаза 1. Загрузка и предобработка данных.

Что делает:
1. Загружает ESS9, отбирает 37 переменных (PVQ-21 + trust + media + демография + meta)
2. Явно заменяет special codes на NaN для каждой группы переменных
3. Строит country-level датасет (GDP PCAP PPP, Gini, CPI за 2018; для Gini -- ближайший год если 2018 нет)
4. Сохраняет data/processed/ess9_analytic.csv и data/processed/country_level.csv
5. Строит и сохраняет графики missingness и страновые размеры выборки
6. Пишет summary в outputs/tables/phase1_summary.txt

Запуск: .venv/bin/python scripts/01_load_and_clean.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data_2018"
ESS_CSV = DATA / "ESS9e03_3" / "ESS9e03_3.csv"
GDP_CSV = DATA / "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_43" / "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_43.csv"
GINI_CSV = DATA / "API_SI.POV.GINI_DS2_en_csv_v2_288" / "API_SI.POV.GINI_DS2_en_csv_v2_288.csv"
CPI_XLSX = DATA / "CPI2018_Full-Results_1801.xlsx"

OUT_DIR = ROOT / "data" / "processed"
OUT_ESS = OUT_DIR / "ess9_analytic.csv"
OUT_COUNTRY = OUT_DIR / "country_level.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# === Переменные ESS ===

PVQ = [
    "ipcrtiv", "imprich", "ipeqopt", "ipshabt", "impsafe", "impdiff",
    "ipfrule", "ipudrst", "ipmodst", "ipgdtim", "impfree", "iphlppl",
    "ipsuces", "ipstrgv", "ipadvnt", "ipbhprp", "iprspot", "iplylfr",
    "impenv", "imptrad", "impfun",
]  # valid 1-6, special 7/8/9

TRUST = ["ppltrst", "pplfair", "pplhlp"]  # valid 0-10, special 77/88/99
MEDIA_LIKERT = ["netusoft"]  # valid 1-5, special 7/8/9
MEDIA_MINUTES = ["netustm", "nwspol"]  # valid 0-1440, special 6666/7777/8888/9999

DEMO_AGE = ["agea"]  # valid 15-120, special 999
DEMO_GENDER = ["gndr"]  # valid 1-2, special 9 (не встречается на деле)
DEMO_EDU_YEARS = ["eduyrs"]  # valid 0-30, special 77/88/99
DEMO_EISCED = ["eisced"]  # valid 1-7, 55=other, 77/88/99
DEMO_INCOME = ["hinctnta"]  # valid 1-10, special 77/88/99
DEMO_DOMICIL = ["domicil"]  # valid 1-5, special 7/8/9

META = ["cntry", "dweight", "pspwght", "pweight"]

ALL_VARS = (
    PVQ + TRUST + MEDIA_LIKERT + MEDIA_MINUTES
    + DEMO_AGE + DEMO_GENDER + DEMO_EDU_YEARS + DEMO_EISCED + DEMO_INCOME + DEMO_DOMICIL
    + META
)

# === Country code mapping (ESS 2-char → ISO3) ===

CNTRY_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CH": "CHE", "CY": "CYP",
    "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "GB": "GBR", "HR": "HRV", "HU": "HUN",
    "IE": "IRL", "IS": "ISL", "IT": "ITA", "LT": "LTU", "LV": "LVA",
    "ME": "MNE", "NL": "NLD", "NO": "NOR", "PL": "POL", "PT": "PRT",
    "RS": "SRB", "SE": "SWE", "SI": "SVN", "SK": "SVK",
}


def load_ess():
    print("[1] Загрузка ESS9 и отбор переменных")
    df = pd.read_csv(ESS_CSV, usecols=ALL_VARS, low_memory=False)
    print(f"  Строк: {df.shape[0]}, переменных: {df.shape[1]}")
    return df


def clean_special_codes(df):
    print("[2] Обработка special codes")

    def replace(cols, codes, label):
        before = df[cols].isna().sum().sum()
        for c in cols:
            df.loc[df[c].isin(codes), c] = np.nan
        after = df[cols].isna().sum().sum()
        print(f"  {label}: +{after - before} NaN")

    replace(PVQ, [7, 8, 9], f"PVQ-21 (codes 7/8/9)")
    replace(TRUST, [77, 88, 99], "Trust 0-10 (codes 77/88/99)")
    replace(MEDIA_LIKERT, [7, 8, 9], "netusoft 1-5 (codes 7/8/9)")
    replace(MEDIA_MINUTES, [6666, 7777, 8888, 9999], "netustm/nwspol minutes (codes 6666-9999)")
    replace(DEMO_AGE, [999], "agea (code 999)")
    replace(DEMO_GENDER, [9], "gndr (code 9)")
    replace(DEMO_EDU_YEARS, [77, 88, 99], "eduyrs (codes 77/88/99)")
    replace(DEMO_EISCED, [55, 77, 88, 99], "eisced (codes 55/77/88/99)")
    replace(DEMO_INCOME, [77, 88, 99], "hinctnta (codes 77/88/99)")
    replace(DEMO_DOMICIL, [7, 8, 9], "domicil (codes 7/8/9)")

    # netustm и nwspol могут иметь значения > 1440 минут (больше суток) -- явно ошибки ввода.
    # Стандартная практика: top-code на 1440 или NaN. Выбираем NaN.
    for c in MEDIA_MINUTES:
        mask = df[c] > 1440
        n = mask.sum()
        if n > 0:
            df.loc[mask, c] = np.nan
            print(f"  {c}: дополнительно {n} значений > 1440 минут → NaN")

    # eduyrs top-coding: значения > 25 лет встречаются у 0.45% респондентов и обычно
    # являются ошибками ввода. 99-й перцентиль = 24 года. Режем на 25.
    mask = df["eduyrs"] > 25
    n = mask.sum()
    if n > 0:
        df.loc[mask, "eduyrs"] = 25
        print(f"  eduyrs: top-coding {n} значений > 25 лет → 25")

    return df


def summarise_missingness(df):
    print("[3] Анализ пропусков")
    miss = df.isna().mean().sort_values(ascending=False)
    miss_pct = (miss * 100).round(2)

    # Таблица в файл
    miss_pct.to_frame("missing_pct").to_csv(TAB_DIR / "phase1_missingness.csv")

    # График
    fig, ax = plt.subplots(figsize=(12, 6))
    miss_pct.plot(kind="bar", ax=ax)
    ax.set_title("Доля пропусков по переменной после очистки, %")
    ax.set_ylabel("Доля пропусков, %")
    ax.set_xlabel("Переменная")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase1_missingness.png", dpi=120)
    plt.close()

    print("  Top-5 переменных с пропусками:")
    for v, p in miss_pct.head(5).items():
        print(f"    {v}: {p}%")

    return miss_pct


def country_sample_sizes(df):
    print("[4] Размеры выборки по странам")
    counts = df["cntry"].value_counts().sort_values(ascending=False)
    counts.to_csv(TAB_DIR / "phase1_country_n.csv", header=["n"])

    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Размер выборки по странам (ESS Round 9)")
    ax.set_ylabel("Количество респондентов")
    ax.set_xlabel("Страна")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase1_country_n.png", dpi=120)
    plt.close()

    print(f"  Стран: {counts.shape[0]}")
    print(f"  Минимум респондентов: {counts.min()} ({counts.idxmin()})")
    print(f"  Максимум респондентов: {counts.max()} ({counts.idxmax()})")
    return counts


def build_country_level():
    print("[5] Сборка country-level датасета")
    iso3_list = sorted(CNTRY_TO_ISO3.values())

    # GDP per capita PPP, колонка "2018"
    gdp = pd.read_csv(GDP_CSV, skiprows=4)
    gdp_2018 = gdp[["Country Code", "2018"]].rename(columns={"Country Code": "iso3", "2018": "gdp_pcap_ppp"})
    gdp_2018 = gdp_2018[gdp_2018["iso3"].isin(iso3_list)]

    # Gini: пробуем 2018, если пусто -- fallback на ближайшие годы 2017, 2019, 2016, 2020
    gini = pd.read_csv(GINI_CSV, skiprows=4)
    gini_rows = []
    for iso3 in iso3_list:
        row = gini[gini["Country Code"] == iso3]
        if row.empty:
            gini_rows.append({"iso3": iso3, "gini": np.nan, "gini_year": np.nan})
            continue
        year_order = ["2018", "2017", "2019", "2016", "2020", "2015", "2021"]
        chosen_year = None
        val = np.nan
        for y in year_order:
            if y in row.columns and pd.notna(row[y].iloc[0]):
                chosen_year = y
                val = float(row[y].iloc[0])
                break
        gini_rows.append({"iso3": iso3, "gini": val, "gini_year": chosen_year})
    gini_df = pd.DataFrame(gini_rows)

    # CPI: читаем лист CPI2018
    cpi = pd.read_excel(CPI_XLSX, sheet_name="CPI2018", header=2)
    # Ключевая колонка называется с пробелом в конце или без, находим надёжно
    score_col = [c for c in cpi.columns if "CPI Score 2018" in c][0]
    cpi_2018 = cpi[["ISO3", score_col]].rename(columns={"ISO3": "iso3", score_col: "cpi"})
    cpi_2018 = cpi_2018[cpi_2018["iso3"].isin(iso3_list)]

    # Мерж
    iso_df = pd.DataFrame({"iso3": iso3_list})
    iso_df["cntry"] = iso_df["iso3"].map({v: k for k, v in CNTRY_TO_ISO3.items()})
    country = iso_df.merge(gdp_2018, on="iso3", how="left")
    country = country.merge(gini_df, on="iso3", how="left")
    country = country.merge(cpi_2018, on="iso3", how="left")

    # Переупорядочить колонки
    country = country[["cntry", "iso3", "gdp_pcap_ppp", "gini", "gini_year", "cpi"]]

    # Диагностика покрытия
    print("  Покрытие country-level:")
    for col in ["gdp_pcap_ppp", "gini", "cpi"]:
        n = country[col].notna().sum()
        miss = country[col].isna().sum()
        print(f"    {col}: {n} из {len(country)} (пропусков: {miss})")

    # Страны без Gini (если есть) -- выведем явно
    missing_gini = country[country["gini"].isna()]["cntry"].tolist()
    if missing_gini:
        print(f"  Страны без Gini: {missing_gini}")

    country.to_csv(OUT_COUNTRY, index=False)
    print(f"  Сохранено: {OUT_COUNTRY}")
    return country


def write_summary(df, country, miss_pct, counts):
    lines = []
    lines.append("=" * 60)
    lines.append("Фаза 1. Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"ESS9 после очистки: {df.shape[0]} строк, {df.shape[1]} переменных")
    lines.append(f"Стран: {df['cntry'].nunique()}")
    lines.append("")
    lines.append("Топ-10 переменных с пропусками:")
    for v, p in miss_pct.head(10).items():
        lines.append(f"  {v}: {p}%")
    lines.append("")
    lines.append("Размеры выборки по странам:")
    for c, n in counts.items():
        lines.append(f"  {c}: {n}")
    lines.append("")
    lines.append(f"Country-level: {country.shape[0]} стран")
    lines.append(country.to_string(index=False))

    summary_path = TAB_DIR / "phase1_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[6] Summary сохранён в {summary_path}")


def main():
    print("=" * 60)
    print("Фаза 1. Загрузка и предобработка данных")
    print("=" * 60)
    print()

    df = load_ess()
    df = clean_special_codes(df)

    # Сохраняем очищенные индивидуальные данные
    df.to_csv(OUT_ESS, index=False)
    print(f"  Сохранено: {OUT_ESS}")
    print()

    miss_pct = summarise_missingness(df)
    print()

    counts = country_sample_sizes(df)
    print()

    country = build_country_level()
    print()

    write_summary(df, country, miss_pct, counts)
    print()
    print("=" * 60)
    print("Фаза 1 завершена успешно")
    print("=" * 60)


if __name__ == "__main__":
    main()
