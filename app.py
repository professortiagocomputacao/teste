# TP2 de Desenvolvimento Front-End com Python (Streamlit)
# Dados reais do portal Coronavírus Brasil (https://covid.saude.gov.br/).
# Os 12 exercícios estão neste único arquivo, identificados por comentários.
#
# Na primeira execução o app baixa a base histórica oficial (HIST_PAINEL_COVIDBR,
# ~50 MB) pela mesma API usada pelo botão "Arquivo CSV" do portal, além de um CSV
# com as coordenadas dos municípios (IBGE), salvando tudo em ./data/.
#
# Executar com: streamlit run app.py
# Dependências: streamlit, pandas, requests, matplotlib, seaborn, altair, plotly, pydeck

import glob
import io
import os
import zipfile

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="COVID-19 Brasil - TP2 Streamlit", layout="wide")

DATA_DIR = "data"
API_PORTAL = "https://qd28tcd6b5.execute-api.sa-east-1.amazonaws.com/prod/PortalGeral"
PARSE_APP_ID = "unAFkcaNDeXajurGB7LChj8SgQYS2ptm"
MUNICIPIOS_URL = (
    "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
)
MUNICIPIOS_CSV = os.path.join(DATA_DIR, "municipios.csv")

REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


# -------------------------------------------------------------------------
# Download e carga dos dados
# -------------------------------------------------------------------------
def baixar_dados() -> None:
    """Baixa a base histórica do portal e as coordenadas dos municípios (uma vez)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not glob.glob(os.path.join(DATA_DIR, "HIST_PAINEL_COVIDBR*.csv")):
        with st.spinner("Consultando a API do portal Coronavírus Brasil..."):
            resp = requests.get(
                API_PORTAL,
                headers={"X-Parse-Application-Id": PARSE_APP_ID},
                timeout=60,
            )
            resp.raise_for_status()
            url_zip = resp.json()["results"][0]["arquivo"]["url"]

        barra = st.progress(0.0, text="Baixando a base histórica (~50 MB)...")
        resp = requests.get(url_zip, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        buffer = io.BytesIO()
        baixado = 0
        for chunk in resp.iter_content(chunk_size=1 << 20):
            buffer.write(chunk)
            baixado += len(chunk)
            if total:
                barra.progress(
                    min(baixado / total, 1.0),
                    text=f"Baixando a base histórica... {baixado / 1e6:.0f} de {total / 1e6:.0f} MB",
                )
        with zipfile.ZipFile(buffer) as zf:
            zf.extractall(DATA_DIR)
        barra.empty()

    if not os.path.exists(MUNICIPIOS_CSV):
        with st.spinner("Baixando coordenadas dos municípios (IBGE)..."):
            resp = requests.get(MUNICIPIOS_URL, timeout=60)
            resp.raise_for_status()
            with open(MUNICIPIOS_CSV, "wb") as f:
                f.write(resp.content)


def rotulo_semana(datas: pd.Series, semanas: pd.Series) -> pd.Series:
    """Rótulo ordenável 'AAAA-Sww' (ano da data + semana epidemiológica),
    ajustando a virada de ano."""
    ano = datas.dt.year.copy()
    ano[(datas.dt.month == 12) & (semanas == 1)] += 1
    ano[(datas.dt.month == 1) & (semanas >= 52)] -= 1
    return ano.astype(str) + "-S" + semanas.astype(int).astype(str).str.zfill(2)


@st.cache_data(show_spinner="Lendo e agregando a base histórica...")
def carregar_dados() -> dict:
    """Lê os CSVs históricos e devolve apenas os recortes usados pelo painel."""
    usecols = [
        "regiao", "estado", "municipio", "codmun", "data", "semanaEpi",
        "populacaoTCU2019", "casosAcumulado", "casosNovos",
        "obitosAcumulado", "obitosNovos",
    ]
    partes = []
    for caminho in sorted(glob.glob(os.path.join(DATA_DIR, "HIST_PAINEL_COVIDBR*.csv"))):
        partes.append(pd.read_csv(caminho, sep=";", usecols=usecols))
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"])

    # Recorte Brasil (uma linha por dia)
    df_br = df[df["regiao"] == "Brasil"].copy()
    df_br["semana"] = rotulo_semana(df_br["data"], df_br["semanaEpi"])

    # Recorte por UF (linhas de estado, sem município)
    df_uf = df[df["estado"].notna() & df["codmun"].isna()].copy()
    df_uf["semana"] = rotulo_semana(df_uf["data"], df_uf["semanaEpi"])

    # Recorte municipal: só a última data disponível, com lat/lon do IBGE
    e_mun = df["municipio"].notna()
    ultima_data = df.loc[e_mun, "data"].max()
    df_mun = df[e_mun & (df["data"] == ultima_data)].copy()
    coords = pd.read_csv(MUNICIPIOS_CSV)
    coords["codmun"] = coords["codigo_ibge"] // 10  # 7 dígitos IBGE -> 6 da base
    df_mun = df_mun.merge(
        coords[["codmun", "latitude", "longitude"]], on="codmun", how="inner"
    )

    # Agregações semanais (novos: soma da semana; acumulados: valor no fim da semana)
    agg = dict(
        casosNovos=("casosNovos", "sum"),
        obitosNovos=("obitosNovos", "sum"),
        casosAcumulado=("casosAcumulado", "max"),
        obitosAcumulado=("obitosAcumulado", "max"),
        data_ref=("data", "min"),
    )
    sem_br = df_br.groupby("semana").agg(**agg).reset_index()
    sem_uf = df_uf.groupby(["regiao", "estado", "semana"]).agg(**agg).reset_index()
    sem_regiao = (
        df_uf.groupby(["regiao", "semana"])
        .agg(
            casosNovos=("casosNovos", "sum"),
            obitosNovos=("obitosNovos", "sum"),
            data_ref=("data", "min"),
        )
        .reset_index()
    )

    # Totais por UF na última data
    tot_uf = (
        df_uf.sort_values("data")
        .groupby(["regiao", "estado"])
        .agg(
            casosAcumulado=("casosAcumulado", "last"),
            obitosAcumulado=("obitosAcumulado", "last"),
            populacao=("populacaoTCU2019", "last"),
        )
        .reset_index()
    )

    return {
        "sem_br": sem_br,
        "sem_uf": sem_uf,
        "sem_regiao": sem_regiao,
        "tot_uf": tot_uf,
        "df_mun": df_mun,
        "ultima_data": ultima_data,
    }


baixar_dados()
dados = carregar_dados()
sem_br = dados["sem_br"]
sem_uf = dados["sem_uf"]
sem_regiao = dados["sem_regiao"]
tot_uf = dados["tot_uf"]
df_mun = dados["df_mun"]
ultima_data = dados["ultima_data"]
estados = sorted(sem_uf["estado"].unique())

st.title("COVID-19 no Brasil - Painel Coronavírus")
st.markdown(
    "TP2 de Desenvolvimento Front-End com Python. Dados oficiais do portal "
    "[Coronavírus Brasil](https://covid.saude.gov.br/), consolidados pelo "
    "Ministério da Saúde a partir das 27 Secretarias Estaduais de Saúde "
    f"(base histórica até {ultima_data:%d/%m/%Y})."
)

c1, c2, c3 = st.columns(3)
c1.metric("Casos acumulados (Brasil)", f"{tot_uf['casosAcumulado'].sum():,.0f}".replace(",", "."))
c2.metric("Óbitos acumulados (Brasil)", f"{tot_uf['obitosAcumulado'].sum():,.0f}".replace(",", "."))
c3.metric("Semanas epidemiológicas na base", f"{sem_br.shape[0]}")

st.divider()

# =========================================================================
# Exercício 1: Importância da visualização de dados
# =========================================================================
st.header("Exercício 1: Importância da visualização de dados em uma pandemia")
st.markdown(
    """
Durante a pandemia, os dados chegavam como milhões de registros diários repassados por
27 secretarias estaduais. A visualização transforma esse volume em padrões legíveis:
curvas de crescimento, ondas, comparações regionais e concentrações geográficas.

Para gestores de saúde pública, ela orienta decisões urgentes. Curvas semanais mostram
se a epidemia acelera ou desacelera e sustentam a adoção ou o relaxamento de medidas
restritivas. Comparações entre estados guiam a alocação de leitos, equipes e vacinas.
Mapas revelam a interiorização da doença e os corredores de transmissão. E a relação
entre casos e óbitos, defasada em 2 a 4 semanas, permite antecipar a demanda hospitalar.

Para a população, gráficos claros cumprem papel de comunicação de risco: uma curva
subindo comunica a gravidade melhor que qualquer número isolado, incentiva a adesão às
medidas de proteção e dá transparência aos dados oficiais, o que ajuda a combater a
desinformação.
"""
)

st.divider()

# =========================================================================
# Exercício 2: Gráfico de barras com Streamlit (st.bar_chart)
# =========================================================================
st.header("Exercício 2: Casos novos por semana epidemiológica (st.bar_chart)")

uf_ex2 = st.selectbox("Estado", estados, index=estados.index("SP"), key="ex2_uf")
serie_ex2 = (
    sem_uf[sem_uf["estado"] == uf_ex2]
    .set_index("semana")[["casosNovos"]]
    .rename(columns={"casosNovos": "Casos novos"})
)
st.bar_chart(serie_ex2, x_label="Semana epidemiológica", y_label="Casos novos")

st.markdown(
    """
Estado escolhido: São Paulo, o mais populoso do país, com o maior número absoluto de
casos e onde a doença chegou primeiro (26/02/2020). O gráfico mostra as grandes ondas:
a inicial de 2020, a da variante Gama no primeiro semestre de 2021 e o pico da Ômicron
na virada de 2021 para 2022, o maior de toda a série, seguido de ondas menores até o
arrefecimento a partir de 2023. O seletor permite conferir o padrão em outras UFs.
"""
)

st.divider()

# =========================================================================
# Exercício 3: Gráfico de linha com Streamlit (st.line_chart)
# =========================================================================
st.header("Exercício 3: Óbitos acumulados no Brasil (st.line_chart)")

serie_ex3 = (
    sem_br.set_index("semana")[["obitosAcumulado"]]
    .rename(columns={"obitosAcumulado": "Óbitos acumulados"})
)
st.line_chart(serie_ex3, x_label="Semana epidemiológica", y_label="Óbitos acumulados")

st.markdown(
    """
Por ser acumulada, a curva nunca desce; o que importa é a inclinação. Trechos íngremes
indicam semanas com muitos óbitos e trechos quase horizontais indicam arrefecimento.
A subida mais forte ocorre no primeiro semestre de 2021, na onda da variante Gama,
quando o país passou de 20 mil óbitos semanais e acumulou quase metade do total. A
partir de meados de 2022, com a vacinação em massa, a curva se aproxima de um platô:
os casos continuaram ocorrendo, mas a letalidade caiu drasticamente.
"""
)

st.divider()

# =========================================================================
# Exercício 4: Gráfico de área com Streamlit (st.area_chart)
# =========================================================================
st.header("Exercício 4: Casos acumulados em três estados (st.area_chart)")

ufs_ex4 = st.multiselect(
    "Estados (escolha 3)", estados, default=["SP", "MG", "RJ"], key="ex4_ufs"
)
pivo_ex4 = (
    sem_uf[sem_uf["estado"].isin(ufs_ex4)]
    .pivot_table(index="semana", columns="estado", values="casosAcumulado")
)
st.area_chart(
    pivo_ex4, stack=False, x_label="Semana epidemiológica", y_label="Casos acumulados"
)

st.markdown(
    """
Estados comparados: SP, MG e RJ, os três mais populosos do Sudeste (as áreas não estão
empilhadas, para permitir comparação direta). SP lidera em nível absoluto durante toda a
série e termina com cerca de 7 milhões de casos, coerente com sua população. MG
ultrapassa 4 milhões, com um salto na onda Ômicron (início de 2022), quando sua curva se
descola da do RJ. O RJ acumula menos casos que MG apesar da letalidade historicamente
alta, o que sugere subnotificação de casos leves. Os degraus simultâneos nas três curvas
marcam as mesmas ondas nacionais, sinal de que a epidemia avançou de forma sincronizada
entre estados vizinhos.
"""
)

st.divider()

# =========================================================================
# Exercício 5: Mapa com Streamlit (st.map)
# =========================================================================
st.header("Exercício 5: Casos acumulados por município (st.map)")

uf_ex5 = st.selectbox("Estado", estados, index=estados.index("SP"), key="ex5_uf")
mapa_ex5 = df_mun[df_mun["estado"] == uf_ex5].copy()
# Raio em metros proporcional à raiz dos casos, para a capital não dominar a escala
mapa_ex5["tamanho"] = np.sqrt(mapa_ex5["casosAcumulado"].clip(lower=0)) * 40
st.map(mapa_ex5, latitude="latitude", longitude="longitude", size="tamanho")

st.markdown(
    f"""
Cada ponto é um município de {uf_ex5}, com raio proporcional à raiz quadrada dos casos
acumulados até {ultima_data:%d/%m/%Y}. O mapa mostra o que tabelas escondem: a
concentração na capital e na região metropolitana, os corredores de transmissão ao
longo das rodovias e a interiorização da doença, que partiu dos grandes centros. Para a
gestão, isso orienta decisões regionais, como onde reforçar leitos, onde montar
barreiras sanitárias e quais regiões de saúde precisam de apoio.
"""
)

st.divider()

# =========================================================================
# Exercício 6: Visualização com Matplotlib (casos novos x óbitos novos por UF)
# =========================================================================
st.header("Exercício 6: Casos e óbitos novos por estado na última semana (Matplotlib)")

# A última semana do arquivo pode vir sem notificações; usa-se a mais recente com casos
casos_por_semana = sem_uf.groupby("semana")["casosNovos"].sum()
ultima_semana = casos_por_semana[casos_por_semana > 0].index.max()
ex6 = (
    sem_uf[sem_uf["semana"] == ultima_semana]
    .sort_values("casosNovos", ascending=False)
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(ex6))
largura = 0.4
barras_casos = ax.bar(
    x - largura / 2, ex6["casosNovos"], largura, color="#1f77b4", label="Casos novos"
)
ax2 = ax.twinx()
barras_obitos = ax2.bar(
    x + largura / 2, ex6["obitosNovos"], largura, color="#d62728", label="Óbitos novos"
)
ax.set_xticks(x)
ax.set_xticklabels(ex6["estado"], fontsize=8)
ax.set_xlabel("Estado")
ax.set_ylabel("Casos novos", color="#1f77b4")
ax2.set_ylabel("Óbitos novos", color="#d62728")
ax.set_title(f"Casos novos x óbitos novos por UF - semana epidemiológica {ultima_semana}")
ax.legend(handles=[barras_casos, barras_obitos], loc="upper right")
fig.tight_layout()
st.pyplot(fig)

st.markdown(
    f"""
Comparação entre casos novos (azul, eixo esquerdo) e óbitos novos (vermelho, eixo
direito) na semana {ultima_semana}, a mais recente com notificações na base. Os eixos
têm escalas distintas porque as grandezas diferem em ordem de magnitude. Em geral,
estados com mais casos registram mais óbitos, mas a relação não é proporcional: os
óbitos de uma semana refletem infecções de 2 a 4 semanas antes, e a letalidade aparente
varia com a testagem, a estrutura hospitalar e o perfil etário de cada estado. No
período recente, com a população vacinada, a razão óbitos/casos é muito menor do que em
2020 e 2021, evidenciando o desacoplamento entre transmissão e mortalidade.
"""
)

st.divider()

# =========================================================================
# Exercício 7: Boxplot com Seaborn (casos novos semanais por região)
# =========================================================================
st.header("Exercício 7: Distribuição de casos novos semanais por região (Seaborn)")

regioes_ex7 = ["Norte", "Nordeste", "Sudeste"]
ex7 = sem_regiao[sem_regiao["regiao"].isin(regioes_ex7)]

fig7, ax7 = plt.subplots(figsize=(9, 5))
sns.boxplot(data=ex7, x="regiao", y="casosNovos", hue="regiao", order=regioes_ex7, ax=ax7)
ax7.set_xlabel("Região")
ax7.set_ylabel("Casos novos por semana epidemiológica")
ax7.set_title("Distribuição dos casos novos semanais (2020-2025)")
fig7.tight_layout()
st.pyplot(fig7)

st.markdown(
    """
Cada caixa resume os casos novos semanais de 2020 a 2025. O Sudeste tem mediana,
intervalo interquartil e outliers bem maiores: concentra cerca de 40% da população do
país, e suas semanas típicas superam os picos das outras regiões. O Nordeste fica em
posição intermediária, alternando períodos de baixa transmissão com ondas intensas. O
Norte tem os menores valores, coerentes com sua população menor e com a baixa
capacidade de testagem da região, que comprime a distribuição. Nas três, a distribuição
é assimétrica à direita: poucas semanas de onda concentram volumes extremos, que
aparecem como outliers acima das caixas.
"""
)

st.divider()

# =========================================================================
# Exercício 8: Gráfico de área com Altair (casos novos semanais em uma região)
# =========================================================================
st.header("Exercício 8: Casos novos por semana em uma região (Altair)")

regiao_ex8 = st.selectbox("Região", REGIOES, index=REGIOES.index("Sudeste"), key="ex8_regiao")
ex8 = sem_regiao[sem_regiao["regiao"] == regiao_ex8]

grafico_ex8 = (
    alt.Chart(ex8)
    .mark_area(opacity=0.7, line=True)
    .encode(
        x=alt.X("data_ref:T", title="Semana epidemiológica (início da semana)"),
        y=alt.Y("casosNovos:Q", title="Casos novos"),
        tooltip=[
            alt.Tooltip("semana:N", title="Semana"),
            alt.Tooltip("casosNovos:Q", title="Casos novos", format=","),
        ],
    )
    .properties(height=350)
)
st.altair_chart(grafico_ex8, width="stretch")

st.markdown(
    """
Região escolhida: Sudeste, a mais populosa e a que concentra o maior volume de casos,
o que torna suas ondas representativas da dinâmica nacional. Observam-se duas ondas em
2020 e início de 2021 (cepa original e Gama), o pico extremo da Ômicron em janeiro de
2022, várias vezes maior que os anteriores, ondas de rebote menores ao longo de 2022 e
queda sustentada a partir de 2023, com pequenos repiques sazonais. A área preenchida dá
noção do volume de cada onda e o tooltip permite consultar o valor de cada semana.
"""
)

st.divider()

# =========================================================================
# Exercício 9: Heatmap com Altair (correlação entre indicadores de um estado)
# =========================================================================
st.header("Exercício 9: Correlação entre indicadores semanais (Altair)")

uf_ex9 = st.selectbox("Estado", estados, index=estados.index("SP"), key="ex9_uf")
indicadores = {
    "casosNovos": "Casos novos",
    "obitosNovos": "Óbitos novos",
    "casosAcumulado": "Casos acumulados",
    "obitosAcumulado": "Óbitos acumulados",
}
ex9 = (
    sem_uf[sem_uf["estado"] == uf_ex9][list(indicadores)]
    .rename(columns=indicadores)
    .corr()
    .reset_index(names="Indicador A")
    .melt(id_vars="Indicador A", var_name="Indicador B", value_name="correlacao")
)

base_ex9 = alt.Chart(ex9).encode(
    x=alt.X("Indicador A:N", title=None),
    y=alt.Y("Indicador B:N", title=None),
)
heatmap_ex9 = base_ex9.mark_rect().encode(
    color=alt.Color(
        "correlacao:Q",
        title="Correlação",
        scale=alt.Scale(domain=[-1, 1], scheme="redblue", reverse=True),
    ),
    tooltip=[
        "Indicador A", "Indicador B",
        alt.Tooltip("correlacao:Q", title="Correlação", format=".2f"),
    ],
)
rotulos_ex9 = base_ex9.mark_text(fontWeight="bold").encode(
    text=alt.Text("correlacao:Q", format=".2f"),
    color=alt.condition(
        "abs(datum.correlacao) > 0.6", alt.value("white"), alt.value("black")
    ),
)
st.altair_chart((heatmap_ex9 + rotulos_ex9).properties(height=350), width="stretch")

st.markdown(
    """
A base pública do portal não inclui ocupação de leitos, então o heatmap correlaciona os
quatro indicadores epidemiológicos disponíveis, agregados por semana, como o enunciado
prevê ("caso os dados estejam disponíveis").

A correlação mais forte é entre casos e óbitos acumulados (0,98), esperada por
construção: ambas as séries crescem com o tempo. Casos novos e óbitos novos têm
correlação alta (0,88 em SP), mas não perfeita, pela defasagem de 2 a 4 semanas entre
infecção e óbito e pela vacinação, que reduziu muito os óbitos da onda Ômicron. Os
indicadores novos e os acumulados têm correlação negativa moderada (perto de -0,5):
quando o acumulado já é alto, a pandemia está em fase tardia e o fluxo semanal é baixo.
Se houvesse dados de leitos, o esperado seria correlação alta com casos novos, com 1 a
2 semanas de defasagem, relação que permite antecipar a pressão hospitalar.
"""
)

st.divider()

# =========================================================================
# Exercício 10: Gráfico de pizza com Plotly (casos acumulados por região)
# =========================================================================
st.header("Exercício 10: Distribuição dos casos acumulados por região (Plotly)")

ex10 = tot_uf.groupby("regiao", as_index=False)["casosAcumulado"].sum()
fig10 = px.pie(
    ex10,
    names="regiao",
    values="casosAcumulado",
    category_orders={"regiao": REGIOES},
    hole=0.35,
)
fig10.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig10, width="stretch")

st.markdown(
    """
O Sudeste concentra cerca de 40% dos casos acumulados do país, seguido por Sul (21%),
Nordeste (19%), Centro-Oeste (12%) e Norte (8%). A distribuição acompanha em parte a
população, mas não exatamente: o Sul tem 14% da população e 21% dos casos, a maior
incidência per capita do país, o que reflete também maior testagem e notificação. Norte
e Nordeste aparecem abaixo do seu peso populacional, o que a literatura associa à menor
capacidade de testagem, e não necessariamente à menor circulação do vírus. Para
comparar risco entre regiões, o ideal é complementar a pizza com taxas por 100 mil
habitantes.
"""
)

st.divider()

# =========================================================================
# Exercício 11: Subplots com Plotly (casos e óbitos novos em duas regiões)
# =========================================================================
st.header("Exercício 11: Casos e óbitos novos por semana em duas regiões (Plotly subplots)")

col_a, col_b = st.columns(2)
regiao_a = col_a.selectbox("Região A", REGIOES, index=REGIOES.index("Sudeste"), key="ex11_a")
regiao_b = col_b.selectbox("Região B", REGIOES, index=REGIOES.index("Nordeste"), key="ex11_b")

fig11 = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    subplot_titles=(
        f"Casos novos - {regiao_a}", f"Casos novos - {regiao_b}",
        f"Óbitos novos - {regiao_a}", f"Óbitos novos - {regiao_b}",
    ),
    vertical_spacing=0.12,
)
for col, regiao, cor_casos, cor_obitos in [
    (1, regiao_a, "#1f77b4", "#d62728"),
    (2, regiao_b, "#17becf", "#e377c2"),
]:
    dados_regiao = sem_regiao[sem_regiao["regiao"] == regiao]
    fig11.add_trace(
        go.Bar(x=dados_regiao["data_ref"], y=dados_regiao["casosNovos"],
               marker_color=cor_casos, name=f"Casos - {regiao}"),
        row=1, col=col,
    )
    fig11.add_trace(
        go.Bar(x=dados_regiao["data_ref"], y=dados_regiao["obitosNovos"],
               marker_color=cor_obitos, name=f"Óbitos - {regiao}"),
        row=2, col=col,
    )
fig11.update_layout(height=600, showlegend=False, bargap=0)
st.plotly_chart(fig11, width="stretch")

st.markdown(
    """
Os painéis comparam Sudeste e Nordeste: casos novos na linha de cima, óbitos novos na
de baixo. O Sudeste opera em outra ordem de grandeza, com pico Ômicron acima de 500 mil
casos semanais, mais que o dobro do pico nordestino. As grandes ondas atingem as duas
regiões quase ao mesmo tempo, mas o Nordeste teve uma onda própria em meados de 2020,
quando foi um dos epicentros nacionais, e um rebote proporcionalmente mais forte em
meados de 2022. Nas duas regiões os óbitos acompanham os casos em 2020 e 2021; na onda
Ômicron a linha de baixo sobe muito menos que a de cima, efeito da vacinação. O eixo x
compartilhado facilita essa leitura de sincronia entre regiões e entre casos e óbitos.
"""
)

st.divider()

# =========================================================================
# Exercício 12: Mapa interativo com PyDeck (casos por município em uma região)
# =========================================================================
st.header("Exercício 12: Mapa 3D de casos por município em uma região (PyDeck)")

regiao_ex12 = st.selectbox(
    "Região", REGIOES, index=REGIOES.index("Sudeste"), key="ex12_regiao"
)
ex12 = df_mun[df_mun["regiao"] == regiao_ex12].copy()
ex12 = ex12[ex12["populacaoTCU2019"] > 0]
# Incidência por 100 mil habitantes: casos ajustados pelo tamanho da população
ex12["incidencia"] = ex12["casosAcumulado"] / ex12["populacaoTCU2019"] * 1e5
# Cor do amarelo (incidência baixa) ao vermelho (alta), limitada ao percentil 95
teto = ex12["incidencia"].quantile(0.95)
intensidade = (ex12["incidencia"] / teto).clip(0, 1)
ex12["cor_r"] = 255
ex12["cor_g"] = (200 * (1 - intensidade)).astype(int)
ex12["cor_b"] = 40
ex12["casos_fmt"] = ex12["casosAcumulado"].map(lambda v: f"{v:,.0f}".replace(",", "."))
ex12["incidencia_fmt"] = ex12["incidencia"].map(lambda v: f"{v:,.0f}".replace(",", "."))

camada = pdk.Layer(
    "ColumnLayer",
    data=ex12[
        ["municipio", "estado", "latitude", "longitude",
         "casosAcumulado", "cor_r", "cor_g", "cor_b", "casos_fmt", "incidencia_fmt"]
    ],
    get_position=["longitude", "latitude"],
    get_elevation="casosAcumulado",
    elevation_scale=1.5,
    radius=4000,
    get_fill_color=["cor_r", "cor_g", "cor_b", 180],
    pickable=True,
    auto_highlight=True,
)
visao = pdk.ViewState(
    latitude=float(ex12["latitude"].mean()),
    longitude=float(ex12["longitude"].mean()),
    zoom=5,
    pitch=45,
)
st.pydeck_chart(
    pdk.Deck(
        layers=[camada],
        initial_view_state=visao,
        tooltip={
            "text": "{municipio} - {estado}\n"
            "Casos acumulados: {casos_fmt}\n"
            "Incidência: {incidencia_fmt} casos/100 mil hab."
        },
    )
)

st.markdown(
    f"""
Cada coluna 3D é um município da região {regiao_ex12} (dados de
{ultima_data:%d/%m/%Y}): a altura representa os casos acumulados e a cor, a incidência
por 100 mil habitantes, ou seja, os casos ajustados pelo tamanho da população (amarelo
para incidência baixa, vermelho para alta).

A COVID-19 se transmite por contato próximo, então áreas densas, como capitais e
regiões metropolitanas, oferecem mais oportunidades de contágio (transporte lotado,
moradias adensadas, grandes fluxos pendulares) e concentram as colunas mais altas.
Essas áreas também funcionaram como portas de entrada: o vírus chegou pelos aeroportos
das grandes cidades e se interiorizou seguindo os fluxos de pessoas. O ajuste pela
população mostra que volume não é o mesmo que risco: municípios pequenos aparecem
baixos, mas muitas vezes vermelhos, com incidência per capita igual ou maior que a das
capitais. Por isso a análise geográfica combina as duas medidas: o absoluto para
dimensionar recursos e o per capita para medir risco.
"""
)

st.divider()
st.text("Feito com Streamlit para o TP2 de Desenvolvimento Front-End com Python.")
