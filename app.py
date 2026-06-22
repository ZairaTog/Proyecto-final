import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Dashboard Aroma del Sur",
    page_icon="☕",
    layout="wide"
)

# ============================================================
# CARGAR DATOS
# ============================================================
ventas = pd.read_csv("ventas_limpio.csv")
productos = pd.read_csv("productos_limpio.csv")
sucursales = pd.read_csv("sucursales_limpio.csv")
clientes = pd.read_csv("clientes_limpio.csv")

st.title("☕ Dashboard - Aroma del Sur")
st.caption("Por Zaira Karim Torres Goiz · ID 183800 · LAD3012 · UDLAP Verano I 2026")
st.markdown("---")

# ============================================================
# FILTROS (sidebar)
# ============================================================
st.sidebar.header("🔎 Filtros")
sucursal_filtro = st.sidebar.multiselect(
    "Sucursal",
    options=sorted(sucursales["nombre"].unique()),
    default=sorted(sucursales["nombre"].unique())
)
categoria_filtro = st.sidebar.multiselect(
    "Categoría",
    options=sorted(productos["categoria"].unique()),
    default=sorted(productos["categoria"].unique())
)

# Aplicar filtros
ventas_f = ventas.merge(productos, on="producto_id").merge(sucursales, on="sucursal_id")
ventas_f = ventas_f[
    ventas_f["nombre"].isin(sucursal_filtro) &
    ventas_f["categoria"].isin(categoria_filtro)
]

if len(ventas_f) == 0:
    st.warning("No hay datos con esos filtros. Selecciona al menos una opción en cada filtro.")
    st.stop()

# ============================================================
# KPIs (3 métricas grandes)
# ============================================================
col1, col2, col3 = st.columns(3)
col1.metric("Ventas totales", f"${ventas_f['total'].sum():,.0f}")
col2.metric("Ganancia total", f"${(ventas_f['total'] - ventas_f['costo']).sum():,.0f}")
margen = 100 * (ventas_f['total'] - ventas_f['costo']).sum() / ventas_f['total'].sum()
col3.metric("Margen %", f"{margen:.1f}%")

st.markdown("---")

# ============================================================
# GRÁFICOS
# ============================================================

# Ventas por sucursal
st.subheader("Ventas por sucursal")
ventas_sucursal = ventas_f.groupby("nombre")["total"].sum().reset_index()
fig1 = px.bar(ventas_sucursal, x="nombre", y="total", color="nombre")
st.plotly_chart(fig1, use_container_width=True)

# Ventas por categoría
st.subheader("Ventas por categoría")
ventas_categoria = ventas_f.groupby("categoria")["total"].sum().reset_index()
fig2 = px.bar(ventas_categoria, x="categoria", y="total", color="categoria")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================================
# TABLA: Top 10 clientes
# ============================================================
st.subheader("Top 10 clientes por gasto")
top_clientes = clientes.sort_values("total_gastado", ascending=False).head(10)
st.dataframe(top_clientes[["cliente_id","total_gastado","sucursal_favorita"]], use_container_width=True)

st.markdown("---")

# ============================================================
# CLUSTERING DE CLIENTES
# ============================================================
st.subheader("Segmentación de clientes (K-Means)")
X = clientes[["edad","total_gastado"]].dropna()
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(X)
clientes["cluster"] = kmeans.labels_

fig3 = px.scatter(clientes, x="edad", y="total_gastado", color="cluster",
                  title="Clusters de clientes por edad y gasto")
st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# INSIGHT FINAL
# ============================================================
st.subheader("💡 Insight de negocio")
st.info("""
Cluster 0: Clientes leales con gasto alto.
Cluster 1: Clientes de edad media con gasto moderado.
Cluster 2: Clientes ocasionales con gasto bajo.
""")
