import time

import pandas as pd
import plotly.express as px
import pulp
import streamlit as st

st.set_page_config(page_title="CapitalGestión · Selección de proyectos", layout="wide")

# ----------------------------------------------------------------------
# 1. DATOS POR DEFECTO
# ----------------------------------------------------------------------
DEFAULT_DATA = pd.DataFrame(
    {
        "Proyecto": [f"P{i}" for i in range(1, 13)],
        "Nombre": [
            "Automatización", "Analítica de datos", "Energía solar", "Infraestructura",
            "Mantenimiento predictivo", "Optimización logística", "Reutilización de agua",
            "Robotización", "Comercio electrónico", "Capacitación tecnológica",
            "Expansión comercial", "Gestión energética",
        ],
        "Inversión": [80, 55, 70, 45, 60, 40, 50, 75, 65, 30, 85, 35],
        "Beneficio": [95, 75, 80, 60, 85, 65, 70, 90, 88, 42, 105, 48],
        "Horas": [500, 350, 250, 400, 300, 350, 200, 450, 380, 220, 520, 180],
    }
)

DEFAULTS = {"budget": 300, "hours": 1800, "max_p": 6}

POLICIES = {
    "pol4": "4. Si se selecciona P8, también P1  (P8 − P1 ≤ 0)",
    "pol5": "5. P3 y P7 no simultáneos  (P3 + P7 ≤ 1)",
    "pol6": "6. Al menos uno entre P3 o P7  (P3 + P7 ≥ 1)",
    "pol7": "7. Si se selecciona P9, también P2  (P9 − P2 ≤ 0)",
    "pol8": "8. P4 y P11 incompatibles  (P4 + P11 ≤ 1)",
    "pol9": "9. Al menos uno entre P1, P2 o P5  (P1 + P2 + P5 ≥ 1)",
}


# ----------------------------------------------------------------------
# 2. MODELO (PuLP): variables, objetivo, restricciones y solución
# ----------------------------------------------------------------------
def build_model(df, budget, max_hours, max_projects, active):
    """Construye el modelo PuLP a partir de los datos y parámetros recibidos."""
    ids = list(df["Proyecto"])
    inv = dict(zip(ids, df["Inversión"]))
    ben = dict(zip(ids, df["Beneficio"]))
    hrs = dict(zip(ids, df["Horas"]))

    prob = pulp.LpProblem("CapitalGestion_Seleccion_Proyectos", pulp.LpMaximize)

    # Variables de decisión: x_i ∈ {0,1}
    x = {i: pulp.LpVariable(i, cat=pulp.LpBinary) for i in ids}

    # Función objetivo
    prob += pulp.lpSum(ben[i] * x[i] for i in ids), "Beneficio_total"

    # Restricciones de recursos
    prob += pulp.lpSum(inv[i] * x[i] for i in ids) <= budget, "R1_Presupuesto"
    prob += pulp.lpSum(hrs[i] * x[i] for i in ids) <= max_hours, "R2_Horas"
    prob += pulp.lpSum(x.values()) <= max_projects, "R3_Max_proyectos"

    # Políticas estratégicas (activables)
    if active.get("pol4"):
        prob += x["P8"] - x["P1"] <= 0, "Pol4_P8_requiere_P1"
    if active.get("pol5"):
        prob += x["P3"] + x["P7"] <= 1, "Pol5_P3_P7_no_simultaneos"
    if active.get("pol6"):
        prob += x["P3"] + x["P7"] >= 1, "Pol6_al_menos_P3_o_P7"
    if active.get("pol7"):
        prob += x["P9"] - x["P2"] <= 0, "Pol7_P9_requiere_P2"
    if active.get("pol8"):
        prob += x["P4"] + x["P11"] <= 1, "Pol8_P4_P11_incompatibles"
    if active.get("pol9"):
        prob += x["P1"] + x["P2"] + x["P5"] >= 1, "Pol9_al_menos_P1_P2_P5"

    return prob, x


def check_solution(df, selected, budget, max_hours, max_projects, active):
    """Verificación independiente: recalcula cada restricción con la solución del solver."""
    s = set(selected)
    v = lambda p: 1 if p in s else 0
    inv = dict(zip(df["Proyecto"], df["Inversión"]))
    hrs = dict(zip(df["Proyecto"], df["Horas"]))
    rows = [
        ("R1 Presupuesto", sum(inv[p] for p in s), "≤", budget),
        ("R2 Horas de personal", sum(hrs[p] for p in s), "≤", max_hours),
        ("R3 Máx. proyectos", len(s), "≤", max_projects),
    ]
    if active.get("pol4"):
        rows.append(("Pol 4: P8 → P1", v("P8") - v("P1"), "≤", 0))
    if active.get("pol5"):
        rows.append(("Pol 5: P3 + P7", v("P3") + v("P7"), "≤", 1))
    if active.get("pol6"):
        rows.append(("Pol 6: P3 + P7", v("P3") + v("P7"), "≥", 1))
    if active.get("pol7"):
        rows.append(("Pol 7: P9 → P2", v("P9") - v("P2"), "≤", 0))
    if active.get("pol8"):
        rows.append(("Pol 8: P4 + P11", v("P4") + v("P11"), "≤", 1))
    if active.get("pol9"):
        rows.append(("Pol 9: P1 + P2 + P5", v("P1") + v("P2") + v("P5"), "≥", 1))

    out = []
    for name, lhs, op, rhs in rows:
        ok = lhs <= rhs + 1e-9 if op == "≤" else lhs >= rhs - 1e-9
        out.append(
            {"Restricción": name, "Valor (LHS)": lhs, "Sentido": op, "Límite (RHS)": rhs,
             "Holgura": abs(rhs - lhs), "Cumple": "✅" if ok else "❌"}
        )
    return pd.DataFrame(out)


def solve(df, budget, max_hours, max_projects, active):
    """Resuelve con CBC y devuelve un diccionario de resultados.
    Si el estado no es 'Optimal' NO se devuelve ninguna solución."""
    prob, x = build_model(df, budget, max_hours, max_projects, active)
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    elapsed = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]

    res = {
        "status": status,
        "solver": "CBC (PuLP)",
        "time": elapsed,
        "lp_text": str(prob),
        "n_vars": len(prob.variables()),
        "n_cons": len(prob.constraints),
        "selected": None,
        "objective": None,
        "check": None,
    }
    if status != "Optimal":
        return res

    selected = [i for i, var in x.items() if (var.value() or 0) > 0.5]
    check = check_solution(df, selected, budget, max_hours, max_projects, active)
    if (check["Cumple"] == "❌").any():  # salvaguarda: nunca mostrar soluciones inválidas
        res["status"] = "Solución inválida (descartada)"
        return res

    res["selected"] = selected
    res["objective"] = pulp.value(prob.objective)
    res["check"] = check
    return res


# ----------------------------------------------------------------------
# 3. INTERFAZ
# ----------------------------------------------------------------------
if "ver" not in st.session_state:
    st.session_state.ver = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None
ver = st.session_state.ver

st.title("📊 CapitalGestión S.A.S. · Selección óptima de proyectos")
st.markdown(
    "**Contexto.** CapitalGestión debe seleccionar proyectos estratégicos de tecnología, "
    "infraestructura, sostenibilidad y crecimiento. Cada proyecto consume inversión y horas de "
    "personal y genera un beneficio esperado. El modelo de **programación lineal binaria** "
    "maximiza el beneficio total respetando los recursos disponibles y las políticas de selección. "
    "*Valores monetarios en millones de COP.*"
)

# ---- Barra lateral: parámetros y políticas
with st.sidebar:
    st.header("⚙️ Parámetros")
    budget = st.number_input("Presupuesto máximo (millones COP)", min_value=0.0,
                             value=float(DEFAULTS["budget"]), step=10.0, key=f"b{ver}")
    max_hours = st.number_input("Horas de personal disponibles", min_value=0.0,
                                value=float(DEFAULTS["hours"]), step=50.0, key=f"h{ver}")
    max_projects = st.number_input("Máximo de proyectos", min_value=0, max_value=12,
                                   value=DEFAULTS["max_p"], step=1, key=f"m{ver}")

    st.header("📜 Políticas estratégicas")
    active = {k: st.checkbox(label, value=True, key=f"{k}_{ver}") for k, label in POLICIES.items()}

    st.header("🧪 Escenario")
    scen_name = st.text_input("Nombre del escenario", value=f"Escenario {len(st.session_state.history) + 1}",
                              key=f"n{ver}_{len(st.session_state.history)}")
    if st.button("↩️ Restaurar valores originales"):
        st.session_state.ver += 1
        st.rerun()

# ---- Datos editables
st.subheader("1️⃣ Datos de los proyectos (editables)")
edited = st.data_editor(
    DEFAULT_DATA,
    key=f"editor{ver}",
    hide_index=True,
    width="stretch",
    num_rows="fixed",
    disabled=["Proyecto", "Nombre"],
    column_config={
        "Inversión": st.column_config.NumberColumn("Inversión (M COP)", min_value=0.0),
        "Beneficio": st.column_config.NumberColumn("Beneficio (M COP)", min_value=0.0),
        "Horas": st.column_config.NumberColumn("Horas de personal", min_value=0.0),
    },
)

# ---- Ejecución
st.subheader("2️⃣ Optimización")
if st.button("🚀 Ejecutar optimización", type="primary"):
    data = edited.copy()
    if data[["Inversión", "Beneficio", "Horas"]].isnull().any().any():
        st.error("Hay celdas vacías en la tabla de datos. Complétalas antes de ejecutar.")
    else:
        with st.spinner("Resolviendo con CBC..."):
            res = solve(data, budget, max_hours, int(max_projects), active)
        res.update(
            name=scen_name, data=data, budget=budget, hours=max_hours, max_p=int(max_projects),
            n_pol=sum(active.values()),
        )
        st.session_state.last = res
        st.session_state.history.append(res)
        st.rerun()

# ---- Resultados
res = st.session_state.last
if res is None:
    st.info("Edita los datos si lo deseas y pulsa **Ejecutar optimización**.")
else:
    st.subheader(f"3️⃣ Resultados · {res['name']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estado del modelo", res["status"])
    c2.metric("Solver", res["solver"])
    c3.metric("Tiempo de resolución", f"{res['time'] * 1000:.0f} ms")

    if res["selected"] is None:
        if res["status"] == "Infeasible":
            st.error(
                "❌ **Modelo infactible**: no existe ningún portafolio que cumpla simultáneamente todos los "
                "recursos y políticas activas. No se muestra solución. Relaja el presupuesto, las horas, "
                "el máximo de proyectos o desactiva alguna política."
            )
        else:
            st.error(f"❌ El solver no encontró una solución óptima (estado: **{res['status']}**). "
                     "No se muestra ninguna solución.")
    else:
        d = res["data"].set_index("Proyecto")
        sel = res["selected"]
        inv_used = d.loc[sel, "Inversión"].sum()
        hrs_used = d.loc[sel, "Horas"].sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Beneficio total", f"{res['objective']:,.1f} M COP")
        k2.metric("Inversión usada", f"{inv_used:,.0f} / {res['budget']:,.0f}")
        k3.metric("Horas usadas", f"{hrs_used:,.0f} / {res['hours']:,.0f}")
        k4.metric("Proyectos", f"{len(sel)} / {res['max_p']}")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["✅ Proyectos seleccionados", "📈 Uso de recursos", "🧪 Comparación de escenarios", "🔍 Modelo y verificación"]
        )

        with tab1:
            table = d.loc[sel].reset_index()
            table["Beneficio/Inversión"] = (table["Beneficio"] / table["Inversión"]).round(2)
            st.dataframe(table, hide_index=True, width="stretch")
            fig = px.bar(table, x="Proyecto", y="Beneficio", color="Inversión", text="Beneficio",
                         title="Beneficio de cada proyecto seleccionado (M COP)")
            st.plotly_chart(fig, width="stretch")
            full = res["data"].copy()
            full["Seleccionado"] = full["Proyecto"].isin(sel).map({True: "Sí", False: "No"})
            with st.expander("Ver todos los proyectos (seleccionados y no seleccionados)"):
                st.dataframe(full, hide_index=True, width="stretch")

        with tab2:
            usage = pd.DataFrame(
                {
                    "Recurso": ["Presupuesto", "Horas de personal", "Nº de proyectos"],
                    "Usado": [inv_used, hrs_used, len(sel)],
                    "Disponible": [res["budget"], res["hours"], res["max_p"]],
                }
            )
            usage["% de uso"] = (100 * usage["Usado"] / usage["Disponible"].replace(0, float("nan"))).round(1)
            st.dataframe(usage, hide_index=True, width="stretch")
            fig = px.bar(usage, x="Recurso", y="% de uso", text="% de uso",
                         title="Porcentaje de uso de cada recurso")
            fig.update_yaxes(range=[0, 110])
            st.plotly_chart(fig, width="stretch")

        with tab3:
            hist = st.session_state.history
            rows = []
            for h in hist:
                ok = h["selected"] is not None
                rows.append(
                    {
                        "Escenario": h["name"],
                        "Estado": h["status"],
                        "Beneficio total": round(h["objective"], 1) if ok else None,
                        "Inversión": float(h["data"].set_index("Proyecto").loc[h["selected"], "Inversión"].sum()) if ok else None,
                        "Horas": float(h["data"].set_index("Proyecto").loc[h["selected"], "Horas"].sum()) if ok else None,
                        "Proyectos": ", ".join(h["selected"]) if ok else "—",
                        "Presupuesto": h["budget"],
                        "Horas disp.": h["hours"],
                        "Máx. proy.": h["max_p"],
                        "Políticas activas": h["n_pol"],
                    }
                )
            comp = pd.DataFrame(rows)
            st.dataframe(comp, hide_index=True, width="stretch")
            ok_df = comp.dropna(subset=["Beneficio total"])
            if len(ok_df) >= 1:
                fig = px.bar(ok_df, x="Escenario", y="Beneficio total", text="Beneficio total",
                             title="Beneficio total por escenario")
                st.plotly_chart(fig, width="stretch")
            st.caption("Cambia parámetros, datos o políticas, ponle otro nombre al escenario y vuelve a ejecutar.")
            if st.button("🗑️ Borrar historial de escenarios"):
                st.session_state.history = []
                st.session_state.last = None
                st.rerun()

        with tab4:
            st.markdown("**Verificación independiente** (se recalcula cada restricción con la solución devuelta por el solver):")
            st.dataframe(res["check"], hide_index=True, width="stretch")
            st.markdown(f"Variables: **{res['n_vars']}** · Restricciones: **{res['n_cons']}**")
            st.markdown("**Modelo enviado al solver (formato PuLP/LP):**")
            st.code(res["lp_text"], language="text")
