import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import find_peaks, savgol_filter
import io
import os

# Fix : pandas 2.0+ utilise LargeUtf8 (Arrow) incompatible avec le frontend Streamlit
try:
    pd.options.future.infer_string = False
except AttributeError:
    pass

st.set_page_config(page_title="Analyse FTIR", layout="wide", page_icon="🔬")


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes string en object pour éviter l'erreur LargeUtf8 d'Arrow."""
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)
    return df

# Constantes
TOLERANCE = 0.5
X_1047, X_1022 = 1047, 1022
K_INVERSE = 262.5
V0_E = 3650
V0_D = 3600


# ── Fonctions de traitement ──────────────────────────────────────────────────

def calcul_deriv2(x, y, window, polyorder):
    y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)
    d2y = np.full_like(y_smooth, np.nan)
    for j in range(1, len(y_smooth) - 1):
        dx = x[j + 1] - x[j]
        if dx != 0:
            d2y[j] = (y_smooth[j + 1] - 2 * y_smooth[j] + y_smooth[j - 1]) / dx ** 2
    valid = ~np.isnan(d2y)
    if valid.sum() > window:
        d2y[valid] = savgol_filter(d2y[valid], window_length=window, polyorder=polyorder)
    return d2y


def calcul_E_D_RD(df, nu):
    y1047 = df[(df["X"] >= X_1047 - TOLERANCE) & (df["X"] <= X_1047 + TOLERANCE)]["Y"].max()
    y1022 = df[(df["X"] >= X_1022 - TOLERANCE) & (df["X"] <= X_1022 + TOLERANCE)]["Y"].max()
    RD = round(y1047 / y1022, 4) if y1022 > 0 else "N/A"
    E = round(K_INVERSE * (V0_E - nu) / V0_E, 4)
    D = round(2.84 - (V0_D - nu) / 4430, 5)
    return E, D, RD


def detecter_pics(df, zones_pics, window, polyorder):
    x, y = df["X"].values, df["Y"].values
    y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)
    d2y = calcul_deriv2(x, y_smooth, window, polyorder)
    df = df.copy()
    df["d2Y"] = d2y
    pics = []
    for p_start, p_end in zones_pics:
        mask = (df["X"] >= min(p_start, p_end)) & (df["X"] <= max(p_start, p_end))
        x_z = df.loc[mask, "X"].values
        y_z = df.loc[mask, "d2Y"].values
        if len(y_z) == 0:
            continue
        y_inv = -y_z
        threshold = np.max(y_inv) * 0.1
        peaks, _ = find_peaks(y_inv, height=threshold)
        for p in peaks:
            xv = round(x_z[p], 1)
            if xv not in pics:
                pics.append(xv)
    return sorted(pics)


def load_csv(file, debut, fin, window, polyorder):
    df = pd.read_csv(file).iloc[debut:fin].rename(
        columns={"TITLE": "X", "Unnamed: 1": "Y"}
    )
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Y"] = savgol_filter(df["Y"].values, window_length=window, polyorder=polyorder)
    df["d2Y"] = calcul_deriv2(df["X"].values, df["Y"].values, window, polyorder)
    return df


def parse_zones(s):
    zones = []
    for pair in s.split(","):
        try:
            parts = pair.strip().split("-")
            a, b = int(parts[0]), int(parts[1])
            zones.append((min(a, b), max(a, b)))
        except Exception:
            continue
    return zones


def parse_decalages(s, n):
    vals = s.split(",")
    dec = []
    for v in vals:
        try:
            dec.append(float(v.strip()))
        except Exception:
            dec.append(0.01)
    while len(dec) < n:
        dec.append(dec[-1])
    return dec[:n]


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf


def df_to_excel_bytes(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf


# ── Interface ────────────────────────────────────────────────────────────────

st.title("Interface FTIR — Analyse de spectres")

# Barre latérale : paramètres globaux
with st.sidebar:
    st.header("Paramètres globaux")
    ligne_debut = st.number_input("Ligne de début", value=18, min_value=0, step=1)
    ligne_fin = st.number_input("Ligne de fin", value=7489, min_value=1, step=1)
    savgol_window = st.number_input(
        "Fenêtre Savitzky-Golay (doit être impaire)", value=11, min_value=3, step=2
    )
    savgol_polyorder = st.number_input("Ordre polynomial", value=3, min_value=1, step=1)

    # Corrections automatiques
    if int(savgol_window) % 2 == 0:
        savgol_window = int(savgol_window) + 1
        st.warning(f"Fenêtre ajustée à {savgol_window} (doit être impaire).")
    if int(savgol_polyorder) >= int(savgol_window):
        savgol_polyorder = int(savgol_window) - 1
        st.warning(f"Ordre polynomial ajusté à {savgol_polyorder}.")

    W = int(savgol_window)
    P = int(savgol_polyorder)
    D = int(ligne_debut)
    F = int(ligne_fin)

    st.divider()
    st.caption("Zones prédéfinies de liaisons H (cm⁻¹)")
    st.markdown("- **Libre** : 3555–3540\n- **Inter-brin** : 3515–3500\n- **Inter-hélice** : 3290–3275")

tab1, tab2, tab3 = st.tabs(
    ["📈 Visualisation spectres", "🔬 Analyse avancée", "⚡ Analyse automatique"]
)

# ── Onglet 1 : Visualisation ─────────────────────────────────────────────────
with tab1:
    st.subheader("Superposition et zoom des spectres")

    col1, col2 = st.columns(2)
    with col1:
        zoom_str = st.text_input(
            "Zones de zoom (cm⁻¹)", value="400-4000,3330-3850",
            help="Séparez plusieurs zones par une virgule, ex: 400-4000,3330-3850"
        )
        decalage_principal = st.number_input(
            "Décalage vertical entre spectres", value=0.7, step=0.1
        )
    with col2:
        pics_str = st.text_input(
            "Zones de détection des pics (cm⁻¹)", value="3555-3540,3290-3275"
        )
        decalages_zoom_str = st.text_input(
            "Décalages verticaux par zoom", value="0.01,0.02"
        )

    fichiers1 = st.file_uploader(
        "Charger les fichiers CSV", type=["csv"],
        accept_multiple_files=True, key="up1"
    )

    if fichiers1 and st.button("Générer les graphes", key="gen1"):
        zooms = parse_zones(zoom_str)
        zones_pics = parse_zones(pics_str)
        decalages_zoom = parse_decalages(decalages_zoom_str, len(zooms))
        couleurs = [cm.tab10(i % 10) for i in range(len(fichiers1))]

        spectres = []
        for i, f in enumerate(fichiers1):
            try:
                df = load_csv(f, D, F, W, P)
                df["Y_offset"] = df["Y"] + i * float(decalage_principal)
                spectres.append(
                    {"df": df, "nom": os.path.splitext(f.name)[0], "couleur": couleurs[i]}
                )
            except Exception as e:
                st.error(f"{f.name} : {e}")

        if not spectres:
            st.stop()

        # Figure principale
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_yticks([])
        ax.set_ylabel("(a.u)", fontsize=12)
        for sp in spectres:
            ax.plot(sp["df"]["X"], sp["df"]["Y_offset"], color=sp["couleur"])
            ax.text(
                sp["df"]["X"].iloc[-1], sp["df"]["Y_offset"].iloc[-1],
                sp["nom"], fontsize=9
            )
        ax.set_title("Spectres superposés (lissage Savitzky-Golay)")
        ax.set_xlabel("Nombre d'onde (cm⁻¹)")
        ax.invert_xaxis()
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button(
            "Télécharger le graphe principal (.png)",
            fig_to_bytes(fig),
            file_name="spectres_superposes.png",
            mime="image/png",
        )
        plt.close(fig)

        # Zooms
        for idx, (start, end) in enumerate(zooms):
            zones_in_zoom = [
                (ps, pe) for ps, pe in zones_pics
                if min(ps, pe) >= start and max(ps, pe) <= end
            ]
            st.markdown(f"---\n### Zoom {start}–{end} cm⁻¹")
            col_a, col_b = st.columns(2)

            # Sans annotation
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.set_yticks([])
            ax2.set_ylabel("(a.u)", fontsize=12)
            for i, sp in enumerate(spectres):
                mask = (sp["df"]["X"] >= start) & (sp["df"]["X"] <= end)
                if not sp["df"].loc[mask].empty:
                    ax2.plot(
                        sp["df"].loc[mask, "X"],
                        sp["df"].loc[mask, "d2Y"] + i * decalages_zoom[idx],
                        label=sp["nom"], color=sp["couleur"]
                    )
            ax2.set_title(f"Zoom {start}–{end} cm⁻¹ (sans annotation)")
            ax2.invert_xaxis()
            ax2.legend(fontsize=8)
            plt.tight_layout()
            with col_a:
                st.pyplot(fig2)
                st.download_button(
                    "Télécharger (sans annotation)",
                    fig_to_bytes(fig2),
                    file_name=f"zoom_{start}_{end}_sans_annot.png",
                    mime="image/png",
                    key=f"dl_na_{idx}",
                )
            plt.close(fig2)

            # Avec annotation
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.set_yticks([])
            ax3.set_ylabel("(a.u)", fontsize=12)
            for i, sp in enumerate(spectres):
                df = sp["df"]
                mask = (df["X"] >= start) & (df["X"] <= end)
                if df.loc[mask].empty:
                    continue
                y_plot = df.loc[mask, "d2Y"] + i * decalages_zoom[idx]
                ax3.plot(df.loc[mask, "X"], y_plot, label=sp["nom"], color=sp["couleur"])
                for zps, zpe in zones_in_zoom:
                    for x_pic in detecter_pics(df, [(zps, zpe)], W, P):
                        try:
                            y_pic = (
                                float(df.loc[np.isclose(df["X"], x_pic, atol=0.5), "d2Y"].values[0])
                                + i * decalages_zoom[idx]
                            )
                            ax3.plot(x_pic, y_pic, "ro", markersize=5)
                            ax3.text(x_pic, y_pic + 0.002, str(x_pic), color="red", fontsize=8)
                        except IndexError:
                            continue
            ax3.set_title(f"Zoom {start}–{end} cm⁻¹ (avec annotation)")
            ax3.invert_xaxis()
            ax3.legend(fontsize=8)
            plt.tight_layout()
            with col_b:
                st.pyplot(fig3)
                st.download_button(
                    "Télécharger (avec annotation)",
                    fig_to_bytes(fig3),
                    file_name=f"zoom_{start}_{end}_annot.png",
                    mime="image/png",
                    key=f"dl_a_{idx}",
                )
            plt.close(fig3)

# ── Onglet 2 : Analyse avancée ───────────────────────────────────────────────
with tab2:
    st.subheader("Classification manuelle des pics")
    st.info(
        "Chargez vos fichiers CSV, détectez les pics, "
        "puis cochez ceux à inclure et ajustez le type de liaison avant d'exporter."
    )

    fichiers2 = st.file_uploader(
        "Charger les fichiers CSV", type=["csv"],
        accept_multiple_files=True, key="up2"
    )

    if fichiers2 and st.button("Détecter les pics", key="detect2"):
        zones_detection = [(3555, 3540), (3515, 3500), (3290, 3275)]
        rows = []
        dfs_cache = {}
        for f in fichiers2:
            try:
                df = load_csv(f, D, F, W, P)
                nom = os.path.splitext(f.name)[0]
                dfs_cache[nom] = df
                for pic in detecter_pics(df, zones_detection, W, P):
                    rows.append({
                        "Fichier": nom,
                        "Pic (cm⁻¹)": pic,
                        "Type de liaison": "libre",
                        "Utiliser": False,
                    })
            except Exception as e:
                st.error(f"{f.name} : {e}")

        st.session_state["tab2_rows"] = rows
        st.session_state["tab2_dfs"] = dfs_cache

    if "tab2_rows" in st.session_state and st.session_state["tab2_rows"]:
        edited = st.data_editor(
            arrow_safe(pd.DataFrame(st.session_state["tab2_rows"])),
            column_config={
                "Type de liaison": st.column_config.SelectboxColumn(
                    options=["libre", "inter-brin", "inter-hélice", "autre"]
                ),
                "Utiliser": st.column_config.CheckboxColumn(default=False),
            },
            use_container_width=True,
            num_rows="dynamic",
            key="editor2",
        )

        if st.button("Calculer et exporter", key="export2"):
            dfs = st.session_state.get("tab2_dfs", {})
            resultats = []
            for _, row in edited[edited["Utiliser"]].iterrows():
                nom, nu, type_l = row["Fichier"], row["Pic (cm⁻¹)"], row["Type de liaison"]
                if nom in dfs:
                    E, D_val, RD = calcul_E_D_RD(dfs[nom], nu)
                    resultats.append({
                        "Fichier": nom,
                        "ν (cm⁻¹)": nu,
                        "Type de liaison": type_l,
                        "E (kJ/mol)": E,
                        "D (Å)": D_val,
                        "Abs1047/Abs1022": RD,
                    })
            if resultats:
                df_res = arrow_safe(pd.DataFrame(resultats))
                st.dataframe(df_res, use_container_width=True)
                st.download_button(
                    "Télécharger Excel",
                    df_to_excel_bytes(df_res),
                    file_name="resultats_avances_ftir.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.warning("Aucun pic sélectionné (cochez la colonne 'Utiliser').")

# ── Onglet 3 : Analyse automatique ───────────────────────────────────────────
with tab3:
    st.subheader("Analyse automatique des liaisons hydrogène")
    st.info(
        "L'analyse détecte automatiquement le pic dominant dans chaque zone "
        "et calcule E, D et Abs1047/Abs1022 pour chaque type de liaison."
    )

    fichiers3 = st.file_uploader(
        "Charger les fichiers CSV", type=["csv"],
        accept_multiple_files=True, key="up3"
    )

    if fichiers3 and st.button("Lancer l'analyse automatique", key="auto3"):
        zones_liaisons = {
            "libre":        (3555, 3540),
            "inter-brin":   (3515, 3500),
            "inter-hélice": (3290, 3275),
        }
        resultats = []
        for f in fichiers3:
            try:
                df = load_csv(f, D, F, W, P)
                nom = os.path.splitext(f.name)[0]
                for type_l, (p_start, p_end) in zones_liaisons.items():
                    mask = (df["X"] >= min(p_start, p_end)) & (df["X"] <= max(p_start, p_end))
                    if df.loc[mask, "d2Y"].empty:
                        continue
                    idx_min = df.loc[mask, "d2Y"].idxmin()
                    x_pic = round(df.loc[idx_min, "X"], 1)
                    E, D_val, RD = calcul_E_D_RD(df, x_pic)
                    resultats.append({
                        "Fichier": nom,
                        "Type": type_l,
                        "ν (cm⁻¹)": x_pic,
                        "E (kJ/mol)": E,
                        "D (Å)": D_val,
                        "Abs1047/Abs1022": RD,
                    })
            except Exception as e:
                st.error(f"{f.name} : {e}")

        if resultats:
            df_res = arrow_safe(pd.DataFrame(resultats))
            st.dataframe(df_res, use_container_width=True)
            st.download_button(
                "Télécharger Excel",
                df_to_excel_bytes(df_res),
                file_name="resultats_automatiques_ftir.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("Aucun résultat généré. Vérifiez vos fichiers et paramètres.")
