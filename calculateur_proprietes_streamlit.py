import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# FONCTIONS UTILITAIRES ET DONNÉES
# ----------------------------------------------------------------------

# Load CSS
def load_css(file_name):
    """Charge CSS avec gestion d'erreurs"""
    # ... (fonction inchangée)
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            # st.sidebar.success(f"✅ CSS chargé depuis {file_name}")
    except FileNotFoundError:
        # st.sidebar.error(f"❌ Fichier CSS introuvable : {file_name}")
        st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #1e3a8a 0%, #facc15 100%);
                color: #f5f5f4;
            }
            .stButton>button {
                background-color: #facc15;
                color: #1e3a8a;
                padding: 10px 20px;
                border-radius: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

load_css('styles.css')

# Données de ROI pour différents types de rénovations (inchangées)
RENOVATION_TYPES = {
    'Cuisine': {'roi': 0.85, 'cout_moyen': 25000},
    'Salle de bain': {'roi': 0.75, 'cout_moyen': 15000},
    'Sous-sol': {'roi': 0.65, 'cout_moyen': 35000},
    'Efficacité énergétique': {'roi': 0.80, 'cout_moyen': 10000},
    'Peinture et planchers': {'roi': 0.70, 'cout_moyen': 10000},
    'Agrandissement': {'roi': 0.55, 'cout_moyen': 75000},
    'Toiture ou fenêtres': {'roi': 0.70, 'cout_moyen': 20000}
}

# Données fallback (TAL 2025 + Cap Rates Q3 2025) (inchangées, mais moins utilisées pour Maison/Condo)
fallback_data = {
    'Plex/Multiplex': {
        'median_price': 650000.0,
        'yoy': 0.11,
        'hausse_loyer': 0.059,
        'discount_rate': {'Montréal': 0.045, 'Québec (RMR)': 0.050, 'Laurentides': 0.055, 'Autre (Québec moyen)': 0.048}
    },
    'Maison': {
        'median_price': 490000.0,
        'yoy': 0.09,
        'hausse_loyer': 0.045,
        'discount_rate': {'Montréal': 0.040, 'Québec (RMR)': 0.045, 'Laurentides': 0.050, 'Autre (Québec moyen)': 0.043}
    },
    'Condo': {
        'median_price': 399900.0,
        'yoy': 0.05,
        'hausse_loyer': 0.045,
        'discount_rate': {'Montréal': 0.042, 'Québec (RMR)': 0.047, 'Laurentides': 0.052, 'Autre (Québec moyen)': 0.045}
    },
    'Terrain': {
        'median_price': 150000.0,
        'yoy': 0.08,
        'hausse_loyer': 0.0,
        'discount_rate': {'Montréal': 0.050, 'Québec (RMR)': 0.055, 'Laurentides': 0.060, 'Autre (Québec moyen)': 0.053}
    },
    'Autre': {
        'median_price': 400000.0,
        'yoy': 0.07,
        'hausse_loyer': 0.045,
        'discount_rate': {'Montréal': 0.045, 'Québec (RMR)': 0.050, 'Laurentides': 0.055, 'Autre (Québec moyen)': 0.048}
    }
}

# Données réelles pour taux d'appréciation annuel par secteur (basé sur recherches 2025) - Taux récents et volatils
region_yoy = {
    'Montréal': 0.05, 'Québec (RMR)': 0.18, 'Gatineau': 0.18, 'Sherbrooke': 0.09, 'Trois-Rivières': 0.09,
    'Saguenay': 0.09, 'Laval': 0.05, 'Longueuil': 0.05, 'Lévis': 0.18, 'Sainte-Foy–Sillery–Cap-Rouge': 0.18,
    'Québec moyen': 0.11
}

# Variable data_scenarios non définie dans l'original. Initialisation pour éviter erreur dans Évaluation Avancée
data_scenarios = {} 


# ----------------------------------------------------------------------
# NOUVELLES FONCTIONS DE CHARGEMENT ET TRAITEMENT DES DONNÉES CSV
# ----------------------------------------------------------------------

# Cache les données pour la rapidité
@st.cache_data
def load_csv_data(type_propriete):
    """Charge et nettoie les données CSV pour un type de propriété."""
    file_map = {
        'Maison': 'prix_maisons_quebec_1970_2025.csv',
        'Condo': 'prix_condos_quebec_1970_2025.csv',
        'Plex/Multiplex': 'prix_plex_quebec_1970_2025.csv'
    }
    file_name = file_map.get(type_propriete)
    if not file_name:
        return None
        
    try:
        df = pd.read_csv(file_name)
        # Supprimer les colonnes de variation (%) pour n'utiliser que les prix
        cols_to_drop = [col for col in df.columns if 'variation' in col]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        df['Année'] = df['Année'].astype(int)
        df.set_index('Année', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Erreur: Fichier de données {file_name} non trouvé.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de {type_propriete}: {e}")
        return None

@st.cache_data
def calculate_historical_growth(df, secteur, start_year, end_year):
    """Calcule le taux de croissance annuel composé (CAGR) sur une période."""
    if df is None or secteur not in df.columns:
        return 0.055 # Taux conservateur par défaut
        
    try:
        P_start = df.loc[start_year, secteur]
        P_end = df.loc[end_year, secteur]
        N = end_year - start_year
        if N > 0 and P_start > 0:
            cagr = (P_end / P_start)**(1/N) - 1
            return cagr
        return 0.055
    except KeyError:
        # Si une année ou une colonne manque, on revient au taux conservateur
        return 0.055

# ----------------------------------------------------------------------
# FONCTIONS DE CALCUL MISES À JOUR
# ----------------------------------------------------------------------

def estimer_valeur_actuelle_propriete(prix_achat, annee_achat, secteur, type_propriete, scenario='recent'):
    """
    Estime la valeur actuelle (2025) à partir du prix d'achat et de l'année.
    Utilise les données CSV pour les taux historiques (décennaux glissants).
    """
    df = load_csv_data(type_propriete)
    current_year = 2025
    if annee_achat >= current_year or df is None: 
        return prix_achat, []

    # Taux conservateur à long terme (utilisé pour 'average')
    yoy_long_term_avg = 0.055 
    
    # 1. Calcul du taux historique glissant (pour 'recent' et 'advanced')
    # On prend l'appréciation des 10 dernières années ou de la période totale si elle est plus courte.
    start_hist_year = max(annee_achat, current_year - 10)
    
    # Taux basé sur les 10 dernières années historiques disponibles dans le CSV (ou période achat-2025)
    yoy_historical_cagr = calculate_historical_growth(df, secteur, annee_achat, current_year)
    
    # ------------------ CORRECTION APPLIQUÉE : Utilisation des CSV ------------------
    if scenario == 'average':
        # Le scénario 'average' utilise un taux long terme pour des projections plus réalistes sur de longues périodes.
        yoy_used = yoy_long_term_avg
        st.info(f"Scénario **Average** utilisé. Taux d'appréciation long-terme conservateur : {yoy_used*100:.2f}%")
    else: # 'recent'
        # Le scénario 'recent' utilise le Taux de Croissance Annuel Composé (TCAC) de la période d'achat à 2025.
        yoy_used = yoy_historical_cagr
        st.info(f"Scénario **Recent** utilisé. Taux d'appréciation (TCAC {annee_achat}-{current_year}) : {yoy_used*100:.2f}%")
    # ------------------ FIN DE LA CORRECTION ------------------

    # Recalcul de la valeur actuelle basée sur le taux choisi
    valeur = prix_achat * (1 + yoy_used) ** (current_year - annee_achat)
    
    # Génération des valeurs historiques pour le graphique
    valeurs_historiques = [(annee_achat + i, prix_achat * (1 + yoy_used) ** i) for i in range(current_year - annee_achat + 1)]
    return round(valeur, 2), valeurs_historiques

def get_valeur_marche_2025(secteur, type_propriete):
    """Récupère la valeur médiane du marché en 2025 (depuis CSV si possible, sinon Fallback)."""
    df = load_csv_data(type_propriete)
    if df is not None and secteur in df.columns and 2025 in df.index:
        try:
            valeur = df.loc[2025, secteur]
            return valeur
        except:
            pass # Continue pour fallback
    
    # Fallback si CSV non disponible ou erreur
    return fallback_data.get(type_propriete, fallback_data['Autre'])['median_price']

def compute_projection(V0, annee_achat, g_override, T, renos, alpha=0.10, secteur=None, type_propriete=None):
    """
    Calcule les projections année par année (Maison/Condo).
    Le taux de croissance 'g' est désormais passé en paramètre et provient de l'historique
    ou d'un taux conservateur forcé.
    """
    
    # Si le taux d'appréciation est très élevé, on le tempère à 5.5% pour la projection à long terme
    # Cela évite les projections extrêmes
    if g_override > 0.08:
        g = 0.055
        st.warning(f"⚠️ Taux d'appréciation ramené à {g*100:.2f}% pour la projection à long terme (éviter les extrêmes).")
    else:
        g = g_override
    
    annees = list(range(0, T + 1))
    valeurs_sans_reno = [0] * (T + 1)
    valeurs_avec_reno = [0] * (T + 1)
    cash_flows = [0] * (T + 1)
    valeurs_sans_reno[0] = V0
    valeurs_avec_reno[0] = V0
    
    for t in range(1, T + 1):
        # Utilisation du taux de croissance 'g' tempéré
        valeurs_sans_reno[t] = valeurs_sans_reno[t-1] * (1 + g)
        valeurs_avec_reno[t] = valeurs_avec_reno[t-1] * (1 + g)
    
    for ren in renos:
        t = ren['year']
        if 1 <= t <= T:
            increment = ren['cost'] * ren['r']
            valeurs_avec_reno[t] += increment
            cash_flows[t] -= ren['cost']
    
    # Logique d'ajustement (limitée à +10% de la valeur sans réno)
    for t in range(1, T + 1):
        max_valeur = valeurs_sans_reno[t] * (1 + alpha)
        valeurs_avec_reno[t] = min(valeurs_avec_reno[t], max_valeur)
    
    df = pd.DataFrame({
        'Année': [annee_achat + a for a in annees],
        'Valeur sans réno': [round(v, 2) for v in valeurs_sans_reno], # Rounding for display
        'Valeur avec réno': [round(v, 2) for v in valeurs_avec_reno], # Rounding for display
        'Cash Flow': [round(cf, 2) for cf in cash_flows] # Rounding for display
    })
    summary = {
        'end_no': valeurs_sans_reno[-1], 'end_with': valeurs_avec_reno[-1],
        'gain': valeurs_avec_reno[-1] - valeurs_sans_reno[-1],
        'total_cost': sum(ren['cost'] for ren in renos),
        'roi': (valeurs_avec_reno[-1] - valeurs_sans_reno[-1]) / sum(ren['cost'] for ren in renos) if sum(ren['cost'] for ren in renos) > 0 else 0
    }
    return df, summary, g


# Le reste des fonctions (calculateur_rentabilite_plex, simulation_probabiliste)
# reste inchangé car la problématique était sur Maison/Condo.

def calculateur_rentabilite_plex(valeur_actuelle, taux_appreciation, nb_annees, revenus_locatifs_annuels,
                                taux_augmentation_loyers, taxes_municipales, taxes_scolaires, assurances,
                                entretien, frais_gestion, autres_depenses, taux_inflation_depenses,
                                paiements_hypothecaires_annuels, ajustements_valeur=0.0, revenus_projected_annuels=None,
                                taux_actualisation=0.048, renovations=None):
    """Calcule la rentabilité avec cash flows, NPV et ROI (Logiciel Plex)."""
    # ... (code inchangé)
    annees = np.arange(0, nb_annees + 1)
    valeurs = valeur_actuelle * (1 + taux_appreciation) ** annees
    
    revenus_current_annual = revenus_locatifs_annuels
    revenus_array = np.zeros(nb_annees + 1)
    
    # Calcul des revenus pour l'année 1 à N
    for i in range(1, nb_annees + 1):
        if revenus_projected_annuels is not None and i == 1:
            # Utilise le revenu projeté pour l'Année 1 (mode Court terme)
            revenus_array[i] = revenus_projected_annuels
        elif revenus_projected_annuels is not None and i > 1:
            # Projection à partir du revenu projeté de l'Année 1
            revenus_array[i] = revenus_array[i-1] * (1 + taux_augmentation_loyers)
        else:
            # Projection à partir du revenu actuel (Moyen/Long terme)
            revenus_array[i] = revenus_current_annual * (1 + taux_augmentation_loyers) ** i
            
    revenus = revenus_array
    
    depenses_initiales = taxes_municipales + taxes_scolaires + assurances + entretien + frais_gestion + autres_depenses
    depenses = depenses_initiales * (1 + taux_inflation_depenses) ** annees
    
    cash_flows = np.zeros(nb_annees + 1)
    
    # Cash flow année 1 à N
    for i in range(1, nb_annees + 1):
        cash_flows[i] = revenus[i] - depenses[i] - paiements_hypothecaires_annuels
        
    cash_flow_cumule = np.cumsum(cash_flows)
    
    # Logique de rénovation
    cout_renovations = 0.0
    valeur_ajoutee_renovations = 0.0
    
    if renovations:
        for reno_type, reno_cout in renovations.items():
            if reno_type in RENOVATION_TYPES:
                roi = RENOVATION_TYPES[reno_type]['roi']
                valeur_ajoutee = reno_cout * roi
                cout_renovations += reno_cout
                valeur_ajoutee_renovations += valeur_ajoutee

    # Ajustements finaux (inclut la perte due à la vacance locative ajustements_valeur < 0)
    valeurs[-1] += ajustements_valeur + valeur_ajoutee_renovations

    investissement_initial = valeur_actuelle + cout_renovations
    prise_valeur_totale = valeurs[-1] - valeur_actuelle
    rendement_total = cash_flows[1:].sum() + prise_valeur_totale
    roi_moyen_annuel = (rendement_total / investissement_initial / nb_annees) * 100 if nb_annees > 0 and investissement_initial > 0 else 0

    cash_flows_for_npv = cash_flows[1:]
    npv_cash_flows = sum([cf / (1 + taux_actualisation)**t for t, cf in enumerate(cash_flows_for_npv, 1)])
    npv_valeur_finale = valeurs[-1] / (1 + taux_actualisation)**nb_annees
    npv_rendement = npv_cash_flows + npv_valeur_finale
    npv_roi = (npv_rendement / investissement_initial) * 100 if investissement_initial > 0 else 0
    
    return {
        'annees': annees, 'valeurs': valeurs, 'revenus': revenus, 'depenses': depenses,
        'cash_flows': cash_flows, 'cash_flow_cumule': cash_flow_cumule,
        'prise_valeur_totale': prise_valeur_totale, 'rendement_total': rendement_total,
        'roi_moyen_annuel': roi_moyen_annuel, 'depenses_initiales': depenses_initiales,
        'npv_rendement': npv_rendement, 'npv_roi': npv_roi,
        'cout_renovations': cout_renovations,
        'valeur_ajoutee_renovations': valeur_ajoutee_renovations
    }

def simulation_probabiliste(valeur_actuelle, taux_appreciation, nb_annees, ajustements_valeur, ecart_type, renovations=None, n_simulations=1000):
    """Simule les prises de valeur avec Monte Carlo (Logiciel Plex)."""
    # ... (code inchangé)
    taux_simules = np.random.normal(taux_appreciation, ecart_type, n_simulations)
    valeurs_futures = valeur_actuelle * np.power(1 + taux_simules, nb_annees) + ajustements_valeur
    
    valeur_ajoutee_renovations = 0.0
    if renovations:
        if isinstance(renovations, dict): # Plex/Multiplex style
            valeur_ajoutee_renos = sum(reno_cout * RENOVATION_TYPES.get(reno_type, {'roi': 0.0})['roi'] for reno_type, reno_cout in renovations.items())
        else: # Maison/Condo style (list of dict)
            valeur_ajoutee_renovations = sum(ren['cost'] * ren['r'] for ren in renovations)
            
    valeurs_futures += valeur_ajoutee_renovations
    
    prises_valeur_simulees = valeurs_futures - valeur_actuelle
    
    p10 = np.percentile(prises_valeur_simulees, 10)
    p50 = np.percentile(prises_valeur_simulees, 50)
    p90 = np.percentile(prises_valeur_simulees, 90)
    
    return prises_valeur_simulees, p10, p50, p90, valeurs_futures

# ----------------------------------------------------------------------
# INTERFACE PRINCIPALE STREAMLIT (Mise à jour des appels de fonctions)
# ----------------------------------------------------------------------

st.title("🧭 Calculateur de Propriétés Immobilières au Québec 2025")
st.write("Découvrez la valeur et la rentabilité de votre bien immobilier en 2025 !")

# Initialisation de session_state
if 'step' not in st.session_state: st.session_state.step = 1
if 'params' not in st.session_state: st.session_state.params = {}
if 'type_propriete' not in st.session_state: st.session_state.type_propriete = None
if 'secteur' not in st.session_state: st.session_state.secteur = None

# Barre de progression
progress = st.session_state.step / 3 * 100
st.markdown(f'<div class="progress-bar"><div class="progress" style="width: {progress}%;">{st.session_state.step}/3</div></div>', unsafe_allow_html=True)


# Étape 1 : Choix du type de propriété et région
if st.session_state.step == 1:
    st.header("Étape 1/3 : Choisissez votre type de propriété")
    type_propriete = st.radio("Type de propriété", ['Plex/Multiplex', 'Maison', 'Condo', 'Terrain', 'Autre'], index=0, key='type_radio', horizontal=True)
    st.session_state.type_propriete = type_propriete
    
    st.subheader("Région")
    secteur = st.selectbox("Sélectionnez votre région", ['Montréal', 'Québec (RMR)', 'Gatineau', 'Sherbrooke', 'Trois-Rivières', 'Saguenay', 'Laval', 'Longueuil', 'Lévis', 'Sainte-Foy–Sillery–Cap-Rouge', 'Québec moyen'], index=0)
    st.session_state.secteur = secteur
    
    if st.button("Suivant"):
        st.session_state.step = 2
        st.rerun()

# ----------------------------------------------------------------------
# Étape 2 : Détails de la propriété (Interface améliorée)
# ----------------------------------------------------------------------
elif st.session_state.step == 2:
    st.write("2/3")
    st.write(f"Entrez les détails pour votre {st.session_state.type_propriete.lower()}.")
    
    if st.session_state.type_propriete in ['Maison', 'Condo']:
        
        type_prop = st.session_state.type_propriete # Alias
        
        mode_maison_condo = st.radio("Mode d'évaluation", ['Estimation Rapide (Historique)', 'Valeur Marché 2025', 'Évaluation Avancée'], index=0, key=f'mode_{type_prop.lower()}', horizontal=True)
        st.session_state.params['mode'] = mode_maison_condo
        
        if mode_maison_condo == 'Estimation Rapide (Historique)':
            annee_achat = st.number_input("Année d'achat", min_value=1970, max_value=2025, value=2010, key=f'quick_annee_{type_prop.lower()}')
            prix_achat = st.number_input("Prix d'achat initial ($)", min_value=0.0, value=300000.0, key=f'quick_prix_{type_prop.lower()}')
            scenario = st.selectbox("Scénario d'appréciation", ['recent', 'average'], index=0, key=f'quick_scenario_{type_prop.lower()}') 
            
            if st.button("Estimer et Continuer"):
                valeur, historique = estimer_valeur_actuelle_propriete(prix_achat, annee_achat, st.session_state.secteur, type_prop, scenario)
                if valeur is not None:
                    st.session_state.params['valeur_actuelle'] = valeur
                    st.session_state['last_quick_result'] = {"valeur": valeur, "historique": historique, "scenario": scenario, "prix_achat": prix_achat, "annee_achat": annee_achat}
                    st.session_state.step = 3
                    st.rerun()

        elif mode_maison_condo == 'Valeur Marché 2025':
            valeur_marche = get_valeur_marche_2025(st.session_state.secteur, type_prop)
            st.session_state.params['valeur_actuelle'] = valeur_marche
            st.success(f"Valeur médiane du marché en 2025 ({st.session_state.secteur}) : **${valeur_marche:,.0f}**") 
            
            if st.button("Continuer"):
                st.session_state.step = 3
                st.rerun()
        
        elif mode_maison_condo == 'Évaluation Avancée':
            st.subheader("Évaluation avancée — Prise de valeur après rénovation")
            st.markdown("Ce module calcule l'impact de rénovations sur la valeur finale du bien. La prise de valeur est limitée à +10 % de la valeur projetée sans rénovations.")
            
            # --- Paramètres de Projection ---
            horizon = st.number_input("Horizon de projection (années)", min_value=1, max_value=50, value=st.session_state.get('nb_annees', 10), step=1, key=f'adv_horizon_{type_prop.lower()}')
            st.session_state.params['nb_annees'] = horizon
            
            # Taux de croissance de la valeur (g)
            df = load_csv_data(type_prop)
            if df is not None:
                # On utilise la croissance historique sur 10 ans glissants (2015-2025)
                g_hist = calculate_historical_growth(df, st.session_state.secteur, 2015, 2025)
            else:
                g_hist = 0.055

            g_override = st.number_input(f"Taux d'appréciation annuel (g) % (Historique : {g_hist*100:.2f}%)", 
                                         min_value=0.0, max_value=20.0, value=g_hist*100, step=0.1) / 100
            
            st.session_state.params['taux_appreciation'] = g_override

            # --- Rénovations dynamique ---
            reno_key = f"adv_renos_{type_prop.lower()}"
            if reno_key not in st.session_state:
                st.session_state[reno_key] = [{"year":3, "cost":55000.0, "r":0.55}]
            st.markdown("### Rénovations (année, coût, taux de récupération r)")
            if st.button("Ajouter une rénovation", key=f"adv_add_reno_{type_prop.lower()}"):
                st.session_state[reno_key].append({"year":3, "cost":10000.0, "r":0.55})
            
            current_renos = list(st.session_state[reno_key])
            new_renos = []
            for i, ren in enumerate(current_renos):
                c1, c2, c3, c4 = st.columns([1,2,2,1])
                with c1: yr = st.number_input(f"Année #{i+1}", min_value=1, max_value=horizon, value=ren.get("year",3), key=f"adv_yr_{type_prop.lower()}_{i}")
                with c2: cost = st.number_input(f"Coût #{i+1} ($)", min_value=0.0, value=ren.get("cost",55000.0), step=100.0, key=f"adv_cost_{type_prop.lower()}_{i}")
                with c3: r = st.slider(f"Taux r #{i+1}", 0.0, 1.0, ren.get("r",0.55), key=f"adv_r_{type_prop.lower()}_{i}")
                with c4:
                    if st.button("Suppr", key=f"adv_suppr_{type_prop.lower()}_{i}"):
                        current_renos.pop(i)
                        st.session_state[reno_key] = current_renos
                        st.rerun()
                    new_renos.append({"year": yr, "cost": cost, "r": r})
            st.session_state[reno_key] = new_renos
            st.session_state.params['renovations'] = st.session_state[reno_key]
            # --- Fin Rénovations dynamique ---
            
            st.markdown("### Valeur initiale (VO)")
            use_specific = st.checkbox("Utiliser un prix d'achat spécifique (sinon la médiane du marché sera utilisée)", value=True, key=f'adv_use_specific_{type_prop.lower()}')
            
            if use_specific:
                annee_achat_v0 = st.number_input("Année d'achat initial", min_value=1970, max_value=2025, value=2010, key=f'adv_annee_v0_{type_prop.lower()}')
                prix_init = st.number_input("Prix d'achat initial ($)", value=220000.0, key=f'adv_prix_v0_{type_prop.lower()}')
                
                # On estime la valeur actuelle (V0) en 2025 à partir du prix d'achat et du TCAC historique
                V0_estimé, _ = estimer_valeur_actuelle_propriete(prix_init, annee_achat_v0, st.session_state.secteur, type_prop, scenario='recent')
                st.info(f"Valeur de départ utilisée (V0 en 2025, estimée): **${V0_estimé:,.0f}**")
                V0 = V0_estimé
            else:
                V0 = get_valeur_marche_2025(st.session_state.secteur, type_prop)
                st.info(f"Valeur de départ utilisée (V0 en 2025, prix médian): **${V0:,.0f}**")
                annee_achat_v0 = 2025  # Si médiane marché, année = 2025
            
            st.session_state.params['valeur_actuelle'] = V0
            st.session_state.params['annee_achat'] = annee_achat_v0
            
            # Affichage de la projection pour feedback immédiat
            # On passe le 'g' override au compute projection
            df_proj, summary, g_used = compute_projection(V0=float(V0), annee_achat=annee_achat_v0, g_override=g_override, T=horizon, renos=st.session_state[reno_key], alpha=0.10, secteur=st.session_state.secteur, type_propriete=type_prop)
            st.info(f"Taux de croissance annuel utilisé pour la projection (g) : {g_used*100:.4f}%")
            st.write("Aperçu de la Projection (Tableau mis à jour):")
            st.dataframe(df_proj.style.format("{:,.2f}"))
            
            # Stockage des résultats pour l'étape 3
            st.session_state['last_advanced_result'] = {"df": df_proj, "summary": summary}
            
            if st.button("Lancer la Simulation"):
                st.session_state.step = 3
                st.rerun()
    
    # ---------------------------------------------------
    # LOGIQUE PLEX / MULTIPLEX (inchangée)
    # ---------------------------------------------------
    elif st.session_state.type_propriete == 'Plex/Multiplex':
        # ... (Logique Plex inchangée, utilise les fallbacks pour les taux)
        st.subheader("1. Horizon de Projection et Valeur")
        periode = st.selectbox("Période de Projection", ['Court terme (1-3 ans)', 'Moyen terme (4-7 ans)', 'Long terme (8+ ans)'], index=0)
        
        if periode == 'Court terme (1-3 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (1-3)", min_value=1, max_value=3, value=st.session_state.params.get('nb_annees', 2), step=1, format="%d", key='plex_nb_annees_ct')
        elif periode == 'Moyen terme (4-7 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (4-7)", min_value=4, max_value=7, value=st.session_state.params.get('nb_annees', 5), step=1, format="%d", key='plex_nb_annees_mt')
        else:
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (8+)", min_value=8, value=st.session_state.params.get('nb_annees', 10), step=1, format="%d", key='plex_nb_annees_lt')

        st.session_state.params['valeur_actuelle'] = st.number_input("Valeur actuelle ($)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'median_price': 650000.0})['median_price']), step=1000.0)
        st.session_state.params['taux_appreciation'] = st.number_input("Taux d’appréciation annuel (%)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'yoy': 0.11})['yoy'] * 100), step=0.1) / 100
        taux_actualisation_default = fallback_data.get(st.session_state.type_propriete, {'discount_rate': {'Autre (Québec moyen)': 0.048}})['discount_rate'].get(st.session_state.secteur, 0.048)
        st.session_state.params['taux_actualisation'] = st.number_input("Taux d'actualisation global (%) (Cap Rate ajusté)", min_value=0.0, value=taux_actualisation_default * 100, step=0.1) / 100
        st.session_state.params['paiements_hypothecaires_annuels'] = st.number_input("Paiements hypothécaires annuels ($)", min_value=0.0, value=15000.0, step=100.0)
        
        st.subheader("2. Détails des Loyers et Revenus")
        st.session_state.params['nb_logements'] = st.number_input("Nombre de logements", min_value=1, value=st.session_state.params.get('nb_logements', 4), step=1, format="%d")
        
        if 'loyers_actuels' not in st.session_state.params or len(st.session_state.params['loyers_actuels']) != st.session_state.params['nb_logements']:
            st.session_state.params['loyers_actuels'] = [1000.0] * st.session_state.params['nb_logements']

        st.session_state.params['loyers_actuels'] = [
            st.number_input(f"Logement {i+1} - Loyer actuel ($/mois)", min_value=0.0, value=st.session_state.params['loyers_actuels'][i], step=50.0, key=f'loyer_actuel_{i}') 
            for i in range(st.session_state.params['nb_logements'])
        ]
        
        if periode == 'Court terme (1-3 ans)':
            st.markdown("**Prévision d'augmentation pour l'année 1 (loyer projeté)**")
            
            if 'loyers_projectes' not in st.session_state.params or len(st.session_state.params['loyers_projectes']) != st.session_state.params['nb_logements']:
                st.session_state.params['loyers_projectes'] = [1100.0] * st.session_state.params['nb_logements']
                
            st.session_state.params['loyers_projectes'] = [
                st.number_input(f"Logement {i+1} - Loyer projeté ($/mois)", min_value=0.0, value=st.session_state.params['loyers_projectes'][i], step=50.0, key=f'loyer_projete_{i}') 
                for i in range(st.session_state.params['nb_logements'])
            ]
        else:
            st.session_state.params.pop('loyers_projectes', None) 
        
        st.session_state.params['taux_augmentation_loyers'] = st.number_input("Augmentation loyers annuelle post-an 1 (%) (TAL 2025 : 5.9%)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'hausse_loyer': 0.059})['hausse_loyer'] * 100), step=0.1) / 100
        
        st.subheader("3. Dépenses Annuelles et Risque")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.session_state.params['taxes_municipales'] = st.number_input("Taxes municipales ($)", min_value=0.0, value=4000.0, step=100.0)
            st.session_state.params['assurances'] = st.number_input("Assurances ($)", min_value=0.0, value=1200.0, step=100.0)
            st.session_state.params['frais_gestion'] = st.number_input("Gestion (%)", min_value=0.0, value=2000.0, step=100.0)
            st.session_state.params['taux_inflation_depenses'] = st.number_input("Inflation dépenses (%)", min_value=0.0, value=2.0, step=0.1) / 100
        with col_exp2:
            st.session_state.params['taxes_scolaires'] = st.number_input("Taxes scolaires ($)", min_value=0.0, value=800.0, step=100.0)
            st.session_state.params['entretien'] = st.number_input("Entretien ($)", min_value=0.0, value=3000.0, step=100.0)
            st.session_state.params['autres_depenses'] = st.number_input("Autres ($)", min_value=0.0, value=1000.0, step=100.0)
            st.session_state.params['vacance_rate'] = st.number_input("Vacance locative (%)", min_value=0.0, value=5.0, step=0.1) / 100
        
        st.subheader("4. Rénovations (Optionnel)")
        renovations = {}
        selected_renovations = st.multiselect("Types de rénovations", list(RENOVATION_TYPES.keys()), help="Sélectionnez les rénovations prévues", key='plex_renos_multiselect')
        for reno_type in selected_renovations:
            default_cost = RENOVATION_TYPES[reno_type]['cout_moyen']
            cout = st.number_input(f"Coût pour {reno_type} ($)", min_value=0.0, value=float(default_cost), step=1000.0, key=f"plex_cout_{reno_type}")
            if cout > 0:
                renovations[reno_type] = cout
        
        st.session_state.params.update({
            'valeur_actuelle': st.session_state.params['valeur_actuelle'], 
            'taux_appreciation': st.session_state.params['taux_appreciation'], 
            'nb_annees': st.session_state.params['nb_annees'], 
            'taux_actualisation': st.session_state.params['taux_actualisation'],
            'paiements_hypothecaires_annuels': st.session_state.params['paiements_hypothecaires_annuels'],
            'nb_logements': st.session_state.params['nb_logements'], 
            'loyers_actuels': st.session_state.params['loyers_actuels'],
            'loyers_projectes': st.session_state.params.get('loyers_projectes'), 
            'taux_augmentation_loyers': st.session_state.params['taux_augmentation_loyers'],
            'taxes_municipales': st.session_state.params['taxes_municipales'], 
            'taxes_scolaires': st.session_state.params['taxes_scolaires'], 
            'assurances': st.session_state.params['assurances'],
            'entretien': st.session_state.params['entretien'], 
            'frais_gestion': st.session_state.params['frais_gestion'], 
            'autres_depenses': st.session_state.params['autres_depenses'],
            'taux_inflation_depenses': st.session_state.params['taux_inflation_depenses'], 
            'vacance_rate': st.session_state.params['vacance_rate'],
            'renovations_for_plex': renovations,
            'mode': 'plex_advanced'
        })
        
        if st.button("Lancer la Simulation de Rentabilité"):
            st.session_state.step = 3
            st.rerun()

    # ---------------------------------------------------
    # LOGIQUE TERRAIN / AUTRE (inchangée)
    # ---------------------------------------------------
    elif st.session_state.type_propriete in ['Terrain', 'Autre']:
        st.subheader(f"Projection de valeur pour {st.session_state.type_propriete}")
        valeur_actuelle = st.number_input("Valeur actuelle du bien ($)", min_value=0.0, value=fallback_data.get(st.session_state.type_propriete, fallback_data['Autre'])['median_price'])
        taux_appreciation_defaut = fallback_data.get(st.session_state.type_propriete, fallback_data['Autre'])['yoy']
        taux_appreciation = st.number_input("Taux d'appréciation annuel (g) (%)", min_value=0.0, max_value=20.0, value=taux_appreciation_defaut * 100) / 100
        nb_annees = st.number_input("Horizon de projection (années)", min_value=1, max_value=50, value=10)
        
        st.session_state.params.update({
            'valeur_actuelle': valeur_actuelle, 'taux_appreciation': taux_appreciation, 'nb_annees': nb_annees, 'mode': 'simple_projection'
        })

        if st.button("Lancer la Projection"):
            st.session_state.step = 3
            st.rerun()

# ----------------------------------------------------------------------
# Étape 3 : Résultats de la simulation (Logique inchangée, utilise les nouveaux résultats)
# ----------------------------------------------------------------------
elif st.session_state.step == 3:
    st.header("Étape 3/3 : Résultats de la simulation")
    
    if st.button("Modifier les paramètres"):
        st.session_state.step = 2
        st.rerun()

    if st.session_state.type_propriete == 'Plex/Multiplex':
        # ... (Logique d'affichage Plex inchangée)
        params = st.session_state.params
        revenus_locatifs_annuels = sum(params['loyers_actuels']) * 12
        revenus_projected_annuels = sum(params['loyers_projectes']) * 12 if params.get('loyers_projectes') else None
        ajustements_valeur = -revenus_locatifs_annuels * params['vacance_rate'] * params['nb_annees']
        
        resultats = calculateur_rentabilite_plex(
            valeur_actuelle=params['valeur_actuelle'], taux_appreciation=params['taux_appreciation'],
            nb_annees=params['nb_annees'], revenus_locatifs_annuels=revenus_locatifs_annuels,
            taux_augmentation_loyers=params['taux_augmentation_loyers'], taxes_municipales=params['taxes_municipales'],
            taxes_scolaires=params['taxes_scolaires'], assurances=params['assurances'], entretien=params['entretien'],
            frais_gestion=params['frais_gestion'], autres_depenses=params['autres_depenses'],
            taux_inflation_depenses=params['taux_inflation_depenses'], paiements_hypothecaires_annuels=params['paiements_hypothecaires_annuels'],
            ajustements_valeur=ajustements_valeur, taux_actualisation=params['taux_actualisation'],
            revenus_projected_annuels=revenus_projected_annuels,
            renovations=params['renovations_for_plex']
        )
        prises_simulees, p10, p50, p90, valeurs_futures = simulation_probabiliste(
            valeur_actuelle=params['valeur_actuelle'], taux_appreciation=params['taux_appreciation'],
            nb_annees=params['nb_annees'], ajustements_valeur=ajustements_valeur,
            ecart_type=params.get('ecart_type', 0.025), renovations=params['renovations_for_plex'], n_simulations=1000
        )

        st.header("🌟 Résultats Globaux (Probabilistes)")
        st.success(f"**🎉 Focus : Projection sur {params['nb_annees']} ans**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valeur Actuelle", f"${resultats['valeurs'][0]:,.0f}")
        col2.metric("Valeur Future Médiane", f"${np.median(valeurs_futures):,.0f}")
        col3.metric("Prise de Valeur Médiane", f"${p50:,.0f}")
        col4.metric("**NPV ROI (Actualisé)**", f"**{resultats['npv_roi']:.2f}%**")
        
        st.write(f"🔍 P10 (Pessimiste): ${p10:,.0f} | P90 (Optimiste): ${p90:,.0f}")
        st.write(f"💸 Cash Flow Cumulé: ${resultats['cash_flow_cumule'][-1]:,.0f}")
        st.write(f"🎯 ROI Brut Médian: {resultats['roi_moyen_annuel']:.2f}% | Taux Actualisation: {params['taux_actualisation']*100:.1f}%")

        df_resultat = pd.DataFrame({
            'Année': [2025 + a for a in resultats['annees']], 
            'Valeur': resultats['valeurs'],
            'Revenus Bruts': resultats['revenus'], 
            'Dépenses Totales': resultats['depenses'],
            'Cash Flow Net': resultats['cash_flows'], 
            'Cash Flow Cumulé': resultats['cash_flow_cumule']
        })
        st.dataframe(df_resultat.style.format(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) and x >= 1000 else f"{x:.2f}" if isinstance(x, (int, float)) else x), hide_index=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(df_resultat['Année'], df_resultat['Valeur'], label='Valeur', color='#facc15')
        ax1.plot(df_resultat['Année'], df_resultat['Cash Flow Cumulé'], label='Cash Flow Cumulé', color='#34d399')
        ax1.set_title("📈 Évolution Base (Valeur & Cash Flow)", color='#f5f5f4')
        ax1.legend()
        ax1.grid(True)

        ax2.hist(prises_simulees, bins=50, alpha=0.7, color='#60a5fa')
        ax2.axvline(p50, color='red', linestyle='--', label=f'Médiane ${p50:,.0f}')
        ax2.set_title("📊 Distribution de la Prise de Valeur", color='#f5f5f4')
        ax2.legend()
        ax2.grid(True)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e3a8a')
            ax.tick_params(axis='x', colors='#f5f5f4')
            ax.tick_params(axis='y', colors='#f5f5f4')
            ax.spines['bottom'].set_color('#f5f5f4')
            ax.spines['left'].set_color('#f5f5f4')
            for item in ([ax.title] + ax.get_xticklabels() + ax.get_yticklabels()): item.set_color('#f5f5f4')
        
        fig.patch.set_facecolor('#1e3a8a')
        st.pyplot(fig)
    
    elif st.session_state.type_propriete in ['Maison', 'Condo']:
        
        mode = st.session_state.params.get('mode')
        
        if mode == 'Évaluation Avancée' and 'last_advanced_result' in st.session_state:
            st.header(f"📈 Résultats de l'Évaluation Avancée ({st.session_state.type_propriete})")
            results = st.session_state['last_advanced_result']
            df_proj = results['df']
            summary = results['summary']

            st.write(f"**Horizon de projection:** {len(df_proj)-1} ans")
            st.success(f"Valeur sans réno (finale) : **${summary['end_no']:,.0f}**")
            st.success(f"Valeur avec réno (finale) : **${summary['end_with']:,.0f}**")
            st.info(f"Gain attribuable aux rénovations : ${summary['gain']:,.0f}")
            st.info(f"ROI total des rénovations : {summary['roi']:.2%} (coût total ${summary['total_cost']:,.0f})")
            
            st.subheader("Projection détaillée")
            st.dataframe(df_proj.style.format("{:,.2f}"), hide_index=True)
            
            # --- PLOT (Graphique de croissance) ---
            fig_adv, ax_adv = plt.subplots(figsize=(10, 6))
            ax_adv.plot(df_proj['Année'], df_proj['Valeur sans réno'], label='Valeur sans Rénovations', linestyle='--', color='#60a5fa')
            ax_adv.plot(df_proj['Année'], df_proj['Valeur avec réno'], label='Valeur avec Rénovations', color='#facc15', linewidth=3)
            ax_adv.set_title(f"Évolution de la Valeur de la {st.session_state.type_propriete}", color='#f5f5f4')
            ax_adv.set_xlabel("Année", color='#f5f5f4')
            ax_adv.set_ylabel("Valeur ($)", color='#f5f5f4')
            ax_adv.legend()
            ax_adv.grid(True, linestyle=':', alpha=0.6)

            ax_adv.set_facecolor('#1e3a8a')
            ax_adv.tick_params(axis='x', colors='#f5f5f4')
            ax_adv.tick_params(axis='y', colors='#f5f5f4')
            ax_adv.spines['bottom'].set_color('#f5f5f4')
            ax_adv.spines['left'].set_color('#f5f5f4')
            fig_adv.patch.set_facecolor('#1e3a8a')
            for item in ([ax_adv.title, ax_adv.xaxis.label, ax_adv.yaxis.label] + ax_adv.get_xticklabels() + ax_adv.get_yticklabels()): item.set_color('#f5f5f4')
            
            st.pyplot(fig_adv)
            # -----------------------------------

        elif mode == 'Estimation Rapide (Historique)' and 'last_quick_result' in st.session_state:
            st.header(f"🔍 Résultats de l'Estimation Rapide ({st.session_state.type_propriete})")
            results = st.session_state['last_quick_result']
            
            st.success(f"Valeur estimée en 2025 (Scénario **{results['scenario'].upper()}**) : **${results['valeur']:,.0f}**")
            st.info(f"Basé sur un prix d'achat de ${results['prix_achat']:,.0f} en {results['annee_achat']}.")

            df_hist = pd.DataFrame(results['historique'], columns=['Année', 'Valeur Estimée'])
            st.subheader("Historique de Valeur Estimée")
            st.dataframe(df_hist.style.format("{:,.2f}"), hide_index=True)
            
            # --- PLOT (Graphique de croissance simplifié) ---
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            ax_hist.plot(df_hist['Année'], df_hist['Valeur Estimée'], label='Valeur Estimée', color='#facc15', linewidth=3)
            ax_hist.scatter([results['annee_achat']], [results['prix_achat']], color='#e11d48', zorder=5, label='Prix Achat Initial')
            ax_hist.scatter([2025], [results['valeur']], color='#34d399', zorder=5, label='Valeur Estimée 2025')
            
            ax_hist.set_title(f"Projection de Valeur ({results['scenario'].upper()})", color='#f5f5f4')
            ax_hist.set_xlabel("Année", color='#f5f5f4')
            ax_hist.set_ylabel("Valeur ($)", color='#f5f5f4')
            ax_hist.legend()
            ax_hist.grid(True, linestyle=':', alpha=0.6)

            ax_hist.set_facecolor('#1e3a8a')
            ax_hist.tick_params(axis='x', colors='#f5f5f4')
            ax_hist.tick_params(axis='y', colors='#f5f5f4')
            ax_hist.spines['bottom'].set_color('#f5f5f4')
            ax_hist.spines['left'].set_color('#f5f5f4')
            fig_hist.patch.set_facecolor('#1e3a8a')
            for item in ([ax_hist.title, ax_hist.xaxis.label, ax_hist.yaxis.label] + ax_hist.get_xticklabels() + ax_hist.get_yticklabels()): item.set_color('#f5f5f4')
            
            st.pyplot(fig_hist)
            # ---------------------------------------------
            
        elif mode == 'Valeur Marché 2025':
            st.header(f"🔍 Résultat de Valeur Marché 2025 ({st.session_state.type_propriete})")
            valeur_marche = st.session_state.params.get('valeur_actuelle')
            st.success(f"La valeur médiane du marché en 2025 pour un {st.session_state.type_propriete.lower()} dans votre région est estimée à : **${valeur_marche:,.0f}**")
        else:
            st.error("Aucune simulation n'a été lancée ou le mode sélectionné n'est pas supporté pour l'affichage des résultats.")

    elif st.session_state.type_propriete in ['Terrain', 'Autre']:
        # ... (Logique d'affichage Terrain/Autre inchangée)
        params = st.session_state.params
        V0 = params['valeur_actuelle']
        g = params['taux_appreciation']
        T = params['nb_annees']
        
        annees = list(range(0, T + 1))
        valeurs = [V0 * (1 + g) ** t for t in annees]
        
        df_proj = pd.DataFrame({
            'Année': [2025 + a for a in annees],
            'Valeur Projetée': [round(v, 2) for v in valeurs],
        })
        
        st.header(f"📈 Projection Simple de Valeur pour {st.session_state.type_propriete}")
        st.success(f"Valeur actuelle (Année 0) : ${V0:,.2f}")
        st.success(f"Valeur projetée après {T} ans : **${valeurs[-1]:,.2f}**")
        st.info(f"Taux d'appréciation annuel (g) : {g*100:.2f}%")
        
        st.subheader("Projection détaillée")
        st.dataframe(df_proj.style.format("{:,.2f}"), hide_index=True)
        
        # --- PLOT (Graphique de croissance) ---
        fig_simple, ax_simple = plt.subplots(figsize=(10, 6))
        ax_simple.plot(df_proj['Année'], df_proj['Valeur Projetée'], label='Valeur Projetée', color='#facc15', linewidth=3)
        ax_simple.set_title(f"Projection de Valeur pour {st.session_state.type_propriete}", color='#f5f5f4')
        ax_simple.set_xlabel("Année", color='#f5f5f4')
        ax_simple.set_ylabel("Valeur ($)", color='#f5f5f4')
        ax_simple.legend()
        ax_simple.grid(True, linestyle=':', alpha=0.6)

        ax_simple.set_facecolor('#1e3a8a')
        ax_simple.tick_params(axis='x', colors='#f5f5f4')
        ax_simple.tick_params(axis='y', colors='#f5f5f4')
        ax_simple.spines['bottom'].set_color('#f5f5f4')
        ax_simple.spines['left'].set_color('#f5f5f4')
        fig_simple.patch.set_facecolor('#1e3a8a')
        for item in ([ax_simple.title, ax_simple.xaxis.label, ax_simple.yaxis.label] + ax_simple.get_xticklabels() + ax_simple.get_yticklabels()): item.set_color('#f5f5f4')
        
        st.pyplot(fig_simple)
        # -----------------------------------
