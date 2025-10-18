import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load CSS
def load_css(file_name):
    """Charge CSS avec gestion d'erreurs"""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            st.sidebar.success(f"✅ CSS chargé depuis {file_name}")
    except FileNotFoundError:
        st.sidebar.error(f"❌ Fichier CSS introuvable : {file_name}")
        st.sidebar.info(f"📂 Dossier actuel : {os.getcwd()}")
        # CSS par défaut si fichier absent
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

# Données de ROI pour différents types de rénovations
RENOVATION_TYPES = {
    'Cuisine': {'roi': 0.85, 'cout_moyen': 25000},
    'Salle de bain': {'roi': 0.75, 'cout_moyen': 15000},
    'Sous-sol': {'roi': 0.65, 'cout_moyen': 35000},
    'Efficacité énergétique': {'roi': 0.80, 'cout_moyen': 10000},
    'Peinture et planchers': {'roi': 0.70, 'cout_moyen': 10000},
    'Agrandissement': {'roi': 0.55, 'cout_moyen': 75000},
    'Toiture ou fenêtres': {'roi': 0.70, 'cout_moyen': 20000}
}

# Chargement des fichiers CSV pour les données historiques
@st.cache_data
def load_real_estate_data():
    """Charge et nettoie les données immobilières avec gestion des doublons pour maisons et condos"""
    data_scenarios = {}
    
    try:
        df_maisons = pd.read_csv("prix_maisons_quebec_1970_2025.csv")
        data_scenarios['Maison'] = process_wide_csv_data(df_maisons)
    except FileNotFoundError:
        st.warning("prix_maisons_quebec_1970_2025.csv non trouvé. Utilisation de fallback pour Maison.")
        data_scenarios['Maison'] = {}
    
    try:
        df_condos = pd.read_csv("prix_condos_quebec_1970_2025.csv")
        data_scenarios['Condo'] = process_wide_csv_data(df_condos)
    except FileNotFoundError:
        st.warning("prix_condos_quebec_1970_2025.csv non trouvé. Utilisation de fallback pour Condo.")
        data_scenarios['Condo'] = {}
    
    return data_scenarios

def process_wide_csv_data(df):
    """Processus pour transformer le format large en format long avec scénarios"""
    regions = ['Montréal', 'Québec (RMR)', 'Gatineau', 'Sherbrooke', 'Trois-Rivières', 'Saguenay', 'Laval', 'Longueuil', 'Lévis', 'Sainte-Foy–Sillery–Cap-Rouge', 'Québec moyen']
    yoy_columns = [f'{r} - variation (%)' for r in regions]
    
    scenarios = {}
    for i, region in enumerate(regions):
        region_data = {}
        for _, row in df.iterrows():
            year = int(row['Année'])
            median_price = row[region]
            yoy = row[yoy_columns[i]] / 100 if pd.notna(row[yoy_columns[i]]) else 0.0
            
            if year not in region_data:
                region_data[year] = {}
            
            region_data[year]['average'] = {'yoy_increase': yoy, 'median_price': median_price}
            
            if year >= 2021:
                yoy_recent = yoy * 1.2
                median_price_recent = median_price * (1 + yoy_recent - yoy)
                region_data[year]['recent'] = {'yoy_increase': yoy_recent, 'median_price': median_price_recent}
        
        scenarios[region] = region_data
    
    for region in scenarios:
        if 2025 in scenarios[region]:
            scenarios[region]['valeur_2025_recent'] = scenarios[region][2025].get('recent', scenarios[region][2025]['average'])['median_price']
            scenarios[region]['valeur_2025_average'] = scenarios[region][2025]['average']['median_price']
        else:
            scenarios[region]['valeur_2025_recent'] = 0
            scenarios[region]['valeur_2025_average'] = 0
    
    return scenarios

data_scenarios = load_real_estate_data()

# Données fallback (TAL 2025 + Cap Rates Q3 2025)
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

# Données réelles pour taux d'appréciation annuel par secteur (basé sur recherches 2025)
region_yoy = {
    'Montréal': 0.05,
    'Québec (RMR)': 0.18,
    'Gatineau': 0.18,
    'Sherbrooke': 0.09,
    'Trois-Rivières': 0.09,
    'Saguenay': 0.09,
    'Laval': 0.05,
    'Longueuil': 0.05,
    'Lévis': 0.18,
    'Sainte-Foy–Sillery–Cap-Rouge': 0.18,
    'Québec moyen': 0.11
}

def estimer_valeur_actuelle_propriete(prix_achat, annee_achat, secteur, scenario='recent'):
    """Estime la valeur actuelle (2025) à partir du prix d'achat et de l'année avec données CSV"""
    type_propriete = st.session_state.type_propriete
    current_year = 2025
    
    if not isinstance(prix_achat, (int, float)) or prix_achat <= 0:
        st.error("Erreur : Prix d'achat invalide.")
        return 0.0, []
    if not isinstance(annee_achat, int) or annee_achat < 1970 or annee_achat > current_year:
        st.error(f"Erreur : Année d'achat {annee_achat} hors limites (1970-2025).")
        return prix_achat, []
    
    if type_propriete not in data_scenarios or secteur not in data_scenarios[type_propriete]:
        st.warning(f"Données non disponibles pour {type_propriete} à {secteur}, utilisation du taux d'appréciation par défaut.")
        yoy_default = region_yoy.get(secteur, 0.09 if type_propriete == 'Maison' else 0.05)
        valeur = prix_achat * (1 + yoy_default) ** (current_year - annee_achat)
        valeurs_historiques = [(annee_achat + i, prix_achat * (1 + yoy_default) ** i) for i in range(current_year - annee_achat + 1)]
        return round(valeur, 2), valeurs_historiques
    
    region_data = data_scenarios[type_propriete][secteur]
    numeric_years = [year for year in region_data if isinstance(year, int)]
    
    if not numeric_years:
        st.warning(f"Aucune année disponible pour {type_propriete} à {secteur}, utilisation du taux par défaut.")
        yoy_default = region_yoy.get(secteur, 0.09 if type_propriete == 'Maison' else 0.05)
        valeur = prix_achat * (1 + yoy_default) ** (current_year - annee_achat)
        valeurs_historiques = [(annee_achat + i, prix_achat * (1 + yoy_default) ** i) for i in range(current_year - annee_achat + 1)]
        return round(valeur, 2), valeurs_historiques
    
    if annee_achat < min(numeric_years) or annee_achat > current_year:
        st.warning(f"Année d'achat {annee_achat} hors des données disponibles, utilisation du taux par défaut.")
        yoy_default = region_yoy.get(secteur, 0.09 if type_propriete == 'Maison' else 0.05)
        valeur = prix_achat * (1 + yoy_default) ** (current_year - annee_achat)
        valeurs_historiques = [(annee_achat + i, prix_achat * (1 + yoy_default) ** i) for i in range(current_year - annee_achat + 1)]
        return round(valeur, 2), valeurs_historiques
    
    years_available = sorted([y for y in numeric_years if y >= annee_achat])
    
    if not years_available:
        st.warning("Aucune année disponible pour le calcul, retour du prix d'achat.")
        return prix_achat, []
    
    valeur = prix_achat
    valeurs_historiques = [(annee_achat, valeur)]
    
    for i in range(1, len(years_available)):
        year_prev = years_available[i-1]
        year_curr = years_available[i]
        yoy = region_data[year_prev].get(scenario, region_data[year_prev]['average'])['yoy_increase']
        valeur *= (1 + yoy)
        valeurs_historiques.append((year_curr, valeur))
    
    # Compléter jusqu'à 2025 si nécessaire
    if years_available and years_available[-1] < current_year:
        yoy_default = region_yoy.get(secteur, 0.09 if type_propriete == 'Maison' else 0.05)
        last_year = years_available[-1]
        for year in range(last_year + 1, current_year + 1):
            valeur *= (1 + yoy_default)
            valeurs_historiques.append((year, valeur))
    
    return round(valeur, 2), valeurs_historiques

def get_valeur_marche_2025(secteur, scenario='recent'):
    """Retourne la valeur médiane du marché en 2025 pour un secteur"""
    type_propriete = st.session_state.type_propriete
    if type_propriete in data_scenarios and secteur in data_scenarios[type_propriete]:
        return data_scenarios[type_propriete][secteur][f'valeur_2025_{scenario}']
    return fallback_data.get(type_propriete, fallback_data['Autre'])['median_price']

def calculateur_rentabilite_plex(valeur_actuelle, taux_appreciation, nb_annees, revenus_locatifs_annuels,
                                taux_augmentation_loyers, taxes_municipales, taxes_scolaires, assurances,
                                entretien, frais_gestion, autres_depenses, taux_inflation_depenses,
                                paiements_hypothecaires_annuels, ajustements_valeur=0.0, revenus_projected_annuels=None,
                                taux_actualisation=0.048, renovations=None):
    annees = np.arange(0, nb_annees + 1)
    valeurs = valeur_actuelle * (1 + taux_appreciation) ** annees
    prise_valeur_totale = valeurs[-1] - valeur_actuelle + ajustements_valeur
    
    revenus = np.full(nb_annees + 1, revenus_locatifs_annuels)
    if revenus_projected_annuels is not None and nb_annees >= 1:
        revenus[1:] = revenus_projected_annuels
    else:
        revenus = revenus_locatifs_annuels * (1 + taux_augmentation_loyers) ** annees
    
    depenses_initiales = taxes_municipales + taxes_scolaires + assurances + entretien + frais_gestion + autres_depenses
    depenses = depenses_initiales * (1 + taux_inflation_depenses) ** annees
    
    cash_flows = np.zeros(nb_annees + 1)
    for i in range(1, nb_annees + 1):
        cash_flows[i] = revenus[i] - depenses[i] - paiements_hypothecaires_annuels
    cash_flow_cumule = np.cumsum(cash_flows)
    
    rendement_total = cash_flow_cumule[-1] + prise_valeur_totale
    investissement_initial = valeur_actuelle
    roi_moyen_annuel = (rendement_total / investissement_initial / nb_annees) * 100 if nb_annees > 0 else 0
    
    npv_rendement = sum([cf / (1 + taux_actualisation)**t for t, cf in enumerate(cash_flows[1:], 1)]) + prise_valeur_totale / (1 + taux_actualisation)**nb_annees
    npv_roi = (npv_rendement / investissement_initial) * 100 if investissement_initial > 0 else 0
    
    # Calcul pour les rénovations si présentes
    cout_renovations = 0.0
    valeur_ajoutee_renovations = 0.0
    prise_valeur_renovations = 0.0
    renovations_details = []
    
    if renovations:
        for reno_type, reno_cout in renovations.items():
            if reno_type in RENOVATION_TYPES:
                roi = RENOVATION_TYPES[reno_type]['roi']
                valeur_ajoutee = reno_cout * roi
                prise_nette = valeur_ajoutee - reno_cout
                cout_renovations += reno_cout
                valeur_ajoutee_renovations += valeur_ajoutee
                prise_valeur_renovations += prise_nette
                renovations_details.append({
                    'type': reno_type,
                    'cout': reno_cout,
                    'valeur_ajoutee': valeur_ajoutee,
                    'prise_nette': prise_nette,
                    'roi': roi
                })
        # Ajuster la valeur finale avec les rénovations (ajoutée à la fin de la période)
        valeurs[-1] += valeur_ajoutee_renovations
        prise_valeur_totale += prise_valeur_renovations
        investissement_initial += cout_renovations
        rendement_total = cash_flow_cumule[-1] + prise_valeur_totale
        roi_moyen_annuel = (rendement_total / investissement_initial / nb_annees) * 100 if nb_annees > 0 else 0
        npv_rendement = sum([cf / (1 + taux_actualisation)**t for t, cf in enumerate(cash_flows[1:], 1)]) + prise_valeur_totale / (1 + taux_actualisation)**nb_annees
        npv_roi = (npv_rendement / investissement_initial) * 100 if investissement_initial > 0 else 0
    
    return {
        'annees': annees, 'valeurs': valeurs, 'revenus': revenus, 'depenses': depenses,
        'cash_flows': cash_flows, 'cash_flow_cumule': cash_flow_cumule,
        'prise_valeur_totale': prise_valeur_totale, 'rendement_total': rendement_total,
        'roi_moyen_annuel': roi_moyen_annuel, 'depenses_initiales': depenses_initiales,
        'npv_rendement': npv_rendement, 'npv_roi': npv_roi,
        'cout_renovations': cout_renovations,
        'valeur_ajoutee_renovations': valeur_ajoutee_renovations,
        'prise_valeur_renovations': prise_valeur_renovations,
        'renovations_details': renovations_details
    }

def simulation_probabiliste(valeur_actuelle, taux_appreciation, nb_annees, ajustements_valeur, ecart_type, renovations=None, n_simulations=1000):
    taux_simules = np.random.normal(taux_appreciation, ecart_type, n_simulations)
    valeurs_futures = valeur_actuelle * np.power(1 + taux_simules, nb_annees) + ajustements_valeur
    
    valeur_ajoutee_renovations = 0.0
    if renovations:
        for reno_type, reno_cout in renovations.items():
            if reno_type in RENOVATION_TYPES:
                roi = RENOVATION_TYPES[reno_type]['roi']
                valeur_ajoutee = reno_cout * roi
                valeur_ajoutee_renovations += valeur_ajoutee
    valeurs_futures += valeur_ajoutee_renovations
    
    prises_valeur_simulees = valeurs_futures - valeur_actuelle
    
    p10 = np.percentile(prises_valeur_simulees, 10)
    p50 = np.percentile(prises_valeur_simulees, 50)
    p90 = np.percentile(prises_valeur_simulees, 90)
    
    return prises_valeur_simulees, p10, p50, p90, valeurs_futures

# Gestion des étapes
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.type_propriete = None
    st.session_state.secteur = None
    st.session_state.params = {}

st.markdown('<div class="stHeader">🚀 Découvrez la Valeur de Votre Propriété en 2025 !</div>', unsafe_allow_html=True)

# Barre de progression
progress = st.session_state.step / 3 * 100
st.markdown(f'<div class="progress-bar"><div class="progress" style="width: {progress}%;">{st.session_state.step}/3</div></div>', unsafe_allow_html=True)

# Étape 1 : Sélection du type et du secteur
if st.session_state.step == 1:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.type_propriete = st.selectbox("Type de propriété", ['Plex/Multiplex', 'Maison', 'Condo', 'Terrain', 'Autre'], help="Choisissez votre type de bien pour une aventure unique !")
    with col2:
        secteurs_populaires = [
            'Montréal', 'Québec (RMR)', 'Gatineau', 'Sherbrooke', 'Trois-Rivières', 
            'Saguenay', 'Laval', 'Longueuil', 'Lévis', 'Sainte-Foy–Sillery–Cap-Rouge'
        ]
        st.session_state.secteur = st.selectbox("Secteur/Région (Top 10 Populaires)", secteurs_populaires + ['Québec moyen'], help="Où se trouve votre trésor ?")
    if st.button("Suivant"):
        st.session_state.step = 2
        st.session_state.params = {}
        st.rerun()

# Étape 2 : Paramètres spécifiques
elif st.session_state.step == 2:
    st.write(f"Entrez les détails pour votre {st.session_state.type_propriete.lower()}.")
    # Réinitialisation des paramètres spécifiques au mode
    if 'mode' not in st.session_state.params:
        st.session_state.params['mode'] = None
        st.session_state.params.pop('annee_achat', None)
        st.session_state.params.pop('prix_achat', None)
        st.session_state.params.pop('scenario', None)
        st.session_state.params.pop('valeur_actuelle', None)
        st.session_state.params.pop('nb_annees', None)
        st.session_state.params.pop('taux_appreciation', None)
        st.session_state.params.pop('taux_actualisation', None)
        st.session_state.params.pop('paiements_hypothecaires_annuels', None)
        st.session_state.params.pop('renovations', None)

    if st.session_state.type_propriete == 'Maison':
        mode_maison = st.radio("Mode d'évaluation", ['Estimation Rapide (Historique)', 'Valeur Marché 2025', 'Évaluation Avancée'], index=0, key='mode_maison', horizontal=True)
        st.session_state.params['mode'] = mode_maison
        if mode_maison == 'Estimation Rapide (Historique)':
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.params['annee_achat'] = st.number_input("Année d'achat", min_value=1970, max_value=2025, value=2000, step=1)
            with col2:
                st.session_state.params['prix_achat'] = st.number_input("Prix d'achat ($)", min_value=0.0, value=250000.0, step=5000.0)
            scenario = st.selectbox("Scénario", ['recent (taux projetés élevés)', 'average (moyenne historique)'], index=0)
            st.session_state.params['scenario'] = scenario.split()[0]  # Store 'recent' or 'average'
            # Aperçu rapide
            valeur_estimee, _ = estimer_valeur_actuelle_propriete(
                st.session_state.params['prix_achat'],
                st.session_state.params['annee_achat'],
                st.session_state.secteur,
                st.session_state.params['scenario']
            )
            st.info(f"📊 **Estimation rapide : ${valeur_estimee:,.0f}** en 2025")
        elif mode_maison == 'Valeur Marché 2025':
            scenario = st.selectbox("Scénario", ['recent', 'average'], index=0)
            st.session_state.params['scenario'] = scenario
            valeur_marche = get_valeur_marche_2025(st.session_state.secteur, scenario)
            st.success(f"🏠 **Prix médian {st.session_state.secteur} 2025 : ${valeur_marche:,.0f}**")
            st.session_state.params['valeur_actuelle'] = valeur_marche
            st.session_state.params['mode'] = 'marche'
        else:  # Évaluation Avancée
            st.session_state.params['valeur_actuelle'] = st.number_input("Valeur actuelle ($)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'median_price': 490000.0})['median_price']), step=1000.0)
            periode = st.selectbox("Période de Projection", ['Court terme (1-3 ans)', 'Moyen terme (4-7 ans)', 'Long terme (8+ ans)'], index=0)
            if periode == 'Court terme (1-3 ans)':
                st.session_state.params['nb_annees'] = st.number_input("Années de projection (1-3)", min_value=1, max_value=3, value=2, step=1, format="%d")
            elif periode == 'Moyen terme (4-7 ans)':
                st.session_state.params['nb_annees'] = st.number_input("Années de projection (4-7)", min_value=4, max_value=7, value=5, step=1, format="%d")
            else:
                st.session_state.params['nb_annees'] = st.number_input("Années de projection (8+)", min_value=8, value=10, step=1, format="%d")
            st.write(f"Taux d’appréciation annuel : {region_yoy.get(st.session_state.secteur, 0.09) * 100:.1f}% (données fiables 2025)")
            taux_actualisation_default = fallback_data.get(st.session_state.type_propriete, {'discount_rate': {'Autre (Québec moyen)': 0.043}})['discount_rate'].get(st.session_state.secteur, 0.043)
            st.write(f"Taux d'actualisation global : {taux_actualisation_default * 100:.1f}% (Cap Rate ajusté)")
            st.session_state.params['taux_appreciation'] = region_yoy.get(st.session_state.secteur, 0.09)
            st.session_state.params['taux_actualisation'] = taux_actualisation_default
            st.session_state.params['paiements_hypothecaires_annuels'] = st.number_input("Paiements hypothécaires annuels ($)", min_value=0.0, value=15000.0, step=100.0)
            # Intégration de la formule de prise de valeur après rénovation
            st.subheader("Rénovations prévues")
            renovations = {}
            selected_renovations = st.multiselect("Types de rénovations", list(RENOVATION_TYPES.keys()), help="Sélectionnez les rénovations prévues")
            for reno_type in selected_renovations:
                default_cost = RENOVATION_TYPES[reno_type]['cout_moyen']
                cout = st.number_input(f"Coût pour {reno_type} ($)", min_value=0.0, value=float(default_cost), step=1000.0, key=f"cout_{reno_type}")
                renovations[reno_type] = cout
            st.session_state.params['renovations'] = renovations
    elif st.session_state.type_propriete == 'Condo':
        mode_condo = st.radio("Mode d'évaluation", ['Estimation Rapide (Historique)', 'Valeur Marché 2025', 'Évaluation Avancée'], index=0, key='mode_condo', horizontal=True)
        st.session_state.params['mode'] = mode_condo
        if mode_condo == 'Estimation Rapide (Historique)':
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.params['annee_achat'] = st.number_input("Année d'achat", min_value=1970, max_value=2025, value=2000, step=1)
            with col2:
                st.session_state.params['prix_achat'] = st.number_input("Prix d'achat ($)", min_value=0.0, value=250000.0, step=5000.0)
            scenario = st.selectbox("Scénario", ['recent (taux projetés élevés)', 'average (moyenne historique)'], index=0)
            st.session_state.params['scenario'] = scenario.split()[0]  # Store 'recent' or 'average'
            # Aperçu rapide
            valeur_estimee, _ = estimer_valeur_actuelle_propriete(
                st.session_state.params['prix_achat'],
                st.session_state.params['annee_achat'],
                st.session_state.secteur,
                st.session_state.params['scenario']
            )
            st.info(f"📊 **Estimation rapide : ${valeur_estimee:,.0f}** en 2025")
        elif mode_condo == 'Valeur Marché 2025':
            scenario = st.selectbox("Scénario", ['recent', 'average'], index=0)
            st.session_state.params['scenario'] = scenario
            valeur_marche = get_valeur_marche_2025(st.session_state.secteur, scenario)
            st.success(f"🏠 **Prix médian {st.session_state.secteur} 2025 : ${valeur_marche:,.0f}**")
            st.session_state.params['valeur_actuelle'] = valeur_marche
            st.session_state.params['mode'] = 'marche'
        else:  # Évaluation Avancée
            st.session_state.params['valeur_actuelle'] = st.number_input("Valeur actuelle ($)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'median_price': 399900.0})['median_price']), step=1000.0)
            periode = st.selectbox("Période de Projection", ['Court terme (1-3 ans)', 'Moyen terme (4-7 ans)', 'Long terme (8+ ans)'], index=0)
            if periode == 'Court terme (1-3 ans)':
                st.session_state.params['nb_annees'] = st.number_input("Années de projection (1-3)", min_value=1, max_value=3, value=2, step=1, format="%d")
            elif periode == 'Moyen terme (4-7 ans)':
                st.session_state.params['nb_annees'] = st.number_input("Années de projection (4-7)", min_value=4, max_value=7, value=5, step=1, format="%d")
            else:
                st.session_state.params['nb_annees'] = st.number_input("Années de projection (8+)", min_value=8, value=10, step=1, format="%d")
            st.write(f"Taux d’appréciation annuel : {region_yoy.get(st.session_state.secteur, 0.05) * 100:.1f}% (données fiables 2025)")
            taux_actualisation_default = fallback_data.get(st.session_state.type_propriete, {'discount_rate': {'Autre (Québec moyen)': 0.045}})['discount_rate'].get(st.session_state.secteur, 0.045)
            st.write(f"Taux d'actualisation global : {taux_actualisation_default * 100:.1f}% (Cap Rate ajusté)")
            st.session_state.params['taux_appreciation'] = region_yoy.get(st.session_state.secteur, 0.05)
            st.session_state.params['taux_actualisation'] = taux_actualisation_default
            st.session_state.params['paiements_hypothecaires_annuels'] = st.number_input("Paiements hypothécaires annuels ($)", min_value=0.0, value=15000.0, step=100.0)
            # Intégration de la formule de prise de valeur après rénovation
            st.subheader("Rénovations prévues")
            renovations = {}
            selected_renovations = st.multiselect("Types de rénovations", list(RENOVATION_TYPES.keys()), help="Sélectionnez les rénovations prévues")
            for reno_type in selected_renovations:
                default_cost = RENOVATION_TYPES[reno_type]['cout_moyen']
                cout = st.number_input(f"Coût pour {reno_type} ($)", min_value=0.0, value=float(default_cost), step=1000.0, key=f"cout_{reno_type}")
                renovations[reno_type] = cout
            st.session_state.params['renovations'] = renovations
    elif st.session_state.type_propriete == 'Plex/Multiplex':
        periode = st.selectbox("Période de Projection", ['Court terme (1-3 ans)', 'Moyen terme (4-7 ans)', 'Long terme (8+ ans)'], index=0)
        if periode == 'Court terme (1-3 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (1-3)", min_value=1, max_value=3, value=2, step=1, format="%d")
        elif periode == 'Moyen terme (4-7 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (4-7)", min_value=4, max_value=7, value=5, step=1, format="%d")
        else:
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (8+)", min_value=8, value=10, step=1, format="%d")
        st.session_state.params['valeur_actuelle'] = st.number_input("Valeur actuelle ($)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'median_price': 650000.0})['median_price']), step=1000.0)
        st.session_state.params['taux_appreciation'] = st.number_input("Taux d’appréciation annuel (%)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'yoy': 0.11})['yoy'] * 100), step=0.1) / 100
        taux_actualisation_default = fallback_data.get(st.session_state.type_propriete, {'discount_rate': {'Autre (Québec moyen)': 0.048}})['discount_rate'].get(st.session_state.secteur, 0.048)
        st.session_state.params['taux_actualisation'] = st.number_input("Taux d'actualisation global (%) (Cap Rate ajusté)", min_value=0.0, value=taux_actualisation_default * 100, step=0.1) / 100
        st.session_state.params['paiements_hypothecaires_annuels'] = st.number_input("Paiements hypothécaires annuels ($)", min_value=0.0, value=15000.0, step=100.0)
        st.session_state.params['nb_logements'] = st.number_input("Nombre de logements", min_value=1, value=4, step=1, format="%d")
        st.session_state.params['loyers_actuels'] = [st.number_input(f"Logement {i+1} - Loyer actuel ($/mois)", min_value=0.0, value=1000.0, step=50.0) for i in range(st.session_state.params['nb_logements'])]
        if periode == 'Court terme (1-3 ans)':
            st.session_state.params['loyers_projectes'] = [st.number_input(f"Logement {i+1} - Loyer projeté ($/mois)", min_value=0.0, value=1100.0, step=50.0) for i in range(st.session_state.params['nb_logements'])]
        st.session_state.params['taux_augmentation_loyers'] = st.number_input("Augmentation loyers annuelle post-an 1 (%) (TAL 2025 : 5.9%)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'hausse_loyer': 0.059})['hausse_loyer'] * 100), step=0.1) / 100
        st.session_state.params['taxes_municipales'] = st.number_input("Taxes municipales ($)", min_value=0.0, value=4000.0, step=100.0)
        st.session_state.params['taxes_scolaires'] = st.number_input("Taxes scolaires ($)", min_value=0.0, value=800.0, step=100.0)
        st.session_state.params['assurances'] = st.number_input("Assurances ($)", min_value=0.0, value=1200.0, step=100.0)
        st.session_state.params['entretien'] = st.number_input("Entretien ($)", min_value=0.0, value=3000.0, step=100.0)
        st.session_state.params['frais_gestion'] = st.number_input("Gestion ($)", min_value=0.0, value=2000.0, step=100.0)
        st.session_state.params['autres_depenses'] = st.number_input("Autres ($)", min_value=0.0, value=1000.0, step=100.0)
        st.session_state.params['taux_inflation_depenses'] = st.number_input("Inflation dépenses (%)", min_value=0.0, value=2.0, step=0.1) / 100
        st.session_state.params['vacance_rate'] = st.number_input("Vacance locative (%)", min_value=0.0, value=5.0, step=0.1) / 100
    elif st.session_state.type_propriete == 'Terrain':
        periode = st.selectbox("Période de Projection", ['Court terme (1-3 ans)', 'Moyen terme (4-7 ans)', 'Long terme (8+ ans)'], index=0)
        if periode == 'Court terme (1-3 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (1-3)", min_value=1, max_value=3, value=2, step=1, format="%d")
        elif periode == 'Moyen terme (4-7 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (4-7)", min_value=4, max_value=7, value=5, step=1, format="%d")
        else:
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (8+)", min_value=8, value=10, step=1, format="%d")
        st.session_state.params['valeur_actuelle'] = st.number_input("Valeur actuelle ($)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'median_price': 150000.0})['median_price']), step=1000.0)
        st.session_state.params['taux_appreciation'] = st.number_input("Taux d’appréciation annuel (%)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'yoy': 0.08})['yoy'] * 100), step=0.1) / 100
        taux_actualisation_default = fallback_data.get(st.session_state.type_propriete, {'discount_rate': {'Autre (Québec moyen)': 0.053}})['discount_rate'].get(st.session_state.secteur, 0.053)
        st.session_state.params['taux_actualisation'] = st.number_input("Taux d'actualisation global (%) (Cap Rate ajusté)", min_value=0.0, value=taux_actualisation_default * 100, step=0.1) / 100
        st.session_state.params['paiements_hypothecaires_annuels'] = st.number_input("Paiements hypothécaires annuels ($)", min_value=0.0, value=5000.0, step=100.0)
        st.session_state.params['taxes_foncieres'] = st.number_input("Taxes foncières ($)", min_value=0.0, value=500.0, step=100.0)
        st.session_state.params['entretien'] = st.number_input("Entretien ($)", min_value=0.0, value=200.0, step=100.0)
        st.session_state.params['autres_depenses'] = st.number_input("Autres ($)", min_value=0.0, value=300.0, step=100.0)
        st.session_state.params['taux_inflation_depenses'] = st.number_input("Inflation dépenses (%)", min_value=0.0, value=2.0, step=0.1) / 100
    elif st.session_state.type_propriete == 'Autre':
        periode = st.selectbox("Période de Projection", ['Court terme (1-3 ans)', 'Moyen terme (4-7 ans)', 'Long terme (8+ ans)'], index=0)
        if periode == 'Court terme (1-3 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (1-3)", min_value=1, max_value=3, value=2, step=1, format="%d")
        elif periode == 'Moyen terme (4-7 ans)':
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (4-7)", min_value=4, max_value=7, value=5, step=1, format="%d")
        else:
            st.session_state.params['nb_annees'] = st.number_input("Années de projection (8+)", min_value=8, value=10, step=1, format="%d")
        st.session_state.params['valeur_actuelle'] = st.number_input("Valeur actuelle ($)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'median_price': 400000.0})['median_price']), step=1000.0)
        st.session_state.params['taux_appreciation'] = st.number_input("Taux d’appréciation annuel (%)", min_value=0.0, value=float(fallback_data.get(st.session_state.type_propriete, {'yoy': 0.07})['yoy'] * 100), step=0.1) / 100
        taux_actualisation_default = fallback_data.get(st.session_state.type_propriete, {'discount_rate': {'Autre (Québec moyen)': 0.048}})['discount_rate'].get(st.session_state.secteur, 0.048)
        st.session_state.params['taux_actualisation'] = st.number_input("Taux d'actualisation global (%) (Cap Rate ajusté)", min_value=0.0, value=taux_actualisation_default * 100, step=0.1) / 100
        st.session_state.params['paiements_hypothecaires_annuels'] = st.number_input("Paiements hypothécaires annuels ($)", min_value=0.0, value=15000.0, step=100.0)
        st.session_state.params['taxes_municipales'] = st.number_input("Taxes municipales ($)", min_value=0.0, value=3000.0, step=100.0)
        st.session_state.params['taxes_scolaires'] = st.number_input("Taxes scolaires ($)", min_value=0.0, value=600.0, step=100.0)
        st.session_state.params['assurances'] = st.number_input("Assurances ($)", min_value=0.0, value=1000.0, step=100.0)
        st.session_state.params['entretien'] = st.number_input("Entretien ($)", min_value=0.0, value=2000.0, step=100.0)
        st.session_state.params['autres_depenses'] = st.number_input("Autres ($)", min_value=0.0, value=500.0, step=100.0)
        st.session_state.params['taux_inflation_depenses'] = st.number_input("Inflation dépenses (%)", min_value=0.0, value=2.0, step=0.1) / 100
    if st.button("Suivant"):
        st.session_state.step = 3
        st.rerun()

# Étape 3 : Résultats
elif st.session_state.step == 3:
    st.write(f"Résultats pour votre {st.session_state.type_propriete.lower()} à {st.session_state.secteur}.")
    if st.session_state.type_propriete in ['Maison', 'Condo']:
        if 'annee_achat' in st.session_state.params and st.session_state.params['mode'] in ['Estimation Rapide (Historique)', 'Valeur Marché 2025']:
            scenario = st.session_state.params.get('scenario', 'recent')
            valeur_actuelle, valeurs_historiques = estimer_valeur_actuelle_propriete(
                st.session_state.params['prix_achat'],
                st.session_state.params['annee_achat'],
                st.session_state.secteur,
                scenario
            )
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"🎉 **Valeur estimée 2025** : ${valeur_actuelle:,.0f}")
            with col2:
                gain = ((valeur_actuelle / st.session_state.params['prix_achat']) - 1) * 100 if st.session_state.params['prix_achat'] > 0 else 0
                st.metric("💰 Gain Total", f"+{gain:.1f}%")
            st.write(f"📈 Basé sur données historiques réelles ({scenario}) pour {st.session_state.secteur}")
            if len(valeurs_historiques) > 1:
                years, values = zip(*valeurs_historiques)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(years, values, 'o-', color='#facc15', linewidth=3, markersize=6)
                ax.axhline(y=valeur_actuelle, color='red', linestyle='--', alpha=0.7, label=f'2025: ${valeur_actuelle:,.0f}')
                ax.set_title(f"Évolution de votre propriété - {st.session_state.secteur}", fontsize=16, color='white')
                ax.set_xlabel("Année", color='white')
                ax.set_ylabel("Valeur ($)", color='white')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.tick_params(colors='white')
                plt.tight_layout()
                st.pyplot(fig)
        elif st.session_state.params['mode'] == 'Évaluation Avancée':
            renovations = st.session_state.params.get('renovations', {})
            resultats = calculateur_rentabilite_plex(
                st.session_state.params['valeur_actuelle'],
                st.session_state.params['taux_appreciation'],
                st.session_state.params['nb_annees'],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                st.session_state.params['paiements_hypothecaires_annuels'],
                0.0,
                taux_actualisation=st.session_state.params['taux_actualisation'],
                renovations=renovations
            )
            prises_simulees, p10, p50, p90, valeurs_futures = simulation_probabiliste(
                st.session_state.params['valeur_actuelle'],
                st.session_state.params['taux_appreciation'],
                st.session_state.params['nb_annees'],
                0.0,
                0.025,
                renovations=renovations,
                n_simulations=1000
            )
            # Affichage de la valeur estimée à la fin de la période (incluant rénovations)
            valeur_estimee = resultats['valeurs'][-1]
            gain = ((valeur_estimee / st.session_state.params['valeur_actuelle']) - 1) * 100 if st.session_state.params['valeur_actuelle'] > 0 else 0
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"🎉 **Valeur estimée à la fin de la période** : ${valeur_estimee:,.0f}")
            with col2:
                st.metric("💰 Gain Total", f"+{gain:.1f}%")
            
            st.header("🌟 Résultats Globaux (Probabilistes)")
            st.success(f"**🎉 Focus Court Terme : Projection sur {st.session_state.params['nb_annees']} ans !**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Valeur Actuelle", f"${resultats['valeurs'][0]:,.0f}")
            col2.metric("Valeur Future Médiane", f"${np.median(valeurs_futures):,.0f}")
            col3.metric("Prise de Valeur Médiane", f"${p50:,.0f}")
            col4.metric("NPV ROI (Actualisé)", f"{resultats['npv_roi']:.2f}%")
            st.write(f"🔍 P10 (Pessimiste): ${p10:,.0f} | P90 (Optimiste): ${p90:,.0f}")
            st.write(f"💸 Cash Flow Cumulé: ${resultats['cash_flow_cumule'][-1]:,.0f}")
            st.write(f"🎯 ROI Brut Médian: {resultats['roi_moyen_annuel']:.2f}% | Taux Actualisation: {st.session_state.params['taux_actualisation']*100:.1f}%")
            
            # Afficher les détails des rénovations
            if resultats['renovations_details']:
                st.subheader("Détails des Rénovations")
                df_renovations = pd.DataFrame(resultats['renovations_details'])
                df_renovations['cout'] = df_renovations['cout'].map('{:,.0f}$'.format)
                df_renovations['valeur_ajoutee'] = df_renovations['valeur_ajoutee'].map('{:,.0f}$'.format)
                df_renovations['prise_nette'] = df_renovations['prise_nette'].map('{:,.0f}$'.format)
                df_renovations['roi'] = df_renovations['roi'].map('{:.1%}'.format)
                df_renovations.columns = ['Type', 'Coût ($)', 'Valeur ajoutée ($)', 'Prise nette ($)', 'ROI']
                st.table(df_renovations)
                st.write(f"**Total Coût des rénovations** : ${resultats['cout_renovations']:,.0f}")
                st.write(f"**Total Valeur ajoutée** : ${resultats['valeur_ajoutee_renovations']:,.0f}")
                st.write(f"**Total Prise nette des rénovations** : ${resultats['prise_valeur_renovations']:,.0f}")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(resultats['annees'], resultats['valeurs'], label='Valeur', color='#facc15')
            ax1.plot(resultats['annees'], resultats['cash_flow_cumule'], label='Cash Flow', color='#34d399')
            ax1.set_title("📈 Évolution Base", color='#f5f5f4')
            ax1.legend()
            ax1.grid(True)
            ax2.hist(prises_simulees, bins=50, alpha=0.7, color='#60a5fa')
            ax2.axvline(p50, color='red', linestyle='--', label=f'Médiane ${p50:,.0f}')
            ax2.axvline(p10, color='orange', linestyle='--', label=f'P10 ${p10:,.0f}')
            ax2.axvline(p90, color='green', linestyle='--', label=f'P90 ${p90:,.0f}')
            ax2.set_title("🎲 Simulations (1000)", color='#f5f5f4')
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig)
    elif st.session_state.type_propriete == 'Plex/Multiplex':
        revenus_locatifs_annuels = sum(st.session_state.params['loyers_actuels']) * 12
        revenus_projected_annuels = sum(st.session_state.params['loyers_projectes']) * 12 if 'loyers_projectes' in st.session_state.params else None
        ajustements_valeur = -revenus_locatifs_annuels * st.session_state.params['vacance_rate'] * st.session_state.params['nb_annees']
        resultats = calculateur_rentabilite_plex(
            st.session_state.params['valeur_actuelle'],
            st.session_state.params['taux_appreciation'],
            st.session_state.params['nb_annees'],
            revenus_locatifs_annuels,
            st.session_state.params['taux_augmentation_loyers'],
            st.session_state.params['taxes_municipales'],
            st.session_state.params['taxes_scolaires'],
            st.session_state.params['assurances'],
            st.session_state.params['entretien'],
            st.session_state.params['frais_gestion'],
            st.session_state.params['autres_depenses'],
            st.session_state.params['taux_inflation_depenses'],
            st.session_state.params['paiements_hypothecaires_annuels'],
            ajustements_valeur,
            revenus_projected_annuels=revenus_projected_annuels,
            taux_actualisation=st.session_state.params['taux_actualisation'],
            renovations=st.session_state.params.get('renovations', {})
        )
        prises_simulees, p10, p50, p90, valeurs_futures = simulation_probabiliste(
            st.session_state.params['valeur_actuelle'],
            st.session_state.params['taux_appreciation'],
            st.session_state.params['nb_annees'],
            ajustements_valeur,
            0.025,
            renovations=st.session_state.params.get('renovations', {}),
            n_simulations=1000
        )
        st.header("🌟 Résultats Globaux (Probabilistes)")
        st.success(f"**🎉 Focus Court Terme : Projection sur {st.session_state.params['nb_annees']} ans avec loyers projetés !**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valeur Actuelle", f"${resultats['valeurs'][0]:,.0f}")
        col2.metric("Valeur Future Médiane", f"${np.median(valeurs_futures):,.0f}")
        col3.metric("Prise de Valeur Médiane", f"${p50:,.0f}")
        col4.metric("NPV ROI (Actualisé)", f"{resultats['npv_roi']:.2f}%")
        st.write(f"🔍 P10 (Pessimiste): ${p10:,.0f} | P90 (Optimiste): ${p90:,.0f}")
        st.write(f"💸 Cash Flow Cumulé: ${resultats['cash_flow_cumule'][-1]:,.0f}")
        st.write(f"🎯 ROI Brut Médian: {resultats['roi_moyen_annuel']:.2f}% | Taux Actualisation: {st.session_state.params['taux_actualisation']*100:.1f}%")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(resultats['annees'], resultats['valeurs'], label='Valeur', color='#facc15')
        ax1.plot(resultats['annees'], resultats['cash_flow_cumule'], label='Cash Flow', color='#34d399')
        ax1.set_title("📈 Évolution Base", color='#f5f5f4')
        ax1.legend()
        ax1.grid(True)
        ax2.hist(prises_simulees, bins=50, alpha=0.7, color='#60a5fa')
        ax2.axvline(p50, color='red', linestyle='--', label=f'Médiane ${p50:,.0f}')
        ax2.axvline(p10, color='orange', linestyle='--', label=f'P10 ${p10:,.0f}')
        ax2.axvline(p90, color='green', linestyle='--', label=f'P90 ${p90:,.0f}')
        ax2.set_title("🎲 Simulations (1000)", color='#f5f5f4')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig)
    elif st.session_state.type_propriete == 'Terrain':
        resultats = calculateur_rentabilite_plex(
            st.session_state.params['valeur_actuelle'],
            st.session_state.params['taux_appreciation'],
            st.session_state.params['nb_annees'],
            0.0,
            0.0,
            st.session_state.params['taxes_foncieres'],
            0.0,
            0.0,
            st.session_state.params['entretien'],
            0.0,
            st.session_state.params['autres_depenses'],
            st.session_state.params['taux_inflation_depenses'],
            st.session_state.params['paiements_hypothecaires_annuels'],
            0.0,
            taux_actualisation=st.session_state.params['taux_actualisation'],
            renovations=st.session_state.params.get('renovations', {})
        )
        prises_simulees, p10, p50, p90, valeurs_futures = simulation_probabiliste(
            st.session_state.params['valeur_actuelle'],
            st.session_state.params['taux_appreciation'],
            st.session_state.params['nb_annees'],
            0.0,
            0.025,
            renovations=st.session_state.params.get('renovations', {}),
            n_simulations=1000
        )
        st.header("🌟 Résultats Globaux (Probabilistes)")
        st.success(f"**🎉 Focus Court Terme : Projection sur {st.session_state.params['nb_annees']} ans !**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valeur Actuelle", f"${resultats['valeurs'][0]:,.0f}")
        col2.metric("Valeur Future Médiane", f"${np.median(valeurs_futures):,.0f}")
        col3.metric("Prise de Valeur Médiane", f"${p50:,.0f}")
        col4.metric("NPV ROI (Actualisé)", f"{resultats['npv_roi']:.2f}%")
        st.write(f"🔍 P10 (Pessimiste): ${p10:,.0f} | P90 (Optimiste): ${p90:,.0f}")
        st.write(f"💸 Cash Flow Cumulé: ${resultats['cash_flow_cumule'][-1]:,.0f}")
        st.write(f"🎯 ROI Brut Médian: {resultats['roi_moyen_annuel']:.2f}% | Taux Actualisation: {st.session_state.params['taux_actualisation']*100:.1f}%")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(resultats['annees'], resultats['valeurs'], label='Valeur', color='#facc15')
        ax1.plot(resultats['annees'], resultats['cash_flow_cumule'], label='Cash Flow', color='#34d399')
        ax1.set_title("📈 Évolution Base", color='#f5f5f4')
        ax1.legend()
        ax1.grid(True)
        ax2.hist(prises_simulees, bins=50, alpha=0.7, color='#60a5fa')
        ax2.axvline(p50, color='red', linestyle='--', label=f'Médiane ${p50:,.0f}')
        ax2.axvline(p10, color='orange', linestyle='--', label=f'P10 ${p10:,.0f}')
        ax2.axvline(p90, color='green', linestyle='--', label=f'P90 ${p90:,.0f}')
        ax2.set_title("🎲 Simulations (1000)", color='#f5f5f4')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig)
    elif st.session_state.type_propriete == 'Autre':
        resultats = calculateur_rentabilite_plex(
            st.session_state.params['valeur_actuelle'],
            st.session_state.params['taux_appreciation'],
            st.session_state.params['nb_annees'],
            0.0,
            0.0,
            st.session_state.params['taxes_municipales'],
            st.session_state.params['taxes_scolaires'],
            st.session_state.params['assurances'],
            st.session_state.params['entretien'],
            0.0,
            st.session_state.params['autres_depenses'],
            st.session_state.params['taux_inflation_depenses'],
            st.session_state.params['paiements_hypothecaires_annuels'],
            0.0,
            taux_actualisation=st.session_state.params['taux_actualisation'],
            renovations=st.session_state.params.get('renovations', {})
        )
        prises_simulees, p10, p50, p90, valeurs_futures = simulation_probabiliste(
            st.session_state.params['valeur_actuelle'],
            st.session_state.params['taux_appreciation'],
            st.session_state.params['nb_annees'],
            0.0,
            0.025,
            renovations=st.session_state.params.get('renovations', {}),
            n_simulations=1000
        )
        st.header("🌟 Résultats Globaux (Probabilistes)")
        st.success(f"**🎉 Focus Court Terme : Projection sur {st.session_state.params['nb_annees']} ans !**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valeur Actuelle", f"${resultats['valeurs'][0]:,.0f}")
        col2.metric("Valeur Future Médiane", f"${np.median(valeurs_futures):,.0f}")
        col3.metric("Prise de Valeur Médiane", f"${p50:,.0f}")
        col4.metric("NPV ROI (Actualisé)", f"{resultats['npv_roi']:.2f}%")
        st.write(f"🔍 P10 (Pessimiste): ${p10:,.0f} | P90 (Optimiste): ${p90:,.0f}")
        st.write(f"💸 Cash Flow Cumulé: ${resultats['cash_flow_cumule'][-1]:,.0f}")
        st.write(f"🎯 ROI Brut Médian: {resultats['roi_moyen_annuel']:.2f}% | Taux Actualisation: {st.session_state.params['taux_actualisation']*100:.1f}%")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(resultats['annees'], resultats['valeurs'], label='Valeur', color='#facc15')
        ax1.plot(resultats['annees'], resultats['cash_flow_cumule'], label='Cash Flow', color='#34d399')
        ax1.set_title("📈 Évolution Base", color='#f5f5f4')
        ax1.legend()
        ax1.grid(True)
        ax2.hist(prises_simulees, bins=50, alpha=0.7, color='#60a5fa')
        ax2.axvline(p50, color='red', linestyle='--', label=f'Médiane ${p50:,.0f}')
        ax2.axvline(p10, color='orange', linestyle='--', label=f'P10 ${p10:,.0f}')
        ax2.axvline(p90, color='green', linestyle='--', label=f'P90 ${p90:,.0f}')
        ax2.set_title("🎲 Simulations (1000)", color='#f5f5f4')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig)
    if st.button("Recommencer"):
        st.session_state.step = 1
        st.session_state.type_propriete = None
        st.session_state.secteur = None
        st.session_state.params = {}
        st.rerun()
