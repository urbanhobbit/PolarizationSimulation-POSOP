# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import math
from agents import Agents
from scenarios import SCENARIOS
from config import *
from polarisation import calculate_polarisation_metrics, calculate_social_choice_winners
from deliberation import deliberation_step_matched
from ui_components import setup_ui, render_visualizations, render_landing_page
from excel_utils import export_to_excel

# --- Simulation Function ---
@st.cache_data
def run_simulation(N_AGENTS, T, N_FRAMES, party_positions, delta_matrix, N_PROFILES, preference_update_mode="dynamic"):
    np.random.seed(42)
    N_PARTIES = len(party_positions)
    expected_N_PROFILES = math.factorial(N_PARTIES)

    # Validate N_PROFILES consistency
    if N_PROFILES != expected_N_PROFILES:
        raise ValueError(f"N_PROFILES ({N_PROFILES}) does not match expected value ({expected_N_PROFILES}) for {N_PARTIES} parties.")

    # Validate delta_matrix shape before proceeding
    if delta_matrix.shape != (N_PROFILES, N_PROFILES):
        raise ValueError(f"delta_matrix shape {delta_matrix.shape} does not match required shape ({N_PROFILES}, {N_PROFILES}) for {N_PARTIES} parties.")

    agents = Agents(N_AGENTS, OPINION_SPACE_SIZE)
    profiles = np.random.choice(N_PROFILES, N_AGENTS)
    agents.pref_indices = profiles

    step_interval = max(1, T // N_FRAMES)
    record_iters = list(range(step_interval, T + 1, step_interval))

    positions_record = []
    polarisation_records = []
    voting_records = []
    social_choice_records = []

    for t in range(1, T + 1):
        if st.session_state.get('cancel_simulation', False):
            return None, None, None, None

        agents.positions, agents.pref_indices = deliberation_step_matched(
            agents.positions, agents.pref_indices, delta_matrix, t, party_positions,
            opinion_space_size=OPINION_SPACE_SIZE,
            mu_a=MU_ATTRACTION,
            mu_r=MU_REACTION,
            discount_coeff=DISCOUNT_COEFF,
            interaction_rate=1.0
        )

        if t in record_iters:
            preferences = agents.update_preferences(party_positions)
            first_choices = preferences[:, 0]
            pref_indices = agents.pref_indices

            frame = pd.DataFrame({
                "x": agents.positions[:, 0],
                "y": agents.positions[:, 1],
                "FirstChoice": first_choices,
                "PrefIndex": pref_indices,
                "Iteration": t
            })

            party_centers = []
            for i in range(N_PARTIES):
                supporters = agents.positions[first_choices == i]
                center = supporters.mean(axis=0) if len(supporters) > 0 else np.array([np.nan, np.nan])
                party_centers.append(center)

            society_center = agents.positions.mean(axis=0)

            frame_party_centers = pd.DataFrame(party_centers, columns=["x", "y"])
            frame_party_centers["Party"] = [str(i) for i in range(N_PARTIES)]
            frame_party_centers["Iteration"] = t

            frame_society_center = pd.DataFrame({
                "x": [society_center[0]],
                "y": [society_center[1]],
                "Iteration": [t]
            })

            try:
                party_polar, pref_polar, binary_polar, kemeny_polar = calculate_polarisation_metrics(
                    agents.positions, preferences, first_choices, party_positions, N_AGENTS, OPINION_SPACE_SIZE, agents.pref_indices
                )
            except TypeError:
                return None, None, None, None

            polarisation_records.append({
                "Iteration": t,
                "PartyPolarisation": party_polar,
                "PrefPolarisation": pref_polar,
                "BinaryPolarisation": binary_polar,
                "KemenyPolarisation": kemeny_polar
            })

            voting_share = pd.Series(first_choices).value_counts(normalize=True).sort_index()
            voting_records.append({"Iteration": t, **voting_share.to_dict()})

            social_choice_results = calculate_social_choice_winners(preferences, N_AGENTS, N_PARTIES)
            social_choice_record = {
                "Iteration": t,
                "PluralityWinner": social_choice_results["plurality_winner"],
                "BordaWinner": social_choice_results["borda_winner"],
                "MajCompWinner": social_choice_results["maj_comp_winner"],
                "CopelandWinner": social_choice_results["copeland_winner"]
            }
            for i in range(N_PARTIES):
                social_choice_record[f"PluralityVotes{i}"] = social_choice_results["plurality_votes"][i]
                social_choice_record[f"BordaScores{i}"] = social_choice_results["borda_scores"][i]
                social_choice_record[f"MajCompScores{i}"] = social_choice_results["maj_comp_scores"][i]
                social_choice_record[f"CopelandScores{i}"] = social_choice_results["copeland_scores"][i]
            social_choice_records.append(social_choice_record)

            positions_record.append((frame, frame_party_centers, frame_society_center))

    polarisation_df = pd.DataFrame(polarisation_records)
    voting_df = pd.DataFrame(voting_records).fillna(0)
    social_choice_df = pd.DataFrame(social_choice_records)

    return positions_record, polarisation_df, voting_df, social_choice_df

# --- Streamlit Setup ---
st.set_page_config(page_title="Polarisation Simulation", layout="wide")
st.title("🥉 Agent-Based Polarisation Simulation")
st.subheader("POlarization viewed from a SOcial choice Perspective")

# Initialize session state
if 'saved_simulations' not in st.session_state:
    st.session_state['saved_simulations'] = []

# Setup UI and get simulation parameters
simulation_params = setup_ui()

# Run simulation if triggered
if st.session_state.get('run_simulation', False):
    try:
        # Filter out parameters not needed by run_simulation
        sim_params = {k: v for k, v in simulation_params.items() if k in [
            'N_AGENTS', 'T', 'N_FRAMES', 'party_positions', 'delta_matrix', 
            'N_PROFILES', 'preference_update_mode'
        ]}
        positions_record, polarisation_df, voting_df, social_choice_df = run_simulation(**sim_params)
        if positions_record is not None:
            st.session_state.positions_record = positions_record
            st.session_state.polarisation_df = polarisation_df
            st.session_state.voting_df = voting_df
            st.session_state.social_choice_df = social_choice_df
            st.session_state.N_PARTIES = len(simulation_params['party_positions'])
            st.session_state.party_positions = simulation_params['party_positions']
            st.session_state.current_scenario = simulation_params['selected_scenario']
            st.session_state.current_delta_matrix = simulation_params['delta_matrix']
            st.session_state.frame_duration = simulation_params['frame_duration']
            st.session_state.simulation_running = False
    except ValueError as e:
        st.error(f"Simulation failed: {str(e)}")
        st.session_state.simulation_running = False

# Render visualizations or landing page
if 'positions_record' in st.session_state and st.session_state.positions_record is not None:
    render_visualizations(
        st.session_state.positions_record,
        st.session_state.polarisation_df,
        st.session_state.voting_df,
        st.session_state.social_choice_df,
        st.session_state.N_PARTIES,
        st.session_state.party_positions,
        st.session_state.get('frame_duration', 500)  # Default to 500 if not set
    )
else:
    render_landing_page()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; padding: 10px;'>"
    "developed by Emre Erdoğan, Pınar Uyan-Semerci, Ayça Ebru Giritligil, Onur Doğan & Giray Girengir-2025"
    "</div>",
    unsafe_allow_html=True
)