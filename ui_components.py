import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import time
from scenarios import SCENARIOS
from config import OPINION_SPACE_SIZE
from excel_utils import export_to_excel

def setup_ui():
    st.sidebar.header("Simulation Settings")

    # Party positions
    party_input_mode = st.sidebar.radio(
        "How to define party positions?",
        options=["Preset Scenario", "Manual Entry"],
        index=0
    )

    if party_input_mode == "Preset Scenario":
        selected_scenario = st.sidebar.selectbox("Select Scenario", list(SCENARIOS.keys()))
        party_positions = np.array(SCENARIOS[selected_scenario]["party_positions"])
        N_PARTIES = len(party_positions)
    else:
        N_PARTIES = st.sidebar.slider("Number of Parties", 2, 5, 3)
        manual_positions = []
        for i in range(N_PARTIES):
            x = st.sidebar.number_input(f"Party {i+1} - X Coordinate", value=float(i*2-2), step=0.1)
            y = st.sidebar.number_input(f"Party {i+1} - Y Coordinate", value=0.0, step=0.1)
            manual_positions.append([x, y])
        party_positions = np.array(manual_positions)

    # Compute N_PROFILES
    N_PROFILES = math.factorial(N_PARTIES)

    # Delta matrix
    if party_input_mode == "Preset Scenario":
        delta_matrix = SCENARIOS[selected_scenario]["delta_matrix"]
        if delta_matrix.shape != (N_PROFILES, N_PROFILES):
            st.sidebar.warning(f"Selected scenario's delta_matrix is {delta_matrix.shape}, but {N_PARTIES} parties require a {N_PROFILES}x{N_PROFILES} matrix. Using default identity matrix.")
            delta_matrix = np.eye(N_PROFILES)
    else:
        st.sidebar.info("Since party positions are manually defined, a default identity delta matrix is used.")
        delta_matrix = np.eye(N_PROFILES)

    # Configuration display
    with st.sidebar.expander("🔍 View Selected Configuration"):
        st.subheader("Party Positions")
        party_positions_df = pd.DataFrame(
            party_positions,
            columns=["X", "Y"],
            index=[f"Party {i+1}" for i in range(N_PARTIES)]
        )
        st.dataframe(party_positions_df)

        st.subheader("Delta Matrix")
        delta_matrix_df = pd.DataFrame(
            delta_matrix,
            columns=[f"Profile {i}" for i in range(N_PROFILES)],
            index=[f"Profile {i}" for i in range(N_PROFILES)]
        )
        st.dataframe(delta_matrix_df)

    # Simulation parameters
    frame_duration = st.sidebar.slider("Animation Speed (ms)", 100, 2000, 500, step=100)
    N_AGENTS = st.sidebar.slider("Number of Agents", 50, 2000, 200, step=10)
    T = st.sidebar.slider("Number of Iterations", 50, 1000, 300, step=50)
    N_FRAMES = st.sidebar.slider("Number of Frames", 10, 100, 30, step=5)
    preference_update_mode = "dynamic"

    # Simulation controls
    progress_container = st.empty()
    status_container = st.empty()
    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ Start", key="start_simulation"):
        st.session_state['run_simulation'] = True
        st.session_state['cancel_simulation'] = False
        st.session_state['simulation_running'] = True
        progress_bar = progress_container.progress(0.0)
        status_container.write("Starting simulation...")
        # Simulate progress updates (actual progress handled in app_final_v06.py)
        progress_update_interval = max(1, T // 50)
        for t in range(1, T + 1):
            if t % progress_update_interval == 0 or t == T:
                progress_bar.progress(t / T)
                status_container.write(f"Running iteration {t} of {T}")
                time.sleep(0.05)
            if st.session_state.get('cancel_simulation', False):
                status_container.write("Simulation cancelled.")
                progress_bar.progress(0.0)
                st.session_state.simulation_running = False
                break
        if not st.session_state.get('cancel_simulation', False):
            status_container.write("Simulation completed!")

    if col2.button("⏹️ Stop/Cancel", key="stop_cancel_simulation"):
        if st.session_state.get('simulation_running', False):
            st.session_state['cancel_simulation'] = True
            status_container.write("Cancelling simulation...")
        else:
            st.session_state.positions_record = None
            st.session_state.party_positions = None
            progress_container.empty()
            status_container.empty()

    # Reset parameters
    if st.sidebar.button("🔄 Reset Parameters", key="reset_parameters"):
        st.session_state['N_AGENTS'] = 200
        st.session_state['T'] = 300
        st.session_state['N_FRAMES'] = 30
        st.session_state['frame_duration'] = 500
        st.cache_data.clear()
        st.rerun()

    # Results management
    with st.sidebar.expander("💾 Results Management"):
        if 'positions_record' in st.session_state and st.session_state.positions_record is not None:
            simulation_name = st.text_input("Simulation Name", value=f"Run {len(st.session_state['saved_simulations']) + 1}")
            if st.button("Save Results", key="save_results"):
                if simulation_name:
                    saved_simulation = {
                        "name": simulation_name,
                        "scenario": st.session_state.current_scenario,
                        "party_positions": st.session_state.party_positions,
                        "delta_matrix": st.session_state.current_delta_matrix,
                        "polarisation_df": st.session_state.polarisation_df,
                        "voting_df": st.session_state.voting_df,
                        "social_choice_df": st.session_state.social_choice_df
                    }
                    st.session_state['saved_simulations'].append(saved_simulation)
                    st.success(f"Saved simulation: {simulation_name}")
                else:
                    st.warning("Please enter a simulation name.")

        st.write(f"Saved Simulations: {len(st.session_state['saved_simulations'])}")
        if st.session_state['saved_simulations']:
            saved_names = [sim["name"] for sim in st.session_state['saved_simulations']]
            st.write("Saved Runs:", ", ".join(saved_names))
            st.info("Note: Excel sheet names are truncated to fit the 31-character limit (e.g., 'Scenario 1 - Ideal W_party_pos').")

            st.write("Select Simulations to Export:")
            selected_simulations = []
            for idx, sim in enumerate(st.session_state['saved_simulations']):
                if st.checkbox(f"Export '{sim['name']}'", value=True, key=f"export_sim_{idx}"):
                    selected_simulations.append(sim)

            if st.button("Export Selected Results", key="export_selected_results"):
                if not selected_simulations:
                    st.warning("Please select at least one simulation to export.")
                else:
                    excel_buffer = export_to_excel(selected_simulations)
                    if excel_buffer:
                        excel_buffer.seek(0)
                        st.download_button(
                            label="Download Selected Results",
                            data=excel_buffer,
                            file_name="selected_simulation_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_selected_results"
                        )

        st.write("---")
        st.write("Clear Saved Simulations")
        confirm_reset = st.checkbox("Confirm Clear All Simulations")
        if st.button("Clear Saved Simulations", key="clear_simulations"):
            if confirm_reset:
                st.session_state['saved_simulations'] = []
                st.success("All saved simulations have been cleared.")
            else:
                st.warning("Please confirm to clear all saved simulations.")

    # Documentation
    with st.sidebar.expander("📜 Documentation"):
        st.write("Download the user manual for this simulation dashboard.")
        try:
            with open("user_manual.md", "r") as file:
                manual_content = file.read()
            st.download_button(
                label="Download User Manual",
                data=manual_content,
                file_name="user_manual.md",
                mime="text/markdown",
                key="download_manual"
            )
        except FileNotFoundError:
            st.error("User manual not found. Please ensure 'user_manual.md' is in the project directory.")

        st.write("Download the technical document (DOCX) for detailed formulae and methodology.")
        try:
            with open("technical_document.docx", "rb") as docx_file:
                docx_data = docx_file.read()
            st.download_button(
                label="Download Technical Document (DOCX)",
                data=docx_data,
                file_name="technical_document.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_technical_doc"
            )
        except FileNotFoundError:
            st.warning("Technical document DOCX not found. Please ensure 'technical_document.docx' is in the project directory (/mount/src/polarizationsimulation/).")
            st.info("To generate the DOCX file:\n"
                    "1. Save the Markdown file 'technical_document.md' provided in the documentation.\n"
                    "2. Install Pandoc (download from pandoc.org/installing.html) and ensure it's added to your PATH.\n"
                    "3. Open a terminal in the project directory and run:\n"
                    "   `pandoc technical_document.md -o technical_document.docx --mathml`\n"
                    "4. The file 'technical_document.docx' will be created in the directory.\n"
                    "Alternatively, use an online Markdown-to-DOCX converter or manually convert the Markdown to DOCX in Microsoft Word, inserting equations as needed.")

    # Advanced controls
    with st.sidebar.expander("⚙️ Advanced Controls"):
        if st.button("🧹 Clear Cache", key="clear_cache"):
            st.cache_data.clear()
            st.rerun()

    return {
        'N_AGENTS': N_AGENTS,
        'T': T,
        'N_FRAMES': N_FRAMES,
        'party_positions': party_positions,
        'delta_matrix': delta_matrix,
        'N_PROFILES': N_PROFILES,
        'preference_update_mode': preference_update_mode,
        'frame_duration': frame_duration,
        'selected_scenario': selected_scenario if party_input_mode == "Preset Scenario" else "Manual"
    }

def render_visualizations(positions_record, polarisation_df, voting_df, social_choice_df, N_PARTIES, party_positions, frame_duration):
    # Create four tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Scatter Animation", "Polarisation Metrics", "Voting Results", "Social Choice Results"])

    # Define color palettes
    party_colors = px.colors.qualitative.Plotly[:N_PARTIES]  # For other tabs
    social_choice_colors = px.colors.qualitative.Set2[:N_PARTIES]  # Distinct colors for social choice tab

    with tab1:
        st.subheader("Scatter Animation of Agents, Party Positions, Party Centers, and Society Center")
        agent_frames, party_center_frames, society_center_frames = zip(*positions_record)
        positions_df = pd.concat(agent_frames, ignore_index=True)
        positions_df["FirstChoice"] = positions_df["FirstChoice"].astype(str)
        party_centers_df = pd.concat(party_center_frames, ignore_index=True)
        society_centers_df = pd.concat(society_center_frames, ignore_index=True)

        frames = []
        unique_iterations = sorted(positions_df['Iteration'].unique())

        initial_data = []
        for party in [str(i) for i in range(N_PARTIES)]:
            df_party = positions_df[positions_df["Iteration"] == unique_iterations[0]]
            df_party = df_party[df_party["FirstChoice"] == party]
            initial_data.append(go.Scatter(
                x=df_party["x"], y=df_party["y"],
                mode="markers",
                marker=dict(size=6, color=party_colors[int(party)], symbol="circle"),
                name=f"Party {int(party)+1} Agents"
            ))
        initial_data.append(go.Scatter(
            x=party_positions[:, 0], y=party_positions[:, 1],
            mode="markers",
            marker=dict(size=12, color=party_colors[:N_PARTIES], symbol="star", line=dict(width=2, color="black")),
            name="Party Positions"
        ))
        for i in range(N_PARTIES):
            center = party_centers_df[party_centers_df["Iteration"] == unique_iterations[0]]
            center = center[center["Party"] == str(i)]
            initial_data.append(go.Scatter(
                x=center["x"], y=center["y"],
                mode="markers",
                marker=dict(size=10, color=party_colors[i], symbol="diamond", line=dict(width=1, color="black")),
                name=f"Party {i+1} Center"
            ))
        df_society = society_centers_df[society_centers_df["Iteration"] == unique_iterations[0]]
        initial_data.append(go.Scatter(
            x=df_society["x"], y=df_society["y"],
            mode="markers",
            marker=dict(size=12, color="black", symbol="square", line=dict(width=2, color="white")),
            name="Society Center"
        ))

        for iteration in unique_iterations:
            frame_data = []
            df_iter = positions_df[positions_df['Iteration'] == iteration]
            for party in [str(i) for i in range(N_PARTIES)]:
                df_party = df_iter[df_iter["FirstChoice"] == party]
                frame_data.append(go.Scatter(
                    x=df_party["x"], y=df_party["y"],
                    mode="markers",
                    marker=dict(size=6, color=party_colors[int(party)], symbol="circle"),
                    name=f"Party {int(party)+1} Agents"
                ))
            frame_data.append(go.Scatter(
                x=party_positions[:, 0], y=party_positions[:, 1],
                mode="markers",
                marker=dict(size=12, color=party_colors[:N_PARTIES], symbol="star", line=dict(width=2, color="black")),
                name="Party Positions"
            ))
            for i in range(N_PARTIES):
                center = party_centers_df[party_centers_df["Iteration"] == iteration]
                center = center[center["Party"] == str(i)]
                frame_data.append(go.Scatter(
                    x=center["x"], y=center["y"],
                    mode="markers",
                    marker=dict(size=10, color=party_colors[i], symbol="diamond", line=dict(width=1, color="black")),
                    name=f"Party {i+1} Center"
                ))
            df_society = society_centers_df[society_centers_df["Iteration"] == iteration]
            frame_data.append(go.Scatter(
                x=df_society["x"], y=df_society["y"],
                mode="markers",
                marker=dict(size=12, color="black", symbol="square", line=dict(width=2, color="white")),
                name="Society Center"
            ))
            frames.append(go.Frame(
                data=frame_data,
                name=str(iteration),
                layout=go.Layout(title=f"Opinion Space: Iteration {iteration}")
            ))

        fig = go.Figure(data=initial_data, frames=frames)
        fig.update_layout(
            xaxis=dict(range=[-OPINION_SPACE_SIZE/2 - 0.5, OPINION_SPACE_SIZE/2 + 0.5], title="X Opinion"),
            yaxis=dict(range=[-OPINION_SPACE_SIZE/2 - 0.5, OPINION_SPACE_SIZE/2 + 0.5], title="Y Opinion"),
            width=800,
            height=800,
            title=f"Opinion Space: Iteration {unique_iterations[0]}",
            showlegend=True,
            shapes=[
                dict(
                    type="rect",
                    x0=-OPINION_SPACE_SIZE/2,
                    y0=-OPINION_SPACE_SIZE/2,
                    x1=OPINION_SPACE_SIZE/2,
                    y1=OPINION_SPACE_SIZE/2,
                    line=dict(color="black", width=2, dash="dash"),
                    layer="below"
                )
            ],
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "⏸️ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]
                    }
                ]
            }]
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Polarisation Metrics Over Time")
        fig = go.Figure()
        metrics = ["PartyPolarisation", "PrefPolarisation", "BinaryPolarisation", "KemenyPolarisation"]
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=polarisation_df["Iteration"],
                y=polarisation_df[metric],
                name=metric,
                visible=(metric == metrics[0])
            ))
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=metric,
                            method="update",
                            args=[{"visible": [m == metric for m in metrics]}]
                        ) for metric in metrics
                    ] + [dict(
                        label="All Metrics",
                        method="update",
                        args=[{"visible": [True] * len(metrics)}]
                    )],
                    direction="down",
                    showactive=True,
                )
            ],
            xaxis_title="Iteration",
            yaxis_title="Polarisation Value",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Party Support Over Time")
        fig = go.Figure()
        for i in range(N_PARTIES):
            party = f"Party {i+1}"
            col = i if i in voting_df.columns else None
            if col is not None:
                fig.add_trace(go.Scatter(
                    x=voting_df["Iteration"],
                    y=voting_df[i],
                    name=party,
                    line=dict(color=party_colors[i]),
                    visible=(i == 0)
                ))
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=f"Party {i+1}",
                            method="update",
                            args=[{"visible": [j == i for j in range(N_PARTIES)]}]
                        ) for i in range(N_PARTIES)
                    ] + [dict(
                        label="All Parties",
                        method="update",
                        args=[{"visible": [True] * N_PARTIES}]
                    )],
                    direction="down",
                    showactive=True,
                )
            ],
            xaxis_title="Iteration",
            yaxis_title="Voting Share",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Final Support Distribution")
        final_support = voting_df[voting_df["Iteration"] == voting_df["Iteration"].max()].drop(columns=["Iteration"]).T
        final_support.columns = ["Support"]
        final_support.index = [f"Party {i+1}" for i in range(N_PARTIES)]
        fig_bar = px.bar(
            final_support,
            x=final_support.index,
            y="Support",
            title="Final Party Support",
            color=final_support.index,
            color_discrete_sequence=party_colors[:N_PARTIES]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Preference Profile Distribution Over Time")
        pref_dist = positions_df.groupby(["Iteration", "PrefIndex"]).size().reset_index(name="Count")
        pref_dist["PrefIndex"] = pref_dist["PrefIndex"].astype(str)
        fig_pref = px.line(pref_dist, x="Iteration", y="Count", color="PrefIndex",
                           title="Preference Profile Support Over Time")
        st.plotly_chart(fig_pref, use_container_width=True)

    with tab4:
        st.subheader("Social Choice Results")

        # Step Plot for Winners Over Time
        winner_df = social_choice_df[["Iteration", "PluralityWinner", "BordaWinner", "MajCompWinner", "CopelandWinner"]]
        winner_df = winner_df.rename(columns={
            "PluralityWinner": "Plurality",
            "BordaWinner": "Borda",
            "MajCompWinner": "Maj. Comp.",
            "CopelandWinner": "Copeland"
        })
        fig_winners = go.Figure()
        rules = ["Plurality", "Borda", "Maj. Comp.", "Copeland"]
        for rule in rules:
            # Map winner indices to colors
            winner_colors = [social_choice_colors[int(w)] for w in winner_df[rule]]
            fig_winners.add_trace(go.Scatter(
                x=winner_df["Iteration"],
                y=winner_df[rule],
                name=rule,
                mode="lines",
                line=dict(shape="hv", width=2),
                marker=dict(color=winner_colors),
                visible=(rule == rules[0])
            ))
        fig_winners.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=rule,
                            method="update",
                            args=[{"visible": [r == rule for r in rules]}]
                        ) for rule in rules
                    ] + [dict(
                        label="All Rules",
                        method="update",
                        args=[{"visible": [True] * len(rules)}]
                    )],
                    direction="down",
                    showactive=True,
                )
            ],
            xaxis_title="Iteration",
            yaxis_title="Winning Party",
            yaxis=dict(tickvals=list(range(N_PARTIES)), ticktext=[f"Party {i+1}" for i in range(N_PARTIES)]),
            title="Social Choice Winners Over Time",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig_winners, use_container_width=True)

        # Heatmap for Scores Over Time
        st.subheader("Scores Across Rules and Parties Over Time")
        score_data = []
        for rule, score_prefix in [
            ("Plurality", "PluralityVotes"),
            ("Borda", "BordaScores"),
            ("Maj. Comp.", "MajCompScores"),
            ("Copeland", "CopelandScores")
        ]:
            for i in range(N_PARTIES):
                score_data.append({
                    "Rule": rule,
                    "Party": f"Party {i+1}",
                    "Iteration": social_choice_df["Iteration"],
                    "Score": social_choice_df[f"{score_prefix}{i}"]
                })
        score_df = pd.concat([pd.DataFrame(d) for d in score_data])
        score_pivot = score_df.pivot_table(index=["Rule", "Party"], columns="Iteration", values="Score")
        # Create custom colors for each Rule-Party combination
        y_labels = [f"{rule} - {party}" for rule, party in score_pivot.index]
        heatmap_colors = []
        for rule, party in score_pivot.index:
            party_idx = int(party.split()[-1]) - 1
            heatmap_colors.append(social_choice_colors[party_idx])
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=score_pivot.values,
            x=score_pivot.columns,
            y=y_labels,
            colorscale=[[0, "rgba(255,255,255,0.1)"], [1, "rgba(255,255,255,1)"]],
            zmin=score_pivot.values.min(),
            zmax=score_pivot.values.max(),
            showscale=True,
            text=score_pivot.values.round(2),
            texttemplate="%{text}",
            textfont=dict(color="black"),
            customdata=np.stack([np.full_like(score_pivot.values, c) for c in heatmap_colors], axis=-1),
            hovertemplate="Iteration: %{x}<br>Rule-Party: %{y}<br>Score: %{z}<br>Color: %{customdata}<extra></extra>"
        ))
        # Apply background colors to cells
        for i, color in enumerate(heatmap_colors):
            fig_heatmap.add_trace(go.Heatmap(
                z=np.full_like(score_pivot.values[i:i+1], 1),
                x=score_pivot.columns,
                y=[y_labels[i]],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                hoverinfo="skip"
            ))
        fig_heatmap.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Rule - Party",
            title="Score Heatmap",
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Interactive Bar Chart for Final Scores
        st.subheader("Final Scores Comparison Across Rules")
        final_scores = []
        for rule, score_prefix in [
            ("Plurality", "PluralityVotes"),
            ("Borda", "BordaScores"),
            ("Maj. Comp.", "MajCompScores"),
            ("Copeland", "CopelandScores")
        ]:
            for i in range(N_PARTIES):
                final_scores.append({
                    "Rule": rule,
                    "Party": f"Party {i+1}",
                    "Score": social_choice_df[social_choice_df["Iteration"] == social_choice_df["Iteration"].max()][f"{score_prefix}{i}"].iloc[0]
                })
        final_scores_df = pd.DataFrame(final_scores)
        fig_bar = px.bar(
            final_scores_df,
            x="Party",
            y="Score",
            color="Party",
            facet_col="Rule",
            title="Final Scores by Rule and Party",
            color_discrete_sequence=social_choice_colors[:N_PARTIES],
            height=400
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📥 Download Simulation Results"):
        excel_buffer_polar = export_to_excel([{"polarisation_df": polarisation_df}], single_df=True, sheet_name="polarisation")
        if excel_buffer_polar:
            excel_buffer_polar.seek(0)
            st.download_button(
                label="Download Polarisation Results",
                data=excel_buffer_polar,
                file_name="polarisation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_polarisation_results"
            )

        excel_buffer_vote = export_to_excel([{"voting_df": voting_df}], single_df=True, sheet_name="voting")
        if excel_buffer_vote:
            excel_buffer_vote.seek(0)
            st.download_button(
                label="Download Voting Results",
                data=excel_buffer_vote,
                file_name="voting_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_voting_results"
            )

        excel_buffer_social = export_to_excel([{"social_choice_df": social_choice_df}], single_df=True, sheet_name="social_choice")
        if excel_buffer_social:
            excel_buffer_social.seek(0)
            st.download_button(
                label="Download Social Choice Results",
                data=excel_buffer_social,
                file_name="social_choice_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_social_choice_results"
            )

def render_landing_page():
    st.markdown("""
    ### Welcome to the Polarisation Simulation Dashboard

    This tool allows you to simulate opinion dynamics in a 2D opinion space, where agents interact, form preferences, and evolve their opinions over time. Explore how polarization emerges, analyze voting behavior, and evaluate social choice outcomes through interactive visualizations.

    **Key Features:**
    - **Interactive Animations:** Watch agents move in a 2D opinion space, with party positions, centers, and the society center dynamically updated.
    - **Polarization Metrics:** Track Party, Preference, Binary, and Kemeny polarization over time.
    - **Voting and Social Choice Results:** Analyze party support, preference profiles, and winners under Plurality, Borda, Majority Comparison, and Copeland rules.
    - **Customizable Scenarios:** Choose from preset scenarios or define your own party positions.
    - **Export Options:** Save and export simulation results as Excel files for further analysis.
    """)

    st.markdown("""
    #### What to Expect

    After running a simulation, you'll see interactive visualizations like the one below:

    *(Preview of the Scatter Animation tab, showing agents, party positions, and centers in a 2D opinion space. Run a simulation to see the real animation with play/pause controls!)*

    **Note:** To include actual screenshots, you can use a tool like ScreenPal to capture the visualization tabs after running a simulation, then embed the images here.
    """)

    st.markdown("""
    #### Get Started Now!

    Run your first simulation to explore opinion dynamics in action! Configure the settings in the sidebar and click the "▶️ Start" button to begin.
    """)

    with st.expander("📖 Quick Start Guide"):
        st.markdown("""
        Follow these steps to get started:

        1. **Configure Settings:** Use the sidebar to select a scenario, define party positions, and adjust parameters like the number of agents (50-2000), iterations (50-1000), and animation speed.
        2. **Run the Simulation:** Click the "▶️ Start" button in the sidebar to launch the simulation.
        3. **Explore Results:** View the results in four interactive tabs:
           - *Scatter Animation:* See agents move in the opinion space.
           - *Polarisation Metrics:* Analyze polarization trends over time.
           - *Voting Results:* Review party support and preference profiles.
           - *Social Choice Results:* Examine winners and scores under different voting rules.
        4. **Save and Export:** Save your simulation in the "Results Management" section and export results as Excel files.
        5. **Need Help?** Download the user manual from the "Documentation" section in the sidebar.
        """)