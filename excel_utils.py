import pandas as pd
import streamlit as st
import io

def export_to_excel(simulations, single_df=False, sheet_name=None):
    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            if single_df:
                # Export a single DataFrame (e.g., polarisation_df, voting_df)
                df = simulations[0][list(simulations[0].keys())[0]]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Export multiple simulations with metadata
                metadata_rows = []
                for sim in simulations:
                    base_name = sim['name'][:21]
                    metadata_rows.append({
                        "Simulation Name": sim["name"],
                        "Scenario": sim["scenario"]
                    })
                    party_positions_df = pd.DataFrame(
                        sim["party_positions"],
                        columns=["X", "Y"],
                        index=[f"Party {i+1}" for i in range(len(sim["party_positions"]))]
                    )
                    sheet_name = f"{base_name}_party_pos"
                    party_positions_df.to_excel(writer, sheet_name=sheet_name)
                    delta_matrix_df = pd.DataFrame(
                        sim["delta_matrix"],
                        columns=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])],
                        index=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])]
                    )
                    sheet_name = f"{base_name}_delta"
                    delta_matrix_df.to_excel(writer, sheet_name=sheet_name)
                    sim["polarisation_df"].to_excel(writer, sheet_name=f"{base_name}_polar", index=False)
                    sim["voting_df"].to_excel(writer, sheet_name=f"{base_name}_voting", index=False)
                    sim["social_choice_df"].to_excel(writer, sheet_name=f"{base_name}_social", index=False)
                pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)
        return excel_buffer
    except ModuleNotFoundError as e:
        if "xlsxwriter" in str(e):
            st.warning("The 'xlsxwriter' library is not installed. Attempting to use 'openpyxl' instead.")
            try:
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    if single_df:
                        df = simulations[0][list(simulations[0].keys())[0]]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        metadata_rows = []
                        for sim in simulations:
                            base_name = sim['name'][:21]
                            metadata_rows.append({
                                "Simulation Name": sim["name"],
                                "Scenario": sim["scenario"]
                            })
                            party_positions_df = pd.DataFrame(
                                sim["party_positions"],
                                columns=["X", "Y"],
                                index=[f"Party {i+1}" for i in range(len(sim["party_positions"]))]
                            )
                            sheet_name = f"{base_name}_party_pos"
                            party_positions_df.to_excel(writer, sheet_name=sheet_name)
                            delta_matrix_df = pd.DataFrame(
                                sim["delta_matrix"],
                                columns=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])],
                                index=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])]
                            )
                            sheet_name = f"{base_name}_delta"
                            delta_matrix_df.to_excel(writer, sheet_name=sheet_name)
                            sim["polarisation_df"].to_excel(writer, sheet_name=f"{base_name}_polar", index=False)
                            sim["voting_df"].to_excel(writer, sheet_name=f"{base_name}_voting", index=False)
                            sim["social_choice_df"].to_excel(writer, sheet_name=f"{base_name}_social", index=False)
                        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)
                return excel_buffer
            except ModuleNotFoundError:
                st.error("Neither 'xlsxwriter' nor 'openpyxl' is installed. Please install one of them to export Excel files.\n"
                         "Install xlsxwriter: `conda install xlsxwriter` or `pip install xlsxwriter`\n"
                         "Install openpyxl: `conda install openpyxl` or `pip install openpyxl`")
                return None
        else:
            st.error(f"Error during export: {str(e)}")
            return None