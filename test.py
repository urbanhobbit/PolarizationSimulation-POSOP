import numpy as np
from agents import Agents
from deliberation import deliberation_step_matched

N_AGENTS = 10
OPINION_SPACE_SIZE = 10
party_positions = np.array([[1, 1], [-1, -1], [0, 0]])
delta_matrix = np.ones((6, 6))  # Dummy delta matrix
agents = Agents(N_AGENTS, OPINION_SPACE_SIZE)
profiles = np.random.choice(6, N_AGENTS)
agents.pref_indices = profiles

positions, profiles = deliberation_step_matched(
    agents.positions, agents.pref_indices, delta_matrix, 1, party_positions,
    opinion_space_size=OPINION_SPACE_SIZE,
    mu_a=0.05, mu_r=1.4, discount_coeff=0.99, interaction_rate=1.0
)
print("Success:", positions.shape, profiles.shape)