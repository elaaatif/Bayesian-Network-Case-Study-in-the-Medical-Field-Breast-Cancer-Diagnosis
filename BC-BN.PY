import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
import re 

# Load the data from the CSV file into a pandas DataFrame
data = pd.read_csv(r"K:\STUDIES\----- M1 AI\S2\RC2\RC2-LAB\TP5\breast-cancer.csv")

# Define the structure
model = BayesianNetwork([
    ('age', 'menopause'),
    ('menopause', 'node_caps'),
    ('node_caps', 'inv_nodes'),
    ('inv_nodes', 'tumor_size'),
    ('tumor_size', 'deg_malig'),
    ('tumor_size', 'irradiat'),
    ('deg_malig', 'irradiat')
])

# Instantiate the ExpectationMaximization Estimator
estimator = ExpectationMaximization(model, data)

# Run the estimation
#estimator.fit()

# Create a directed graph from the model
graph = nx.DiGraph(model.edges())

# Plot the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(graph, seed=30)  # Set a random seed for consistent layout
nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=12, font_weight="bold", arrowsize=20)
plt.title("Bayesian Network Structure")
plt.show()

# Get the CPD values
cpd_values = estimator.get_parameters()

# Print the CPD values
for node, cpd in cpd_values.items():
    print("CPD for node:", node)
    print(cpd)


# Define some variables and evidence combinations
queries = [
    {'variables': ['tumor_size'], 'evidence': {'age': True, 'menopause': True}},
    {'variables': ['irradiat'], 'evidence': {'node_caps': True, 'inv_nodes': False}},
    {'variables': ['deg_malig'], 'evidence': {'age': False, 'tumor_size': True}},
    {'variables': ['irradiat'], 'evidence': {'node_caps': True}},
]

# Iterate over the queries and print the results
""" 
for query in queries:
    query_result = inference.query(**query)
    print("Inference result for variables {} with evidence {}: ".format(query['variables'], query['evidence']))
    print(query_result)

"""