import pickle
from scripts.build_mesh_counts import generate_utility_matrix

counter_path = 'output/ctg-public-xml.ctdmc'
matrix_path = 'output/ctg-public-xml.utility'

print('Loading saved MeSH counter...')

with open(counter_path, 'rb') as f:
    ctdmc = pickle.load(f)

print('Documents:', ctdmc.num_processed_docs())
print('Generating utility matrix...')

generate_utility_matrix(ctdmc, matrix_path)

print('Utility matrix saved successfully:', matrix_path)