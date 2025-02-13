from src import MACE_MP, Pack, Smiles2Conformers, StructureOptimization

import zntrack

# Initialize the ZnTrack project
project = zntrack.Project()

# Define the MACE-MP model
model = MACE_MP()

# Build the workflow graph
with project:
    etoh = Smiles2Conformers(smiles="CCO", numConfs=32)
    box = Pack(data=[etoh.frames], counts=[32], density=789)
    optm = StructureOptimization(model=model, data=box.frames, data_id=-1, fmax=0.5)

# Execute the workflow
project.repro()
