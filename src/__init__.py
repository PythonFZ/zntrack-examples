from dataclasses import dataclass
from pathlib import Path

import ase.io
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from rdkit2ase import pack, smiles2conformers

import zntrack


class Smiles2Conformers(zntrack.Node):
    smiles: str = zntrack.params()  # A required parameter
    numConfs: int = zntrack.params(32)  # A parameter with a default value

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")  # Output file path

    def run(self) -> None:
        frames = smiles2conformers(smiles=self.smiles, numConfs=self.numConfs)
        ase.io.write(self.frames_path, frames)

    @property
    def frames(self) -> list[ase.Atoms]:
        # Load the frames from the output file using the node's filesystem
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, ":", format="extxyz"))


class Pack(zntrack.Node):
    data: list[list[ase.Atoms]] = zntrack.deps()  # Input dependency (list of ASE Atoms)
    counts: list[int] = zntrack.params()  # Parameter (list of counts)
    density: float = zntrack.params()  # Parameter (density value)

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")  # Output file path

    def run(self) -> None:
        box = pack(data=self.data, counts=self.counts, density=self.density)
        ase.io.write(self.frames_path, box)

    @property
    def frames(self) -> list[ase.Atoms]:
        # Load the packed structure from the output file
        with self.state.fs.open(self.frames_path, "r") as f:
            return list(ase.io.iread(f, ":", format="extxyz"))


# We could hardcode the MACE_MP model into the StructureOptimization Node, but we
# can also define it as a dependency. Since the model doesn't require a `run` method,
# we define it as a `@dataclass`.


@dataclass
class MACE_MP:
    model: str = "medium"  # Default model type

    def get_calculator(self, **kwargs):
        return mace_mp(model=self.model)


class StructureOptimization(zntrack.Node):
    model: MACE_MP = zntrack.deps()  # Dependency (MACE_MP model)
    data: list[ase.Atoms] = zntrack.deps()  # Dependency (list of ASE Atoms)
    data_id: int = zntrack.params()  # Parameter (index of the structure to optimize)
    fmax: float = zntrack.params(0.05)  # Parameter (force convergence threshold)

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.traj")  # Output file path

    def run(self):
        atoms = self.data[self.data_id]
        atoms.calc = self.model.get_calculator()
        dyn = LBFGS(atoms, trajectory=self.frames_path.as_posix())
        dyn.run(fmax=0.5)

    @property
    def frames(self) -> list[ase.Atoms]:
        # Load the optimization trajectory from the output file
        with self.state.fs.open(self.frames_path, "rb") as f:
            return list(ase.io.iread(f, ":", format="traj"))
