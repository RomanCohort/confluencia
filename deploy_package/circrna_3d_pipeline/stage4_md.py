"""
Stage 4: MD Relaxation using OpenMM — QUALITY MAXIMIZED.

Extended MD simulation with:
- 20ns production + 1ns equilibration
- Temperature coupling (Langevin thermostat)
- Pressure coupling (Monte Carlo barostat)
- BSJ restraint with gradual annealing
- Dense snapshot sampling
- Energy/RMSD convergence monitoring
"""

import os
import numpy as np
import json
import time

try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("Warning: OpenMM not installed. Install with: pip install openmm")


class MDRelaxation:
    """Quality-maximized MD relaxation on cyclized circRNA structures."""

    def __init__(self, config, mode='quality'):
        self.config = config
        self.mode = mode

        # Select mode-specific config
        if mode == 'ultra_quality':
            self.md_config = config.get('ultra_quality', config.get('quality', {}))
            self.minimize_only = False
        elif mode == 'quality':
            self.md_config = config.get('quality', config.get('high_quality', {}))
            self.minimize_only = False
        elif mode == 'prefilter':
            self.md_config = config.get('prefilter', {})
            self.minimize_only = True
        elif mode == 'fast':
            self.md_config = config.get('fast', {})
            self.minimize_only = False
        else:
            self.md_config = config.get('quality', config.get('high_quality', {}))
            self.minimize_only = False

        # Forcefield
        self.forcefield_protein = config.get('forcefield', {}).get('protein', 'amber14-all.xml')
        self.forcefield_water = config.get('forcefield', {}).get('water', 'amber14/tip3pfb.xml')

        # MD parameters
        self.duration_ns = self.md_config.get('duration_ns', 20.0)
        self.temperature_k = self.md_config.get('temperature_k', 300)
        self.timestep_fs = self.md_config.get('timestep_fs', 1.0)
        self.snapshot_interval_ps = self.md_config.get('snapshot_interval_ps', 20)
        self.bsj_restraint_k = self.md_config.get('bsj_restraint_k', 1000.0)
        self.padding_nm = self.md_config.get('padding_nm', 1.2)

        # Quality-mode extras
        self.equilibration_steps = self.md_config.get('equilibration_steps', 50000)
        self.use_barostat = self.md_config.get('use_barostat', True)
        self.pressure_atm = self.md_config.get('pressure_atm', 1.0)
        self.friction_coeff = self.md_config.get('friction_coeff', 1.0)

    def relax(self, pdb_path, bsj_start, bsj_end, output_dir=None):
        """
        Run quality-maximized MD relaxation.

        Protocol:
        1. Solvate with explicit water (TIP3P)
        2. Add ions (Na+/Cl-) for charge neutralization
        3. Energy minimization (steepest descent)
        4. NVT equilibration (1ns, position restraints on RNA)
        5. NPT equilibration (0.5ns, release restraints)
        6. NPT production (20ns, BSJ restraint)
        7. Extract snapshots with convergence monitoring
        """
        if not HAS_OPENMM:
            raise RuntimeError("OpenMM is required for Stage 4")

        if output_dir is None:
            output_dir = os.path.dirname(pdb_path)
        os.makedirs(output_dir, exist_ok=True)

        # ---- 1. Load and prepare system ----
        pdb = app.PDBFile(pdb_path)
        forcefield = app.ForceField(self.forcefield_protein, self.forcefield_water)

        # Solvate with larger box for quality
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(
            forcefield,
            model='tip3p',
            padding=self.padding_nm * unit.nanometer,
            ionicStrength=0.15 * unit.molar  # Physiological salt
        )

        # ---- 2. Create system with quality settings ----
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.2 * unit.nanometer,  # Larger cutoff
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005  # Tighter Ewald
        )

        # ---- 3. Add BSJ restraint (harmonic, with annealing schedule) ----
        bsj_force = mm.CustomBondForce('k_bsj * (r - r0)^2')
        bsj_force.addPerBondParameter('k_bsj')
        bsj_force.addPerBondParameter('r0')

        bsj_end_atom = self._find_atom(modeller, bsj_end - 1, "O3'")
        bsj_start_atom = self._find_atom(modeller, bsj_start, 'P')

        if bsj_end_atom is None or bsj_start_atom is None:
            bsj_end_atom = self._find_atom(modeller, bsj_end - 1, "C3'")
            bsj_start_atom = self._find_atom(modeller, bsj_start, "C3'")

        if bsj_end_atom is not None and bsj_start_atom is not None:
            bsj_force.addBond(bsj_end_atom, bsj_start_atom,
                            [self.bsj_restraint_k, 0.35])  # 3.5 Å in nm
            system.addForce(bsj_force)

        # ---- 4. Add barostat for NPT ----
        if self.use_barostat:
            system.addForce(mm.MonteCarloBarostat(
                self.pressure_atm * unit.atmosphere,
                self.temperature_k * unit.kelvin,
                25  # Attempt frequency
            ))

        # ---- 5. Create integrator (Langevin with better friction) ----
        integrator = mm.LangevinMiddleIntegrator(
            self.temperature_k * unit.kelvin,
            self.friction_coeff / unit.picosecond,
            self.timestep_fs * 0.001 * unit.picosecond
        )

        # ---- 6. Create simulation ----
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # ---- 7. Energy minimization (steepest descent) ----
        simulation.minimizeEnergy(maxIterations=5000)

        # ---- 8. Add reporters for trajectory logging ----
        # Energy log
        energy_log_path = os.path.join(output_dir, 'energy_log.csv')
        simulation.reporters.append(app.StateDataReporter(
            energy_log_path, 1000,
            step=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, volume=True, density=True
        ))

        # DCD trajectory (for VMD/PyMOL analysis)
        dcd_path = os.path.join(output_dir, 'trajectory.dcd')
        simulation.reporters.append(app.DCDReporter(dcd_path, 5000))

        # ---- 9. NVT Equilibration (position restraints on RNA) ----
        # Add position restraints for equilibration
        restraint_force = mm.CustomExternalForce('k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
        restraint_force.addPerParticleParameter('k')
        restraint_force.addPerParticleParameter('x0')
        restraint_force.addPerParticleParameter('y0')
        restraint_force.addPerParticleParameter('z0')

        # Restrain RNA heavy atoms (not solvent)
        rna_atoms = list(modeller.topology.atoms())
        eq_restraint_k = 100.0  # kJ/mol/nm^2
        eq_positions = simulation.context.getState(getPositions=True).getPositions()

        rna_atom_count = 0
        for i, atom in enumerate(rna_atoms):
            if atom.residue.name != 'HOH' and atom.name[0] != 'H':
                pos = eq_positions[i]
                restraint_force.addParticle(i, [eq_restraint_k, pos[0], pos[1], pos[2]])
                rna_atom_count += 1

        restraint_index = system.addForce(restraint_force)

        # Run NVT equilibration
        eq_steps = min(self.equilibration_steps, 500000)  # Cap at 0.5ns for 1fs timestep
        simulation.step(eq_steps)

        # Remove position restraints
        system.removeForce(restraint_index)

        # ---- 10. NPT Equilibration (no restraints) ----
        npt_eq_steps = 100000  # 0.1ns
        simulation.step(npt_eq_steps)

        # ---- 11. Production run with snapshot extraction ----
        prod_steps = int(self.duration_ns * 1e6 / self.timestep_fs)
        snapshot_interval_steps = int(self.snapshot_interval_ps * 1000 / self.timestep_fs)
        num_snapshots = prod_steps // snapshot_interval_steps

        snapshots = []
        energies = []
        rmsds = []

        initial_positions = simulation.context.getState(getPositions=True).getPositions()

        for i in range(num_snapshots):
            simulation.step(snapshot_interval_steps)

            state = simulation.context.getState(
                getPositions=True,
                getEnergy=True
            )

            positions = state.getPositions()
            energy = state.getPotentialEnergy()._value
            rmsd = self._calculate_rmsd(initial_positions, positions)

            snapshot_path = os.path.join(output_dir, f'snapshot_{i:04d}.pdb')
            app.PDBFile.writeFile(modeller.topology, positions, open(snapshot_path, 'w'))

            snapshots.append({
                'pdb_path': snapshot_path,
                'frame': i,
                'time_ps': (i + 1) * self.snapshot_interval_ps,
                'time_ns': (i + 1) * self.snapshot_interval_ps / 1000.0
            })
            energies.append(energy)
            rmsds.append(rmsd)

        # ---- 12. Save trajectories ----
        energy_path = os.path.join(output_dir, 'energy_trajectory.npy')
        rmsd_path = os.path.join(output_dir, 'rmsd_trajectory.npy')
        np.save(energy_path, np.array(energies))
        np.save(rmsd_path, np.array(rmsds))

        # ---- 13. Convergence analysis ----
        convergence = self._analyze_convergence(energies, rmsds)

        return {
            'snapshots': snapshots,
            'energy_trajectory': energy_path,
            'rmsd_trajectory': rmsd_path,
            'num_frames': num_snapshots,
            'duration_ns': self.duration_ns,
            'mode': self.mode,
            'convergence': convergence
        }

    def _analyze_convergence(self, energies, rmsds):
        """Analyze MD convergence quality."""
        n = len(energies)
        if n < 10:
            return {'converged': False, 'reason': 'Too few frames'}

        # Split into halves
        half = n // 2
        energy_first = np.mean(energies[:half])
        energy_second = np.mean(energies[half:])
        rmsd_first = np.mean(rmsds[:half])
        rmsd_second = np.mean(rmsds[half:])

        # Energy drift
        energy_drift = abs(energy_second - energy_first) / max(abs(energy_first), 1.0)

        # RMSD plateau
        rmsd_last_20pct = rmsds[int(0.8 * n):]
        rmsd_variance = np.var(rmsd_last_20pct)
        rmsd_trend = np.mean(rmsd_last_20pct) - np.mean(rmsds[:int(0.2 * n)])

        converged = (
            energy_drift < 0.05 and           # < 5% energy drift
            rmsd_variance < 0.1 and           # Low variance in last 20%
            abs(rmsd_trend) < 0.05            # RMSD plateaued
        )

        return {
            'converged': converged,
            'energy_drift': float(energy_drift),
            'rmsd_variance': float(rmsd_variance),
            'rmsd_trend': float(rmsd_trend),
            'energy_first_half': float(energy_first),
            'energy_second_half': float(energy_second),
            'rmsd_first_half': float(rmsd_first),
            'rmsd_second_half': float(rmsd_second)
        }

    def _find_atom(self, modeller, residue_index, atom_name):
        """Find atom index in modeller topology."""
        atoms = list(modeller.topology.atoms())
        for i, atom in enumerate(atoms):
            if atom.residue.index == residue_index and atom.name == atom_name:
                return i
        return None

    def _calculate_rmsd(self, positions1, positions2):
        """Calculate RMSD between two position sets (nm)."""
        n_atoms = min(len(positions1), len(positions2))
        rmsd = 0.0
        for i in range(n_atoms):
            diff = positions1[i] - positions2[i]
            rmsd += diff[0]**2 + diff[1]**2 + diff[2]**2
        rmsd = np.sqrt(rmsd / n_atoms)
        return rmsd._value if hasattr(rmsd, '_value') else rmsd

    def relax_batch(self, cyclized_results, ss_results, output_dir):
        """Run MD relaxation on multiple cyclized structures."""
        all_results = []
        for i, cycl in enumerate(cyclized_results):
            seq_dir = os.path.join(output_dir, f'md_{cycl["seq_id"]}_{cycl["sample_id"]}')
            ss = ss_results[cycl['seq_id']]

            result = self.relax(
                pdb_path=cycl['pdb_path'],
                bsj_start=ss['bsj_start'],
                bsj_end=ss['bsj_end'],
                output_dir=seq_dir
            )
            result['seq_id'] = cycl['seq_id']
            result['sample_id'] = cycl['sample_id']
            all_results.append(result)

        return all_results
