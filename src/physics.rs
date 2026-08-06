//! Microphysics (equation of state, opacity, and energy generation) traits that specify
//! constitutive relations for the stellar structure equations.
//!
//! Users may define their own functions for each trait, or use one of the pre-built
//! implementations.

use crate::constants::*;
use crate::shell::{Composition, Shell};
use std::f64::consts::PI;

/// Relates pressure, density, and temperature for a given composition, and provides the
/// adiabatic temperature gradient used by the Schwarzschild convection criterion.
///
/// Implementations should be self-consistent: `density(pressure(ρ, T, c), T, c) = ρ`, since
/// the solver relies on inverting `pressure` via `density` to recover the state at each mass shell.
pub trait EquationOfState {
    /// Pressure as a function of density, temperature, and composition.
    fn pressure(&self, density: f64, temperature: f64, composition: Composition) -> f64;

    /// The adiabatic temperature gradient dln(T)/dln(P), used as the convective bound in the
    /// Schwarzschild criterion (see [`temperature_gradient`]).
    fn adiabatic_gradient(&self, density: f64, temperature: f64, composition: Composition) -> f64;

    /// Density as a function of pressure, temperature, and composition. Defaults to inverting
    /// [`Self::pressure`] by Newton's method. Override with a closed form where one exists (e.g.,
    /// [`IdealGasEos`]).
    fn density(&self, pressure: f64, temperature: f64, composition: Composition) -> f64 {
        const MAX_ITERS: usize = 50;
        const TOL: f64 = 1e-10;

        // Initial estimate: ideal gas approximation
        let mut rho = composition.mean_molecular_weight() * pressure / (R_GAS * temperature);

        for _ in 0..MAX_ITERS {
            let p_trial = self.pressure(rho, temperature, composition);
            let dp_drho = self.dpressure_drho(rho, temperature, composition);
            let delta = (p_trial - pressure) / dp_drho;
            rho -= delta;

            if (delta / rho).abs() < TOL {
                break;
            }
        }

        rho
    }

    /// ∂P/∂ρ at fixed T, used by the default Newton inversion in [`Self::density`]. Defaults to a
    /// central finite difference over [`Self::pressure`].
    fn dpressure_drho(&self, density: f64, temperature: f64, composition: Composition) -> f64 {
        let h = density * 1e-6;
        let left = self.pressure(density - h, temperature, composition);
        let right = self.pressure(density + h, temperature, composition);

        (right - left) / (2.0 * h)
    }
}

/// Relates opacity to density, temperature, and composition.
pub trait Opacity {
    /// Opacity as a function of density, temperature, and composition.
    fn opacity(&self, density: f64, temperature: f64, composition: Composition) -> f64;
}

/// Relates nuclear energy generation rate to density, temperature, and composition.
pub trait EnergyGeneration {
    /// Energy generation rate as a function of density, temperature, and composition.
    fn rate(&self, density: f64, temperature: f64, composition: Composition) -> f64;
}

/// Computes dln(T)/dln(P) at a shell using the Schwarzschild criterion: the smaller of the
/// radiative and [adiabatic][`EquationOfState::adiabatic_gradient`] gradients.
pub(crate) fn temperature_gradient<E: EquationOfState, O: Opacity>(
    shell: &Shell,
    density: f64,
    mass: f64,
    eos: &E,
    opacity: &O,
) -> f64 {
    let kappa = opacity.opacity(density, shell.temperature, shell.composition);
    let adiabatic_gradient = eos.adiabatic_gradient(density, shell.temperature, shell.composition);
    let radiative_gradient = 3.0 * kappa * shell.luminosity * shell.pressure
        / (16.0 * PI * A_RAD * C * G * mass * shell.temperature.powi(4));

    adiabatic_gradient.min(radiative_gradient)
}

/// A bundle of microphysics implementations (equation of state, opacity, and energy generation)
/// used to close the stellar structure equations.
pub struct Microphysics<E, O, N> {
    /// Equation of state relating pressure, density, and temperature.
    pub(crate) eos: E,
    /// Opacity law used in the radiative temperature gradient.
    pub(crate) opacity: O,
    /// Nuclear energy generation rate.
    pub(crate) nuclear: N,
}

impl<E: EquationOfState, O: Opacity, N: EnergyGeneration> Microphysics<E, O, N> {
    /// Bundles an equation of state, opacity law, and energy generation rate together.
    pub fn new(eos: E, opacity: O, nuclear: N) -> Self {
        Microphysics {
            eos,
            opacity,
            nuclear,
        }
    }

    /// Density at the given shell, per this bundle's equation of state.
    pub fn density(&self, shell: &Shell) -> f64 {
        self.eos
            .density(shell.pressure, shell.temperature, shell.composition)
    }
}

/// An ideal gas equation of state, assuming full ionization.
pub struct IdealGasEos {
    /// Adiabatic index
    gamma: f64,
}

impl IdealGasEos {
    /// Constructs an ideal gas EOS with adiabatic index `gamma`.
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl EquationOfState for IdealGasEos {
    fn pressure(&self, density: f64, temperature: f64, composition: Composition) -> f64 {
        (density * R_GAS * temperature) / composition.mean_molecular_weight()
    }

    fn adiabatic_gradient(
        &self,
        _density: f64,
        _temperature: f64,
        _composition: Composition,
    ) -> f64 {
        (self.gamma - 1.0) / self.gamma
    }

    fn density(&self, pressure: f64, temperature: f64, composition: Composition) -> f64 {
        composition.mean_molecular_weight() * pressure / (R_GAS * temperature)
    }
}

/// Kramers' opacity law, appropriate for bound-free and free-free absorption in moderately high
/// temperature stellar interiors.
pub struct KramersOpacity;

impl Opacity for KramersOpacity {
    fn opacity(&self, density: f64, temperature: f64, composition: Composition) -> f64 {
        let x = composition.x;
        let z = composition.z;
        4e25 * (1.0 + x) * z * density * temperature.powf(-3.5)
    }
}

/// The proton-proton chain, the dominant hydrogen-burning process in stars near or below the Sun's
/// mass.
pub struct PpChain;

impl EnergyGeneration for PpChain {
    /// Energy generation from the pp1 branch of the proton-proton chain. Follows the analytic fit
    /// of Kippenhahn & Weigert (2012, *Stellar Structure and Evolution*, §18.5.1).
    fn rate(&self, density: f64, temperature: f64, composition: Composition) -> f64 {
        let x = composition.x;
        let t9 = temperature / 1e9;

        let psi = 1.0; // Energy generation correction from pp2 and pp3 branches
        let f11 = 1.0; // Shielding factor
        let g11 = 1.0 + 3.82 * t9 + 1.51 * t9.powi(2) + 0.144 * t9.powi(3) - 0.0114 * t9.powi(4);

        2.57e4
            * psi
            * f11
            * g11
            * density
            * x
            * x
            * t9.powf(-2.0 / 3.0)
            * (-3.381 / t9.powf(1.0 / 3.0)).exp()
    }
}
