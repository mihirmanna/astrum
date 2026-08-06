//! Physical constants used in astrophysics calculations, following CODATA 2022 values.

/// Ideal gas constant measured in erg/(mol.K)
pub(crate) const R_GAS: f64 = 8.31446218e7;

/// Radiation density constant measured in erg/(cm^3.K^4)
pub(crate) const A_RAD: f64 = 7.5657333e-15;

/// Speed of light in vacuum measured in cm/s
pub(crate) const C: f64 = 2.99792458e10;

/// Newton's gravitational constant measured in dyn.cm^2/g^2
pub(crate) const G: f64 = 6.6743e-8;

/// Boltzmann constant measured in erg/K
pub(crate) const K_B: f64 = 1.380649e-16;

/// Planck constant measured in erg/Hz
pub(crate) const H_PLANCK: f64 = 6.62607015e-27;

/// Electron mass measured in g
pub(crate) const M_E: f64 = 9.1093837139e-28;

/// Avogadro number measured in 1/mol
pub(crate) const N_A: f64 = 6.02214076e23;

/// Hydrogen ionization energy measured in erg
pub(crate) const PHI_H: f64 = 2.179872361103e-11;
