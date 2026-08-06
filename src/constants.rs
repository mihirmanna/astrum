//! Physical constants used in astrophysics calculations, following CODATA 2022 values.

/// Ideal gas constant measured in erg/(mol.K)
pub(crate) const R_GAS: f64 = 8.31446218e7;

/// Radiation density constant measured in erg/(cm^3.K^4)
pub(crate) const A_RAD: f64 = 7.5657333e-15;

/// Speed of light in vacuum measured in cm/s
pub(crate) const C: f64 = 2.99792458e10;

/// Newton's gravitational constant measured in dyn.cm^2/g^2
pub(crate) const G: f64 = 6.6743e-8;
