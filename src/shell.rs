//! Stellar shell representation used to hold radius, pressure, luminosity, temperature, and
//! composition information at a specific mass coordinate.

use std::ops::{Add, Mul};
use thiserror::Error;

/// The chemical composition of a [`Shell`], described by the hydrogen fraction X, helium fraction
/// Y, and metals fraction Z.
#[derive(Clone, Copy)]
pub struct Composition {
    /// Hydrogen fraction
    pub(crate) x: f64,
    /// Helium fraction
    pub(crate) y: f64,
    /// Metals fraction
    pub(crate) z: f64,
}

/// Validation errors that may arise when constructing a new [`Composition`].
#[derive(Debug, Error)]
pub enum CompositionError {
    /// Element fractions do not sum to 1.0.
    #[error("Element fractions {x} + {y} + {z} != 1.0")]
    InvalidSum { x: f64, y: f64, z: f64 },

    /// At least one of the element fractions is negative.
    #[error("Negative element fraction {element} = {f}")]
    NegativeFraction { element: &'static str, f: f64 },
}

/// The physical state of a star at a given mass coordinate: enclosed radius, pressure, luminosity,
/// and temperature, plus the chemical composition.
///
/// These are accumulated quantities (e.g., `luminosity` is the total energy flux through the sphere
/// at this mass coordinate). See [`ShellDerivatives`] for the differential rates of each quantity.
#[derive(Clone, Copy)]
pub struct Shell {
    /// Radius of the shell enclosing this mass coordinate (cm).
    pub radius: f64,
    /// Pressure at this mass coordinate (dyn/cm^2).
    pub pressure: f64,
    /// Total luminosity carried outward through this mass coordinate (erg/s).
    pub luminosity: f64,
    /// Temperature at this mass coordinate (K).
    pub temperature: f64,
    /// Chemical composition at this mass coordinate, treated as constant.
    pub composition: Composition,
}

/// The derivative of a [`Shell`]'s physical quantities with respect to mass: `d(radius, pressure,
/// luminosity, temperature)/dm`, as computed by the stellar structure equations at a given mass
/// coordinate.
#[derive(Clone, Copy)]
pub(crate) struct ShellDerivatives {
    /// Derivative of the shell radius with respect to mass.
    pub(crate) dr_dm: f64,
    /// Derivative of the shell pressure with respect to mass.
    pub(crate) dp_dm: f64,
    /// Derivative of the shell luminosity with respect to mass.
    pub(crate) dl_dm: f64,
    /// Derivative of the shell temperature with respect to mass.
    pub(crate) dt_dm: f64,
}

/// Per-field absolute tolerance (atol) floors used alongside a relative tolerance to compute the
/// scaled error norm in the adaptive integration routine. Each field is in the same units as the
/// corresponding [`Shell`] field.
pub struct ShellTolerance {
    /// Absolute tolerance of the shell radius (cm).
    pub radius: f64,
    /// Absolute tolerance of the shell pressure (dyn/cm^2).
    pub pressure: f64,
    /// Absolute tolerance of the shell luminosity (erg/s).
    pub luminosity: f64,
    /// Absolute tolerance of the shell temperature (K).
    pub temperature: f64,
}

impl Composition {
    /// Returns a chemical composition with hydrogen fraction `x`, helium fraction `y`, and metals
    /// fraction `z`.
    ///
    /// # Errors
    ///
    /// An error is returned if the element fractions do not sum to 1.0, or at least one of the
    /// element fractions is negative.
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self, CompositionError> {
        for (element, f) in [("x", x), ("y", y), ("z", z)] {
            if f < 0.0 {
                return Err(CompositionError::NegativeFraction { element, f });
            }
        }

        let eps = 1e-6;
        if (x + y + z - 1.0).abs() > eps {
            Err(CompositionError::InvalidSum { x, y, z })
        } else {
            Ok(Self { x, y, z })
        }
    }

    /// The mean molecular weight corresponding to this composition in the fully ionized limit.
    pub(crate) fn mean_molecular_weight(&self) -> f64 {
        (2.0 * self.x + 0.75 * self.y + 0.5 * self.z).recip()
    }
}

/// Advances the quantities in a [`Shell`] by a differential amount. Composition remains unchanged,
/// since it isn't an integrated quantity.
impl Add<ShellDerivatives> for Shell {
    type Output = Self;
    fn add(self, rhs: ShellDerivatives) -> Self::Output {
        Self::Output {
            radius: self.radius + rhs.dr_dm,
            pressure: self.pressure + rhs.dp_dm,
            luminosity: self.luminosity + rhs.dl_dm,
            temperature: self.temperature + rhs.dt_dm,
            composition: self.composition,
        }
    }
}

/// Combines two derivatives, e.g. when summing weighted Runge-Kutta stages.
impl Add for ShellDerivatives {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            dr_dm: self.dr_dm + rhs.dr_dm,
            dp_dm: self.dp_dm + rhs.dp_dm,
            dl_dm: self.dl_dm + rhs.dl_dm,
            dt_dm: self.dt_dm + rhs.dt_dm,
        }
    }
}

/// Scales a derivative by a step size or weighting coefficient.
impl Mul<f64> for ShellDerivatives {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            dr_dm: self.dr_dm * rhs,
            dp_dm: self.dp_dm * rhs,
            dl_dm: self.dl_dm * rhs,
            dt_dm: self.dt_dm * rhs,
        }
    }
}

/// Scales a derivative by a step size or weighting coefficient (reverse order).
impl Mul<ShellDerivatives> for f64 {
    type Output = ShellDerivatives;
    fn mul(self, rhs: ShellDerivatives) -> Self::Output {
        Self::Output {
            dr_dm: self * rhs.dr_dm,
            dp_dm: self * rhs.dp_dm,
            dl_dm: self * rhs.dl_dm,
            dt_dm: self * rhs.dt_dm,
        }
    }
}
