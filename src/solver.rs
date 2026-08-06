//! Adaptive Cash-Karp integration of the stellar structure equations, from a starting [`Shell`] at
//! some mass coordinate towards a target mass.

use crate::constants::G;
use crate::physics::{
    EnergyGeneration, EquationOfState, Microphysics, Opacity, temperature_gradient,
};
use crate::shell::{Shell, ShellDerivatives, ShellTolerance};
use std::f64::consts::PI;
use thiserror::Error;

/// Errors that can occur while integrating the stellar structure equations.
#[derive(Debug, Error)]
pub enum IntegrationError {
    /// A shell's state became non-physical (negative or zero radius, pressure, etc.) during a step.
    ///
    /// This is expected occasionally, as the adaptive step-size controller probes step sizes that
    /// turn out to be too large. [`integrate`] treats it as a rejected step and retries with a
    /// smaller step, so this variant only appears when that retry loop is exhausted.
    #[error("Non-physical {field} = {value} at mass = {mass}")]
    NonPhysicalState {
        field: &'static str,
        value: f64,
        mass: f64,
    },

    /// The step size was shrunk below a configured minimum without producing an acceptable step.
    #[error("Step size collapsed to {h} at mass = {mass}")]
    StepSizeTooSmall { mass: f64, h: f64 },

    /// A single step was rejected and retried more times than allowed without converging on an
    /// acceptable step size.
    #[error("Exceeded max iterations at mass = {mass}")]
    MaxIterationsExceeded { mass: f64 },
}

/// Evaluates the stellar structure equations at a shell, returning d(radius, pressure,
/// luminosity, temperature)/dm.
///
/// # Errors
///
/// An error is returned if the shell's state is non-physical (negative or zero radius, pressure,
/// temperature, or density).
fn derivatives<E, O, N>(
    mass: f64,
    shell: Shell,
    physics: &Microphysics<E, O, N>,
) -> Result<ShellDerivatives, IntegrationError>
where
    E: EquationOfState,
    O: Opacity,
    N: EnergyGeneration,
{
    debug_assert!(mass > 0.0);
    if shell.radius <= 0.0 {
        return Err(IntegrationError::NonPhysicalState {
            field: "radius",
            value: shell.radius,
            mass,
        });
    }
    if shell.pressure <= 0.0 {
        return Err(IntegrationError::NonPhysicalState {
            field: "pressure",
            value: shell.pressure,
            mass,
        });
    }
    if shell.temperature <= 0.0 {
        return Err(IntegrationError::NonPhysicalState {
            field: "temperature",
            value: shell.temperature,
            mass,
        });
    }

    let density = physics
        .eos
        .density(shell.pressure, shell.temperature, shell.composition);
    if density <= 0.0 {
        return Err(IntegrationError::NonPhysicalState {
            field: "density",
            value: density,
            mass,
        });
    }

    let nuclear_rate = physics
        .nuclear
        .rate(density, shell.temperature, shell.composition);
    let grad = temperature_gradient(&shell, density, mass, &physics.eos, &physics.opacity);

    let dr_dm = (4.0 * PI * shell.radius * shell.radius * density).recip();
    let dp_dm = -G * mass / (4.0 * PI * shell.radius.powi(4));
    let dl_dm = nuclear_rate;
    let dt_dm =
        -G * mass * shell.temperature / (4.0 * PI * shell.radius.powi(4) * shell.pressure) * grad;

    Ok(ShellDerivatives {
        dr_dm,
        dp_dm,
        dl_dm,
        dt_dm,
    })
}

/// The result of a single Cash-Karp step: the 5th-order solution, and a scaled error estimate
/// comparing it against the embedded 4th-order solution. See [`step_error`].
struct StepResult {
    shell: Shell,
    error: f64,
}

/// Advances a shell at the given mass coordinate by one Cash-Karp step of size `h`, returning the
/// 5th-order solution along with a scaled error estimate used to accept or reject the step.
///
/// # Errors
///
/// An error is returned if any of the six stage evaluations hits a non-physical intermediate state.
/// The caller should treat this the same as an error estimate that's too large, i.e., shrink `h`
/// and retry.
fn step<E, O, N>(
    mass: f64,
    shell: Shell,
    h: f64,
    physics: &Microphysics<E, O, N>,
    rtol: f64,
    atol: &ShellTolerance,
) -> Result<StepResult, IntegrationError>
where
    E: EquationOfState,
    O: Opacity,
    N: EnergyGeneration,
{
    // Cash-Karp Butcher tableau values (Cash & Karp 1990)
    #[rustfmt::skip]
    let a = [
        [0.2,              0.0,           0.0,             0.0,                0.0,            0.0],
        [0.075,            0.225,         0.0,             0.0,                0.0,            0.0],
        [0.3,              -0.9,          1.2,             0.0,                0.0,            0.0],
        [-11.0 / 54.0,     2.5,           -70.0 / 27.0,    35.0 / 27.0,        0.0,            0.0],
        [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0, 0.0],
    ];
    // 5th-order solution weights
    let b5 = [
        37.0 / 378.0,
        0.0,
        250.0 / 621.0,
        125.0 / 594.0,
        0.0,
        512.0 / 1771.0,
    ];
    // 4th-order solution weights, used to estimate error against b5
    let b4 = [
        2825.0 / 27648.0,
        0.0,
        18575.0 / 48384.0,
        13525.0 / 55296.0,
        277.0 / 14336.0,
        0.25,
    ];
    let c = [0.0, 0.2, 0.3, 0.6, 1.0, 0.875];

    let deriv = |m: f64, y: Shell| derivatives(m, y, physics);
    let y0 = shell;

    // Compute the 6 stage derivatives k[0..6]. Stage i is evaluated at coordinate (mass + c[i] * h),
    // using a sum of the already-computed stages k[0..i] weighted by tableau row a[i-1].
    let mut k: Vec<ShellDerivatives> = Vec::with_capacity(6);
    for i in 0..6 {
        let y_stage = k[..i]
            .iter()
            .zip(&a[i.saturating_sub(1)]) // First stage (i = 0) is just y0
            .fold(y0, |acc, (&k_j, &a_j)| acc + k_j * a_j * h);

        let d = deriv(mass + c[i] * h, y_stage)?;
        k.push(d);
    }

    let y5 = (0..6).fold(y0, |acc, i| acc + k[i] * b5[i] * h);
    let y4 = (0..6).fold(y0, |acc, i| acc + k[i] * b4[i] * h);
    let err = step_error(y5, y4, y0, k[0], h, rtol, atol);

    Ok(StepResult {
        shell: y5,
        error: err,
    })
}

/// Computes a scaled error estimate comparing the 5th- and 4th-order Cash-Karp solutions.
///
/// Each of the four shell variables `v_i` is scaled by its own tolerance `atol_i + rtol * (typical
/// magnitude of v_i over this step)` before combining. This accounts for order-of-magnitude
/// differences between variables.
///
/// # Returns
///
/// The root-mean-square of the four scaled errors. A result of `1.0` means the step exactly meets
/// tolerance; [`integrate`] accepts steps at or below this threshold and rejects steps above it.
fn step_error(
    y5: Shell,
    y4: Shell,
    y0: Shell,
    k1: ShellDerivatives,
    h: f64,
    rtol: f64,
    atol: &ShellTolerance,
) -> f64 {
    // Scale to the larger of (1) the value at the start of the step, or (2) a one-step Euler
    // estimate of its value at the end of the step
    let scale = |y0_i: f64, k1_i: f64, atol_i: f64| {
        atol_i + rtol * f64::max(y0_i.abs(), (y0_i + k1_i * h).abs())
    };

    let sc_r = scale(y0.radius, k1.dr_dm, atol.radius);
    let sc_p = scale(y0.pressure, k1.dp_dm, atol.pressure);
    let sc_l = scale(y0.luminosity, k1.dl_dm, atol.luminosity);
    let sc_t = scale(y0.temperature, k1.dt_dm, atol.temperature);

    let e_r = (y5.radius - y4.radius) / sc_r;
    let e_p = (y5.pressure - y4.pressure) / sc_p;
    let e_l = (y5.luminosity - y4.luminosity) / sc_l;
    let e_t = (y5.temperature - y4.temperature) / sc_t;

    // Use RMS of the errors
    ((e_r * e_r + e_p * e_p + e_l * e_l + e_t * e_t) / 4.0).sqrt()
}

/// A solution to the stellar structure equations, represented as a sequence of [`Shell`] objects
/// and their corresponding mass coordinates.
pub struct StellarModel {
    /// Mass coordinates of this solution.
    pub masses: Vec<f64>,
    /// Shells associated to this solution's mass coordinates.
    pub shells: Vec<Shell>,
}

impl StellarModel {
    /// Returns a stellar model with empty mass and shell sequences.
    fn new() -> Self {
        StellarModel {
            masses: Vec::new(),
            shells: Vec::new(),
        }
    }

    /// Pushes the given mass coordinate and shell to this model.
    fn push(&mut self, mass: f64, shell: Shell) {
        self.masses.push(mass);
        self.shells.push(shell);
    }
}

/// Integrates the stellar structure equations from `initial_shell` at `initial_mass` towards
/// `final_mass`, using adaptive Cash-Karp step-size control.
///
/// Step size starts at `initial_step` and is adjusted based on the scaled error estimate from
/// [`step_error`]. Positive/negative step sizes are used to integrate outwards/inwards. Steps
/// producing a non-physical intermediate state (see [`derivatives`]) are treated the same as steps
/// exceeding error tolerance, i.e., retried with a smaller step.
///
/// # Errors
///
/// An error is returned if the step size falls below `MIN_STEP` without producing an acceptable
/// step, or if a single step is rejected more than `MAX_REJECTIONS` times without converging.
pub fn integrate<E, O, N>(
    initial_mass: f64,
    final_mass: f64,
    initial_step: f64,
    initial_shell: Shell,
    physics: &Microphysics<E, O, N>,
    rtol: f64,
    atol: &ShellTolerance,
) -> Result<StellarModel, IntegrationError>
where
    E: EquationOfState,
    O: Opacity,
    N: EnergyGeneration,
{
    const MIN_STEP: f64 = 1e10; // TODO: Make configurable? Depends on problem scale
    const MAX_REJECTIONS: usize = 50;

    // Integration direction: +1.0 for outward (increasing mass), -1.0 for inward
    let direction = (final_mass - initial_mass).signum();
    debug_assert_eq!(
        initial_step.signum(),
        direction,
        "Initial step must match integration direction"
    );

    let mut mass = initial_mass;
    let mut shell = initial_shell;
    let mut h = initial_step;
    let mut model = StellarModel::new();

    while (final_mass - mass) * direction > 0.0 {
        let mut rejections = 0;

        loop {
            // Don't overshoot near boundary (direction-aware)
            let remaining = final_mass - mass;
            let h_trial = if h.abs() < remaining.abs() {
                h
            } else {
                remaining
            };
            if h_trial.abs() < MIN_STEP {
                return Err(IntegrationError::StepSizeTooSmall { mass, h: h_trial });
            }

            match step(mass, shell, h_trial, physics, rtol, atol) {
                Ok(result) if result.error <= 1.0 => {
                    model.push(mass, shell);
                    shell = result.shell;
                    mass += h_trial;

                    // Raise to power 1/5 for RK4. Clamped to avoid oscillations
                    let growth = (1.0 / result.error.max(1e-10)).powf(0.2).min(10.0);
                    h = h_trial * growth;

                    break;
                }
                Ok(result) => {
                    // Clamped to avoid oscillations
                    let shrink = (1.0 / result.error).powf(0.25).max(0.1);
                    h = h_trial * shrink;
                }
                Err(_) => {
                    // Non-physical intermediate state, treat as oversized step
                    h *= 0.25;
                }
            }

            rejections += 1;
            if rejections > MAX_REJECTIONS {
                return Err(IntegrationError::MaxIterationsExceeded { mass });
            }
        }
    }

    model.push(mass, shell); // Final shell
    Ok(model)
}
