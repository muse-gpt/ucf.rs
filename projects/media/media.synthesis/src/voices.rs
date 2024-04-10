//! One synthesis routine per family.
//!
//! Every voice is a pure function of `(spec, sample_rate)`: no clock, no
//! global RNG beyond the per-sound `Rng` stream, no lookahead. That is what
//! makes the catalogue reproducible. The constants below are the OpenSFX
//! values preserved one by one, and the per-family random-consumption order
//! (three draws per glitch grid cell, two per sparkle ping) is part of the
//! waveform contract — "optimising" any of it changes the samples and fails
//! the golden tests.

use media_dsp::{Biquad, Rng, env_ad, glide, pan_gains};
use media_types::Error;

use crate::spec::{Family, SfxSpec};

/// Renders `spec` into interleaved stereo float samples.
///
/// `sample_rate` is an explicit input contract (hertz, finite and positive):
/// the caller decides the rate, nothing is resampled behind this function's
/// back. Out-of-range specs fail with `Error::invalid_input`.
pub fn render(spec: &SfxSpec, sample_rate: f32) -> Result<Vec<[f32; 2]>, Error> {
    spec.validate()?;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(Error::invalid_input(format!("sfx sample rate {sample_rate} must be finite and positive")));
    }
    let frames = (spec.duration.max(0.01) * sample_rate).round().max(1.0) as usize;
    let mono = render_mono(spec, sample_rate, frames);
    let (left, right) = pan_gains(spec.pan);
    Ok(mono.into_iter().map(|sample| [sample * left * spec.intensity, sample * right * spec.intensity]).collect())
}

fn render_mono(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    match spec.family {
        Family::Whoosh => whoosh(spec, sample_rate, frames),
        Family::Impact => impact(spec, sample_rate, frames),
        Family::Riser => riser(spec, sample_rate, frames),
        Family::Tick => tick(spec, sample_rate, frames),
        Family::Glitch => glitch(spec, sample_rate, frames),
        Family::Sub => sub(spec, sample_rate, frames),
        Family::Ambience => ambience(spec, sample_rate, frames),
        Family::Sparkle => sparkle(spec, sample_rate, frames),
    }
}

/// Moves air: band-passed noise whose centre sweeps upward, then falls away.
fn whoosh(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let duration = spec.duration.max(0.01);
    let mut out = Vec::with_capacity(frames);
    let mut band = Biquad::band_pass(400.0 * spec.pitch, 1.1, sample_rate);
    let mut low = Biquad::low_pass(6000.0, 0.7, sample_rate);
    let mut noise = 0.0f32;
    for i in 0..frames {
        let t = i as f32 / sample_rate;
        let k = t / duration;
        // Re-create the filter every 64 samples so the sweep is audible without
        // recomputing coefficients per sample.
        if i % 64 == 0 {
            let centre = glide(k, 1.0, 320.0, 1900.0) * spec.pitch;
            band = Biquad::band_pass(centre, 1.1, sample_rate);
        }
        let white = rng.bipolar();
        // A touch of smoothed noise underneath keeps it from sounding thin.
        noise = noise * 0.995 + white * 0.005;
        let body = band.process(white) * (0.55 + spec.brightness * 0.6);
        let air = low.process(noise) * (0.5 - spec.brightness * 0.3);
        let envelope = env_ad(t, duration * 0.45, duration * 0.52);
        out.push((body + air) * envelope);
    }
    out
}

/// Lands a hit: sine drop plus a short filtered noise transient.
fn impact(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let duration = spec.duration.max(0.01);
    let mut low = Biquad::low_pass(900.0 + spec.brightness * 2600.0, 0.8, sample_rate);
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0f32;
    for i in 0..frames {
        let t = i as f32 / sample_rate;
        let frequency = glide(t, duration, 118.0, 36.0) * spec.pitch;
        phase = (phase + std::f32::consts::TAU * frequency / sample_rate) % std::f32::consts::TAU;
        let tone = phase.sin() * env_ad(t, 0.002, duration * 0.55);
        let crack = low.process(rng.bipolar()) * env_ad(t, 0.001, 0.09) * spec.brightness;
        out.push(tone * 0.85 + crack * 0.7);
    }
    out
}

/// Builds tension: pitch climbs and the noise floor rises with it.
fn riser(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let duration = spec.duration.max(0.01);
    let mut band = Biquad::band_pass(800.0, 0.9, sample_rate);
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0f32;
    for i in 0..frames {
        let t = i as f32 / sample_rate;
        let k = t / duration;
        if i % 64 == 0 {
            band = Biquad::band_pass(glide(k, 1.0, 420.0, 3200.0) * spec.pitch, 0.9, sample_rate);
        }
        let frequency = glide(k, 1.0, 160.0, 900.0) * spec.pitch;
        phase = (phase + std::f32::consts::TAU * frequency / sample_rate) % std::f32::consts::TAU;
        // A saw-ish stack reads as a rising tone rather than as one thin sine.
        let tone = phase.sin() * 0.35 + (phase * 2.0).sin() * 0.12;
        let hiss = band.process(rng.bipolar()) * (0.25 + spec.brightness * 0.5);
        let envelope = k.powf(1.6);
        let tail = if duration - t < 0.03 { ((duration - t) / 0.03).max(0.0) } else { 1.0 };
        out.push((tone + hiss) * envelope * tail);
    }
    out
}

/// Marks time: a very short click, bright by design.
fn tick(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let mut high = Biquad::high_pass(2600.0 * spec.pitch, 0.9, sample_rate);
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0f32;
    let frequency = 2400.0 * spec.pitch;
    for i in 0..frames {
        let t = i as f32 / sample_rate;
        phase = (phase + std::f32::consts::TAU * frequency / sample_rate) % std::f32::consts::TAU;
        let body = high.process(rng.bipolar()) * env_ad(t, 0.0005, 0.012);
        let tone = phase.sin() * env_ad(t, 0.0003, 0.008) * 0.5;
        out.push(body * 0.8 + tone);
    }
    out
}

/// Breaks the stream: gated, quantised noise bursts on a random grid.
fn glitch(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let mut band = Biquad::band_pass(1400.0 + spec.brightness * 2600.0, 0.8, sample_rate);
    let mut out = Vec::with_capacity(frames);
    let mut gate = 0.0f32;
    let mut held = 0.0f32;
    let mut countdown = 0u32;
    for _ in 0..frames {
        if countdown == 0 {
            countdown = (sample_rate * (0.02 + rng.unit() * 0.09)) as u32 + 1;
            gate = if rng.unit() > 0.35 { 1.0 } else { 0.0 };
            held = rng.bipolar();
        }
        countdown -= 1;
        let sample = if gate > 0.5 { (band.process(held) * 3.0).clamp(-1.0, 1.0).round() * 0.35 } else { 0.0 };
        out.push(sample);
    }
    out
}

/// Falls away: a low sine dropping out of hearing.
fn sub(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let duration = spec.duration.max(0.01);
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0f32;
    for i in 0..frames {
        let t = i as f32 / sample_rate;
        let frequency = glide(t, duration, 74.0, 28.0) * spec.pitch;
        phase = (phase + std::f32::consts::TAU * frequency / sample_rate) % std::f32::consts::TAU;
        out.push(phase.sin() * env_ad(t, 0.01, duration * 0.5));
    }
    out
}

/// Holds a room: low-passed noise with a slow swell, plus a distant tone.
fn ambience(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let duration = spec.duration.max(0.01);
    let mut low = Biquad::low_pass(700.0 + spec.brightness * 900.0, 0.6, sample_rate);
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0f32;
    for i in 0..frames {
        let t = i as f32 / sample_rate;
        let swell = 0.55 + 0.45 * (t * 0.35).sin();
        let noise = low.process(rng.bipolar()) * 0.16;
        phase = (phase + std::f32::consts::TAU * 110.0 * spec.pitch / sample_rate) % std::f32::consts::TAU;
        let tone = phase.sin() * 0.02 * (1.0 + (t * 0.6).sin());
        let edge = if t < 0.6 { t / 0.6 } else { ((duration - t) / 0.6).clamp(0.0, 1.0) };
        out.push((noise + tone) * swell * edge);
    }
    out
}

/// Scatters light: a handful of high pings on a pentatonic set.
fn sparkle(spec: &SfxSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let duration = spec.duration.max(0.01);
    const STEPS: [f32; 5] = [0.0, 2.0, 4.0, 7.0, 9.0];
    let mut pings: Vec<(f32, f32)> = Vec::new();
    for _ in 0..6 {
        let at = rng.unit() * duration * 0.85;
        let step = STEPS[(rng.unit() * STEPS.len() as f32) as usize % STEPS.len()];
        let frequency = 880.0 * (step / 12.0).exp2() * spec.pitch;
        pings.push((at, frequency));
    }
    let mut out = vec![0.0f32; frames];
    for (at, frequency) in pings {
        let start = (at * sample_rate) as usize;
        let mut phase = 0.0f32;
        for i in start..frames {
            let t = (i - start) as f32 / sample_rate;
            let envelope = env_ad(t, 0.002, 0.35);
            if t > 0.05 && envelope < 1e-4 {
                break;
            }
            phase = (phase + std::f32::consts::TAU * frequency / sample_rate) % std::f32::consts::TAU;
            out[i] += phase.sin() * envelope * 0.22;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{PRESETS, preset};

    /// FNV-1a over the raw little-endian f32 bytes of the interleaved buffer.
    /// A single flipped sample changes the digest, so this is the waveform
    /// lock for the byte-for-byte reproducibility contract.
    fn fnv1a64(frames: &[[f32; 2]]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for frame in frames {
            for sample in frame {
                for byte in sample.to_le_bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        h
    }

    #[derive(Clone, Copy)]
    struct Golden {
        frames: usize,
        digest: u64,
        sum: f64,
        peak: f32,
    }

    fn check(label: &str, got: &[[f32; 2]], want: Golden) {
        assert_eq!(got.len(), want.frames, "{label}: frame count");
        let mut sum: f64 = 0.0;
        let mut peak: f32 = 0.0;
        for frame in got {
            for sample in frame {
                assert!(sample.is_finite(), "{label}: non-finite sample {sample}");
                sum += f64::from(*sample);
                peak = peak.max(sample.abs());
            }
        }
        assert_eq!(fnv1a64(got), want.digest, "{label}: waveform digest");
        assert_eq!(sum, want.sum, "{label}: sample sum");
        assert_eq!(peak, want.peak, "{label}: peak");
    }

    const SR: f32 = 44_100.0;

    /// Golden digests are pinned on Windows MSVC float math.
    fn skip_unless_windows_golden(label: &str) -> bool {
        if cfg!(windows) {
            true
        } else {
            eprintln!("SKIP: {label} waveform golden is Windows-pinned");
            false
        }
    }

    /// Golden values captured from the OpenSFX core at its baseline HEAD
    /// (`6ecfb695`, pure xorshift32 stream, verbatim constants).
    #[test]
    fn every_family_matches_the_source_waveform() {
        if !skip_unless_windows_golden("every_family_matches_the_source_waveform") {
            return;
        }
        let cases = [
            (
                Family::Whoosh,
                Golden { frames: 26_460, digest: 0x6b10_3865_b612_7dfd, sum: 19.544_872_251_184_53, peak: 0.204_734_76 },
            ),
            (
                Family::Impact,
                Golden { frames: 26_460, digest: 0xdcb6_9391_5d4f_76e9, sum: 44.622_097_919_723_83, peak: 0.658_452_1 },
            ),
            (
                Family::Riser,
                Golden { frames: 26_460, digest: 0x67b3_b998_d4fd_4065, sum: 2.468_728_439_973_07, peak: 0.419_208_68 },
            ),
            (
                Family::Tick,
                Golden { frames: 26_460, digest: 0x090d_50bb_dc41_c4e9, sum: -0.754_346_335_420_101_9, peak: 0.696_040_5 },
            ),
            (
                Family::Glitch,
                Golden { frames: 26_460, digest: 0x88a5_ffd9_d099_f48d, sum: -0.494_974_732_398_986_8, peak: 0.247_487_37 },
            ),
            (
                Family::Sub,
                Golden { frames: 26_460, digest: 0x55b6_3f14_2cae_7959, sum: -34.712_438_108_681_07, peak: 0.705_396_6 },
            ),
            (
                Family::Ambience,
                Golden { frames: 26_460, digest: 0x0dc8_b1ff_7747_7c71, sum: 6.672_510_154_879_813, peak: 0.034_578_606 },
            ),
            (
                Family::Sparkle,
                Golden { frames: 26_460, digest: 0x1fca_0dca_5881_0141, sum: -1.145_603_428_921_045_7, peak: 0.410_707_62 },
            ),
        ];
        for (family, want) in cases {
            let got = render(&SfxSpec::new(family, 0.6), SR).expect("valid");
            check(family.as_str(), &got, want);
        }
    }

    #[test]
    fn every_preset_matches_the_source_waveform() {
        if !skip_unless_windows_golden("every_preset_matches_the_source_waveform") {
            return;
        }
        let wants: [Golden; 10] = [
            Golden { frames: 39_690, digest: 0x1cf7_6dfe_b3be_04ed, sum: -100.499_172_275_490_88, peak: 0.180_684_73 },
            Golden { frames: 30_870, digest: 0xab2b_0732_f960_248d, sum: -16.800_137_780_605_304, peak: 0.325_391_77 },
            Golden { frames: 61_740, digest: 0x95dd_c527_72c3_2889, sum: 61.082_310_168_931_5, peak: 0.619_919_84 },
            Golden { frames: 19_845, digest: 0x2ea3_7b95_5716_edd9, sum: 18.470_110_334_824_156, peak: 0.673_640_85 },
            Golden { frames: 264_600, digest: 0x529c_174a_02d9_69f5, sum: -3.818_406_189_539_782_5, peak: 0.550_550_16 },
            Golden { frames: 5_292, digest: 0x095f_15a3_49a7_4b29, sum: -0.715_116_534_979_912_7, peak: 0.718_693_85 },
            Golden { frames: 35_280, digest: 0xdb8f_6bc1_0e35_a9b5, sum: 2.969_848_394_393_921, peak: 0.247_487_37 },
            Golden { frames: 97_020, digest: 0x1292_5abd_4fb1_a48d, sum: -32.809_337_804_983_585, peak: 0.703_485_2 },
            Golden { frames: 352_800, digest: 0x81c6_6d1a_3e30_cbed, sum: -108.324_000_460_130_12, peak: 0.070_884_22 },
            Golden { frames: 70_560, digest: 0x1a7a_b4f2_26d8_c721, sum: 0.285_989_317_016_046_13, peak: 0.194_850_61 },
        ];
        for (entry, want) in PRESETS.iter().zip(&wants) {
            let got = render(&preset(entry.name).expect("preset"), SR).expect("valid");
            check(entry.name, &got, want.clone());
        }
    }

    #[test]
    fn variants_lock_pan_sample_rate_and_random_order() {
        if !skip_unless_windows_golden("variants_lock_pan_sample_rate_and_random_order") {
            return;
        }
        // Pan + intensity at a non-default rate: locks the stereo widening and
        // the explicit sample-rate contract.
        let mut whoosh = SfxSpec::new(Family::Whoosh, 0.9).with(23, 1.35, 0.8);
        whoosh.pan = -0.6;
        whoosh.intensity = 0.7;
        check(
            "whoosh@48k",
            &render(&whoosh, 48_000.0).expect("valid"),
            Golden { frames: 43_200, digest: 0xdac2_5d3c_6f1a_3687, sum: -24.718_436_625_818_434, peak: 0.264_277_85 },
        );
        // Second seeds for the two families whose RNG consumption order is the
        // contract: three draws per glitch grid cell, two per sparkle ping.
        check(
            "glitch@seed2",
            &render(&SfxSpec::new(Family::Glitch, 0.8).with(2, 1.0, 0.7), SR).expect("valid"),
            Golden { frames: 35_280, digest: 0xe180_5405_5fc3_ef55, sum: -3.959_797_859_191_894_5, peak: 0.247_487_37 },
        );
        check(
            "sparkle@seed2",
            &render(&SfxSpec::new(Family::Sparkle, 1.6).with(2, 1.1, 0.85), SR).expect("valid"),
            Golden { frames: 70_560, digest: 0xc51a_9a88_761d_6851, sum: -0.025_610_526_718_679_694, peak: 0.228_936_25 },
        );
        check(
            "impact@seed99",
            &render(&SfxSpec::new(Family::Impact, 1.4).with(99, 0.85, 0.25), SR).expect("valid"),
            Golden { frames: 61_740, digest: 0x202c_73b9_7c21_263d, sum: 61.009_690_255_940_214, peak: 0.608_944_06 },
        );
    }

    #[test]
    fn rng_stream_matches_the_source() {
        let mut rng = Rng::new(1);
        let stream: Vec<f32> = (0..1000).map(|_| rng.bipolar()).collect();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for value in &stream {
            for byte in value.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(h, 0x352a_6430_af0c_d3b4);
        assert_eq!(stream[0], -0.999_874_1);
        assert_eq!(stream[999], -0.874_290_94);
    }

    #[test]
    fn rendering_is_deterministic_per_seed() {
        let spec = SfxSpec::new(Family::Whoosh, 0.5).with(7, 1.0, 0.5);
        let a = render(&spec, SR).expect("valid");
        let b = render(&spec, SR).expect("valid");
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x == y));
    }

    #[test]
    fn different_seeds_change_the_noise() {
        let a = render(&SfxSpec::new(Family::Whoosh, 0.4).with(1, 1.0, 0.5), SR).expect("valid");
        let b = render(&SfxSpec::new(Family::Whoosh, 0.4).with(2, 1.0, 0.5), SR).expect("valid");
        assert!(a.iter().zip(&b).any(|(x, y)| x != y));
    }

    #[test]
    fn every_family_produces_audible_samples() {
        for family in [
            Family::Whoosh,
            Family::Impact,
            Family::Riser,
            Family::Tick,
            Family::Glitch,
            Family::Sub,
            Family::Ambience,
            Family::Sparkle,
        ] {
            let samples = render(&SfxSpec::new(family, 0.6), SR).expect("valid");
            let peak = samples.iter().flat_map(|s| s.iter()).fold(0.0f32, |a, b| a.max(b.abs()));
            assert!(peak > 0.02, "{family:?} too quiet: {peak}");
            assert!(peak <= 1.01, "{family:?} clipped: {peak}");
        }
    }

    #[test]
    fn duration_controls_length() {
        let samples = render(&SfxSpec::new(Family::Tick, 0.25), SR).expect("valid");
        assert_eq!(samples.len(), 11_025);
    }

    #[test]
    fn render_rejects_invalid_input() {
        let mut short = SfxSpec::new(Family::Tick, 0.5);
        short.duration = 0.005;
        assert_eq!(render(&short, SR).expect_err("duration below the contract").code(), "invalid_input");
        assert_eq!(render(&SfxSpec::new(Family::Tick, 0.5), 0.0).expect_err("zero rate").code(), "invalid_input");
        assert_eq!(render(&SfxSpec::new(Family::Tick, 0.5), f32::NAN).expect_err("nan rate").code(), "invalid_input");
    }
}
