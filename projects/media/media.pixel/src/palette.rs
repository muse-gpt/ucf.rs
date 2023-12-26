// 源自 OpenPixel（Apache-2.0, © OpenPixel Authors），索引语义逐值保持。

//! A palette of indexed colors. Index 0 is always transparent.

use media_types::Error;

/// An RGBA colour with 8-bit channels.
///
/// The layout matches the inner array of `geometry::Rgba`; the alias keeps
/// this crate's dependency list to the media base contract.
pub type Rgba = [u8; 4];

/// The colour every transparent read resolves to.
const TRANSPARENT: Rgba = [0, 0, 0, 0];

/// A colour table; sprites store indices into it.
#[derive(Clone, Debug)]
pub struct Palette {
    colors: Vec<Rgba>,
}

impl Default for Palette {
    /// A small, warm sixteen-colour set: enough for sprites and UI chips.
    fn default() -> Self {
        let colors: Vec<Rgba> = vec![
            [0, 0, 0, 0],
            [24, 26, 38, 255],
            [52, 58, 84, 255],
            [96, 106, 140, 255],
            [160, 172, 200, 255],
            [236, 240, 248, 255],
            [120, 40, 52, 255],
            [204, 68, 72, 255],
            [246, 132, 92, 255],
            [250, 196, 112, 255],
            [96, 156, 96, 255],
            [140, 204, 128, 255],
            [72, 132, 200, 255],
            [120, 186, 244, 255],
            [148, 112, 208, 255],
            [222, 168, 236, 255],
        ];
        Self { colors }
    }
}

impl Palette {
    #[must_use]
    pub fn from_colors(colors: Vec<Rgba>) -> Self {
        Self { colors }
    }

    /// Builds a palette from hex strings (`#rgb`, `#rrggbb`, `#rrggbbaa`).
    ///
    /// A malformed colour is an explicit [`Error::invalid_input`]; the
    /// OpenPixel binding layer's silent fallback to transparent is dropped
    /// on purpose so a colour typo never reads back as an eraser.
    pub fn from_hex<S: AsRef<str>>(hexes: &[S]) -> Result<Self, Error> {
        let mut colors = Vec::with_capacity(hexes.len());
        for hex in hexes {
            colors.push(parse_hex(hex.as_ref())?);
        }
        Ok(Self { colors })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Colour for an index; unknown indices fall back to transparent.
    ///
    /// This is the canvas read semantics (an index the palette never
    /// carried renders as transparent), not a parsing error; the write
    /// side [`Palette::set`] has its own silent contract. `from_colors`
    /// and `from_hex` do not force index 0 to be transparent — the rule
    /// is carried by the default table and the read fallback, exactly as
    /// in the absorbed implementation.
    #[must_use]
    pub fn color(&self, index: u8) -> Rgba {
        self.colors.get(index as usize).copied().unwrap_or(TRANSPARENT)
    }

    /// Replaces one slot; used for recolouring a sprite.
    ///
    /// Out-of-range slots are ignored silently — a deliberate second
    /// failure behaviour next to [`Palette::color`]'s transparent read
    /// fallback; the two must not be unified into one rule.
    pub fn set(&mut self, index: u8, color: Rgba) {
        if let Some(slot) = self.colors.get_mut(index as usize) {
            *slot = color;
        }
    }
}

/// Parses `#rgb`, `#rrggbb` and `#rrggbbaa` strings.
///
/// The accepted grammar is the one the OpenPixel binding used via
/// `geometry::parse`; only the failure mode changed from `None` to an
/// explicit input error.
pub fn parse_hex(input: &str) -> Result<Rgba, Error> {
    let invalid = || Error::invalid_input(format!("invalid hex colour `{input}`"));
    let hex = input.strip_prefix('#').ok_or_else(invalid)?;
    let nibble = |c: u8| -> Option<u32> {
        match c {
            b'0'..=b'9' => Some((c - b'0') as u32),
            b'a'..=b'f' => Some((c - b'a') as u32 + 10),
            b'A'..=b'F' => Some((c - b'A') as u32 + 10),
            _ => None,
        }
    };
    let byte = |i: usize, j: usize, double: bool| -> Option<u8> {
        let hi = nibble(*hex.as_bytes().get(i)?)?;
        if double {
            let lo = nibble(*hex.as_bytes().get(j)?)?;
            Some((hi * 16 + lo) as u8)
        } else {
            Some((hi * 17) as u8)
        }
    };
    let color = match hex.len() {
        3 => [
            byte(0, 0, false).ok_or_else(invalid)?,
            byte(1, 1, false).ok_or_else(invalid)?,
            byte(2, 2, false).ok_or_else(invalid)?,
            255,
        ],
        6 => [
            byte(0, 1, true).ok_or_else(invalid)?,
            byte(2, 3, true).ok_or_else(invalid)?,
            byte(4, 5, true).ok_or_else(invalid)?,
            255,
        ],
        8 => [
            byte(0, 1, true).ok_or_else(invalid)?,
            byte(2, 3, true).ok_or_else(invalid)?,
            byte(4, 5, true).ok_or_else(invalid)?,
            byte(6, 7, true).ok_or_else(invalid)?,
        ],
        _ => Err(invalid())?,
    };
    Ok(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_table_is_the_warm_sixteen() {
        let palette = Palette::default();
        assert_eq!(palette.len(), 16);
        let golden: Vec<Rgba> = vec![
            [0, 0, 0, 0],
            [24, 26, 38, 255],
            [52, 58, 84, 255],
            [96, 106, 140, 255],
            [160, 172, 200, 255],
            [236, 240, 248, 255],
            [120, 40, 52, 255],
            [204, 68, 72, 255],
            [246, 132, 92, 255],
            [250, 196, 112, 255],
            [96, 156, 96, 255],
            [140, 204, 128, 255],
            [72, 132, 200, 255],
            [120, 186, 244, 255],
            [148, 112, 208, 255],
            [222, 168, 236, 255],
        ];
        for (index, expected) in golden.iter().enumerate() {
            assert_eq!(palette.color(index as u8), *expected, "slot {index}");
        }
    }

    #[test]
    fn index_zero_of_the_default_table_is_transparent() {
        assert_eq!(Palette::default().color(0), [0, 0, 0, 0]);
    }

    #[test]
    fn unknown_index_reads_transparent() {
        let palette = Palette::default();
        assert_eq!(palette.color(200), [0, 0, 0, 0]);
        assert_eq!(Palette::from_colors(vec![]).color(0), [0, 0, 0, 0]);
    }

    #[test]
    fn out_of_range_slot_set_is_ignored() {
        let mut palette = Palette::from_colors(vec![[1, 2, 3, 255]]);
        palette.set(200, [9, 9, 9, 9]);
        assert_eq!(palette.len(), 1);
        assert_eq!(palette.color(0), [1, 2, 3, 255]);
        palette.set(0, [7, 8, 9, 255]);
        assert_eq!(palette.color(0), [7, 8, 9, 255]);
    }

    #[test]
    fn from_colors_keeps_whatever_index_zero_holds() {
        // Documents the absorbed gap: the transparent-zero rule is a
        // default-table property, not a construction invariant.
        let palette = Palette::from_colors(vec![[1, 2, 3, 255]]);
        assert_eq!(palette.color(0), [1, 2, 3, 255]);
    }

    #[test]
    fn hex_colours_parse_in_three_six_and_eight_forms() {
        assert_eq!(parse_hex("#f00").unwrap(), [255, 0, 0, 255]);
        assert_eq!(parse_hex("#ff0000").unwrap(), [255, 0, 0, 255]);
        assert_eq!(parse_hex("#FF0000").unwrap(), [255, 0, 0, 255]);
        assert_eq!(parse_hex("#ff000080").unwrap(), [255, 0, 0, 128]);
        assert_eq!(parse_hex("#1234").err().unwrap().code(), "invalid_input");
    }

    #[test]
    fn malformed_hex_is_an_explicit_error() {
        for input in ["ff0000", "#ff00", "#gg0000", "", "#", "#ff0000ff00"] {
            let error = parse_hex(input).err().unwrap_or_else(|| panic!("{input} parsed"));
            assert_eq!(error.code(), "invalid_input", "{input}");
        }
    }

    #[test]
    fn from_hex_errors_on_the_first_bad_colour() {
        let error = Palette::from_hex(&["#000".to_string(), "nope".to_string()])
            .err()
            .unwrap();
        assert_eq!(error.code(), "invalid_input");
        let palette = Palette::from_hex(&["#000".to_string(), "#fff".to_string()]).unwrap();
        assert_eq!(palette.color(1), [255, 255, 255, 255]);
    }
}
