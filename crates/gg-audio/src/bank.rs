//! The clips a session can play, as one block of samples the audio thread is
//! handed (§6 M43).
//!
//! This crate parses no audio format and links no pack (§3 — it spends none of
//! its six dependency slots on either). The decode is `ggc`'s, at bake time,
//! for the reason its manifest already gave: a decoder here is what would make
//! `cpal` stop being a rental. So what arrives is a slice of `i16` and a rate,
//! and where those bytes came from is the shell's business.
//!
//! One `Vec` for every clip's samples rather than a `Vec` each, because the
//! audio thread holds this whole structure behind an `Arc` and a table of
//! separate allocations would be a table of pointers the callback chases.

/// Where one clip lives in the bank's block, and how fast it was authored.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    id: u64,
    /// First sample, as an index into [`Bank::block`].
    pub(crate) start: u32,
    pub(crate) frames: u32,
    pub(crate) rate: u32,
}

/// Every clip a session can play. Built once by the shell, handed over whole,
/// and replaced whole when the pack is rebuilt under a running game (§4.6).
#[derive(Default, Debug)]
pub struct Bank {
    samples: Vec<i16>,
    /// Sorted by `id`, kept sorted by [`Bank::add`] — a dozen entries reached by
    /// binary search. Not an `IndexMap`: this never reaches the hash, and a
    /// sorted `Vec` needs no dependency (`Audio::seen`'s argument).
    entries: Vec<Entry>,
}

impl Bank {
    /// An empty bank. What a game with no pack gets, and what makes every clip
    /// in it silent rather than absent — see [`Bank::find`]'s contract.
    #[must_use]
    pub fn new() -> Bank {
        Bank::default()
    }

    /// Add `samples` under `id`, replacing any clip already there.
    ///
    /// `rate` is the clip's own, not the device's: nothing resamples until a
    /// voice reads it, because no bake can know what rate the device turned out
    /// to run at.
    ///
    /// A replaced clip's samples stay in the block. A pack rebuild is an
    /// artist's keystroke, not a frame, and compacting would mean rewriting
    /// every entry's offset to reclaim a few kilobytes.
    pub fn add(&mut self, id: u64, rate: u32, samples: &[i16]) {
        let entry = Entry {
            id,
            start: self.samples.len() as u32,
            frames: samples.len() as u32,
            rate: rate.max(1),
        };
        self.samples.extend_from_slice(samples);
        match self.entries.binary_search_by_key(&id, |e| e.id) {
            Ok(at) => self.entries[at] = entry,
            Err(at) => self.entries.insert(at, entry),
        }
    }

    /// True when `id` names a clip this bank can play.
    ///
    /// The observable a gate needs: "the cue resolved" is a claim about the
    /// bank, provable on a machine with no sound card and no device open.
    #[must_use]
    pub fn holds(&self, id: u64) -> bool {
        self.locate(id).is_some()
    }

    /// How many clips. Reported at load, so a pack that baked nothing is
    /// visible in a log rather than only in the silence.
    #[must_use]
    pub fn clips(&self) -> usize {
        self.entries.len()
    }

    /// Total samples across every clip — the bank's memory, in units the log
    /// can multiply by two and print as bytes.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.samples.len()
    }

    /// Where `id` lives, or `None` — which is what an id no pack ever held
    /// resolves to, and why a missing clip is silence rather than an error
    /// (`Sound::clip`'s contract).
    ///
    /// Offsets rather than a slice, because a voice resolves once and then
    /// indexes per sample: a binary search inside the mix loop would be a
    /// lookup per sample per voice, for an answer that cannot have changed.
    /// What makes that safe is [`Mixer::set_bank`](crate::Mixer::set_bank)
    /// cutting every voice — an offset outlives the block it was measured in.
    pub(crate) fn locate(&self, id: u64) -> Option<Entry> {
        let at = self.entries.binary_search_by_key(&id, |e| e.id).ok()?;
        let entry = self.entries[at];
        // A truncated block would make every later offset a lie; refusing here
        // costs one comparison per trigger and cannot happen through `add`.
        let end = entry.start as usize + entry.frames as usize;
        (end <= self.samples.len()).then_some(entry)
    }

    /// Every clip's samples, end to end. Indexed by [`Entry::start`].
    pub(crate) fn block(&self) -> &[i16] {
        &self.samples
    }

    /// The samples and rate behind `id`. The reader's convenience over
    /// [`Bank::locate`]; the mix loop uses the offsets directly.
    #[cfg(test)]
    fn find(&self, id: u64) -> Option<(&[i16], u32)> {
        let entry = self.locate(id)?;
        let start = entry.start as usize;
        let samples = &self.samples[start..start + entry.frames as usize];
        Some((samples, entry.rate))
    }
}

#[cfg(test)]
mod tests {
    // unwrap is permitted in tests (§2, Error handling row).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn a_clip_comes_back_out_under_the_id_it_went_in_under() {
        let mut bank = Bank::new();
        bank.add(7, 22_050, &[1, 2, 3]);
        bank.add(3, 48_000, &[-1, -2]);
        assert_eq!(bank.clips(), 2);
        assert_eq!(bank.frames(), 5);
        assert_eq!(bank.find(7), Some((&[1, 2, 3][..], 22_050)));
        assert_eq!(bank.find(3), Some((&[-1, -2][..], 48_000)));
        assert!(bank.holds(3) && !bank.holds(4));
    }

    /// The property the whole flat block rests on: entries are found by id and
    /// not by position, so insertion order cannot change what a clip sounds
    /// like. `ggc` writes a pack sorted by id and the shell walks it in that
    /// order, but a bank built any other way must agree.
    #[test]
    fn insertion_order_is_not_a_property_of_the_bank() {
        let mut forward = Bank::new();
        let mut backward = Bank::new();
        for id in [11_u64, 2, 90, 7] {
            forward.add(id, 8_000, &[id as i16; 4]);
        }
        for id in [7_u64, 90, 2, 11] {
            backward.add(id, 8_000, &[id as i16; 4]);
        }
        for id in [11_u64, 2, 90, 7] {
            assert_eq!(forward.find(id), backward.find(id));
        }
    }

    #[test]
    fn a_rebuilt_pack_replaces_a_clip_rather_than_shadowing_it() {
        let mut bank = Bank::new();
        bank.add(1, 8_000, &[9; 3]);
        bank.add(1, 16_000, &[4; 2]);
        assert_eq!(bank.clips(), 1, "the rebuild added an entry");
        assert_eq!(bank.find(1), Some((&[4, 4][..], 16_000)));
    }

    #[test]
    fn an_empty_bank_holds_nothing_rather_than_failing_to_be_asked() {
        let bank = Bank::new();
        assert_eq!(bank.find(1), None);
        assert_eq!(bank.clips(), 0);
    }

    /// A clip of no samples is legal at the pack layer (`gg_assets::clip`), so
    /// it has to be legal here — and it resolves, because "the pack holds it"
    /// and "it makes a sound" are different questions.
    #[test]
    fn a_clip_of_no_samples_still_resolves() {
        let mut bank = Bank::new();
        bank.add(5, 44_100, &[]);
        assert_eq!(bank.find(5), Some((&[][..], 44_100)));
        assert!(bank.holds(5));
    }
}
