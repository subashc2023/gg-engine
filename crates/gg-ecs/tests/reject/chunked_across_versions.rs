//! §4.2.1: the chunked accelerator is version-local, and comparing two versions'
//! values must be a type error rather than a green-looking test.
//!
//! Literal version parameters rather than `chunked::Local`, so the expected
//! error does not move every time this crate's version does.

use gg_ecs::chunked::ChunkedHash;

fn main() {
    let here: ChunkedHash<1> = ChunkedHash::new(1);
    let elsewhere: ChunkedHash<2> = ChunkedHash::new(1);
    assert_eq!(here, elsewhere);
}
