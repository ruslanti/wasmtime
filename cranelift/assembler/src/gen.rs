//! Expose code generated in `build.rs`.

include!(concat!(env!("OUT_DIR"), "/assembler-isle-macro.rs"));
