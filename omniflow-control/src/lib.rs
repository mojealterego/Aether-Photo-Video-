#![forbid(unsafe_code)]
// SYS-AETHERIS-OMNIFLOW-016; native control plane.
// This library is not yet a production HTTP service or multi-host scheduler.
pub mod ir;
pub mod compiler;
pub mod scheduler;
