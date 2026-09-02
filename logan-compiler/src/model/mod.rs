//! Architecture-specific source frontends.

pub mod deepseek_v4;
#[path = "qwen_moe_dispatch.rs"]
pub mod qwen_moe;
#[path = "qwen_moe.rs"]
mod qwen_moe_legacy;
pub mod qwen4_exp;
pub mod qwen_mtp;
