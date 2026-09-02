//! Compiler frontend dispatcher for the Qwen fine-grained MoE family.
//!
//! Keep the existing qwen3_5_moe frontend frozen while allowing Qwen4Exp /
//! Qwen3.8 Flash Next to use its own source-layout adapter.

use crate::{
    error::Result,
    ir::SemanticModel,
    source::SourceInventory,
};

pub struct QwenMoeFrontend;

impl QwenMoeFrontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        if super::qwen4_exp::Qwen4ExpFrontend::probe(source)? {
            return Ok(true);
        }
        super::qwen_moe_legacy::QwenMoeFrontend::probe(source)
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        if super::qwen4_exp::Qwen4ExpFrontend::probe(source)? {
            return super::qwen4_exp::Qwen4ExpFrontend::build(source);
        }
        super::qwen_moe_legacy::QwenMoeFrontend::build(source)
    }
}
