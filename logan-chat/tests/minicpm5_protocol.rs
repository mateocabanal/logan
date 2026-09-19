use logan_chat::openai::{ApiMessage, normalize_openai_string_argument, render_minicpm5_prompt};
use logan_chat::protocol::minicpm5::{
    MINICPM5_EOS_IDS, MiniCpm5Delta, MiniCpm5ParseError, MiniCpm5StreamParser,
};
use logan_chat::runtime::{
    ModelFamily, PromptAdapter, ProtocolOptions, prompt_adapter, select_model_family,
};
use serde_json::json;

#[test]
fn parser_handles_arbitrary_byte_splits_unicode_cdata_and_multiple_calls() {
    let source = "<think>推理</think>answer <function name=\"echo\"><param name=\"text\"><![CDATA[héllo <world>]]></param></function> and <function name=\"sum\"><param name=\"a\">1</param><param name=\"b\">2</param></function>";
    let mut parser = MiniCpm5StreamParser::new();
    let mut deltas = Vec::new();
    for byte in source.as_bytes().chunks(1) {
        deltas.extend(parser.feed(byte).unwrap());
    }
    deltas.extend(parser.finish().unwrap());

    let reasoning = deltas
        .iter()
        .filter_map(|delta| match delta {
            MiniCpm5Delta::Reasoning(payload) => Some(payload.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(reasoning, "推理");
    let text = deltas
        .iter()
        .filter_map(|delta| match delta {
            MiniCpm5Delta::Text(payload) => Some(payload.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(text.contains("answer "));
    assert!(deltas.contains(&MiniCpm5Delta::ToolArgument {
        name: "text".into(),
        value: "héllo <world>".into(),
    }));
    assert!(
        deltas.contains(&MiniCpm5Delta::ToolCallFinished(
            logan_chat::protocol::minicpm5::ToolCall {
                name: "echo".into(),
                arguments: [("text".into(), "héllo <world>".into())]
                    .into_iter()
                    .collect(),
            },
        ))
    );
    assert!(
        deltas
            .iter()
            .filter(|delta| matches!(delta, MiniCpm5Delta::ToolCallFinished(_)))
            .count()
            == 2
    );
}

#[test]
fn tools_render_without_qwen_empty_think_prefix() {
    let messages = vec![ApiMessage {
        role: "user".into(),
        text: "call a tool".into(),
    }];
    let prompt = render_minicpm5_prompt(
        &messages,
        None,
        false,
        Some(&json!([{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object"}
            }
        }])),
    )
    .unwrap();
    assert!(prompt.contains("lookup"));
    assert!(prompt.contains("<function name=\"...\">"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
    assert!(!prompt.contains("<think>\n\n</think>"));
}

#[test]
fn malformed_and_truncated_calls_are_errors_before_execution() {
    let mut malformed = MiniCpm5StreamParser::new();
    assert!(matches!(
        malformed.feed(b"<function name=\"x\"><param name=\"a\">&evil;</param></function>"),
        Err(MiniCpm5ParseError::ExternalEntity)
    ));

    let mut truncated = MiniCpm5StreamParser::new();
    truncated
        .feed(b"<function name=\"x\"><param name=\"a\">value")
        .unwrap();
    assert!(matches!(
        truncated.finish(),
        Err(MiniCpm5ParseError::Truncated(_))
    ));
}

#[test]
fn selector_and_stop_ids_are_explicit() {
    assert_eq!(
        select_model_family(Some("MiniCPM5-8B")),
        ModelFamily::MiniCpm5
    );
    assert_eq!(select_model_family(Some("Qwen4")), ModelFamily::Qwen4);
    assert_eq!(ModelFamily::MiniCpm5.eos_ids(), &MINICPM5_EOS_IDS);
    assert!(matches!(
        prompt_adapter(ModelFamily::MiniCpm5, &ProtocolOptions::default()),
        PromptAdapter::MiniCpm5(_)
    ));
}

#[test]
fn openai_json_string_arguments_are_decoded_once() {
    assert_eq!(
        normalize_openai_string_argument(r#"{"query":"logan"}"#).unwrap(),
        json!({"query": "logan"})
    );
    assert_eq!(
        normalize_openai_string_argument("plain text").unwrap(),
        json!("plain text")
    );
}
