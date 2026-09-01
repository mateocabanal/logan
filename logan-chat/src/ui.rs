use ratatui::buffer::CellWidth;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Gauge, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{generation_rate, App, Role};

const ACCENT: Color = Color::Cyan;
const MUTED: Color = Color::DarkGray;
const GOOD: Color = Color::Green;
const WARN: Color = Color::Yellow;

pub fn draw(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, app, rows[0]);

    let show_side = app.show_stats && rows[1].width >= 96;
    if show_side {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(48), Constraint::Length(42)])
            .split(rows[1]);
        draw_transcript(frame, app, columns[0]);
        draw_stats(frame, app, columns[1]);
    } else {
        draw_transcript(frame, app, rows[1]);
    }

    draw_input(frame, app, rows[2]);
    draw_footer(frame, app, rows[3]);

    if app.show_help {
        draw_help(frame, area);
    }
}

fn draw_header(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let status_style = if app.generating {
        Style::default().fg(WARN)
    } else if app.loaded {
        Style::default().fg(GOOD)
    } else {
        Style::default().fg(WARN)
    };
    let cache = if app.features.prefix_cache {
        "SSD cache"
    } else {
        "cache off"
    };
    let line = Line::from(vec![
        Span::styled(
            " LOGAN CHAT ",
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(&app.model_name, Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(format!("  ·  {cache}  ·  "), Style::default().fg(MUTED)),
        Span::styled(&app.status, status_style),
    ]);
    let paragraph = Paragraph::new(line)
        .block(Block::default().borders(Borders::BOTTOM))
        .alignment(Alignment::Left);
    frame.render_widget(paragraph, area);
}

fn transcript_text(app: &App) -> Text<'static> {
    let mut lines = Vec::new();
    if app.messages.is_empty() {
        lines.push(Line::from(Span::styled(
            "Type a message and press Enter. /help shows commands.",
            Style::default().fg(MUTED),
        )));
        return Text::from(lines);
    }

    for message in &app.messages {
        let (label, style) = match message.role {
            Role::User => (
                "YOU",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Role::Assistant => (
                "QWEN",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Role::Notice => ("LOGAN", Style::default().fg(WARN)),
        };
        lines.push(Line::from(Span::styled(label, style)));
        if message.content.is_empty() && message.role == Role::Assistant && app.generating {
            lines.push(Line::from(Span::styled(
                "▌",
                Style::default().fg(ACCENT),
            )));
        } else {
            for raw in message.content.lines() {
                lines.push(Line::from(raw.to_string()));
            }
            if message.content.ends_with('\n') {
                lines.push(Line::default());
            }
        }
        lines.push(Line::default());
    }
    Text::from(lines)
}

fn draw_transcript(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Conversation ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    let text = transcript_text(app);
    let base = Paragraph::new(text).wrap(Wrap { trim: false });
    let total_lines = base.line_count(inner.width.max(1));
    let visible = inner.height as usize;
    let max_scroll = total_lines.saturating_sub(visible);
    let from_bottom = usize::from(app.transcript_scroll).min(max_scroll);
    let y = max_scroll
        .saturating_sub(from_bottom)
        .min(u16::MAX as usize) as u16;
    let paragraph = base.block(block).scroll((y, 0));
    frame.render_widget(paragraph, area);
}

fn draw_stats(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(4)])
        .split(area);

    let metrics = if app.generating {
        &app.live_metrics
    } else {
        &app.last_metrics
    };
    let context = metrics.context_tokens;
    let limit = app.context_limit.max(1);
    let ratio = (context as f64 / limit as f64).clamp(0.0, 1.0);
    let gauge = Gauge::default()
        .block(Block::default().title(" Context ").borders(Borders::ALL))
        .gauge_style(Style::default().fg(if ratio > 0.9 { WARN } else { ACCENT }))
        .label(format!(
            "{context} / {}  ({:.1}%)",
            app.context_limit,
            ratio * 100.0
        ))
        .ratio(ratio);
    frame.render_widget(gauge, chunks[0]);

    let s = &app.last_stats;
    let forwards = metrics.forward_tokens.max(1) as f64;
    let mut lines = Vec::<Line<'static>>::new();

    section(&mut lines, "TURN");
    kv(&mut lines, "input", format!("{} tok", metrics.input_tokens));
    kv(
        &mut lines,
        "prompt forward",
        format!("{} tok", metrics.forwarded_prompt_tokens),
    );
    kv(
        &mut lines,
        "generated",
        format!("{} tok", metrics.generated_tokens),
    );
    kv(
        &mut lines,
        "decode",
        format!("{:.2} tok/s", generation_rate(metrics)),
    );
    kv(&mut lines, "TTFT", fmt_ms(metrics.first_token_ms));
    kv(&mut lines, "prompt", fmt_ms(metrics.prompt_ms));
    kv(&mut lines, "wall", fmt_ms(metrics.total_ms));
    kv(
        &mut lines,
        "live reused",
        format!("{} tok", metrics.live_reused_tokens),
    );
    kv(
        &mut lines,
        "SSD reused",
        format!("{} tok", metrics.ssd_cached_tokens),
    );
    kv(&mut lines, "SSD restore", fmt_ms(metrics.cache_restore_ms));
    kv(&mut lines, "cache write", fmt_ms(metrics.cache_write_ms));
    if let Some(reason) = &metrics.stop_reason {
        kv(&mut lines, "stop", reason.label().to_string());
    }

    section(&mut lines, "EXPERT LRU");
    kv(
        &mut lines,
        "resident",
        format!("{} / {}", s.expert_resident, s.expert_capacity),
    );
    kv(&mut lines, "hits", s.expert_hits.to_string());
    kv(&mut lines, "misses", s.expert_misses.to_string());
    kv(
        &mut lines,
        "hit rate",
        format!("{:.1}%", s.expert_hit_rate() * 100.0),
    );
    kv(&mut lines, "evictions", s.expert_evictions.to_string());

    section(&mut lines, "METAL");
    kv(&mut lines, "direct", yesno(s.features.metal_direct));
    kv(&mut lines, "overlap", yesno(s.features.metal_overlap));
    kv(&mut lines, "fused calls", s.fused_calls.to_string());
    kv(&mut lines, "fused experts", s.fused_experts.to_string());
    kv(&mut lines, "GDN metal", s.gdn_metal_ok.to_string());
    kv(&mut lines, "kernel", fmt_ns(s.metal_kernel_ns));
    kv(&mut lines, "GPU wait", fmt_ns(s.metal_wait_ns));
    kv(&mut lines, "encode", fmt_ns(s.metal_encode_ns));
    kv(&mut lines, "submit", fmt_ns(s.metal_submit_ns));

    section(&mut lines, "METALIO");
    kv(&mut lines, "loads", s.mio_loads.to_string());
    kv(&mut lines, "read", fmt_bytes(s.mio_bytes));
    kv(&mut lines, "waits", s.mio_waits.to_string());
    kv(&mut lines, "fails", s.mio_fails.to_string());
    kv(
        &mut lines,
        "avg latency",
        format!("{:.2} ms", s.mio_avg_latency_ms()),
    );
    kv(&mut lines, "outstanding", s.mio_outstanding.to_string());
    kv(&mut lines, "peak out", s.mio_peak_outstanding.to_string());
    kv(&mut lines, "prefetch", s.mio_prefetch_loads.to_string());
    kv(&mut lines, "prefetch used", s.mio_prefetch_used.to_string());
    kv(
        &mut lines,
        "prefetch waste",
        s.mio_prefetch_wasted.to_string(),
    );

    section(&mut lines, "PHASE / FORWARD");
    kv(
        &mut lines,
        "GDN",
        format!("{:.1} ms", s.gdn_ms / forwards),
    );
    kv(
        &mut lines,
        "attention",
        format!("{:.1} ms", s.attn_ms / forwards),
    );
    kv(
        &mut lines,
        "HC",
        format!("{:.1} ms", s.hc_ms / forwards),
    );
    kv(
        &mut lines,
        "head",
        format!("{:.1} ms", s.head_ms / forwards),
    );
    kv(
        &mut lines,
        "expert I/O",
        format!("{:.1} ms", s.io_ms / forwards),
    );
    kv(
        &mut lines,
        "shared",
        format!("{:.1} ms", s.shared_ms / forwards),
    );
    kv(
        &mut lines,
        "GPU MoE",
        format!("{:.1} ms", s.gpu_ms / forwards),
    );
    kv(
        &mut lines,
        "route",
        format!("{:.1} ms", s.route_ms / forwards),
    );
    kv(
        &mut lines,
        "fill",
        format!("{:.1} ms", s.fill_ms / forwards),
    );

    section(&mut lines, "CACHE / MEMORY");
    kv(&mut lines, "entries", app.cache_entries.to_string());
    kv(&mut lines, "disk", fmt_bytes(app.cache_bytes));
    kv(&mut lines, "peak RSS", fmt_bytes(app.peak_rss_bytes));

    section(&mut lines, "FAST PATHS");
    kv(&mut lines, "BNNS BF16", yesno(app.features.bnns_bf16));
    kv(
        &mut lines,
        "attention Metal",
        yesno(app.features.attn_metal),
    );
    kv(
        &mut lines,
        "QSA Metal",
        yesno(app.features.qsa_index_metal),
    );
    kv(&mut lines, "GDN Metal", yesno(app.features.gdn_metal));
    kv(
        &mut lines,
        "GDN single copy",
        yesno(app.features.gdn_single_copy),
    );
    kv(
        &mut lines,
        "shared overlap",
        yesno(app.features.shared_io_overlap),
    );
    kv(
        &mut lines,
        "prefix cache",
        yesno(app.features.prefix_cache),
    );
    kv(
        &mut lines,
        "cache writes",
        yesno(app.features.prefix_cache_write),
    );

    section(&mut lines, "SAMPLING");
    kv(&mut lines, "max new", app.settings.max_new.to_string());
    kv(
        &mut lines,
        "temperature",
        format!("{:.2}", app.settings.temperature),
    );
    kv(&mut lines, "top-p", format!("{:.2}", app.settings.top_p));
    kv(&mut lines, "top-k", app.settings.top_k.to_string());
    kv(
        &mut lines,
        "repeat",
        format!("{:.2}", app.settings.repeat_penalty),
    );
    if let Some(token) = app.last_token_id {
        kv(&mut lines, "last token", token.to_string());
    }

    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .title(" Runtime Stats · Shift+PgUp/PgDn ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.stats_scroll, 0));
    frame.render_widget(paragraph, chunks[1]);
}

fn draw_input(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let title = if app.generating {
        " Prompt — generating; Esc cancels "
    } else if !app.loaded {
        " Prompt — model loading "
    } else {
        " Prompt — Enter sends · Ctrl+J newline "
    };
    let style = if app.loaded && !app.generating {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(MUTED)
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if app.generating { WARN } else { ACCENT }));
    let inner = block.inner(area);

    let before = &app.input[..app.cursor.min(app.input.len())];
    let row = before.chars().filter(|&c| c == '\n').count();
    let current_line = before.rsplit('\n').next().unwrap_or("");
    let scroll = row.saturating_sub(inner.height.saturating_sub(1) as usize);
    let paragraph = Paragraph::new(app.input.as_str())
        .style(style)
        .block(block)
        .scroll((scroll.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(paragraph, area);

    if app.loaded && !app.generating && !app.show_help && inner.width > 0 && inner.height > 0 {
        let xoff = current_line.cell_width().min(inner.width.saturating_sub(1));
        let yoff = row
            .saturating_sub(scroll)
            .min(inner.height.saturating_sub(1) as usize) as u16;
        frame.set_cursor_position((inner.x + xoff, inner.y + yoff));
    }
}

fn draw_footer(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let text = if app.generating {
        " Esc cancel   PgUp/PgDn chat   Shift+PgUp/PgDn stats   Tab stats   Ctrl+C quit "
    } else {
        " Enter send   ↑/↓ history   PgUp/PgDn chat   Shift+PgUp/PgDn stats   F1 help   Ctrl+C quit "
    };
    frame.render_widget(
        Paragraph::new(text).style(Style::default().fg(MUTED)),
        area,
    );
}

fn draw_help(frame: &mut Frame<'_>, area: Rect) {
    let popup = centered(area, 74, 78);
    frame.render_widget(Clear, popup);
    let help = Text::from(vec![
        Line::from(Span::styled(
            "KEYS",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("Enter             send message"),
        Line::from("Ctrl+J            insert newline"),
        Line::from("Esc               cancel generation / close help"),
        Line::from("↑ / ↓             prompt history"),
        Line::from("PgUp/PgDn         scroll conversation"),
        Line::from("Shift+PgUp/PgDn   scroll runtime stats"),
        Line::from("Tab               toggle runtime stats"),
        Line::from("Ctrl+W            delete previous word"),
        Line::from("Ctrl+U            clear prompt"),
        Line::from("Ctrl+C            quit"),
        Line::default(),
        Line::from(Span::styled(
            "COMMANDS",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("/clear              new session; preserve system prompt"),
        Line::from("/system TEXT        new session with a new system prompt"),
        Line::from("/max N              maximum generated tokens"),
        Line::from("/temp F             temperature, 0 = greedy"),
        Line::from("/top-p F            nucleus probability"),
        Line::from("/top-k N            top-k candidate limit, 0 = all"),
        Line::from("/repeat F           repeat penalty (1.0..2.0)"),
        Line::from("/greedy             temperature=0, top-k=1"),
        Line::from("/stats              toggle stats panel"),
        Line::from("/save [FILE]        save readable transcript"),
        Line::from("/quit               quit"),
        Line::default(),
        Line::from(Span::styled(
            "Performance defaults are ON; use Logan env vars to opt out before launch.",
            Style::default().fg(WARN),
        )),
    ]);
    frame.render_widget(
        Paragraph::new(help)
            .block(
                Block::default()
                    .title(" Logan Chat Help ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT)),
            )
            .wrap(Wrap { trim: false }),
        popup,
    );
}

fn centered(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn section(lines: &mut Vec<Line<'static>>, title: &str) {
    if !lines.is_empty() {
        lines.push(Line::default());
    }
    lines.push(Line::from(Span::styled(
        title.to_string(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    )));
}

fn kv(lines: &mut Vec<Line<'static>>, key: &str, value: String) {
    lines.push(Line::from(vec![
        Span::styled(format!("{key:<15}"), Style::default().fg(MUTED)),
        Span::raw(value),
    ]));
}

fn yesno(value: bool) -> String {
    if value { "on" } else { "off" }.to_string()
}

fn fmt_ms(ms: f64) -> String {
    if ms <= 0.0 {
        "—".into()
    } else if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else {
        format!("{ms:.1} ms")
    }
}

fn fmt_ns(ns: u64) -> String {
    fmt_ms(ns as f64 / 1e6)
}

fn fmt_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}
