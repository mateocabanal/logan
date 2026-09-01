mod app;
mod engine;
mod ui;

use std::io::{self, stdout};
use std::path::PathBuf;
use std::time::Duration;

use app::{App, UiAction};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use engine::{EngineCommand, GenerationSettings};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

struct Args {
    package: PathBuf,
    system: String,
    settings: GenerationSettings,
    show_stats: bool,
}

fn main() {
    let args = match parse_args() {
        Ok(Some(args)) => args,
        Ok(None) => return,
        Err(error) => {
            eprintln!("logan-chat: {error}\n");
            print_usage();
            std::process::exit(2);
        }
    };

    install_panic_cleanup();
    if let Err(error) = run(args) {
        eprintln!("logan-chat: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let handle = engine::spawn(args.package, args.system.clone());
    let mut app = App::new(args.system, handle.cancel.clone());
    app.settings = args.settings;
    app.show_stats = args.show_stats;

    let mut terminal = enter_terminal()?;
    let result = event_loop(&mut terminal, &mut app, &handle);
    let _ = handle.tx.send(EngineCommand::Shutdown);
    let cleanup = leave_terminal(&mut terminal);

    match (result, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error.into()),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn enter_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut out = stdout();
    if let Err(error) = execute!(
        out,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    ) {
        let _ = disable_raw_mode();
        return Err(error);
    }
    let backend = CrosstermBackend::new(out);
    match Terminal::new(backend) {
        Ok(mut terminal) => {
            terminal.clear()?;
            Ok(terminal)
        }
        Err(error) => {
            let _ = disable_raw_mode();
            let mut out = stdout();
            let _ = execute!(
                out,
                DisableBracketedPaste,
                DisableMouseCapture,
                LeaveAlternateScreen
            );
            Err(error)
        }
    }
}

fn leave_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let raw_result = disable_raw_mode();
    let screen_result = execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    let cursor_result = terminal.show_cursor();
    raw_result?;
    screen_result?;
    cursor_result?;
    Ok(())
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    handle: &engine::EngineHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    let tick = Duration::from_millis(50);

    loop {
        while let Ok(event) = handle.rx.try_recv() {
            app.on_engine_event(event);
        }

        terminal.draw(|frame| ui::draw(frame, app))?;

        if !event::poll(tick)? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if handle_key(app, key, handle)? {
                    break;
                }
            }
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    app.transcript_scroll = app.transcript_scroll.saturating_add(3);
                }
                MouseEventKind::ScrollDown => {
                    app.transcript_scroll = app.transcript_scroll.saturating_sub(3);
                }
                _ => {}
            },
            Event::Paste(text) => {
                if app.loaded && !app.generating && !app.show_help {
                    app.insert_str(&text);
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }
    Ok(())
}

fn handle_key(
    app: &mut App,
    key: KeyEvent,
    handle: &engine::EngineHandle,
) -> Result<bool, Box<dyn std::error::Error>> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Ok(true);
    }
    if key.code == KeyCode::F(1) {
        app.show_help = !app.show_help;
        return Ok(false);
    }
    if app.show_help {
        if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
            app.show_help = false;
        }
        return Ok(false);
    }

    match key.code {
        KeyCode::Esc => {
            app.cancel_generation();
        }
        KeyCode::Tab => {
            app.show_stats = !app.show_stats;
        }
        KeyCode::PageUp if key.modifiers.contains(KeyModifiers::SHIFT) => {
            app.scroll_stats_up();
        }
        KeyCode::PageDown if key.modifiers.contains(KeyModifiers::SHIFT) => {
            app.scroll_stats_down();
        }
        KeyCode::PageUp => {
            app.transcript_scroll = app.transcript_scroll.saturating_add(12);
        }
        KeyCode::PageDown => {
            app.transcript_scroll = app.transcript_scroll.saturating_sub(12);
        }
        KeyCode::Enter
            if key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            if app.loaded && !app.generating {
                app.insert_char('\n');
            }
        }
        KeyCode::Enter => {
            return dispatch(app.submit(), handle);
        }
        KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.loaded && !app.generating {
                app.insert_char('\n');
            }
        }
        KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if !app.generating {
                app.delete_word();
            }
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if !app.generating {
                app.clear_input();
            }
        }
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            if app.loaded && !app.generating {
                app.insert_char(ch);
            }
        }
        KeyCode::Backspace => {
            if !app.generating {
                app.backspace();
            }
        }
        KeyCode::Delete => {
            if !app.generating {
                app.delete();
            }
        }
        KeyCode::Left => {
            if !app.generating {
                app.move_left();
            }
        }
        KeyCode::Right => {
            if !app.generating {
                app.move_right();
            }
        }
        KeyCode::Home => {
            if !app.generating {
                app.cursor = 0;
            }
        }
        KeyCode::End => {
            if !app.generating {
                app.cursor = app.input.len();
            }
        }
        KeyCode::Up => {
            if !app.generating {
                app.history_prev();
            }
        }
        KeyCode::Down => {
            if !app.generating {
                app.history_next();
            }
        }
        _ => {}
    }
    Ok(false)
}

fn dispatch(
    action: UiAction,
    handle: &engine::EngineHandle,
) -> Result<bool, Box<dyn std::error::Error>> {
    match action {
        UiAction::None => Ok(false),
        UiAction::Quit => Ok(true),
        UiAction::Send(command) => {
            handle.tx.send(command).map_err(|_| {
                io::Error::new(io::ErrorKind::BrokenPipe, "inference worker exited")
            })?;
            Ok(false)
        }
    }
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut args = std::env::args().skip(1);
    let Some(first) = args.next() else {
        return Err("missing .coli package path".into());
    };
    if first == "-h" || first == "--help" {
        print_usage();
        return Ok(None);
    }

    let mut out = Args {
        package: PathBuf::from(first),
        system: "You are a helpful assistant.".into(),
        settings: GenerationSettings::default(),
        show_stats: true,
    };

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--system" => out.system = next_value(&mut args, &flag)?,
            "--max-new" => {
                out.settings.max_new = parse_value(&mut args, &flag)?;
            }
            "--temperature" | "--temp" => {
                out.settings.temperature = parse_value(&mut args, &flag)?;
            }
            "--top-p" => out.settings.top_p = parse_value(&mut args, &flag)?,
            "--top-k" => out.settings.top_k = parse_value(&mut args, &flag)?,
            "--repeat-penalty" => {
                out.settings.repeat_penalty = parse_value(&mut args, &flag)?;
            }
            "--greedy" => {
                out.settings.temperature = 0.0;
                out.settings.top_k = 1;
            }
            "--no-stats" => out.show_stats = false,
            "-h" | "--help" => {
                print_usage();
                return Ok(None);
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    if out.settings.max_new == 0 {
        return Err("--max-new must be at least 1".into());
    }
    if !(0.0..=5.0).contains(&out.settings.temperature) {
        return Err("--temperature must be in 0..=5".into());
    }
    if !(0.01..=1.0).contains(&out.settings.top_p) {
        return Err("--top-p must be in 0.01..=1".into());
    }
    if !(1.0..=2.0).contains(&out.settings.repeat_penalty) {
        return Err("--repeat-penalty must be in 1..=2".into());
    }
    Ok(Some(out))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let raw = next_value(args, flag)?;
    raw.parse::<T>()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

fn print_usage() {
    eprintln!(
        "Usage: logan-chat <package.coli> [options]\n\n\
         Interactive Qwen4/Flash-Next chat TUI with live Logan runtime telemetry.\n\n\
         Options:\n\
           --system TEXT            system prompt\n\
           --max-new N             max response tokens (default 256)\n\
           --temperature F         sampling temperature (default 0.7)\n\
           --top-p F               nucleus probability (default 0.9)\n\
           --top-k N               candidate count; 0 = all (default 40)\n\
           --repeat-penalty F      recent-token penalty (default 1.05)\n\
           --greedy                temperature=0, top-k=1\n\
           --no-stats              start with stats panel hidden\n\
           -h, --help              show this help\n\n\
         Performance paths are enabled by default by logan-qwen4. Set the\n\
         corresponding QWEN_* environment variable to 0 before launch to A/B."
    );
}

fn install_panic_cleanup() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let mut out = stdout();
        let _ = execute!(
            out,
            DisableBracketedPaste,
            DisableMouseCapture,
            LeaveAlternateScreen
        );
        previous(info);
    }));
}
