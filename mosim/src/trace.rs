//! Per-event trace logging.
//!
//! The simulator emits one line per scheduling event and per contention
//! recomputation. On a 60-job trace that is ~3.5 million lines / 224 MB, and
//! writing it costs about half the total runtime — so it is off unless the
//! caller asks for it by passing `--trace-log PATH`.
//!
//! This mirrors the existing `--gpu_util_log` contract: give a path to enable,
//! leave it empty to disable. The trace never goes to stdout, so stdout stays
//! reserved for the final metrics summary that a human reads.

use std::cell::RefCell;
use std::fmt::Arguments;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};

/// Checked before touching thread-local state so that a disabled trace costs
/// one relaxed atomic load per call site rather than a TLS lookup.
static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static SINK: RefCell<Option<BufWriter<File>>> = const { RefCell::new(None) };
}

/// Open `path` for the trace log. An empty path leaves tracing disabled.
pub fn init(path: &str) {
    if path.is_empty() {
        return;
    }
    match File::create(path) {
        Ok(file) => {
            SINK.with(|sink| {
                *sink.borrow_mut() = Some(BufWriter::with_capacity(1 << 20, file));
            });
            ENABLED.store(true, Ordering::Relaxed);
        }
        Err(e) => {
            eprintln!("mosim: cannot open trace log '{}': {} (tracing disabled)", path, e);
        }
    }
}

/// Write one trace line. Called by the `trace_log!` macro; returns immediately
/// when tracing is off, and `format_args!` means the arguments are not
/// formatted in that case either.
pub fn write_fmt(args: Arguments) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    SINK.with(|sink| {
        if let Some(writer) = sink.borrow_mut().as_mut() {
            let _ = writeln!(writer, "{}", args);
        }
    });
}

/// Flush the buffered trace. Must run before the process exits or the tail of
/// the log is lost.
pub fn flush() {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    SINK.with(|sink| {
        if let Some(writer) = sink.borrow_mut().as_mut() {
            let _ = writer.flush();
        }
    });
}

/// Emit a per-event trace line. Same formatting as `println!`, but the line
/// goes to the `--trace-log` file and is skipped entirely when that is unset.
macro_rules! trace_log {
    ($($arg:tt)*) => {
        $crate::trace::write_fmt(format_args!($($arg)*))
    };
}
