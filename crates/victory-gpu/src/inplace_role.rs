//! Local (VICTORY_INPLACE_ROLE): title memory in place by its role, not by
//! its size or kind. `VICTORY_INPLACE_ROLE=1` (off by default; with it off
//! nothing here does anything) puts in place (`crate::inplace`) only the
//! host copies that shaders are declared to write, which is the memory the
//! title's CPU reads back; everything else stays a copy, in the card's
//! memory where it is now. The idea is xcl's "coherent window"
//! (`crates/xcl/src/gpu/uma.rs`, `RUNTIME_COMMITTED_CAP`: per-draw
//! arguments and read-back buffers shared in place, the rest in video
//! memory), applied to the copies this layer makes.
//!
//! Where the role is decided. A copy is made at its first use, before
//! anyone knows whether shaders write it; what shaders write is known only
//! when a list declares it (`Shadow::gpu_writes`). So the role is learnt at
//! the declaration and acted on at the next point where it can be acted on
//! safely:
//!
//! 1. a memory of the 64 KiB pages shaders were declared to write (through
//!    copies of at most `VICTORY_INPLACE_ROLE_KB`): a copy made later over
//!    such a page and no bigger than that is made in place at once
//!    ([`place`]; windows let go when idle and made again, buffers the
//!    title makes anew, transient windows);
//! 2. a copy already made that shaders write is turned into an in-place one
//!    by the submitting thread before a submission ([`tick`]), but only when
//!    nothing but its table holds it: no list not yet reset, no flight, no
//!    descriptor slot (a slot has the copy's GPU address baked in, and
//!    `descriptor::host::buffer_copy` finds the copy by the table: a slot
//!    on the old copy and a table on the new would bring up to date and
//!    copy back the wrong one). The check and the swap are under the
//!    table's lock, as `shadow::let_go_idle` does it.
//!
//! What an in-place copy shaders write still does: its ranges are declared
//! to the list as before (`HostList::defer_gpu_write`), so the flights'
//! written ranges (`Line::written`), `shader_written`, the memory waits
//! (`late_met`) and the card copies see them and order what reads that
//! memory behind the flight; what is skipped is the snapshot, the compare,
//! the fetch from the card and the merge (they all return at once for a
//! copy in place). When such a flight lands, the pages are told to the
//! write-watch journal (`watch::written_through_view`), so that any other
//! copy or texture of the same bytes takes them again: the card's writes
//! there are not seen by write watching.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const K64: u64 = 64 << 10;

/// `VICTORY_INPLACE_ROLE=1`.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_INPLACE_ROLE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                kb = most_one() >> 10,
                mb = most_all() >> 20,
                "copies shaders write are put in place (title memory as the card's own), the rest stays copied (VICTORY_INPLACE_ROLE=1)"
            );
        }
        on
    })
}

/// The biggest copy put in place by role (`VICTORY_INPLACE_ROLE_KB`, 1024).
fn most_one() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_INPLACE_ROLE_KB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1024) << 10)
}

/// The most bytes in place by role at once (`VICTORY_INPLACE_ROLE_MB`, 512).
fn most_all() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_INPLACE_ROLE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(512) << 20)
}

struct Memory {
    /// 64 KiB pages (address / 64 KiB) shaders were declared to write
    /// through copies no bigger than [`most_one`].
    written: BTreeSet<u64>,
    /// Copies (start, bytes, window) shaders write that are not in place,
    /// to turn into in-place ones when nothing but the table holds them.
    wanted: HashMap<(u64, u64), bool>,
    /// Copies that could not be opened in place (with why): not asked
    /// about again.
    refused: HashMap<(u64, u64), crate::inplace::Unopened>,
    /// In-place copies made here and alive, by (start, bytes).
    live: HashMap<(u64, u64), u32>,
}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

fn with<T>(f: impl FnOnce(&mut Memory) -> T) -> T {
    let mut m = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    f(m.get_or_insert_with(|| Memory {
        written: BTreeSet::new(),
        wanted: HashMap::new(),
        refused: HashMap::new(),
        live: HashMap::new(),
    }))
}

/// Counts: made in place at creation, turned in place from a copy, alive
/// now, their bytes now; turned down: over the KB cap, over the MB cap,
/// view memory, not private memory committed whole, refused by the host;
/// copies not turned yet because a list or a slot held them at the last
/// look; declared shader-written ranges in place since the last line (no
/// fetch, no merge), and still copied back.
static STATS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
const MADE: usize = 0;
const TURNED: usize = 1;
const ALIVE: usize = 2;
const ALIVE_BYTES: usize = 3;
const OVER_KB: usize = 4;
const OVER_MB: usize = 5;
const VIEW: usize = 6;
const PRIVATE: usize = 7;
const HOST: usize = 8;
const HELD: usize = 9;
const FREE_RANGES: usize = 10;
const BACK_RANGES: usize = 11;

fn refused_count(why: crate::inplace::Unopened) {
    let at = match why {
        crate::inplace::Unopened::View => VIEW,
        crate::inplace::Unopened::Private => PRIVATE,
        crate::inplace::Unopened::Host => HOST,
    };
    STATS[at].fetch_add(1, Ordering::Relaxed);
}

fn pages(at: u64, len: u64) -> std::ops::Range<u64> {
    at / K64..(at.saturating_add(len.max(1))).div_ceil(K64)
}

/// Shaders of a list are declared to write `[address, address + len)` of
/// the copy of `[start, start + bytes)` (`Shadow::gpu_writes`); `placed`
/// if that copy is in place.
pub(crate) fn declared(start: u64, bytes: u64, address: u64, len: u64, window: bool, placed: bool) {
    if !on() {
        return;
    }
    if placed {
        STATS[FREE_RANGES].fetch_add(1, Ordering::Relaxed);
        return;
    }
    STATS[BACK_RANGES].fetch_add(1, Ordering::Relaxed);
    if bytes > most_one() {
        return;
    }
    with(|m| {
        if m.refused.contains_key(&(start, bytes)) {
            return;
        }
        for page in pages(address, len) {
            m.written.insert(page);
        }
        m.wanted.insert((start, bytes), window);
    });
}

/// The copy of `[start, start + len)` about to be made, in place if shaders
/// were declared to write any of its pages and it is small enough; `None`
/// leaves the caller to make the copy as before.
pub(crate) fn place(start: u64, len: u64, window: bool) -> Option<crate::inplace::Placed> {
    if !on() {
        return None;
    }
    let written = with(|m| m.written.range(pages(start, len)).next().is_some() && !m.refused.contains_key(&(start, len)));
    if !written {
        return None;
    }
    if len > most_one() {
        STATS[OVER_KB].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if STATS[ALIVE_BYTES].load(Ordering::Relaxed) + len > most_all() {
        STATS[OVER_MB].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    match crate::inplace::open_for(start, len) {
        Ok(placed) => {
            kept(start, len, window);
            STATS[MADE].fetch_add(1, Ordering::Relaxed);
            Some(placed)
        }
        Err(why) => {
            refused_count(why);
            with(|m| m.refused.insert((start, len), why));
            None
        }
    }
}

/// A placement of ours became a copy.
fn kept(start: u64, len: u64, window: bool) {
    crate::inplace::counted(window, len);
    STATS[ALIVE].fetch_add(1, Ordering::Relaxed);
    STATS[ALIVE_BYTES].fetch_add(len, Ordering::Relaxed);
    with(|m| {
        *m.live.entry((start, len)).or_default() += 1;
        m.wanted.remove(&(start, len));
    });
}

/// An in-place copy of `[start, start + len)` is gone (its drop): if it
/// was one of ours, it is no longer counted against [`most_all`].
pub(crate) fn gone(start: u64, len: u64) {
    if !on() {
        return;
    }
    let ours = with(|m| match m.live.get_mut(&(start, len)) {
        Some(n) => {
            *n -= 1;
            if *n == 0 {
                m.live.remove(&(start, len));
            }
            true
        }
        None => false,
    });
    if ours {
        STATS[ALIVE].fetch_sub(1, Ordering::Relaxed);
        STATS[ALIVE_BYTES].fetch_sub(len, Ordering::Relaxed);
    }
}

/// A flight whose shaders were declared to write `ranges` of an in-place
/// copy has landed (`Shadow::copy_back_done`): the card wrote title memory
/// itself, which write watching does not see. The pages are told to the
/// journal, so that other copies and textures of the same bytes take them
/// again (what the copy-back's own write to title memory did before).
pub(crate) fn landed(ranges: &[(usize, usize)]) {
    if !on() {
        return;
    }
    for &(from, to) in ranges {
        crate::watch::written_through_view(from, to.saturating_sub(from));
    }
}

/// Before a submission, on the submitting thread (`crate::inplace::tick`
/// from `host::sweep`): copies shaders write are turned into in-place ones
/// where nothing but their table holds them; and the line every ten
/// seconds.
pub(crate) fn tick() {
    if !on() {
        return;
    }
    turn();
    say();
}

/// At most this many copies looked at per submission.
const TURNS: usize = 16;

fn turn() {
    // From a turning place in the table: held copies do not keep the others
    // from being looked at.
    static CURSOR: AtomicU64 = AtomicU64::new(0);
    let wanted: Vec<((u64, u64), bool)> = with(|m| {
        if m.wanted.is_empty() {
            return Vec::new();
        }
        let skip = CURSOR.fetch_add(TURNS as u64, Ordering::Relaxed) as usize % m.wanted.len();
        m.wanted.iter().cycle().skip(skip).take(TURNS.min(m.wanted.len())).map(|(k, w)| (*k, *w)).collect()
    });
    for ((start, len), window) in wanted {
        if STATS[ALIVE_BYTES].load(Ordering::Relaxed) + len > most_all() {
            STATS[OVER_MB].fetch_add(1, Ordering::Relaxed);
            // Looked at again once some go.
            continue;
        }
        let mut why = None;
        let swap = crate::shadow::role_swap(start, len, window, || match crate::inplace::open_for(start, len) {
            Ok(placed) => Some(placed),
            Err(e) => {
                why = Some(e);
                None
            }
        });
        match swap {
            crate::shadow::RoleSwap::Swapped => {
                kept(start, len, window);
                STATS[TURNED].fetch_add(1, Ordering::Relaxed);
            }
            crate::shadow::RoleSwap::Held => {
                STATS[HELD].fetch_add(1, Ordering::Relaxed);
            }
            crate::shadow::RoleSwap::Gone => {
                with(|m| m.wanted.remove(&(start, len)));
            }
            crate::shadow::RoleSwap::Refused => {
                let why = why.unwrap_or(crate::inplace::Unopened::Host);
                refused_count(why);
                with(|m| {
                    m.wanted.remove(&(start, len));
                    m.refused.insert((start, len), why);
                });
            }
        }
    }
}

fn say() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    static FRAME: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 10_000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let frame = crate::frame::current();
    let frames = frame.saturating_sub(FRAME.swap(frame, Ordering::Relaxed)).max(1) as f64;
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let free = STATS[FREE_RANGES].swap(0, Ordering::Relaxed) as f64 / frames;
    let back = STATS[BACK_RANGES].swap(0, Ordering::Relaxed) as f64 / frames;
    let (wanted, pages) = with(|m| (m.wanted.len(), m.written.len()));
    tracing::info!(
        "in place by role (VICTORY_INPLACE_ROLE): {} copies alive {:.1} MB ({} made in place, {} turned from copies, since the start); \
         turned down: {} over VICTORY_INPLACE_ROLE_KB, {} over VICTORY_INPLACE_ROLE_MB, {} view memory (scattered, not committed or its section refused), \
         {} not private memory committed whole, {} refused by the host; {} copies waiting until no list or slot holds them ({} looks found one held); \
         {} pages of 64 KiB known shader-written; shader-written range declarations per frame: {:.0} in place (no fetch, no merge), {:.0} still copied back",
        s[ALIVE],
        s[ALIVE_BYTES] as f64 / (1u64 << 20) as f64,
        s[MADE],
        s[TURNED],
        s[OVER_KB],
        s[OVER_MB],
        s[VIEW],
        s[PRIVATE],
        s[HOST],
        wanted,
        s[HELD],
        pages,
        free,
        back
    );
}
