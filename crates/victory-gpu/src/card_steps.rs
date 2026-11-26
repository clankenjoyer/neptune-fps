//! Steps of a submission that read title memory flights in the air may
//! still write, done on the card in the flights' order instead of on the
//! CPU after a wait for them (`VICTORY_CARD_STEPS`, see [`stage`]).
//!
//! From cool_fox (github.com/codec-d/cool_fox, GPL-3.0-or-later):
//! `crates/victory-gpu/src/host/card_steps.rs`, ported to Neptune. A child of
//! `crate::host` (declared there with `#[path]`), so `super::` is that module.
//! What differs from cool_fox is marked "(Neptune)":
//! - Neptune has no copies at their place among the steps (cool_fox's
//!   `Step::Placed`, `VICTORY_PLACED_ORDER`, `cut_for`): its title copies on
//!   the card (`crate::list::card_copy`) are commands in the title's lists
//!   with refreshes and declared writes, which the plan sees as such.
//! - What a step is "asked" for (flights stop it from being done on the CPU)
//!   is what Neptune's `blocked_by` waits for (`VICTORY_FILL_NO_WAIT`, fills
//!   of constants that do not wait), plus, from stage 1 on, the constant
//!   buffers of descriptor tables (their slot is known) that read memory
//!   flights may write: Neptune reads those without a wait, a frame late;
//!   here they are read on the card behind the flights. If the plan refuses
//!   a submission that Neptune would not have waited for, nothing changes:
//!   no wait, the steps on the CPU as before.
//! - Uses of the tiled image (`VICTORY_TILED`, `Deferred::tiled`) count as
//!   refreshes of one host copy, the image.
//! - Work on the CPU that reads title memory names it (`Deferred::host_reads`,
//!   [`HostList::defer_host_read`]: the rectangles' vertices).
//! - No timeline spans, no spread of the slots over threads, no diagnostics
//!   of refusals (Neptune has neither `crate::timeline` nor `crate::land`).

use std::sync::Arc;

use super::*;
use crate::shadow::Shadow;

/// The stage (`VICTORY_CARD_STEPS`; off by default): `1` constant buffers
/// of descriptor tables (`Step::Fill` with a slot of its own), `2` the
/// title's buffer copies (`Step::Copy`) too, with the steps of the same
/// submission that depend on them. Only with the relay's second stage
/// (`relay_exact`).
///
/// Why: eleven dispatches a frame (the skinning passes, in two early
/// submissions) write through a view of the title's whole 174 MB heap, at
/// places they read out of job records in buffers: no bound of what they
/// write is known before they ran. The title's ring of constants lies in
/// the same heap. One constant buffer of 480 bytes and one copy of 20 KiB a
/// frame, at a new place of the ring each time, waited for these passes to
/// land: 25-30 ms of a 105 ms frame, the submitting thread standing.
///
/// What is done: a list of ours goes to the card in front of the
/// submission's lists, in the same call. It copies the bytes read from the
/// host copy that the flights write into a buffer of ours, and from there
/// to the destination's host copy (a copy), or the fill's descriptor slot
/// is pointed at that buffer (a fill; the slot is the fill's alone, the
/// root signatures say such descriptors are volatile, and the list that
/// names it has not gone to the card).
///
/// Why that is the value the CPU would have read after the wait. After the
/// wait title memory holds, for every byte read, what the title wrote, or
/// what the last of the flights that wrote it wrote (the landings put it
/// there, in order). The host copy on the card holds the same when our list
/// runs: every flight that may write one of these bytes writes this same
/// host copy ([`holder`]: a write through another host copy, a read-back
/// or a value written behind a flight there, and the old wait is taken),
/// they are all ahead of our list on the queue, and the title's own bytes
/// are brought up to date by a refresh step of our list, which is prepared
/// like any other list's (also under flights' declared writes:
/// `stale_under_flights`, which settles if it cannot). Nothing of this
/// submission writes these bytes earlier than our list: the steps are done
/// in front of all its lists in either case, and among themselves in the
/// order they stand in wherever one depends on another: [`plan`] takes
/// every copy and fill of the submission that writes what a taken one
/// reads or writes, or reads what a taken one writes. Where a page is not
/// mapped the host copy may hold other bytes than the CPU's copy would have
/// left: refused.
///
/// For a copy's destination our list declares a shader write, exactly the
/// bytes copied, after a refresh of them: from there on it is what any
/// shader's write is. The relay compares, and puts what changed into title
/// memory before the title is told anything of this flight (its
/// bottom-of-pipe writes and fences come after, in order); CPU steps of
/// later submissions that read it find it declared and wait, or are done
/// here. Refused: a step of the submission that cannot be taken (a texture,
/// a fill of memory of another kind) reads the destination behind the copy,
/// or a refresh brings another host copy of it up to date ([`plan`],
/// [`elsewhere`]). While such a copy is in the air a submission that
/// refreshes another host copy of its destination, or reads it in a step
/// that is not checked otherwise, waits for it ([`copied_read`]); with the
/// copy done on the CPU it was there at once.
///
/// What stage 2 does not see: code of ours that reads title memory away
/// from a submission's steps (the pass over textures picked by index, while
/// a list is recorded). With the CPU's copy such a read of the destination
/// found the bytes at once; now it finds them when the flight has landed.
/// This is why stage 2 is its own.
pub(super) fn stage() -> u32 {
    static STAGE: OnceLock<u32> = OnceLock::new();
    *STAGE.get_or_init(|| {
        let stage = match std::env::var("VICTORY_CARD_STEPS").ok().as_deref().map(str::trim) {
            Some("1") => 1,
            Some("2") => 2,
            _ => 0,
        };
        if stage > 0 {
            tracing::info!(
                "steps that read memory flights may write are done on the card behind them (VICTORY_CARD_STEPS={stage}{})",
                if relay_exact() { "" } else { "; off: it needs the relay's second stage, VICTORY_RELAY=2" }
            );
        }
        stage
    })
}

/// The bytes of our buffer between reading and writing: what one submission
/// may do this way (each place read once, however many fills take it).
const BETWEEN: u64 = 4 << 20;

/// The most steps of a submission done this way.
const MOST: usize = 12288;

/// (Neptune) The host copy that stands for the tiled image (`crate::tiled`)
/// in [`Does::Refresh`]: every use of it is one copy of title memory.
const TILED: usize = 1;

/// How a fill's bytes get where its list reads them, if the card brings
/// them.
#[derive(Clone)]
pub(super) enum FillTo {
    /// A descriptor slot that shows the fill's memory, and nothing else does
    /// (title heap, slot): it is made to show our buffer instead.
    Slot(usize, u32),
    /// A buffer the card can write, filled from its beginning.
    Buffer(ID3D12Resource),
}

/// Title memory something in the air writes: the host copy it goes through
/// (`None`: none, the bytes reach title memory when the flight lands), from,
/// to, and whether it is the destination of a copy done here.
pub(super) struct Writer {
    shadow: Option<Arc<Shadow>>,
    from: usize,
    to: usize,
    copy: bool,
}

/// What `items` of the line write.
pub(super) fn writers(items: &[Late]) -> Vec<Writer> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Late::Run(flight) => {
                for (shadow, ranges) in &flight.back {
                    out.extend(ranges.iter().map(|&(from, to)| Writer {
                        shadow: Some(shadow.clone()),
                        from,
                        to,
                        copy: false,
                    }));
                }
                for &i in &flight.order {
                    // (Neptune) Its read-backs: five fields.
                    for &(_, at, [row, rows, pitch], ..) in &flight.lists[i].host().deferred().backs {
                        out.push(Writer {
                            shadow: None,
                            from: at,
                            to: at + rows.saturating_sub(1) * pitch + row,
                            copy: false,
                        });
                    }
                }
                out.extend(flight.own.iter().flat_map(Taken::copied));
            }
            Late::Write(at, _, bytes) => out.push(Writer {
                shadow: None,
                from: *at,
                to: *at + *bytes,
                copy: false,
            }),
            Late::Signal(..) => {}
        }
    }
    out
}

/// The destinations of copies done here that have not landed.
pub(super) fn copies(line: &Line) -> Vec<Writer> {
    if stage() < 2 {
        return Vec::new();
    }
    let flying = line.line.iter().filter_map(|l| match l {
        Late::Run(flight) => flight.own.as_ref(),
        _ => None,
    });
    let landing = line.landing.iter().filter(|w| w.copy).map(|w| Writer {
        shadow: w.shadow.clone(),
        ..*w
    });
    flying.flat_map(Taken::copied).chain(landing).collect()
}

/// Whether `runs` refresh another host copy of a destination in `copies`,
/// or read it in a step that flights are not waited for otherwise (work
/// on the CPU that names what it reads): that reads title memory, where
/// the copy is not yet.
pub(super) fn copied_read(copies: &[Writer], runs: &[&HostList]) -> bool {
    if copies.is_empty() {
        return false;
    }
    runs.iter().any(|list| {
        let deferred = list.deferred();
        deferred.copies.iter().any(|step| match step {
            Step::Refresh(shadow, at, len) => copies
                .iter()
                .any(|w| w.from < *at + *len && *at < w.to && w.shadow.as_ref().is_none_or(|s| !Arc::ptr_eq(s, shadow))),
            _ => false,
        })
            // (Neptune) The tiled image's uses are refreshes of it.
            || deferred.tiled.iter().any(|&(at, len)| {
                copies.iter().any(|w| w.from < at + len && at < w.to && w.shadow.as_ref().is_none_or(|s| !s.tiled()))
            })
            || deferred.host_reads.iter().any(|&(_, from, to)| copies.iter().any(|w| w.from < to && from < w.to))
    })
}

/// What a step of a submission does to title memory, as the plan sees it.
/// `asked`: flights in the air stop it from being done on the CPU now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Does {
    Copy { dst: usize, src: usize, len: usize, asked: bool },
    /// `to`: its memory can be given the bytes on the card.
    Fill { src: usize, len: usize, asked: bool, to: bool },
    /// Another read on the CPU (a texture, vertices expanded).
    Read { from: usize, to: usize, asked: bool },
    /// A refresh of a host copy (named by its address, or [`TILED`]).
    Refresh { shadow: usize, from: usize, to: usize },
    /// Anything, anywhere.
    Run,
}

fn meet(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// The bytes `d` reads and writes as a copy or a fill (empty ranges for
/// what it does not).
fn reads_writes(d: &Does) -> Option<((usize, usize), (usize, usize))> {
    match *d {
        Does::Copy { dst, src, len, .. } => Some(((src, src + len), (dst, dst + len))),
        Does::Fill { src, len, .. } => Some(((src, src + len), (0, 0))),
        _ => None,
    }
}

fn touched(ranges: &[(usize, usize)], r: (usize, usize)) -> bool {
    touches(ranges, r.0, r.1)
}

/// The steps of `does` (by place) to be done on the card, in order, each
/// with its level; or why not. All that are asked for, and every copy and
/// fill that is in the way of one that is done there: it writes what that
/// one reads or writes, or reads what it writes, whichever comes first. So
/// what is left on the CPU writes none of their bytes, and reads of them
/// only what a taken copy has not been told to write yet: bytes a step
/// ahead of it reads, in title memory as it is.
///
/// Steps of a level are done together, every read before every write; a
/// step that writes what an earlier one reads or writes, or reads what it
/// writes, is of a higher level. Our list does them level by level.
fn plan(does: &[Does], stage: u32) -> Result<Vec<(usize, u32)>, &'static str> {
    let takes = |d: &Does| match *d {
        Does::Copy { .. } if stage < 2 => Err("a buffer copy (VICTORY_CARD_STEPS=2 does it on the card)"),
        Does::Fill { to: false, .. } => Err("a fill of memory the card cannot be given the bytes of"),
        _ => Ok(()),
    };
    let mut taken = vec![false; does.len()];
    let (mut read, mut written) = (Vec::new(), Vec::new());
    let mut more = Vec::new();
    for (i, d) in does.iter().enumerate() {
        match *d {
            Does::Run => return Err("a step of ours runs"),
            Does::Read { asked: true, .. } => return Err("a texture reads it"),
            Does::Copy { asked: true, .. } | Does::Fill { asked: true, .. } => more.push(i),
            _ => {}
        }
    }
    while !more.is_empty() {
        // (Joined once for the round: one by one the 2,800 fills of a frame
        // sorted the ranges each time, 16 ms of the submitting thread.)
        let (mut reads, mut writes) = (Vec::new(), Vec::new());
        for i in std::mem::take(&mut more) {
            takes(&does[i])?;
            taken[i] = true;
            if let Some((r, w)) = reads_writes(&does[i]) {
                if w.0 < w.1 && meet(r, w) {
                    return Err("a copy onto itself");
                }
                reads.push(r);
                writes.push(w);
            }
        }
        join_ranges(&mut read, reads);
        if writes.iter().any(|w| w.0 < w.1) {
            join_ranges(&mut written, writes);
        }
        if taken.iter().filter(|t| **t).count() > MOST {
            return Err("too many steps depend on it");
        }
        for (i, d) in does.iter().enumerate() {
            // A fill that cannot be taken is looked at below.
            if taken[i] || matches!(d, Does::Fill { to: false, .. }) {
                continue;
            }
            if let Some((r, w)) = reads_writes(d)
                && (touched(&written, r) || touched(&written, w) || touched(&read, w))
            {
                more.push(i);
            }
        }
    }
    let order: Vec<usize> = (0..does.len()).filter(|&i| taken[i]).collect();
    // What stays on the CPU and reads what an earlier taken copy writes
    // would read it too early.
    let mut behind: Vec<(usize, usize)> = Vec::new();
    for (i, d) in does.iter().enumerate() {
        if taken[i] {
            if let Some((_, w)) = reads_writes(d)
                && w.0 < w.1
            {
                join_ranges(&mut behind, [w]);
            }
            continue;
        }
        match *d {
            Does::Read { from, to, .. } if touched(&behind, (from, to)) => {
                return Err("a texture or expanded vertices read what it writes");
            }
            Does::Fill { src, len, .. } if touched(&behind, (src, src + len)) => {
                return Err("a fill of memory the card cannot be given the bytes of reads what it writes");
            }
            _ => {}
        }
    }
    let mut out: Vec<(usize, u32)> = Vec::with_capacity(order.len());
    // The copies among them, which alone put others behind them or have to
    // stand behind others.
    let mut copies: Vec<usize> = Vec::new();
    for &i in &order {
        let Some((r, w)) = reads_writes(&does[i]) else {
            continue;
        };
        let writes = w.0 < w.1;
        let mut level = 0;
        // Behind what writes what this reads or writes, and, if this
        // writes, behind what reads it.
        let ahead: Box<dyn Iterator<Item = usize>> =
            if writes { Box::new(0..out.len()) } else { Box::new(copies.iter().copied()) };
        for k in ahead {
            let (j, at): (usize, u32) = out[k];
            let Some((jr, jw)) = reads_writes(&does[j]) else {
                continue;
            };
            if at >= level && (meet(jw, r) || meet(jw, w) || meet(jr, w)) {
                level = at + 1;
            }
        }
        if writes {
            copies.push(out.len());
        }
        out.push((i, level));
    }
    Ok(out)
}

/// Whether a refresh of `does` brings a host copy other than `shadow` up to
/// date in `[from, to)`.
fn elsewhere(does: &[Does], shadow: usize, from: usize, to: usize) -> bool {
    does.iter().any(|d| match *d {
        Does::Refresh { shadow: s, from: a, to: b } => s != shadow && meet((a, b), (from, to)),
        _ => false,
    })
}

/// (Neptune) How [`Does::Refresh`] names `shadow`: every use of the tiled
/// image is one host copy.
fn copy_id(shadow: &Arc<Shadow>) -> usize {
    if shadow.tiled() { TILED } else { Arc::as_ptr(shadow) as usize }
}

/// The host copy that holds `[from, to)` on the card once what `writers`
/// write is there: the one they all write it through, or without any the
/// one shaders would be given.
fn holder(writers: &[Writer], from: usize, to: usize) -> Result<Arc<Shadow>, &'static str> {
    let mut found: Option<&Arc<Shadow>> = None;
    for w in writers.iter().filter(|w| meet((w.from, w.to), (from, to))) {
        match (&w.shadow, found) {
            (None, _) => return Err("a read-back or a value behind a flight is written there"),
            (Some(s), Some(f)) if !Arc::ptr_eq(s, f) => return Err("flights write it through two host copies"),
            (Some(s), _) => found = Some(s),
        }
    }
    let shadow = match found {
        Some(s) => s.clone(),
        None => crate::shadow::find(from as u64, (to - from) as u64).ok_or("no host copy of it")?,
    };
    let (start, len) = shadow.span();
    if shadow.in_place() || (from as u64) < start || to as u64 > start + len {
        return Err("no host copy holds all of it");
    }
    // (Neptune) A window in a GPU upload heap (`VICTORY_WINDOW_REBAR`) is read
    // by the card in place: not a copy of ours to write.
    if shadow.rebar() {
        return Err("a window in a GPU upload heap");
    }
    Ok(shadow)
}

/// What the steps of `runs` do, and which step each is (its list by address,
/// its place in it). `written`: what flights may write.
fn does_of(written: &[(usize, usize)], runs: &[&HostList]) -> (Vec<Does>, Vec<(usize, usize)>) {
    // (Neptune) What `blocked_by` waits for.
    let wait_reads = !fill_ahead::on();
    let mut does = Vec::new();
    // Which step each is. One list's lock at a time: a submission may
    // name a list twice.
    let mut steps = Vec::new();
    for list in runs {
        let at = std::ptr::from_ref::<HostList>(list) as usize;
        let deferred = list.deferred();
        let mut host_reads = deferred.host_reads.iter().peekable();
        for (i, step) in deferred.copies.iter().enumerate() {
            does.push(match step {
                &Step::Copy(dst, src, len) => Does::Copy {
                    dst,
                    src,
                    len,
                    asked: touches(written, dst, dst + len) || wait_reads && touches(written, src, src + len),
                },
                Step::Fill(_, src, len, constants, to) => Does::Fill {
                    src: *src,
                    len: *len,
                    // (Neptune) A constant buffer of a table is read on the
                    // card; other constants as Neptune reads them.
                    asked: touches(written, *src, *src + *len)
                        && (matches!(to, Some(FillTo::Slot(..))) || !*constants && wait_reads),
                    to: to.is_some(),
                },
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    Does::Read {
                        from: at,
                        to: at + len,
                        asked: touches(written, at, at + len),
                    }
                }
                Step::Refresh(shadow, at, len) => Does::Refresh {
                    shadow: copy_id(shadow),
                    from: *at,
                    to: *at + *len,
                },
                Step::Run(_) => Does::Run,
                Step::Host(_) => {
                    // (Neptune) Work that names what it reads.
                    while host_reads.next_if(|r| r.0 < i).is_some() {}
                    match host_reads.next_if(|r| r.0 == i) {
                        Some(&(_, from, to)) if from < to => Does::Read { from, to, asked: false },
                        _ => continue,
                    }
                }
                Step::GpuWrite(..) => continue,
            });
            steps.push((at, i));
        }
        // (Neptune) The tiled image's uses, brought up to date ahead of every
        // list of the submission.
        for &(from, len) in &deferred.tiled {
            does.push(Does::Refresh { shadow: TILED, from, to: from + len });
            steps.push((usize::MAX, usize::MAX));
        }
    }
    (does, steps)
}

/// (Neptune) Whether `runs` fill a constant buffer of a table from memory the
/// flights may write: Neptune does not wait for those, so [`Host::card_steps`]
/// is asked about them although nothing else stops the submission.
fn tables_read(written: &[(usize, usize)], runs: &[&HostList]) -> bool {
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| {
            matches!(step, Step::Fill(_, src, len, _, Some(FillTo::Slot(..))) if touches(written, *src, *src + *len))
        })
    })
}

/// (Neptune) `VICTORY_GPU_BACK_MAX_KB` as `HostList::defer_gpu_write` reads
/// it: a declared write over it is not copied back, so a copy done here
/// that big would never reach title memory.
fn back_most() -> Option<usize> {
    static MOST: OnceLock<Option<usize>> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_GPU_BACK_MAX_KB").ok().and_then(|v| v.trim().parse::<usize>().ok()).map(|kb| kb << 10)
    })
}

/// Whether every page of `[from, to)` can be read (through the kernel:
/// asking Windows walks the page tables, milliseconds in the title's heap).
/// `known`: pages that could be, of the same submission: its fills' places
/// lie eight to a page, and each was read whole to find out.
fn mapped(from: usize, to: usize, known: &mut std::collections::HashSet<usize>) -> bool {
    const PAGE: usize = 4096;
    (from / PAGE..to.div_ceil(PAGE)).all(|page| {
        known.contains(&page) || {
            let there = read_into((page * PAGE).max(from) as *const u8, &mut [0]);
            if there {
                known.insert(page);
            }
            there
        }
    })
}

/// Where the bytes of the fill that is `step` (its list by address, its place
/// in it) go.
fn fill_to(runs: &[&HostList], step: (usize, usize)) -> Option<FillTo> {
    let list = runs.iter().find(|l| std::ptr::from_ref::<HostList>(l) as usize == step.0)?;
    match list.deferred().copies.get(step.1)? {
        Step::Fill(_, _, _, _, to) => to.clone(),
        _ => None,
    }
}

/// A step done on the card.
struct Op {
    from: (Arc<Shadow>, usize),
    len: usize,
    /// Its place in the buffer between.
    at: u64,
    level: u32,
    to: To,
}

enum To {
    /// A host copy, and the title address.
    Copy(Arc<Shadow>, usize),
    /// A descriptor slot (title heap, slot).
    Slot(usize, u32),
    /// A buffer's beginning.
    Buffer(ID3D12Resource),
}

/// Records a level of what [`stage`] describes into `list` (open): `reads`
/// are copied into `between` (source, offset in it, offset in `between`,
/// bytes), then `writes` out of it (destination, offset in it, offset in
/// `between`, bytes). `between` is left readable as constants (and must be
/// so, unless this is the `first` level), the others in the common state,
/// as the submission's lists expect them.
fn record(
    first: bool,
    list: &ID3D12GraphicsCommandList,
    between: &ID3D12Resource,
    reads: &[(&ID3D12Resource, u64, u64, u64)],
    writes: &[(&ID3D12Resource, u64, u64, u64)],
) {
    let common = D3D12_RESOURCE_STATE_COMMON;
    let (source, dest) = (D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut sources: Vec<&ID3D12Resource> = Vec::new();
    let mut dests: Vec<&ID3D12Resource> = Vec::new();
    for (all, list) in [(reads, &mut sources), (writes, &mut dests)] {
        for &(buffer, ..) in all {
            if !list.iter().any(|b| b.as_raw() == buffer.as_raw()) {
                list.push(buffer);
            }
        }
    }
    let both = |b: &ID3D12Resource| dests.iter().any(|d| d.as_raw() == b.as_raw());
    let constants = source | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    let mut barriers = vec![transition(between, dest, constants)];
    barriers.extend(sources.iter().map(|b| transition(b, source, if both(b) { dest } else { common })));
    let done: Vec<D3D12_RESOURCE_BARRIER> = dests.iter().map(|b| transition(b, dest, common)).collect();
    // SAFETY: an open list; live buffers, and the callers' ranges lie in
    // them. The buffers start in the common state (the list is the first of
    // its call) and are promoted by their first use.
    unsafe {
        if first {
            // What earlier lists' shaders wrote, first.
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            }]);
        } else {
            list.ResourceBarrier(&[transition(between, constants, dest)]);
        }
        for &(buffer, offset, at, len) in reads {
            list.CopyBufferRegion(between, at, buffer, offset, len);
        }
        list.ResourceBarrier(&barriers);
        for &(buffer, offset, at, len) in writes {
            list.CopyBufferRegion(buffer, offset, between, at, len);
        }
        if !done.is_empty() {
            list.ResourceBarrier(&done);
        }
    }
}

/// The list in front of a submission's, and its buffer ([`BETWEEN`] bytes).
struct Own {
    list: HostList,
    between: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded, and the list is touched by one
// thread at a time (the submitting one until it is sent, then the one that
// lands it, then nobody until it is taken out of `POOL`).
unsafe impl Send for Own {}

/// Those that have landed: closed, their steps dropped.
static POOL: Mutex<Vec<Own>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: fills and copies done on the card, their
/// bytes, and submissions that waited instead, refused by a check.
static DONE: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What [`Host::card_steps`] took off a submission: the list that does it,
/// for the submission to run first, and the steps the CPU leaves out (their
/// list by address, their place in it).
pub(super) struct Taken {
    own: Own,
    steps: std::collections::HashSet<(usize, usize)>,
    copied: Vec<(Arc<Shadow>, usize, usize)>,
}

impl Taken {
    pub(super) fn list(&self) -> &HostList {
        &self.own.list
    }

    /// Whether step `i` of `list` is done by this.
    pub(super) fn takes(&self, list: &HostList, i: usize) -> bool {
        self.steps.contains(&(std::ptr::from_ref(list) as usize, i))
    }

    fn copied(&self) -> impl Iterator<Item = Writer> {
        self.copied.iter().map(|(shadow, from, to)| Writer {
            shadow: Some(shadow.clone()),
            from: *from,
            to: *to,
            copy: true,
        })
    }

    /// The submission has run and landed: the list and its buffer are free.
    pub(super) fn landed(self) {
        // The host copies its steps held go; its commands stay until the
        // next use resets it.
        drop(std::mem::take(&mut *self.own.list.deferred()));
        POOL.lock().unwrap_or_else(|e| e.into_inner()).push(self.own);
    }
}

impl HostList {
    /// (Neptune) As [`HostList::defer_host`], for work that reads `len` bytes
    /// of title memory at `from`: what [`Host::card_steps`] must not leave
    /// behind a copy it does on the card.
    pub(crate) fn defer_host_read(&self, from: usize, len: usize, work: impl Fn() + Send + 'static) {
        self.defer_host(work);
        let mut deferred = self.deferred();
        let at = deferred.copies.len() - 1;
        deferred.host_reads.push((at, from, from + len));
    }

    /// (Neptune) As [`HostList::defer_constants`], for the constant buffer that
    /// only `slot` (a title heap's address, and the slot) shows: with
    /// [`stage`] the card may bring its bytes instead.
    pub(crate) fn defer_constants_in(&self, dst: *mut u8, src: *const u8, len: usize, slot: (usize, u32)) {
        self.defer_constants(dst, src, len);
        if stage() > 0
            && let Some(Step::Fill(.., to)) = self.deferred().copies.last_mut()
        {
            *to = Some(FillTo::Slot(slot.0, slot.1));
        }
    }
}

impl Host {
    /// With flights in the air: if what keeps `runs` from being prepared
    /// now is reads of title memory that can all be done on the card
    /// ([`stage`]), the list that does them. `None` if nothing is in the
    /// way or the old wait is due. Under the submission lock.
    pub(super) fn card_steps(&self, runs: &[&HostList]) -> Option<Taken> {
        if stage() == 0 || !relay_exact() {
            return None;
        }
        let (written, copies) = {
            let line = line();
            if line.written.is_empty() {
                return None;
            }
            (line.written.clone(), copies(&line))
        };
        // (Neptune) Or tables' constants read under the flights' writes,
        // which Neptune does not wait for.
        let blocked = blocked_by(&written, runs, false).is_some();
        if !blocked && !tables_read(&written, runs) {
            return None;
        }
        let taken = if copied_read(&copies, runs) {
            Err("a copy done on the card is read through another host copy")
        } else {
            self.take_steps(&written, runs)
        };
        match taken {
            Ok(taken) => taken,
            Err(_why) => {
                if blocked {
                    DONE[3].fetch_add(1, Ordering::Relaxed);
                }
                None
            }
        }
    }

    fn take_steps(&self, written: &[(usize, usize)], runs: &[&HostList]) -> Result<Option<Taken>, &'static str> {
        let (does, steps) = does_of(written, runs);
        let chosen = plan(&does, stage())?;
        if chosen.is_empty() {
            return Ok(None);
        }
        let mut writers = {
            let line = line();
            let mut all = writers(&line.line);
            all.extend(line.landing.iter().map(|w| Writer {
                shadow: w.shadow.clone(),
                ..*w
            }));
            all
        };
        let mut ops: Vec<Op> = Vec::with_capacity(chosen.len());
        // The places read (host copy, address, bytes, level): where in the
        // buffer between their bytes are.
        let mut places: std::collections::HashMap<(usize, usize, usize, u32), u64> = std::collections::HashMap::new();
        // Some 2,800 fills a frame, of a few hundred places: what holds each
        // is looked for once, as long as no copy is added to the writers.
        let mut held: std::collections::HashMap<(usize, usize), Arc<Shadow>> = std::collections::HashMap::new();
        let mut room = std::collections::HashSet::new();
        let mut end = 0u64;
        for &(i, level) in &chosen {
            let Some(((src, _), (dst, _))) = reads_writes(&does[i]) else {
                continue;
            };
            let (Does::Copy { len, .. } | Does::Fill { len, .. }) = does[i] else {
                continue;
            };
            if len == 0 {
                return Err("nothing to read");
            }
            let from = match held.get(&(src, len)) {
                Some(from) => from.clone(),
                None => {
                    let from = holder(&writers, src, src + len)?;
                    held.insert((src, len), from.clone());
                    from
                }
            };
            let place = (Arc::as_ptr(&from) as usize, src, len, level);
            let at = match places.get(&place) {
                Some(&at) => at,
                None => {
                    if end + len as u64 > BETWEEN {
                        return Err("too much at once");
                    }
                    if !mapped(src, src + len, &mut room) {
                        return Err("not all of what it reads is mapped");
                    }
                    let at = end;
                    end = (end + len as u64).next_multiple_of(256);
                    places.insert(place, at);
                    at
                }
            };
            let to = match does[i] {
                Does::Copy { .. } => {
                    // (Neptune) A declared write over `VICTORY_GPU_BACK_MAX_KB`
                    // is not copied back.
                    if back_most().is_some_and(|most| len > most || most == 0) {
                        return Err("its destination would not be copied back (VICTORY_GPU_BACK_MAX_KB)");
                    }
                    let to = holder(&writers, dst, dst + len)?;
                    if elsewhere(&does, copy_id(&to), dst, dst + len) {
                        return Err("another host copy of what it writes is read");
                    }
                    if !mapped(dst, dst + len, &mut room) {
                        return Err("not all of what it writes is mapped");
                    }
                    // What the steps behind it read of it is in that host
                    // copy, or nowhere.
                    writers.push(Writer {
                        shadow: Some(to.clone()),
                        from: dst,
                        to: dst + len,
                        copy: true,
                    });
                    held.clear();
                    To::Copy(to, dst)
                }
                _ => match fill_to(runs, steps[i]) {
                    Some(FillTo::Slot(heap, index)) => To::Slot(heap, index),
                    Some(FillTo::Buffer(buffer)) => To::Buffer(buffer),
                    None => continue,
                },
            };
            ops.push(Op {
                from: (from, src),
                len,
                at,
                level,
                to,
            });
        }
        let Some(own) = self.own_list() else {
            return Err("no host list or buffer for it");
        };
        // First every refresh, then the writes: the bytes compared with are
        // the title's. (One step for each host copy, over all that is read of
        // it: a flush brings a copy up to date from the first range asked for
        // to the last, whatever the steps.)
        let mut spans: std::collections::HashMap<usize, (Arc<Shadow>, usize, usize)> = std::collections::HashMap::new();
        for op in &ops {
            let to = match &op.to {
                To::Copy(to, dst) => Some((to, *dst)),
                _ => None,
            };
            for (shadow, at) in std::iter::once((&op.from.0, op.from.1)).chain(to) {
                let span = spans.entry(Arc::as_ptr(shadow) as usize).or_insert_with(|| (shadow.clone(), at, at + op.len));
                (span.1, span.2) = (span.1.min(at), span.2.max(at + op.len));
            }
        }
        for (shadow, from, to) in spans.into_values() {
            shadow.refresh(&own.list, from as u64, (to - from) as u64);
        }
        let mut copied = Vec::new();
        for op in &ops {
            if let To::Copy(to, dst) = &op.to {
                to.gpu_writes(&own.list, *dst as u64, op.len as u64);
                copied.push((to.clone(), *dst, *dst + op.len));
            }
        }
        let levels = ops.iter().map(|op| op.level).max().unwrap_or(0);
        for level in 0..=levels {
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            for op in ops.iter().filter(|op| op.level == level) {
                let len = op.len as u64;
                let read = (&op.from.0.buffer, op.from.0.offset(op.from.1 as u64), op.at, len);
                // (Each place once.)
                if places.remove(&(Arc::as_ptr(&op.from.0) as usize, op.from.1, op.len, level)).is_some() {
                    reads.push(read);
                }
                match &op.to {
                    To::Copy(to, dst) => writes.push((&to.buffer, to.offset(*dst as u64), op.at, len)),
                    To::Buffer(buffer) => writes.push((buffer, 0, op.at, len)),
                    To::Slot(..) => {}
                }
            }
            record(level == 0, &own.list.list, &own.between, &reads, &writes);
        }
        own.list.close();
        // SAFETY: a live buffer.
        let gpu = unsafe { own.between.GetGPUVirtualAddress() };
        // (Neptune: on this thread, in the steps' order.)
        for op in &ops {
            if let To::Slot(heap, index) = &op.to {
                // (A heap that is gone: nothing can read the slot.)
                crate::descriptor::constant_slot(*heap, *index, gpu + op.at, op.len as u32);
            }
        }
        let copies = ops.iter().filter(|op| matches!(op.to, To::Copy(..))).count();
        DONE[0].fetch_add((ops.len() - copies) as u64, Ordering::Relaxed);
        DONE[1].fetch_add(copies as u64, Ordering::Relaxed);
        DONE[2].fetch_add(ops.iter().map(|op| op.len as u64).sum(), Ordering::Relaxed);
        Ok(Some(Taken {
            own,
            steps: chosen.iter().map(|&(i, _)| steps[i]).collect(),
            copied,
        }))
    }

    /// A list of ours open for recording, and its buffer.
    fn own_list(&self) -> Option<Own> {
        let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(own) = pooled {
            own.list.reset();
            return Some(own);
        }
        let between = self.uav_buffer(BETWEEN)?;
        name(&between, || "copies of title memory on the card".into());
        let list = self.new_list().ok()?;
        Some(Own { list, between })
    }
}

/// (Neptune) With [`stage`] and the relay's second stage, for a submission
/// that refreshes ranges of host copies flights may write which the title
/// wrote too (`stale_under_flights`): only the flights that may write in
/// what `runs` refresh of those copies land, not everything in the air, and
/// the bytes are looked at again (`ranges`: the submission's one asking,
/// made again). Whether that was enough; if not, the caller settles as
/// before. From cool_fox's `Host::settle_stale`; Neptune's
/// `stale_under_flights` says only whether, so the flights waited for are
/// those of every refreshed range of a copy shaders may be writing.
pub(super) fn stale_landed(runs: &[&HostList], ranges: &[(usize, usize)]) -> bool {
    if stage() == 0 || !relay_exact() {
        return false;
    }
    for _ in 0..3 {
        let mut spans: Vec<(Arc<Shadow>, usize, usize)> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::Refresh(shadow, address, len) = step
                    && shadow.writes_pending_in(*address, *address + *len)
                {
                    match spans.iter_mut().find(|s| Arc::ptr_eq(&s.0, shadow)) {
                        Some(s) => (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len)),
                        None => spans.push((shadow.clone(), *address, *address + *len)),
                    }
                }
            }
        }
        if spans.is_empty() {
            // (A window in a GPU upload heap, which waits for its readers.)
            return false;
        }
        relay_wait(WHY_STALE, |_| spans.iter().all(|(shadow, from, to)| !shadow.writes_pending_in(*from, *to)));
        crate::watch::ask_once(ranges);
        if !stale_under_flights(runs) {
            return true;
        }
    }
    false
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    if stage() == 0 {
        return;
    }
    let d: Vec<f64> = DONE.iter().map(|d| d.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "steps done on the card instead of waiting for flights, per frame (VICTORY_CARD_STEPS={}): {:.2} fills, {:.2} buffer \
         copies, {:.1} KB; {:.2} submissions refused and waited",
        stage(),
        d[0],
        d[1],
        d[2] / 1024.0,
        d[3],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(dst: usize, src: usize, len: usize, asked: bool) -> Does {
        Does::Copy { dst, src, len, asked }
    }

    fn fill(src: usize, len: usize, asked: bool) -> Does {
        Does::Fill { src, len, asked, to: true }
    }

    fn stuck(src: usize, len: usize) -> Does {
        Does::Fill { src, len, asked: false, to: false }
    }

    /// What is done on the card, as (place, level), in stage 2.
    fn planned(does: &[Does]) -> Vec<(usize, u32)> {
        plan(does, 2).unwrap()
    }

    /// Only what is asked for is taken, all of it, and a step that stands
    /// in none of their ways stays where it is.
    #[test]
    fn steps_apart_from_one_another_are_taken() {
        let does = [
            copy(1000, 5000, 100, false),
            fill(2000, 64, true),
            Does::Refresh { shadow: 1, from: 0, to: 9000 },
            copy(3000, 6000, 100, true),
            fill(1000, 64, false),
            Does::Read { from: 3100, to: 4000, asked: false },
            stuck(3100, 5),
        ];
        assert_eq!(planned(&does), [(1, 0), (3, 0)]);
        // Two fills of the same bytes.
        assert_eq!(plan(&[fill(0, 64, true), fill(32, 64, true)], 1), Ok(vec![(0, 0), (1, 0)]));
        assert_eq!(plan(&[fill(0, 64, false)], 1), Ok(vec![]));
        // Ends that touch are apart.
        let does = [
            copy(1000, 5000, 100, true),
            copy(900, 9000, 100, false),
            copy(1100, 9000, 5, false),
            copy(4900, 9000, 100, false),
        ];
        assert_eq!(planned(&does), [(0, 0)]);
    }

    /// What depends on a taken step, or it on it, is taken too, on the level
    /// the order of the steps asks for.
    #[test]
    fn steps_that_depend_on_one_another_are_taken_in_order() {
        let taken = copy(1000, 5000, 100, true);
        // Reads what it writes, behind and ahead of it.
        assert_eq!(planned(&[taken, fill(1099, 64, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[fill(1099, 64, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(7000, 1050, 10, false)]), [(0, 0), (1, 1)]);
        // Writes what it reads, and what it writes.
        assert_eq!(planned(&[copy(5099, 9000, 1, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(5099, 9000, 1, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(999, 9000, 2, false)]), [(0, 0), (1, 1)]);
        // The same list twice.
        assert_eq!(planned(&[taken, taken]), [(0, 0), (1, 1)]);
        // A chain, with steps apart from it in between, and two fills of one
        // copy on one level.
        let does = [
            copy(8000, 7000, 50, false),
            taken,
            copy(2000, 1010, 20, false),
            fill(90_000, 10, false),
            fill(2000, 20, false),
            fill(2010, 20, true),
            copy(1000, 2005, 5, false),
            fill(1000, 100, false),
        ];
        assert_eq!(planned(&does), [(1, 0), (2, 1), (4, 2), (5, 2), (6, 2), (7, 3)]);
        // A fill is in nobody's way.
        assert_eq!(planned(&[taken, fill(5000, 100, false)]), [(0, 0)]);
        // A copy ahead that a taken one only shares its source with.
        assert_eq!(planned(&[copy(7000, 5000, 10, false), taken]), [(1, 0)]);
    }

    /// Whatever the steps, done level by level (in a level every read
    /// before every write) they read and leave what they read and leave one
    /// after the other, the CPU's steps ahead of all of them.
    #[test]
    fn levels_do_what_the_order_does() {
        let mut seed = 12345u32;
        let mut random = move |n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % n
        };
        let (mut done, mut refused) = (0, 0);
        for _ in 0..4000 {
            let does: Vec<Does> = (0..2 + random(9))
                .map(|_| {
                    let (at, other, len) = (random(60) as usize, random(60) as usize, 1 + random(8) as usize);
                    let asked = random(4) == 0;
                    match random(8) {
                        0..4 => copy(at, other, len, asked),
                        4..6 => fill(at, len, asked),
                        6 => stuck(at, len),
                        _ => Does::Read { from: at, to: at + len, asked: false },
                    }
                })
                .collect();
            let Ok(chosen) = plan(&does, 2) else {
                refused += 1;
                continue;
            };
            done += 1;
            let start: Vec<u32> = (0..80).collect();
            // One after the other: what each reads, and what is left.
            let mut memory = start.clone();
            let mut wanted = Vec::new();
            for d in &does {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                wanted.push(memory[from..to].to_vec());
                if let Does::Copy { dst, src, len, .. } = *d {
                    memory.copy_within(src..src + len, dst);
                }
            }
            // The CPU's, then the card's.
            let mut cpu = start.clone();
            let taken = |i: usize| chosen.iter().any(|c| c.0 == i);
            for (i, d) in does.iter().enumerate().filter(|&(i, _)| !taken(i)) {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                assert_eq!(cpu[from..to], wanted[i][..], "step {i} of {does:?}");
                if let Does::Copy { dst, src, len, .. } = *d {
                    cpu.copy_within(src..src + len, dst);
                }
            }
            let mut card = cpu.clone();
            for level in 0..=chosen.iter().map(|c| c.1).max().unwrap_or(0) {
                let of: Vec<usize> = chosen.iter().filter(|c| c.1 == level).map(|c| c.0).collect();
                let before = card.clone();
                for i in of {
                    let ((from, to), (dst, _)) = reads_writes(&does[i]).unwrap();
                    assert_eq!(before[from..to], wanted[i][..], "step {i} of {does:?}: {chosen:?}");
                    if matches!(does[i], Does::Copy { .. }) {
                        card[dst..dst + to - from].copy_from_slice(&before[from..to]);
                    }
                }
            }
            assert_eq!(card, memory, "{does:?}: {chosen:?}");
        }
        assert!(done > 500 && refused > 100, "{done} {refused}");
    }

    /// What cannot be done on the card refuses the lot, where it stands in
    /// the way.
    #[test]
    fn steps_in_the_way_are_refused() {
        let taken = copy(1000, 5000, 100, true);
        assert!(plan(&[taken, Does::Run], 2).is_err());
        // A texture that itself waits for flights; a fill that cannot be
        // taken but waits.
        assert!(plan(&[taken, Does::Read { from: 1, to: 2, asked: true }], 2).is_err());
        assert!(plan(&[Does::Fill { src: 1, len: 1, asked: true, to: false }], 2).is_err());
        // Reads on the CPU of what it writes: behind it, not ahead of it.
        for read in [Does::Read { from: 900, to: 1001, asked: false }, stuck(1050, 3)] {
            assert!(plan(&[taken, read], 2).is_err(), "{read:?} behind");
            assert_eq!(planned(&[read, taken]), [(1, 0)], "{read:?} ahead");
        }
        // Also of what is taken along.
        assert!(plan(&[taken, copy(3000, 1000, 10, false), stuck(3005, 1)], 2).is_err());
        // Into itself, and copies without stage 2.
        assert!(plan(&[copy(1000, 1050, 100, true)], 2).is_err());
        assert!(plan(&[taken], 1).is_err());
        assert!(plan(&[fill(1000, 4, true), copy(1000, 5000, 100, false)], 1).is_err());
    }

    #[test]
    fn a_refresh_of_another_host_copy_is_seen() {
        let does = [Does::Refresh { shadow: 7, from: 100, to: 200 }];
        assert!(!elsewhere(&does, 7, 100, 200));
        assert!(elsewhere(&does, 8, 199, 300));
        assert!(!elsewhere(&does, 8, 200, 300));
    }

    /// A value written behind a flight, or a read-back, is in no host copy
    /// before it lands.
    #[test]
    fn what_no_host_copy_holds_is_refused() {
        let writers = [Writer { shadow: None, from: 100, to: 108, copy: false }];
        assert!(holder(&writers, 104, 120).is_err());
    }
}
