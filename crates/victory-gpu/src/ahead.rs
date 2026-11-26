//! Local (VICTORY_ASYNC_REFRESH): a submission's refreshes of host copies done off the
//! submitting thread, with the host queue waiting for them on the card.
//!
//! Design taken from xcl (`crates/xcl/src/hle/d3d12x/sparse.rs`, GPL-2.0, module doc and
//! `before_execute` / `submit_slot`): the CPU side does not stall for the upload; the bytes
//! go up on a queue of their own, and the title's queue `Wait`s for that queue's fence before
//! the title's lists. No code of xcl is copied here.
//!
//! What changes with `VICTORY_ASYNC_REFRESH=1` (off by default; with it off nothing here runs
//! and `plan` / `claimed` / `send` / `drain` return at once):
//!
//! A submission sent ahead (`late`) hands the refreshes at its end to one thread of ours (the
//! "worker") and returns. The worker asks the write watch about them, copies what changed
//! into the copies (on the wide threads with `VICTORY_WIDE_REFRESH=1`, as the flush does),
//! puts the changed ranges of copies in the card's memory on a direct queue of its own
//! (`Side::queue`) behind a wait for everything the host queue had before this submission's
//! lists, and signals `Side::done` on that queue. The host queue was told, before the lists,
//! to signal `Side::reached` and then to wait for `Side::done`: the lists run only once the
//! copies are in. The next submission (any thread) waits on the CPU for the worker first
//! ([`drain`]): one job at most is open, submissions stay in order, and nothing else touches
//! a copy or the write-watch journal while the worker does.
//!
//! Which refreshes go (see [`plan`]): those after the submission's last step that writes
//! title memory (`Step::Copy`, `Step::Run`: before such a step every pending refresh is done
//! by the submitting thread, as before), of copies that this submission's shaders do not
//! write (no `Step::GpuWrite` of the copy), that no flight's shaders may be writing
//! (`pending` empty: the merges, `merge_under_writes`, `VICTORY_LATE_WAIT` stage 2 and the
//! relay's notes all assume the submitting thread's order), that are not in place, and that
//! have no "before" twin on the card (`Host::card_before` hashes such a copy's pieces again
//! as they go up). Everything else is refreshed as before, on the submitting thread, and
//! sent by `Host::card_before` on the host queue.
//!
//! What it relies on: the title does not change memory a submission reads between its
//! `ExecuteCommandLists` and that submission's fence (on the console the card reads it when
//! the lists run, which is later still). Pages the title writes meanwhile beside what the
//! lists read (the next slot of a ring) may be taken by the worker as they are then; the
//! write watch has their pages marked again, so the next refresh takes them again.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{HostList, Step};
use crate::shadow::{AheadSend, Shadow};

/// `VICTORY_ASYNC_REFRESH=1`.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_REFRESH").is_some_and(|v| v == "1"))
}

/// One submission's refreshes for the worker: per copy the range that covers them (as
/// `Batch` merges them), and the number the queues signal and wait for.
struct Job {
    refreshes: Vec<(Arc<Shadow>, usize, usize)>,
    number: u64,
    sent: std::time::Instant,
}

/// The upload path of our own and the worker's line.
struct Side {
    queue: ID3D12CommandQueue,
    /// Signalled by the host queue before it waits: everything it had before the lists of
    /// submission `n` is done when this reaches `n`.
    reached: ID3D12Fence,
    /// Signalled by [`Side::queue`] when submission `n`'s refreshes and their sends are in.
    done: ID3D12Fence,
    jobs: Mutex<std::sync::mpsc::Sender<Job>>,
}

// SAFETY: D3D12 queues and fences are free-threaded; the sender is behind a lock.
unsafe impl Send for Side {}
// SAFETY: as above.
unsafe impl Sync for Side {}

/// The side, made at the first submission that has something for it; `None` (and every
/// refresh on the submitting thread, as without the switch) if it cannot be made.
fn side() -> Option<&'static Side> {
    static SIDE: OnceLock<Option<Side>> = OnceLock::new();
    SIDE.get_or_init(|| {
        let host = super::host()?;
        let made = (|| -> windows::core::Result<(Side, std::sync::mpsc::Receiver<Job>)> {
            // SAFETY: plain object creation on our device.
            unsafe {
                let queue: ID3D12CommandQueue = host.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
                let reached: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let done: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let (tx, rx) = std::sync::mpsc::channel();
                Ok((Side { queue, reached, done, jobs: Mutex::new(tx) }, rx))
            }
        })();
        match made {
            Ok((side, rx)) => {
                let spawned = std::thread::Builder::new().name("victory-ahead".into()).spawn(move || worker(rx));
                if let Err(e) = spawned {
                    tracing::warn!(error = %e, "VICTORY_ASYNC_REFRESH=1: no worker thread; refreshes stay on the submitting thread");
                    return None;
                }
                tracing::info!(
                    "refreshes at the end of a submission sent ahead are done by a thread of ours, the host queue waits for them \
                     on the card (VICTORY_ASYNC_REFRESH=1)"
                );
                Some(side)
            }
            Err(e) => {
                tracing::warn!(error = %e, "VICTORY_ASYNC_REFRESH=1: no queue or fences of its own; refreshes stay on the submitting thread");
                None
            }
        }
    })
    .as_ref()
}

thread_local! {
    /// The submitting thread's plan: the addresses of the claimed `Step::Refresh` steps, and
    /// the refreshes per copy (copy address to place in the list). Set by [`plan`] at every
    /// submission (emptied first), taken by [`send`].
    static CLAIMED: std::cell::RefCell<HashSet<usize>> = std::cell::RefCell::new(HashSet::new());
    static REFRESHES: std::cell::RefCell<(Vec<(Arc<Shadow>, usize, usize)>, HashMap<usize, usize>)> =
        std::cell::RefCell::new((Vec::new(), HashMap::new()));
}

/// Jobs sent and not finished by the worker, and the word that one finished.
static OPEN: (Mutex<Vec<(u64, HashSet<usize>)>>, Condvar) = (Mutex::new(Vec::new()), Condvar::new());

/// Local (VICTORY_ASYNC_OVERLAP=1, with VICTORY_ASYNC_REFRESH=1; off by default): a submission
/// sent ahead waits for open jobs only where they touch a copy it names too (a refresh, a
/// declared write); one with steps that write title memory (`Step::Copy`, `Step::Run`), or one
/// that is not sent ahead, waits for all of them, as without the switch. The worker then asks
/// per copy, not once per job (the journal's one standing asking is the submission's).
fn overlap() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_OVERLAP").is_some_and(|v| v == "1"))
}

/// The number of the last job ([`Job::number`]): raised under the submission lock only.
static NUMBER: AtomicU64 = AtomicU64::new(0);

/// Since the last log line: submissions sent ahead, of them with a job; refresh steps of
/// those submissions, of them claimed; copies refreshed by the worker; bytes it copied into
/// copies; bytes it sent up; worker microseconds in all, of them asking, of them sending;
/// microseconds the submitting thread spent planning and sending; microseconds submissions
/// waited for the worker ([`drain`]) and how many waited; jobs that ended after the host
/// queue had reached their wait, and the microseconds from send to end of those (the most
/// the card can have stood for them); jobs that failed (a panic).
static STATS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
const S_SUBS: usize = 0;
const S_JOBS: usize = 1;
const S_STEPS: usize = 2;
const S_CLAIMED: usize = 3;
const S_COPIES: usize = 4;
const S_COPIED: usize = 5;
const S_SENT: usize = 6;
const S_WORK_US: usize = 7;
const S_ASK_US: usize = 8;
const S_SEND_US: usize = 9;
const S_PLAN_US: usize = 10;
const S_DRAIN_US: usize = 11;
const S_DRAINS: usize = 12;
const S_LATE: usize = 13;
const S_LATE_US: usize = 14;
const S_FAILED: usize = 15;
/// Local (VICTORY_ASYNC_OVERLAP): submissions that went on beside open jobs.
const S_PASSED: usize = 16;

fn add(i: usize, n: u64) {
    STATS[i].fetch_add(n, Ordering::Relaxed);
}

/// Before a submission is prepared (under the submission lock): waits until the worker has
/// finished the job of the one before, if it has one.
pub(super) fn drain(runs: &[&HostList], late: bool) {
    if !on() {
        return;
    }
    // Local (VICTORY_ASYNC_OVERLAP): the copies this submission names, unless it must wait
    // for every job.
    let touched: Option<HashSet<usize>> = (overlap() && late)
        .then(|| {
            let mut set = HashSet::new();
            for list in runs {
                for step in &list.deferred().copies {
                    match step {
                        Step::Copy(..) | Step::Run(_) => return None,
                        Step::Refresh(shadow, ..) | Step::GpuWrite(shadow, ..) => {
                            set.insert(Arc::as_ptr(shadow) as usize);
                        }
                        _ => {}
                    }
                }
            }
            Some(set)
        })
        .flatten();
    let blocks = |open: &Vec<(u64, HashSet<usize>)>| match &touched {
        None => !open.is_empty(),
        Some(t) => open.iter().any(|(_, keys)| !keys.is_disjoint(t)),
    };
    let mut open = OPEN.0.lock().unwrap_or_else(|e| e.into_inner());
    if !blocks(&open) {
        if !open.is_empty() {
            add(S_PASSED, 1);
        }
        return;
    }
    let waiting = std::time::Instant::now();
    while blocks(&open) {
        open = OPEN.1.wait(open).unwrap_or_else(|e| e.into_inner());
    }
    add(S_DRAIN_US, waiting.elapsed().as_micros() as u64);
    add(S_DRAINS, 1);
}

/// At the start of a submission's preparing: which of its refreshes the worker takes
/// (module doc). Only for a submission sent ahead (`late`); otherwise, and with the switch
/// off, the plan is empty.
pub(super) fn plan(runs: &[&HostList], late: bool) {
    if !on() {
        return;
    }
    let started = std::time::Instant::now();
    CLAIMED.with(|c| c.borrow_mut().clear());
    REFRESHES.with(|r| {
        let mut r = r.borrow_mut();
        r.0.clear();
        r.1.clear();
    });
    if !late {
        return;
    }
    add(S_SUBS, 1);
    // In submission order: each refresh with its place, the last step that writes title
    // memory, the copies shaders write.
    let mut candidates: Vec<(usize, usize, Arc<Shadow>, usize, usize)> = Vec::new();
    let mut barrier: Option<usize> = None;
    let mut written: HashSet<usize> = HashSet::new();
    let mut index = 0usize;
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                Step::Copy(..) | Step::Run(_) => barrier = Some(index),
                Step::GpuWrite(shadow, ..) => {
                    written.insert(Arc::as_ptr(shadow) as usize);
                }
                Step::Refresh(shadow, address, len) => {
                    candidates.push((index, step as *const Step as usize, shadow.clone(), *address, *len));
                }
                Step::Fill(..) | Step::Host(_) | Step::Texture(_) => {}
            }
            index += 1;
        }
    }
    add(S_STEPS, candidates.len() as u64);
    if candidates.is_empty() || side().is_none() {
        add(S_PLAN_US, started.elapsed().as_micros() as u64);
        return;
    }
    // Per copy, once: whether it may go.
    let mut eligible: HashMap<usize, bool> = HashMap::new();
    let mut claimed = 0u64;
    CLAIMED.with(|c| {
        REFRESHES.with(|r| {
            let mut c = c.borrow_mut();
            let mut r = r.borrow_mut();
            let (list, places) = &mut *r;
            for (at, step, shadow, address, len) in candidates {
                if barrier.is_some_and(|b| at < b) || len == 0 {
                    continue;
                }
                let key = Arc::as_ptr(&shadow) as usize;
                if written.contains(&key) || !*eligible.entry(key).or_insert_with(|| shadow.ahead_eligible()) {
                    continue;
                }
                c.insert(step);
                claimed += 1;
                let end = address + len;
                match places.get(&key) {
                    Some(&i) => {
                        let e = &mut list[i];
                        (e.1, e.2) = (e.1.min(address), e.2.max(end));
                    }
                    None => {
                        places.insert(key, list.len());
                        list.push((shadow, address, end));
                    }
                }
            }
        })
    });
    add(S_CLAIMED, claimed);
    add(S_PLAN_US, started.elapsed().as_micros() as u64);
}

/// Whether `step` (a `Step::Refresh` of the submission being prepared) is the worker's.
pub(super) fn claimed(step: &Step) -> bool {
    if !on() {
        return false;
    }
    CLAIMED.with(|c| {
        let c = c.borrow();
        !c.is_empty() && c.contains(&(step as *const Step as usize))
    })
}

/// Right before the submission's lists go to the host queue (submission lock held): the
/// claimed refreshes go to the worker, and the queue is told to wait for them.
pub(super) fn send(host: &super::Host) {
    if !on() {
        return;
    }
    let started = std::time::Instant::now();
    CLAIMED.with(|c| c.borrow_mut().clear());
    let refreshes = REFRESHES.with(|r| {
        let mut r = r.borrow_mut();
        r.1.clear();
        std::mem::take(&mut r.0)
    });
    if refreshes.is_empty() {
        return;
    }
    let Some(side) = side() else {
        // (Not reached: `plan` claims nothing without the side.)
        run_here(refreshes);
        return;
    };
    let number = NUMBER.fetch_add(1, Ordering::AcqRel) + 1;
    // SAFETY: our fences, the host's queue (under the submission lock, as every use of it).
    let told = unsafe {
        host.queue
            .Signal(&side.reached, number)
            .and_then(|()| host.queue.Wait(&side.done, number))
    };
    if let Err(e) = told {
        // The lists cannot be held for the worker: the refreshes are done here, now (what
        // they change on card copies goes up with the next submission's send; the device
        // is most likely gone anyway).
        tracing::warn!(error = %e, "VICTORY_ASYNC_REFRESH=1: the host queue could not be made to wait; refreshed on the submitting thread");
        run_here(refreshes);
        return;
    }
    let keys: HashSet<usize> = refreshes.iter().map(|r| Arc::as_ptr(&r.0) as usize).collect();
    OPEN.0.lock().unwrap_or_else(|e| e.into_inner()).push((number, keys));
    let job = Job {
        refreshes,
        number,
        sent: std::time::Instant::now(),
    };
    let gone = side.jobs.lock().unwrap_or_else(|e| e.into_inner()).send(job);
    if let Err(std::sync::mpsc::SendError(job)) = gone {
        // The worker is gone: done here, and the queue let through.
        tracing::warn!("VICTORY_ASYNC_REFRESH=1: the worker is gone; refreshed on the submitting thread");
        let mut slots = Vec::new();
        run(side, &mut slots, &job);
        // The card may still be reading the list: kept for the rest of the run.
        std::mem::forget(slots);
        finished(job.number);
    }
    add(S_JOBS, 1);
    add(S_PLAN_US, started.elapsed().as_micros() as u64);
    log_line();
}

/// The refreshes on this thread, as the flush does them (fallback only).
fn run_here(refreshes: Vec<(Arc<Shadow>, usize, usize)>) {
    for (shadow, from, to) in refreshes {
        super::Batch::update(&shadow, from, to);
    }
}

/// A list of the side's and what it reads, kept until the side's fence passes its number.
struct Slot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    number: u64,
    keep: Vec<Arc<Shadow>>,
    chunks: Vec<crate::windowup::Chunk>,
}

fn worker(jobs: std::sync::mpsc::Receiver<Job>) {
    // (`side` starts this thread from inside its own making: this waits for that to end.)
    let Some(side) = side() else {
        return;
    };
    let mut slots: Vec<Slot> = Vec::new();
    for job in jobs {
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(side, &mut slots, &job)));
        if ran.is_err() {
            add(S_FAILED, 1);
            tracing::error!(
                number = job.number,
                "VICTORY_ASYNC_REFRESH=1: a job of refreshes failed; its lists run with whatever the copies hold"
            );
            // The host queue waits for this number: it must come.
            // SAFETY: our queue and fence.
            let _ = unsafe { side.queue.Signal(&side.done, job.number) };
        }
        finished(job.number);
    }
}

/// A job is done: the next submission may be prepared.
fn finished(number: u64) {
    let mut open = OPEN.0.lock().unwrap_or_else(|e| e.into_inner());
    open.retain(|(n, _)| *n != number);
    drop(open);
    OPEN.1.notify_all();
}

/// Does `job`: asks, refreshes, sends what changed in card copies on the side's queue, and
/// signals the side's fence with the job's number (always, also with nothing to send).
fn run(side: &Side, slots: &mut Vec<Slot>, job: &Job) {
    let started = std::time::Instant::now();
    // What earlier sends read can go now.
    // SAFETY: a plain query of our fence.
    let completed = unsafe { side.done.GetCompletedValue() };
    for slot in slots.iter_mut() {
        if slot.number <= completed {
            slot.keep.clear();
            slot.chunks.clear();
        }
    }
    // The asking, as the submission's own (`crate::watch::ask_once`), for these ranges.
    let ranges: Vec<(usize, usize)> = job.refreshes.iter().filter(|r| !r.0.unwatched()).map(|r| (r.1, r.2)).collect();
    // Local (VICTORY_ASYNC_OVERLAP): not while submissions are prepared beside it.
    let once = crate::watch::once() && !overlap();
    if once {
        crate::watch::ask_once(&ranges);
    } else if ranges.len() > 1 && !overlap() {
        crate::watch::ask_ahead(&ranges);
    }
    add(S_ASK_US, started.elapsed().as_micros() as u64);
    // The refreshes, as `Batch::flush` does them.
    let copied = Arc::new(AtomicU64::new(0));
    if super::wide_refresh() && job.refreshes.len() >= 4 {
        let jobs: Vec<Box<dyn FnOnce() + Send>> = job
            .refreshes
            .iter()
            .map(|(shadow, from, to)| {
                let (shadow, from, to, copied) = (shadow.clone(), *from, *to, copied.clone());
                Box::new(move || {
                    copied.fetch_add(shadow.update(from, to - from) as u64, Ordering::Relaxed);
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        crate::shadow::wide_run(jobs);
    } else {
        for (shadow, from, to) in &job.refreshes {
            copied.fetch_add(shadow.update(*from, *to - *from) as u64, Ordering::Relaxed);
        }
    }
    if once {
        crate::watch::ask_once_end();
    } else if ranges.len() > 1 && !overlap() {
        crate::watch::ask_ahead_done();
    }
    add(S_COPIES, job.refreshes.len() as u64);
    add(S_COPIED, copied.load(Ordering::Relaxed));
    // What changed in copies in the card's memory goes up on our queue, behind everything
    // the host queue had before this submission's lists (`reached`): the same place in the
    // card's order as `Host::card_before`'s send has.
    let sending = std::time::Instant::now();
    let mut sends: Vec<(Arc<Shadow>, AheadSend)> = Vec::new();
    for (shadow, ..) in &job.refreshes {
        if shadow.carded()
            && let Some(send) = shadow.ahead_take()
        {
            sends.push((shadow.clone(), send));
        }
    }
    if !sends.is_empty() {
        record_and_send(side, slots, job.number, sends);
    }
    // SAFETY: our queue and fence.
    if let Err(e) = unsafe { side.queue.Signal(&side.done, job.number) } {
        tracing::warn!(error = %e, "VICTORY_ASYNC_REFRESH=1: the side's fence could not be signalled");
    }
    add(S_SEND_US, sending.elapsed().as_micros() as u64);
    // Had the host queue already reached its wait? Then the card stood for us, at most since
    // the job was sent.
    // SAFETY: a plain query of our fence.
    if unsafe { side.reached.GetCompletedValue() } >= job.number {
        add(S_LATE, 1);
        add(S_LATE_US, job.sent.elapsed().as_micros() as u64);
    }
    add(S_WORK_US, started.elapsed().as_micros() as u64);
}

/// Records the sends into a free slot's list and puts it on the side's queue behind a wait
/// for `reached`.
fn record_and_send(side: &Side, slots: &mut Vec<Slot>, number: u64, sends: Vec<(Arc<Shadow>, AheadSend)>) {
    // SAFETY: a plain query of our fence.
    let completed = unsafe { side.done.GetCompletedValue() };
    let free = slots.iter().position(|s| s.number <= completed);
    let at = match free {
        Some(i) => i,
        None => {
            let Some(host) = super::host() else {
                return;
            };
            // SAFETY: plain object creation on our device.
            let made = unsafe {
                (|| -> windows::core::Result<Slot> {
                    let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList =
                        host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    Ok(Slot { allocator, list, number: 0, keep: Vec::new(), chunks: Vec::new() })
                })()
            };
            match made {
                Ok(slot) => {
                    slots.push(slot);
                    slots.len() - 1
                }
                Err(e) => super::fail(format_args!("VICTORY_ASYNC_REFRESH=1: no list for the side's sends ({e})")),
            }
        }
    };
    let slot = &mut slots[at];
    slot.keep.clear();
    slot.chunks.clear();
    let mut bytes = 0u64;
    let mut ranges = 0u64;
    // SAFETY: the slot's last run has finished (its number is passed); live buffers, and the
    // ranges lie in them (offsets in the copy, as `Host::card_before` sends them).
    let recorded = unsafe {
        (|| -> windows::core::Result<()> {
            slot.allocator.Reset()?;
            slot.list.Reset(&slot.allocator, None)?;
            for (shadow, send) in &sends {
                match send {
                    AheadSend::Ranges(staging, list) => {
                        for &(a, b) in list {
                            slot.list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                            bytes += (b - a) as u64;
                        }
                        ranges += list.len() as u64;
                    }
                    AheadSend::Chunks(parts) => {
                        for (a, b, chunk) in parts {
                            slot.list.CopyBufferRegion(&shadow.buffer, *a as u64, chunk.buffer(), chunk.offset, (b - a) as u64);
                            crate::windowup::SENT.fetch_add((b - a) as u64, Ordering::Relaxed);
                            bytes += (b - a) as u64;
                        }
                        ranges += parts.len() as u64;
                    }
                }
            }
            slot.list.Close()?;
            side.queue.Wait(&side.reached, number)?;
            side.queue.ExecuteCommandLists(&[Some(slot.list.cast()?)]);
            Ok(())
        })()
    };
    if let Err(e) = recorded {
        super::fail(format_args!("VICTORY_ASYNC_REFRESH=1: host copies in the card's memory could not be sent up ({e})"));
    }
    crate::shadow::CARD_STATS[0].fetch_add(bytes, Ordering::Relaxed);
    crate::shadow::CARD_STATS[1].fetch_add(ranges, Ordering::Relaxed);
    add(S_SENT, bytes);
    slot.number = number;
    for (shadow, send) in sends {
        if let AheadSend::Chunks(parts) = send {
            slot.chunks.extend(parts.into_iter().map(|p| p.2));
        }
        slot.keep.push(shadow);
    }
}

/// Every four seconds or so: the numbers per frame.
fn log_line() {
    static LAST: Mutex<Option<(std::time::Instant, u64)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let frames_now = crate::queue::presented();
    let Some((since, frames_then)) = *last else {
        *last = Some((std::time::Instant::now(), frames_now));
        return;
    };
    if since.elapsed() < std::time::Duration::from_secs(4) {
        return;
    }
    *last = Some((std::time::Instant::now(), frames_now));
    drop(last);
    let s: Vec<u64> = STATS.iter().map(|x| x.swap(0, Ordering::Relaxed)).collect();
    let frames = frames_now.saturating_sub(frames_then).max(1) as f64;
    let per = |i: usize| s[i] as f64 / frames;
    let ms = |i: usize| s[i] as f64 / 1000.0 / frames;
    tracing::info!(
        "refreshes off the submitting thread per frame (VICTORY_ASYNC_REFRESH=1, {frames:.0} frames): {:.1} of {:.1} submissions sent ahead \
         had a job; refresh steps {:.0} of {:.0} done by the worker ({:.0} on the submitting thread; that time is the frame line's \"copies\"), \
         {:.0} copies, {:.1} MB copied, {:.1} MB sent up on its own queue; worker {:.1} ms (asking {:.1}, sending {:.1}); submitting thread \
         {:.2} ms planning and handing over, {:.1} ms waiting for the worker in {:.1} submissions; {:.1} jobs ended after the host queue \
         was at its wait (the card stood at most {:.1} ms for them); {} failed; {:.1} submissions went on beside open jobs (VICTORY_ASYNC_OVERLAP)",
        per(S_JOBS),
        per(S_SUBS),
        per(S_CLAIMED),
        per(S_STEPS),
        per(S_STEPS) - per(S_CLAIMED),
        per(S_COPIES),
        s[S_COPIED] as f64 / 1e6 / frames,
        s[S_SENT] as f64 / 1e6 / frames,
        ms(S_WORK_US),
        ms(S_ASK_US),
        ms(S_SEND_US),
        ms(S_PLAN_US),
        ms(S_DRAIN_US),
        per(S_DRAINS),
        per(S_LATE),
        ms(S_LATE_US),
        s[S_FAILED],
        per(S_PASSED),
    );
}
