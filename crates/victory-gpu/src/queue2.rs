// Ported from our build (VICTORY_QUEUE2): the whole module (its WARP tests left out).
//! More queues on the card, for copies only (`VICTORY_QUEUE2`; off by
//! default): work of ours that does not depend on the order of the title's
//! draws goes to a copy queue of its own and no longer stands in the one
//! direct queue behind them. A copy queue runs on the card's copy engine,
//! beside the engine that draws.
//!
//! Two lanes, each a copy queue with a fence of its own, so that one never
//! stands behind the other:
//!
//! - **down** (bit 1): the fetch of the pieces shaders wrote in copies in
//!   the card's memory (`Host::relay_card_back`, `Host::card_back`). With
//!   the relay that fetch stood behind every flight sent since: 114-157 ms
//!   a frame of the landing thread's 176-229 (run 118).
//! - **up** (bit 2): the title bytes that changed on the CPU's side of such
//!   copies cross to the card's memory (`Host::card_before`), into a
//!   landing buffer of ours there; the direct queue waits for them and
//!   moves them on into the copies, inside the card's memory.
//!
//! What is synchronised, and with what:
//!
//! - A job of a lane may wait for fences of the direct queue
//!   ([`Lane::send`]'s `after`: `ID3D12CommandQueue::Wait` on the lane's
//!   queue). A fetch names the fence value behind the last flight it lands,
//!   so it runs after that flight's lists and its compare.
//! - The direct queue waits for a job of the up lane ([`Second::hold`]:
//!   `Wait` on the direct queue for the lane's fence), so nothing sent to
//!   the direct queue afterwards runs before the bytes have landed.
//! - No wait can stand for ever: a job only ever waits for a value already
//!   signalled on the direct queue, and the direct queue only for a job
//!   already sent, so each wait's partner lies ahead of it in its queue.
//! - Only buffers go through the lanes. A buffer needs no state change for
//!   a copy queue (it is promoted from the common state and falls back to
//!   it), and it may be READ on one queue while another queue's work on it
//!   is in the air; a texture may not, which is why no texture goes this
//!   way.
//! - No lane WRITES a buffer that the direct queue's work in the air
//!   names: the host's debug layer calls that an error ("not safe to start
//!   write GPU operations now on this Command Queue", seen on the software
//!   adapter with a first draft that copied title bytes straight into the
//!   copies). Hence the landing buffers ([`Second::landing`]): each is
//!   written by one job of the up lane and read by the direct queue behind
//!   its wait for that job, and is given out again only when the card has
//!   passed that read. The copies themselves change only on the direct
//!   queue, in its order, as before.
//! - With the relay the sends of title bytes are numbered
//!   (`crate::host::QUEUE_ORDER`). The direct queue signals each number
//!   behind its send ([`Second::sent_up`]), and a fetch on the down lane
//!   takes for "the sends ahead of me" those the card has finished
//!   ([`Second::order_done`]). A later send that the card happens to run
//!   before the fetch is left out by the merge as one that came after
//!   (`Shadow::card_merge_relay`): the title's bytes stand, as they would.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

/// `VICTORY_QUEUE2`, a sum of bits; unset or 0: off, everything goes
/// through the one direct queue as before.
///
/// - 1: fetches of what shaders wrote to copies in the card's memory go to
///   a copy queue (the gain is with `VICTORY_RELAY=1`/`2`; without the
///   relay nothing else is queued when they run, so it only shows that
///   the path works);
/// - 2: the title bytes sent up to copies in the card's memory cross from
///   the PC's memory on a copy queue, into landing buffers in the card's
///   memory ([`landing_most`] says how much of it), and the direct queue
///   moves them on from there.
///
/// Read once, when the first submission asks.
pub(crate) fn bits() -> u32 {
    static BITS: OnceLock<u32> = OnceLock::new();
    *BITS.get_or_init(|| std::env::var("VICTORY_QUEUE2").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(0) & 3)
}

/// The most bytes of landing buffers in the card's memory, for title
/// bytes on their way to copies there (`VICTORY_QUEUE2_LANDING_MB`, 96;
/// a frame sent up 45-76 MB in runs 116-120, and a buffer is free again as
/// soon as the card has run the submission it was for). A send that finds
/// no room goes straight from the PC's memory on the direct queue, as
/// without the switch; the frame line counts those.
fn landing_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_QUEUE2_LANDING_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(96) << 20
    })
}

/// A buffer in the card's memory that title bytes land in.
struct Landing {
    buffer: ID3D12Resource,
    size: u64,
    /// The value of the host's fence that the card passes after the direct
    /// queue's last read of the buffer: free once it is reached.
    until: u64,
}

/// Which landing buffer (its size, and the fence value it is held until)
/// takes `bytes` when the host's fence has reached `completed`: the
/// smallest free one that holds them.
fn pick_landing(landings: &[(u64, u64)], bytes: u64, completed: u64) -> Option<usize> {
    landings
        .iter()
        .enumerate()
        .filter(|&(_, &(size, until))| until <= completed && size >= bytes)
        .min_by_key(|&(_, &(size, _))| size)
        .map(|(at, _)| at)
}

/// An allocator and a list of the copy type.
pub(crate) type CopyList = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Since the last frame-rate line, of one lane: jobs, their bytes,
/// microseconds the lane had a job (sent and not finished: its time on the
/// card and its waits for the direct queue's fences), microseconds a thread
/// of ours stood waiting for its jobs.
#[derive(Default)]
pub(crate) struct LaneStats {
    pub(crate) jobs: AtomicU64,
    pub(crate) bytes: AtomicU64,
    pub(crate) busy_us: AtomicU64,
    pub(crate) waited_us: AtomicU64,
}

struct LaneState {
    /// Jobs sent: the last value signalled on the lane's fence.
    sent: u64,
    /// Lists not in use, each with the job it last ran in: free for the
    /// next job once the card has run that one.
    spare: Vec<(u64, CopyList)>,
    /// To the lane's clock: a job and when it was sent.
    clock: Option<Sender<(u64, Instant)>>,
}

/// A copy queue with a fence of its own; its jobs run in the order sent.
pub(crate) struct Lane {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    /// Signalled on `queue` behind each job with the job's number.
    fence: ID3D12Fence,
    state: Mutex<LaneState>,
    pub(crate) stats: LaneStats,
}

// SAFETY: D3D12 devices, queues, fences, allocators and lists are
// free-threaded objects; the lists are recorded by the thread that took
// them and otherwise touched under `state`.
unsafe impl Send for Lane {}
// SAFETY: as above.
unsafe impl Sync for Lane {}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A lane's busy time grows by the stretch from `sent` (or from when the
/// job before it finished, `before`, if that was later: the lane was busy
/// with that one until then) to `done`. Returns the microseconds.
fn busy_stretch(before: Option<Instant>, sent: Instant, done: Instant) -> u64 {
    let from = before.map_or(sent, |b| b.max(sent));
    done.saturating_duration_since(from).as_micros() as u64
}

impl Lane {
    fn new(device: &ID3D12Device, name: &str) -> windows::core::Result<Lane> {
        // SAFETY: plain object creation with valid descriptions.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let wide: Vec<u16> = format!("victory copy queue ({name})").encode_utf16().chain([0]).collect();
            let _ = queue.SetName(windows::core::PCWSTR(wide.as_ptr()));
            Ok(Lane {
                device: device.clone(),
                queue,
                fence,
                state: Mutex::new(LaneState {
                    sent: 0,
                    spare: Vec::new(),
                    clock: None,
                }),
                stats: LaneStats::default(),
            })
        }
    }

    /// Starts the lane's clock: a thread that waits for each job in turn
    /// and adds up how long the lane had one ([`LaneStats::busy_us`]).
    fn clocked(&'static self, name: &str) {
        let (tx, rx) = channel();
        let started = std::thread::Builder::new().name(format!("victory-queue2-{name}")).spawn(move || self.watch(&rx));
        match started {
            Ok(_) => lock(&self.state).clock = Some(tx),
            Err(e) => tracing::warn!(error = %e, "no thread for a copy queue's clock: its milliseconds stay 0"),
        }
    }

    fn watch(&self, jobs: &Receiver<(u64, Instant)>) {
        // SAFETY: a plain event of our own.
        let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
            return;
        };
        let mut before = None;
        while let Ok((job, sent)) = jobs.recv() {
            if self.wait(job, event).is_err() {
                return;
            }
            let done = Instant::now();
            self.stats.busy_us.fetch_add(busy_stretch(before, sent, done), Ordering::Relaxed);
            before = Some(done);
        }
    }

    /// An open list of the copy type: one whose last job the card has run,
    /// or a new one.
    pub(crate) fn list(&self) -> windows::core::Result<CopyList> {
        // SAFETY: a plain query; a list is reset only when the job it ran
        // in is done; plain object creation otherwise.
        unsafe {
            let done = self.fence.GetCompletedValue();
            let free = {
                let mut state = lock(&self.state);
                state.spare.iter().position(|(job, _)| *job <= done).map(|at| state.spare.swap_remove(at).1)
            };
            if let Some((allocator, list)) = free {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                return Ok((allocator, list));
            }
            let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            Ok((allocator, list))
        }
    }

    /// Closes `list` (from [`Lane::list`], with copies recorded) and sends
    /// it as the lane's next job, of `bytes` bytes. The job runs behind
    /// the lane's earlier jobs and not before each fence of `after` (fences
    /// of other queues) has reached its value: every value must have been
    /// signalled on its queue already. Returns the job's number, for
    /// [`Lane::wait`], [`Lane::done`] and [`Second::hold`].
    pub(crate) fn send(&self, list: CopyList, after: &[(&ID3D12Fence, u64)], bytes: u64) -> windows::core::Result<u64> {
        // SAFETY: an open list of ours; the lane's queue and fence; live
        // fences of the same device.
        unsafe {
            list.1.Close()?;
            let lists = [Some(list.1.cast::<ID3D12CommandList>()?)];
            // One job at a time from its waits to its signal: the jobs'
            // numbers are the order they run in.
            let mut state = lock(&self.state);
            for &(fence, value) in after {
                // (A fence is at 0 before anything is signalled on it:
                // nothing to wait for.)
                if value != 0 {
                    self.queue.Wait(fence, value)?;
                }
            }
            self.queue.ExecuteCommandLists(&lists);
            state.sent += 1;
            let job = state.sent;
            self.queue.Signal(&self.fence, job)?;
            state.spare.push((job, list));
            if let Some(clock) = &state.clock {
                let _ = clock.send((job, Instant::now()));
            }
            self.stats.jobs.fetch_add(1, Ordering::Relaxed);
            self.stats.bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(job)
        }
    }

    /// Whether the card has run `job`.
    pub(crate) fn done(&self, job: u64) -> bool {
        // SAFETY: a plain query.
        unsafe { self.fence.GetCompletedValue() >= job }
    }

    /// Waits on this thread, with its `event`, until the card has run
    /// `job`.
    pub(crate) fn wait(&self, job: u64, event: HANDLE) -> windows::core::Result<()> {
        // SAFETY: the lane's fence and the caller's event.
        unsafe {
            if self.fence.GetCompletedValue() < job {
                self.fence.SetEventOnCompletion(job, event)?;
                WaitForSingleObject(event, INFINITE);
            }
        }
        Ok(())
    }
}

/// The copy queues and what ties them to the direct queue.
pub(crate) struct Second {
    bits: u32,
    device: ID3D12Device,
    /// The landing buffers made so far ([`Second::landing`]).
    landings: Mutex<Vec<Landing>>,
    /// The landing buffers' bytes alive; since the last frame-rate line,
    /// the sends that found no room in them.
    pub(crate) landing: [AtomicU64; 2],
    /// Fetches (bit 1).
    pub(crate) down: Lane,
    /// Sends of title bytes to copies in the card's memory (bit 2).
    pub(crate) up: Lane,
    /// Signalled on the DIRECT queue behind each numbered send of title
    /// bytes, with its number ([`Second::sent_up`]).
    order: ID3D12Fence,
    /// Signalled on the DIRECT queue just ahead of each wait for the up
    /// lane ([`Second::hold`]), for the clock: when the direct queue got
    /// there. Behind the lock: the holds made so far, and the way to the
    /// thread that times them.
    reached: ID3D12Fence,
    holds: Mutex<(u64, Option<Sender<(u64, u64)>>)>,
    /// Since the last frame-rate line: waits of the direct queue for the up
    /// lane, those of them at which it stood, and for how many
    /// microseconds.
    pub(crate) held: [AtomicU64; 3],
}

// SAFETY: fences (free-threaded) and lanes.
unsafe impl Send for Second {}
// SAFETY: as above.
unsafe impl Sync for Second {}

impl Second {
    /// The lanes on `device`. `bits` as [`bits`].
    pub(crate) fn new(device: &ID3D12Device, bits: u32) -> windows::core::Result<Second> {
        // SAFETY: plain fences on the device.
        unsafe {
            Ok(Second {
                bits,
                device: device.clone(),
                landings: Mutex::new(Vec::new()),
                landing: [const { AtomicU64::new(0) }; 2],
                down: Lane::new(device, "down")?,
                up: Lane::new(device, "up")?,
                order: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                reached: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                holds: Mutex::new((0, None)),
                held: [const { AtomicU64::new(0) }; 3],
            })
        }
    }

    /// Starts the clocks: the lanes' and the one that times the direct
    /// queue's waits ([`Second::held`]).
    pub(crate) fn clocked(&'static self) {
        self.down.clocked("down");
        self.up.clocked("up");
        let (tx, rx) = channel();
        match std::thread::Builder::new().name("victory-queue2-holds".into()).spawn(move || self.watch_holds(&rx)) {
            Ok(_) => lock(&self.holds).1 = Some(tx),
            Err(e) => tracing::warn!(error = %e, "no thread to time the direct queue's waits for the copy queue: they stay 0"),
        }
    }

    /// Whether fetches go to the down lane.
    pub(crate) fn fetches(&self) -> bool {
        self.bits & 1 != 0
    }

    /// Whether title bytes go up through the up lane.
    pub(crate) fn uploads(&self) -> bool {
        self.bits & 2 != 0
    }

    /// A buffer in the card's memory for `bytes` title bytes on their way
    /// up, for a job of the up lane to write and the direct queue to read
    /// behind its wait for that job ([`Second::hold`]). `completed`: the
    /// value the host's fence has reached; `until`: the value it reaches
    /// when the card has run that read (the next one signalled behind
    /// it). The buffer is not given out again before then, so no job
    /// writes it while anything in the air names it. `None`: no room under
    /// [`landing_most`], or no memory; the caller sends the bytes the old
    /// way.
    pub(crate) fn landing(&self, bytes: u64, completed: u64, until: u64) -> Option<ID3D12Resource> {
        let mut landings = lock(&self.landings);
        let sizes: Vec<(u64, u64)> = landings.iter().map(|l| (l.size, l.until)).collect();
        if let Some(at) = pick_landing(&sizes, bytes, completed) {
            landings[at].until = until;
            return Some(landings[at].buffer.clone());
        }
        // A new one; free ones that are too small make room for it (the
        // card has passed their last use: nothing names them).
        let size = bytes.next_multiple_of(1 << 20);
        let alive = |landings: &[Landing]| landings.iter().map(|l| l.size).sum::<u64>();
        if alive(&landings) + size > landing_most() {
            landings.retain(|l| l.until > completed);
        }
        let made = if alive(&landings) + size > landing_most() {
            None
        } else {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a buffer in the card's memory.
            let made = unsafe {
                self.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
            };
            match (made, buffer) {
                (Ok(()), Some(buffer)) => {
                    let wide: Vec<u16> = "victory landing buffer (title bytes on their way up)".encode_utf16().chain([0]).collect();
                    // SAFETY: a live buffer and a terminated string.
                    let _ = unsafe { buffer.SetName(windows::core::PCWSTR(wide.as_ptr())) };
                    landings.push(Landing {
                        buffer: buffer.clone(),
                        size,
                        until,
                    });
                    Some(buffer)
                }
                _ => None,
            }
        };
        self.landing[0].store(alive(&landings), Ordering::Relaxed);
        if made.is_none() {
            self.landing[1].fetch_add(1, Ordering::Relaxed);
        }
        made
    }

    /// The numbered send `number` of title bytes has just gone to `direct`
    /// (or there was nothing to send under that number): the number is
    /// signalled behind it.
    pub(crate) fn sent_up(&self, direct: &ID3D12CommandQueue, number: u64) {
        // SAFETY: a live queue and our fence.
        let _ = unsafe { direct.Signal(&self.order, number) };
    }

    /// The last numbered send of title bytes the card has finished: every
    /// send up to it is in what a fetch sent from now on brings down.
    pub(crate) fn order_done(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.order.GetCompletedValue() }
    }

    /// Makes `direct` wait for `job` of the up lane: nothing sent to it
    /// from now on runs before the card has run the job. The job must have
    /// been sent ([`Lane::send`]).
    pub(crate) fn hold(&self, direct: &ID3D12CommandQueue, job: u64) -> windows::core::Result<()> {
        let mut holds = lock(&self.holds);
        holds.0 += 1;
        // SAFETY: a live queue and our fences.
        unsafe {
            // For the clock only: the moment the queue gets to the wait.
            let _ = direct.Signal(&self.reached, holds.0);
            direct.Wait(&self.up.fence, job)?;
        }
        self.held[0].fetch_add(1, Ordering::Relaxed);
        if let Some(clock) = &holds.1 {
            let _ = clock.send((holds.0, job));
        }
        Ok(())
    }

    /// Times the holds, in the order they were made: from the moment the
    /// direct queue got to one until the job it waits for is done. (While
    /// it stands at one it gets to no later one, so one thread sees each
    /// in time.)
    fn watch_holds(&self, holds: &Receiver<(u64, u64)>) {
        // SAFETY: a plain event of our own.
        let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
            return;
        };
        while let Ok((hold, job)) = holds.recv() {
            // SAFETY: our fence and this thread's event.
            unsafe {
                if self.reached.GetCompletedValue() < hold {
                    if self.reached.SetEventOnCompletion(hold, event).is_err() {
                        return;
                    }
                    WaitForSingleObject(event, INFINITE);
                }
            }
            if self.up.done(job) {
                continue;
            }
            let standing = Instant::now();
            if self.up.wait(job, event).is_err() {
                return;
            }
            self.held[1].fetch_add(1, Ordering::Relaxed);
            self.held[2].fetch_add(standing.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }
}

/// The copy queues, made at the first ask if [`bits`] says so; `None` when
/// the switch is off or they could not be made (everything then goes
/// through the direct queue, as before).
pub(crate) fn second() -> Option<&'static Second> {
    static SECOND: OnceLock<Option<Second>> = OnceLock::new();
    static CLOCKS: std::sync::Once = std::sync::Once::new();
    if bits() == 0 {
        return None;
    }
    let second = SECOND
        .get_or_init(|| {
            let host = crate::host::host()?;
            match Second::new(&host.device, bits()) {
                Ok(second) => {
                    tracing::info!(
                        fetches = second.fetches(),
                        uploads = second.uploads(),
                        landing_mb = landing_most() >> 20,
                        "copy queues on the card beside the direct queue (VICTORY_QUEUE2={})",
                        bits()
                    );
                    Some(second)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no copy queues (VICTORY_QUEUE2): everything goes through the direct queue");
                    None
                }
            }
        })
        .as_ref()?;
    CLOCKS.call_once(|| second.clocked());
    Some(second)
}

/// For the frame-rate line: what the copy queues did per frame since the
/// last one. "On their queue" is from a job's send to its end with each
/// moment counted once: its time on the card and any wait of its queue
/// for the direct queue's fences. The direct queue "stood" at a wait if
/// the job was not done when it got there.
pub(crate) fn line(frames: f64) {
    let Some(second) = second() else {
        return;
    };
    let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / frames;
    let ms = |a: &AtomicU64| take(a) / 1000.0;
    let mb = |a: &AtomicU64| take(a) / (1 << 20) as f64;
    let (down, up) = (&second.down.stats, &second.up.stats);
    let (down_jobs, down_mb, down_busy, down_waited) = (take(&down.jobs), mb(&down.bytes), ms(&down.busy_us), ms(&down.waited_us));
    let (up_jobs, up_mb, up_busy) = (take(&up.jobs), mb(&up.bytes), ms(&up.busy_us));
    let [holds, stood, stood_us] = [0, 1, 2].map(|i| take(&second.held[i]));
    let (landing_mb, no_room) = (second.landing[0].load(Ordering::Relaxed) >> 20, take(&second.landing[1]));
    tracing::info!(
        "the copy queues per frame (VICTORY_QUEUE2={}): fetches of what shaders wrote: {down_jobs:.1} jobs, {down_mb:.1} MB, \
         {down_busy:.1} ms on their queue, our threads stood {down_waited:.1} ms waiting for them; title bytes sent up: \
         {up_jobs:.1} jobs, {up_mb:.1} MB, {up_busy:.1} ms on their queue; the direct queue was told to wait for them \
         {holds:.1} times and stood at {stood:.1} of those for {:.1} ms in all; landing buffers on the card: {landing_mb} MB, \
         {no_room:.1} sends found no room in them and went through the direct queue (VICTORY_QUEUE2_LANDING_MB)",
        second.bits,
        stood_us / 1000.0
    );
}

