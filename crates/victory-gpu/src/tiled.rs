//! (Local, `VICTORY_TILED=1`; off by default) One card-side image of the title's private
//! graphics memory, in place of host copies of it.
//!
//! What it is. Every `XMemVirtualAlloc` reservation the runtime makes for the title with write
//! watching (`victory-hle` xmem; not the page pool's placeholders) is a *stretch* of title
//! addresses. When a title buffer or a window outside buffers that lies wholly inside one stretch
//! is first used, the stretch gets a reserved (tiled) buffer on the card whose offset `x` is title
//! address `base + x`, and the 64 KiB tiles under the use are mapped onto video-memory heaps of
//! ours (`UpdateTileMappings`). The use is then a [`crate::shadow::Shadow`] whose buffer is that
//! reserved buffer and whose GPU address is the image's (`Shadow::tiled_image`): no host copy,
//! no per-range refresh, no per-range copy-back. Everything that binds, views and keeps copies
//! works on it unchanged, as it does for `crate::inplace`.
//!
//! CPU to card, once per submission for the ranges its lists use (`HostList::defer_tiled_use`;
//! the tiles of new uses are mapped then too, in one call per heap): the pages the title wrote are asked of
//! the write-watch journal (`crate::watch::writes`, shared with every other reader, so no reader
//! steals another's answer), each written 4 KiB page is compared by 64-byte chunk with hashes of
//! what was last sent, and only the changed chunks go up, through a 64 MiB ring of upload memory,
//! on a queue of our own. The host's queue waits for that queue's fence on the card
//! (`ID3D12CommandQueue::Wait`): the CPU does not wait for the upload.
//!
//! Card to CPU: shaders write the image in place. For the ranges a submission declares written
//! (the same declarations host copies use, `Shadow::gpu_writes`), the image's bytes are copied
//! into a scratch buffer on the card ahead of the submission's lists, and a compare pass of ours
//! after them marks each 4 KiB page that changed. When the submission has landed
//! ([`hand_back`], called where host copies are copied back), only the marked pages are read
//! back and merged into title memory by 64-byte chunk: a chunk the CPU changed since it was sent
//! keeps the CPU's bytes (they go up at the next submission), every other chunk takes the card's.
//!
//! Large parts are taken from xcl's `hle/d3d12x/sparse.rs` (GPL-2.0, allowed here for local use;
//! marked "origin: xcl sparse.rs" below): tile entries, the pool of 64 MiB heaps, batched tile
//! mappings, the upload ring and slots, chunk hashing and the read-back staging.
//!
//! Decisions (see the report that came with this prototype for the reasoning):
//! - Stretches are cut at [`stretch_max`] (4 GiB by default) from the reservation's base, so a
//!   buffer view's byte offset never needs more than 32 bits of elements; a use that straddles two
//!   pieces of one reservation is not covered (it gets a copy, counted).
//! - Many title buffers are one host resource here. Buffers have no layout, and every use of the
//!   image relies on implicit promotion from the common state, as host copies do; the layer's
//!   existing global UAV barriers order shader writes. The debug layer may still report states.
//! - Tiles of memory the title leaves uncommitted stay unmapped (reads give zeros, tier 2).
//! - Memory the image does not cover (view memory of the page pool, reservations below
//!   [`min_bytes`], uses that straddle pieces, past [`budget`]) falls back to the existing copies.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

/// The card's tile, and the journal's page (`crate::watch::PAGE`).
const TILE: usize = 0x1_0000;
/// The kernel's page: the unit of hashing, of the compare and of read-backs.
const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = 64;
const CHUNKS: usize = PAGE / CHUNK;
/// Tiles of a heap of the pool (64 MiB).
const POOL_TILES: u32 = 1024;
/// Bytes of the upload ring.
const STAGING: usize = 64 << 20;
const RING_ALIGN: usize = 64;
const UPLOAD_SLOTS: usize = 16;
/// Bytes of the zero buffer (tile clears, mask clears).
const ZERO_BUF: usize = 1 << 20;
/// Bytes of the read-back staging buffer.
const READBACK_STAGING: usize = 16 << 20;
/// The compare's output in CPU-visible memory: a header (the count of changed chunks), then
/// `cap` chunk numbers, then `cap` chunks of 64 bytes.
const OUT_HEAD: usize = 256;
/// The most changed chunks one compare hands back through its output (4 MiB of bytes); past
/// it the compared ranges are read back by page, as before.
const OUT_MOST: usize = 1 << 16;

/// Whether the image is on (`VICTORY_TILED=1`). Read once.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TILED").is_some_and(|v| v == "1"))
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
}

/// The most bytes of one stretch piece (`VICTORY_TILED_STRETCH_MB`, 4096), a multiple of 64 KiB.
fn stretch_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| ((env_u64("VICTORY_TILED_STRETCH_MB", 4096).max(1) << 20) as usize).next_multiple_of(TILE))
}

/// The least reservation covered (`VICTORY_TILED_MIN_KB`, 1024).
fn min_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| (env_u64("VICTORY_TILED_MIN_KB", 1024) << 10) as usize)
}

/// The most video memory in tiles (`VICTORY_TILED_MB`, 4096).
fn budget() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| (env_u64("VICTORY_TILED_MB", 4096) << 20) as usize)
}

/// The most bytes of declared shader-written ranges compared on the card per submission
/// (`VICTORY_TILED_BEFORE_MB`, 1024); past it a range is read back whole.
fn before_most() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| (env_u64("VICTORY_TILED_BEFORE_MB", 1024) << 20) as usize)
}

// ---------------------------------------------------------------------------------------------
// Counters for the ten-second line.

/// Indices of [`STATS`].
mod at {
    pub const UP_BYTES: usize = 0;
    pub const UP_RUNS: usize = 1;
    pub const BACK_PAGES: usize = 2;
    pub const BACK_BYTES: usize = 3;
    pub const CHECKS: usize = 4;
    pub const COMPARED: usize = 5;
    pub const WHOLE_BACK: usize = 6;
    pub const SCAN_US: usize = 7;
    pub const MAPS: usize = 8;
    pub const PLACED: usize = 9;
    pub const UNWATCHED: usize = 10;
    // fallbacks, by reason
    pub const NOT_COVERED: usize = 11;
    pub const STRADDLES: usize = 12;
    pub const OVER_BUDGET: usize = 13;
    pub const NO_DEVICE: usize = 14;
    pub const SAME_RESOURCE_COPY: usize = 15;
    pub const LEAKED_TILES: usize = 16;
    pub const CONFLICT_CHUNKS: usize = 17;
    pub const MAP_US: usize = 18;
    pub const ASK_US: usize = 19;
    pub const HASHED: usize = 20;
    pub const LAND_US: usize = 21;
    pub const UP_US: usize = 22;
    pub const CARD_CHUNKS: usize = 23;
    pub const OVERFLOWS: usize = 24;
    pub const VIEW_PLACED: usize = 25;
    pub const VIEW_UNMAPPED: usize = 26;
    pub const VIEW_ALIASED: usize = 27;
    pub const FORCED: usize = 28;
    pub const LAND_LOCK_US: usize = 29;
    pub const LAND_TAKE_US: usize = 30;
    pub const WRITE_US: usize = 31;
}
const N_STATS: usize = 32;
static STATS: [AtomicU64; N_STATS] = [const { AtomicU64::new(0) }; N_STATS];

fn add(i: usize, n: u64) {
    STATS[i].fetch_add(n, Ordering::Relaxed);
}

/// A copy-to-copy GPU copy inside the image that was done on the CPU instead
/// (`list.rs` `card_copy`).
pub(crate) fn note_same_resource_copy() {
    add(at::SAME_RESOURCE_COPY, 1);
}

// ---------------------------------------------------------------------------------------------
// Reservations (known before the device exists).

/// `XMemVirtualAlloc` reservations with write watching: base → (bytes (64 KiB multiple),
/// whether it is a placeholder for physical-page views: Local (VICTORY_TILED views)).
static RESERVED: Mutex<BTreeMap<usize, (usize, bool)>> = Mutex::new(BTreeMap::new());

/// Local (VICTORY_TILED views): whether page-pool view memory is on the image too
/// (`VICTORY_TILED_VIEWS`): 0 (default) no; 1, pages the title shows at one address only;
/// 2, also pages it shows at several (those are hashed whole at every submission that uses
/// them, and a merge into one tells the others).
fn views_mode() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_u64("VICTORY_TILED_VIEWS", 0).min(2) as u8)
}

/// Local (VICTORY_TILED views): 64 KiB pages of title view memory that are (or were) shown at
/// more than one address, by address.
static ALIASED: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());

/// Local (VICTORY_TILED views): the runtime reserved `[base, base + len)` for physical-page
/// views (a placeholder, `victory-hle` xmem `XMemVirtualAlloc`).
pub fn view_reserved(base: usize, len: usize) {
    if on() && views_mode() > 0 {
        reserve(base, len, true);
    }
}

/// Local (VICTORY_TILED views): the 64 KiB pages at `addresses` show one physical page
/// (`crate::inplace::title_memory_view`).
pub(crate) fn view_aliased(addresses: &[usize]) {
    if !on() || views_mode() == 0 {
        return;
    }
    let mut a = ALIASED.lock().unwrap_or_else(|e| e.into_inner());
    a.extend(addresses.iter().map(|&x| x & !(TILE - 1)));
}

/// Local (VICTORY_TILED views): the most other addresses of a page whose writes are asked
/// about one by one; a page shown at more is hashed whole at every use, and is not placed on
/// the image at all past [`ALIASES_PLACED`].
const ALIASES_ASKED: usize = 8;
const ALIASES_PLACED: usize = 64;

fn is_aliased(t: usize) -> bool {
    views_mode() > 0 && ALIASED.lock().unwrap_or_else(|e| e.into_inner()).contains(&t)
}

/// The runtime reserved `[base, base + len)` for the title, with write watching
/// (`victory-hle` xmem `XMemVirtualAlloc`). A reservation that overlaps older ones replaces
/// them (the title freed those without telling us).
pub fn reserved(base: usize, len: usize) {
    if on() {
        reserve(base, len, false);
    }
}

fn reserve(base: usize, len: usize, view: bool) {
    if len < min_bytes() || base % TILE != 0 {
        return;
    }
    let len = len.next_multiple_of(TILE);
    let end = base.saturating_add(len);
    let gone: Vec<(usize, usize)> = {
        let mut r = RESERVED.lock().unwrap_or_else(|e| e.into_inner());
        let gone: Vec<(usize, usize)> = r.range(..end).filter(|&(&b, &(l, _))| b + l > base).map(|(&b, &(l, _))| (b, l)).collect();
        for (b, _) in &gone {
            r.remove(b);
        }
        r.insert(base, (len, view));
        gone
    };
    if !gone.is_empty() {
        let mut g = lock_state();
        if let Some(s) = g.as_mut() {
            for (b, l) in gone {
                s.forget(b, l);
            }
        }
    }
}

/// The reservation holding `[start, end)`, as (base, bytes, view memory).
fn reservation(start: usize, end: usize) -> Option<(usize, usize, bool)> {
    let r = RESERVED.lock().unwrap_or_else(|e| e.into_inner());
    let (&b, &(l, view)) = r.range(..=start).next_back()?;
    (end <= b + l).then_some((b, l, view))
}

// ---------------------------------------------------------------------------------------------
// origin: xcl sparse.rs — hashing and change detection by 64-byte chunk.

#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    std::array::from_fn(|i| u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8])))
}

#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

/// The 64-byte chunks of the 4 KiB page `page` (title address `pg`) that differ from what was
/// sent (`slot`; `None`: all zeros), appended to `out` as runs; `slot` updated.
fn scan_page(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    if let Some(sent) = slot.as_deref_mut() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h != sent[k] {
                sent[k] = h;
                push_run(out, pg + k * CHUNK, CHUNK);
            }
        }
        return;
    }
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let sent = slot.insert(Box::new(ZERO_HASHES));
        sent[k] = hash_words(w);
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 != ZERO_HASH {
                sent[k2] = h2;
                push_run(out, pg + k2 * CHUNK, CHUNK);
            }
        }
        break;
    }
}

/// Local (VICTORY_TILED read-back): the 64-byte chunk hashes of title memory's 4 KiB pages
/// `pages` (`None`: not readable), without the image's lock; on the wide threads
/// (`crate::shadow::wide_run`) when there are many.
fn hash_pages(pages: &[usize]) -> Vec<Option<[u64; CHUNKS]>> {
    fn one(pg: usize) -> Option<[u64; CHUNKS]> {
        let mut page = [0u8; PAGE];
        if !crate::host::read_into(pg as *const u8, &mut page) {
            return None;
        }
        Some(std::array::from_fn(|c| chunk_hash(&page[c * CHUNK..(c + 1) * CHUNK])))
    }
    const PER_JOB: usize = 128;
    let mut out: Vec<Option<[u64; CHUNKS]>> = vec![None; pages.len()];
    if pages.len() <= PER_JOB {
        for (o, &pg) in out.iter_mut().zip(pages) {
            *o = one(pg);
        }
        return out;
    }
    let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::with_capacity(pages.len().div_ceil(PER_JOB));
    for (i, part) in pages.chunks(PER_JOB).enumerate() {
        let part: Vec<usize> = part.to_vec();
        // The slots of this part of `out`: no other job touches them, and `wide_run` returns
        // when every job has run.
        let dst = out[i * PER_JOB..].as_mut_ptr() as usize;
        jobs.push(Box::new(move || {
            let dst = dst as *mut Option<[u64; CHUNKS]>;
            for (j, pg) in part.into_iter().enumerate() {
                // SAFETY: inside this job's slots of `out` (see above).
                unsafe { dst.add(j).write(one(pg)) };
            }
        }));
    }
    crate::shadow::wide_run(jobs);
    out
}

/// origin: xcl sparse.rs `plan_ring_chunk`.
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

/// origin: xcl sparse.rs `query_tile_holes`. Bit `i` set: 4 KiB page `i` of the tile at `tile`
/// is not committed and readable.
fn query_tile_holes(tile: usize) -> u16 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        // SAFETY: a query of address-space metadata into a local structure.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(cur as *const _, &mut mbi, size_of::<MEMORY_BASIC_INFORMATION>()) };
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let readable = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if readable {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------------------------
// State.

/// origin: xcl sparse.rs `TileEntry`, with the journal asking it was last scanned at.
struct TileEntry {
    /// the pool tile under it (kept for giving tiles back and for mirroring page mappings, both
    /// still to come)
    #[allow(dead_code)]
    pool: (u32, u32),
    /// bit `i`: 4 KiB page `i` was not readable when last looked at
    holes: u16,
    /// the write-watch asking this tile was last scanned after (0: never)
    seen: u64,
    /// Local (VICTORY_TILED read-back): the count of merges (`State::merges`) when a merge
    /// last wrote title memory here
    merged: u64,
    /// per 4 KiB page: hashes of the 64-byte chunks the card holds (`None`: all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
}

/// One piece of a reservation: a reserved buffer whose offset `x` is title address `base + x`.
struct Stretch {
    len: usize,
    res: ID3D12Resource,
    gpu: u64,
    /// the lowest and highest title address any use placed here covers: what is scanned
    used: (usize, usize),
    /// Local (VICTORY_TILED views): page-pool view memory, not write-watched
    view: bool,
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

/// Command lists of ours that run on the host's queue around a submission's lists.
struct HostLists {
    alloc: ID3D12CommandAllocator,
    pre: ID3D12GraphicsCommandList,
    check: ID3D12GraphicsCommandList,
    /// the done-fence value after which they may be reset
    done: u64,
}

/// A compare of one submission, to be read back when it has run.
struct Check {
    done: u64,
    /// Local (VICTORY_TILED read-back): the compare's output in CPU-visible memory (buffer,
    /// mapping, room in chunks): the chunks that changed, with their bytes. No copy after the
    /// submission: the copy-back thread merges straight from it.
    out: (ID3D12Resource, *mut u8, usize),
    /// (title address, 64-byte chunks, number of the first chunk) of each compared range
    ranges: Vec<(usize, usize, u32)>,
    /// ranges past [`before_most`]: read back whole
    whole: Vec<(usize, usize)>,
}

struct Compare {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

struct State {
    dev: ID3D12Device,
    /// our queue: tile mappings, uploads, read-backs
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    event: HANDLE,
    /// signalled on the host's queue after a submission whose writes are compared
    done_fence: ID3D12Fence,
    done_value: u64,
    /// the done value the next `after_execute` signals
    to_signal: Option<u64>,
    /// the value of `fence` the host's queue last waited for
    waited: u64,
    stretches: BTreeMap<usize, Stretch>,
    pools: Vec<Option<ID3D12Heap>>,
    free: Vec<(u32, u32)>,
    mapped: FxHashMap<usize, TileEntry>,
    /// Local (VICTORY_TILED fix): tiles taken but not mapped yet ([`State::flush_maps`]).
    pending_maps: Vec<(usize, (u32, u32))>,
    /// Local (VICTORY_TILED fix): pages of a landed check still to read back, and its done value.
    land_left: VecDeque<(usize, usize, u64)>,
    to_zero: Vec<(usize, usize)>,
    staging: Option<(ID3D12Resource, *mut u8)>,
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    inflight: VecDeque<(u64, u64)>,
    zero_buf: Option<ID3D12Resource>,
    slots: Vec<UploadSlot>,
    next_slot: usize,
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last: u64,
    host_lists: Vec<HostLists>,
    before: Option<(ID3D12Resource, usize)>,
    /// the compare's counter of changed chunks (video memory)
    mask: Option<(ID3D12Resource, usize)>,
    /// compare outputs not in use (`Check::out`)
    mask_pool: Vec<(ID3D12Resource, *mut u8, usize)>,
    /// merged pages still to write into title memory: (page, its bytes, runs changed)
    to_write: Vec<(usize, Box<[u8; PAGE]>, Vec<(usize, usize)>)>,
    /// merges that wrote title memory so far (`TileEntry::merged`)
    merges: u64,
    checks: VecDeque<Check>,
    /// resources replaced while lists in the air may use them: dropped past the done value
    graveyard: Vec<(u64, ID3D12Resource)>,
    compare: Option<Compare>,
    /// frame and time of the last line, and the counters then
    line: (u64, u64, [u64; N_STATS]),
}

// SAFETY: D3D12 objects are free-threaded; the raw pointers (mappings) and the event are used
// only under the state's lock.
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Whether making the state failed for good (no device features): not tried again.
static BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn lock_state() -> std::sync::MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

// Debugging aid: what the image's code is doing, and since when, for the monitor thread
// (a stall inside it is logged after 2 s).
static PHASE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const PH_PLACE: usize = 0;
const PH_BEFORE: usize = 1;
const PH_LAND: usize = 2;
const PH_AFTER: usize = 3;

struct Phase(usize);

impl Phase {
    fn enter(i: usize) -> Phase {
        PHASE[i].store(crate::host::now_ms().max(1), Ordering::Relaxed);
        Phase(i)
    }
}

impl Drop for Phase {
    fn drop(&mut self) {
        PHASE[self.0].store(0, Ordering::Relaxed);
    }
}

fn start_monitor() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("tiled image monitor".into()).spawn(|| {
            let mut said = [0u64; 4];
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let now = crate::host::now_ms();
                for i in 0..4 {
                    let since = PHASE[i].load(Ordering::Relaxed);
                    if since != 0 && now.saturating_sub(since) > 2000 && said[i] != since {
                        said[i] = since;
                        let name = ["place_for", "before_execute", "hand_back/land", "after_execute"][i];
                        tracing::warn!("VICTORY_TILED: {name} has run for {} ms", now - since);
                    }
                }
            }
        });
    });
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            // SAFETY: our own event.
            unsafe {
                let _ = windows::Win32::Foundation::CloseHandle(self.event);
            }
        }
    }
}

/// The state, made on first use once the host device exists.
fn ensure_state(g: &mut Option<State>) -> Option<&mut State> {
    if g.is_none() {
        if BROKEN.load(Ordering::Relaxed) {
            return None;
        }
        let dev = crate::host::host()?.device.clone();
        match make_state(&dev) {
            Ok(s) => {
                start_monitor();
                tracing::info!("VICTORY_TILED: the title's private graphics memory is one tiled image on the card");
                *g = Some(s);
            }
            Err(e) => {
                BROKEN.store(true, Ordering::Relaxed);
                tracing::warn!(error = %e, "VICTORY_TILED: no tiled image (copies as before)");
                return None;
            }
        }
    }
    g.as_mut()
}

fn make_state(dev: &ID3D12Device) -> windows::core::Result<State> {
    // Unmapped tiles must read as zeros and drop writes: tier 2.
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: writes `options` of the given size.
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut options).cast(), size_of_val(&options) as u32)? };
    if options.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
        tracing::warn!(tier = options.TiledResourcesTier.0, "VICTORY_TILED: the card's tiled resources tier is below 2");
        return Err(windows::core::Error::empty());
    }
    // SAFETY: plain creation of our own objects.
    unsafe {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY,
            ..Default::default()
        })?;
        crate::host::name(&queue, || "tiled image queue".into());
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
        for _ in 0..UPLOAD_SLOTS {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None)?;
            list.Close()?;
            slots.push(UploadSlot { alloc, list, fence_val: 0 });
        }
        let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
        let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &rb_alloc, None)?;
        rb_list.Close()?;
        Ok(State {
            dev: dev.clone(),
            queue,
            fence,
            value: 0,
            event,
            done_fence,
            done_value: 0,
            to_signal: None,
            waited: 0,
            stretches: BTreeMap::new(),
            pools: Vec::new(),
            free: Vec::new(),
            mapped: FxHashMap::default(),
            pending_maps: Vec::new(),
            land_left: VecDeque::new(),
            to_zero: Vec::new(),
            staging: None,
            ring_write: 0,
            ring_submitted: 0,
            ring_free: 0,
            inflight: VecDeque::new(),
            zero_buf: None,
            slots,
            next_slot: 0,
            readback: None,
            rb_alloc,
            rb_list,
            rb_last: 0,
            host_lists: Vec::new(),
            before: None,
            mask: None,
            mask_pool: Vec::new(),
            to_write: Vec::new(),
            merges: 0,
            checks: VecDeque::new(),
            graveyard: Vec::new(),
            compare: None,
            line: (crate::frame::current(), crate::host::now_ms(), [0; N_STATS]),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Placing uses.

/// A use of title memory on the image: the stretch's buffer, the host GPU address of the use's
/// first byte, and its offset in the buffer.
pub(crate) struct Placed {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) gpu: u64,
    pub(crate) bias: u64,
}

/// Title memory `[start, start + len)` on the image, if the image covers it; `None` leaves the
/// caller to make a copy, as before (counted by reason).
pub(crate) fn place_for(start: u64, len: u64) -> Option<Placed> {
    if !on() {
        return None;
    }
    let (start, end) = (start as usize, (start + len.max(1)) as usize);
    let Some((rb, rl, view)) = reservation(start, end) else {
        add(at::NOT_COVERED, 1);
        return None;
    };
    // Local (VICTORY_TILED views): page-pool view memory only where every 64 KiB page shows a
    // physical page, and (`VICTORY_TILED_VIEWS=1`) shows it at this address alone.
    if view {
        let mut t = start & !(TILE - 1);
        while t < end {
            let shown = crate::inplace::pages_of(t);
            if shown.is_empty() {
                add(at::VIEW_UNMAPPED, 1);
                return None;
            }
            if shown.len() > 1 {
                if views_mode() < 2 || shown.len() > ALIASES_PLACED {
                    add(at::VIEW_ALIASED, 1);
                    return None;
                }
                view_aliased(&shown);
            }
            t += TILE;
        }
    }
    let max = stretch_max();
    let k = (start - rb) / max;
    let (pb, pl) = (rb + k * max, max.min(rl - k * max));
    if end > pb + pl {
        add(at::STRADDLES, 1);
        return None;
    }
    let _phase = Phase::enter(PH_PLACE);
    let mut g = lock_state();
    let Some(s) = ensure_state(&mut g) else {
        add(at::NO_DEVICE, 1);
        return None;
    };
    {
        // Debugging aid: the first uses and the big ones.
        static SAID: AtomicU64 = AtomicU64::new(0);
        if SAID.fetch_add(1, Ordering::Relaxed) < 40 || len >= 16 << 20 {
            tracing::info!(start = format_args!("{start:#x}"), len, stretch = format_args!("{pb:#x}"), "VICTORY_TILED: a use on the image");
        }
    }
    let wanted = (start & !(TILE - 1)..end).step_by(TILE).filter(|t| !s.mapped.contains_key(t)).count();
    if (s.mapped.len() + wanted) * TILE > budget() {
        add(at::OVER_BUDGET, 1);
        return None;
    }
    if !s.stretches.contains_key(&pb) {
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a reserved buffer of ours.
        let made = unsafe {
            s.dev.CreateReservedResource(
                &buffer_desc(pl as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut res,
            )
        };
        let res = match (made, res) {
            (Ok(()), Some(res)) => res,
            (made, _) => {
                tracing::warn!(base = format_args!("{pb:#x}"), bytes = pl, error = ?made.err(), "VICTORY_TILED: no reserved buffer for a stretch");
                add(at::NO_DEVICE, 1);
                return None;
            }
        };
        crate::host::name(&res, || format!("tiled image {pb:#x}+{pl:#x}"));
        // SAFETY: a live buffer.
        let gpu = unsafe { res.GetGPUVirtualAddress() };
        tracing::info!(base = format_args!("{pb:#x}"), bytes = pl, "VICTORY_TILED: a stretch of the image");
        s.stretches.insert(pb, Stretch { len: pl, res, gpu, used: (usize::MAX, 0), view });
    }
    s.ensure_mapped(start, end - start, true);
    let stretch = s.stretches.get_mut(&pb)?;
    stretch.used = (stretch.used.0.min(start & !(TILE - 1)), stretch.used.1.max(end.next_multiple_of(TILE).min(pb + pl)));
    add(at::PLACED, 1);
    if view {
        add(at::VIEW_PLACED, 1);
    }
    Some(Placed {
        buffer: stretch.res.clone(),
        gpu: stretch.gpu + (start - pb) as u64,
        bias: (start - pb) as u64,
    })
}

impl State {
    /// The stretch piece holding `a`: (base, bytes, buffer).
    fn stretch_of(&self, a: usize) -> Option<(usize, usize, &ID3D12Resource)> {
        let (&b, st) = self.stretches.range(..=a).next_back()?;
        (a < b + st.len).then_some((b, st.len, &st.res))
    }

    /// A reservation the title gave up: its stretches are forgotten. Their tiles are not
    /// given back (lists in the air may still read them): leaked, counted.
    fn forget(&mut self, base: usize, len: usize) {
        let end = base + len;
        let pieces: Vec<usize> = self.stretches.range(base..end).map(|(&b, _)| b).collect();
        for b in pieces {
            self.stretches.remove(&b);
        }
        let tiles: Vec<usize> = self.mapped.keys().copied().filter(|&t| t >= base && t < end).collect();
        add(at::LEAKED_TILES, tiles.len() as u64);
        for t in tiles {
            self.mapped.remove(&t);
        }
        self.to_zero.retain(|&(a, _)| a < base || a >= end);
        self.pending_maps.retain(|&(a, _)| a < base || a >= end);
        self.land_left.retain(|&(a, _, _)| a < base || a >= end);
    }

    /// origin: xcl sparse.rs `take_tile` / `add_pool` (no retiring of pools here).
    fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        // Local (VICTORY_TILED fix): the free list only ever holds tiles of fresh heaps (tiles of
        // forgotten stretches are leaked, not given back), which the device zeroes: no clear.
        if let Some(t) = self.free.pop() {
            return Some((t, false));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: u64::from(POOL_TILES) * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation; the out pointer is ours.
        if let Err(e) = unsafe { self.dev.CreateHeap::<ID3D12Heap>(&raw const desc, &raw mut heap) } {
            tracing::warn!(error = %e, "VICTORY_TILED: no video memory for more tiles");
            return None;
        }
        let heap = heap?;
        let h = match self.pools.iter().position(Option::is_none) {
            Some(i) => {
                self.pools[i] = Some(heap);
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pools.len() as u32 - 1
            }
        };
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        Some(((h, 0), false))
    }

    /// The tiles of `[a, a + n)` get video memory, mapped in their stretch. With `skip_empty`
    /// a tile none of whose pages is committed stays unmapped (it reads zeros).
    fn ensure_mapped(&mut self, a: usize, n: usize, skip_empty: bool) {
        let end = a.saturating_add(n);
        let mut t = a & !(TILE - 1);
        let mut new: Vec<(usize, (u32, u32))> = Vec::new();
        while t < end {
            if !self.mapped.contains_key(&t) && self.stretch_of(t).is_some() {
                let holes = query_tile_holes(t);
                if !(skip_empty && holes == 0xFFFF) {
                    let Some((tile, reused)) = self.take_tile() else { break };
                    self.mapped.insert(t, TileEntry { pool: tile, holes, seen: 0, merged: 0, sent: std::array::from_fn(|_| None) });
                    if reused {
                        self.to_zero.push((t, TILE));
                    }
                    new.push((t, tile));
                }
            }
            t += TILE;
        }
        // Local (VICTORY_TILED fix): mapped at the next submission, all at once
        // ([`State::flush_maps`]). One `UpdateTileMappings` per use cost about 3.5 ms each
        // (20,000 calls while the world loaded: the submitting thread stood for seconds and
        // the title's stall watchdog ended it).
        self.pending_maps.extend(new);
    }

    /// The tiles taken since the last submission mapped, one call per heap and stretch.
    fn flush_maps(&mut self) {
        let new = std::mem::take(&mut self.pending_maps);
        if new.is_empty() {
            return;
        }
        let started = std::time::Instant::now();
        let mut by_stretch: BTreeMap<usize, Vec<(usize, (u32, u32))>> = BTreeMap::new();
        for x in new {
            if let Some((b, _, _)) = self.stretch_of(x.0) {
                by_stretch.entry(b).or_default().push(x);
            }
        }
        for (b, tiles) in by_stretch {
            let Some(res) = self.stretches.get(&b).map(|s| s.res.clone()) else { continue };
            self.map_tiles(&res, b, &tiles);
        }
        self.signal();
        add(at::MAP_US, started.elapsed().as_micros() as u64);
    }

    /// origin: xcl sparse.rs `State::map_tiles`: batched per heap, contiguous runs merged.
    fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        let mut by_heap: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let (mut coords, mut regions, mut flags, mut offsets, mut counts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() && pairs[i + run as usize] == (x0 + run, t0 + run) {
                    run += 1;
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            let Some(heap) = self.pools.get(h as usize).and_then(Option::as_ref) else { continue };
            add(at::MAPS, 1);
            // SAFETY: a live reserved buffer and heap of ours; arrays of the lengths given that
            // live through the call.
            unsafe {
                self.queue.UpdateTileMappings(
                    res,
                    coords.len() as u32,
                    Some(coords.as_ptr()),
                    Some(regions.as_ptr()),
                    heap,
                    flags.len() as u32,
                    Some(flags.as_ptr()),
                    Some(offsets.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }
    }

    fn signal(&mut self) {
        self.value += 1;
        // SAFETY: our queue and fence.
        let _ = unsafe { self.queue.Signal(&self.fence, self.value) };
    }

    fn wait_for(&self, target: u64) {
        // SAFETY: our fence and event.
        unsafe {
            if target == 0 || self.fence.GetCompletedValue() >= target {
                return;
            }
            let _ = self.fence.SetEventOnCompletion(target, self.event);
            WaitForSingleObject(self.event, 5000);
            if self.fence.GetCompletedValue() < target {
                tracing::warn!(target, "VICTORY_TILED: our queue has not reached a fence value after 5 s");
            }
        }
    }

    fn reclaim_ring(&mut self) {
        // SAFETY: a plain query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv > completed {
                break;
            }
            self.ring_free = end_pos;
            self.inflight.pop_front();
        }
    }

    /// origin: xcl sparse.rs `begin_slot`.
    fn begin_slot(&mut self) -> usize {
        let idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let fv = self.slots[idx].fence_val;
        // SAFETY: a plain query.
        if fv > 0 && unsafe { self.fence.GetCompletedValue() } < fv {
            self.wait_for(fv);
            self.reclaim_ring();
        }
        let slot = &self.slots[idx];
        // SAFETY: the slot's last work has finished (waited above).
        unsafe {
            let _ = slot.alloc.Reset();
            let _ = slot.list.Reset(&slot.alloc, None);
        }
        idx
    }

    /// origin: xcl sparse.rs `submit_slot`.
    fn submit_slot(&mut self, idx: usize) {
        // SAFETY: a list of ours, recorded.
        unsafe {
            if self.slots[idx].list.Close().is_ok()
                && let Ok(l) = self.slots[idx].list.cast::<ID3D12CommandList>()
            {
                self.queue.ExecuteCommandLists(&[Some(l)]);
            }
        }
        self.signal();
        self.slots[idx].fence_val = self.value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((self.value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
    }

    fn upload_buffer(&self, bytes: usize, kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> Option<(ID3D12Resource, *mut u8)> {
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation and mapping of a CPU-visible buffer of ours.
        unsafe {
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_NONE),
                    state,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            Some((r, p.cast()))
        }
    }

    /// Local (VICTORY_TILED read-back): a buffer in the PC's memory, cached for the CPU, that
    /// shaders write (as `crate::indirect`'s shared buffers), mapped.
    fn shared_buffer(&self, bytes: usize) -> Option<(ID3D12Resource, *mut u8)> {
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation and mapping of a buffer of ours.
        unsafe {
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_CUSTOM,
                        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            Some((r, p.cast()))
        }
    }

    fn device_buffer(&self, bytes: usize) -> Option<ID3D12Resource> {
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer of ours.
        unsafe {
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut r,
                )
                .ok()?;
        }
        r
    }

    fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            self.staging = Some(self.upload_buffer(STAGING, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?);
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let Some((r, p)) = self.upload_buffer(ZERO_BUF, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ) else {
                return false;
            };
            // SAFETY: the mapping holds `ZERO_BUF` bytes.
            unsafe {
                std::ptr::write_bytes(p, 0, ZERO_BUF);
                r.Unmap(0, None);
            }
            self.zero_buf = Some(r);
        }
        true
    }

    fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            self.readback = Some(self.upload_buffer(READBACK_STAGING, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST)?);
        }
        self.readback.clone()
    }

    /// origin: xcl sparse.rs `upload_zeros`: `runs` cleared on our queue from the zero buffer.
    fn upload_zeros(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() || !self.ensure_zero_buf() {
            return;
        }
        let slot = self.begin_slot();
        let Some(zero) = self.zero_buf.clone() else { return };
        for &(a, n) in runs {
            let mut done = 0;
            while done < n {
                let cur = a + done;
                let Some((b, len, res)) = self.stretch_of(cur) else { break };
                let chunk = (n - done).min(ZERO_BUF).min(b + len - cur);
                // SAFETY: an open list of ours; live buffers; inside both.
                unsafe { self.slots[slot].list.CopyBufferRegion(res, (cur - b) as u64, &zero, 0, chunk as u64) };
                done += chunk;
            }
        }
        self.submit_slot(slot);
    }

    /// origin: xcl sparse.rs `upload_dirty`: `runs` copied from title memory into the image
    /// through the ring, on our queue, without a CPU wait unless the ring is full. Title memory
    /// is read through the kernel where a plain read would fault (`read_into`).
    fn upload_dirty(&mut self, runs: &[(usize, usize)]) {
        let Some(p) = self.staging_ptr() else { return };
        let Some(staging) = self.staging.as_ref().map(|s| s.0.clone()) else { return };
        let mut open: Option<usize> = None;
        for &(a, n) in runs {
            let mut done = 0;
            while done < n {
                let cur = a + done;
                let Some((b, len, _)) = self.stretch_of(cur) else { break };
                let (aligned, chunk) = plan_ring_chunk(self.ring_write, (n - done).min(b + len - cur), STAGING);
                if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                    self.reclaim_ring();
                    if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                        if let Some(slot) = open.take() {
                            self.submit_slot(slot);
                            self.reclaim_ring();
                        }
                        while (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                            let Some(&(fv, _)) = self.inflight.front() else { break };
                            self.wait_for(fv);
                            self.reclaim_ring();
                        }
                    }
                }
                let slot = match open {
                    Some(i) => i,
                    None => {
                        let i = self.begin_slot();
                        open = Some(i);
                        i
                    }
                };
                let dst_off = (aligned as usize) % STAGING;
                // SAFETY: inside the ring's mapping (`plan_ring_chunk` keeps the chunk whole).
                let dst = unsafe { std::slice::from_raw_parts_mut(p.add(dst_off), chunk) };
                if !crate::host::read_into(cur as *const u8, dst) {
                    // A page went away since it was scanned: zeros, as the GPU reads unmapped.
                    dst.fill(0);
                }
                let Some((_, _, res)) = self.stretch_of(cur) else { break };
                // SAFETY: an open list of ours; live buffers; inside both.
                unsafe { self.slots[slot].list.CopyBufferRegion(res, (cur - b) as u64, &staging, dst_off as u64, chunk as u64) };
                self.ring_write = aligned + chunk as u64;
                done += chunk;
                add(at::UP_BYTES, chunk as u64);
            }
        }
        add(at::UP_RUNS, runs.len() as u64);
        if let Some(slot) = open {
            self.submit_slot(slot);
        }
    }

    /// The chunks of `uses` (start, end; sorted and joined: what the submission's lists use)
    /// that the title changed since they were sent.
    ///
    /// Local (VICTORY_TILED fix): only what the submission uses is asked about, as copies
    /// are. Asking the kernel about every stretch's whole used span (2.9 GB in the world) at
    /// each submission was most of 240 ms a frame. What goes unasked is not lost: the journal
    /// keeps each page's last write, and a tile's `seen` tells what was sent since.
    fn scan(&mut self, uses: &[(usize, usize)]) -> Vec<usize> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for &(a, e) in uses {
            let mut cur = a;
            while cur < e {
                let Some((b, len, _)) = self.stretch_of(cur) else { break };
                let to = e.min(b + len);
                // Local (VICTORY_TILED views): view memory is asked only where it is all views.
                let near = if self.stretches.get(&b).is_some_and(|s| s.view) { 0 } else { 1 << 20 };
                match spans.last_mut() {
                    // Near enough in the same piece: one asking.
                    Some(last) if last.1 + near >= cur && last.1 > b => last.1 = to,
                    _ => spans.push((cur, to)),
                }
                cur = to;
            }
        }
        // Local (VICTORY_TILED read-back): this step only lists the 4 KiB pages to hash, under
        // the image's lock; they are hashed without it, on the wide threads ([`hash_pages`]),
        // and compared with what was sent under it again ([`State::apply_hashes`]). Hashing
        // under the lock (8 ms a frame) held up the copy-back thread's merges behind it.
        let mut pages: Vec<usize> = Vec::new();
        for (lo, hi) in spans {
            let view = self.stretch_of(lo).is_some_and(|(b, _, _)| self.stretches.get(&b).is_some_and(|s| s.view));
            // Local (VICTORY_TILED views): the aliased tiles of the span, with what they had
            // seen before this asking.
            let alias_old: Vec<(usize, u64)> = if view {
                let aliased: Vec<usize> = ALIASED.lock().unwrap_or_else(|e| e.into_inner()).range(lo & !(TILE - 1)..hi).copied().collect();
                aliased.into_iter().filter_map(|t| self.mapped.get(&t).map(|e| (t, e.seen))).collect()
            } else {
                Vec::new()
            };
            let asking = std::time::Instant::now();
            // Local (VICTORY_TILED views): view memory is not write-watched; it is asked as
            // copies of it are (fault tracking, or the guards).
            let found = crate::watch::writes(lo, hi, view);
            add(at::ASK_US, asking.elapsed().as_micros() as u64);
            let Some((now, written)) = found else {
                add(at::UNWATCHED, 1);
                continue;
            };
            for (p, asking) in written {
                let t = p * TILE;
                if t + TILE <= lo || t >= hi {
                    continue;
                }
                if !self.mapped.contains_key(&t) {
                    // Committed since its use was placed: video memory now.
                    self.ensure_mapped(t, TILE, true);
                }
                let Some(entry) = self.mapped.get_mut(&t) else { continue };
                if asking <= entry.seen {
                    continue;
                }
                let mask = crate::watch::stale_mask(p, entry.seen, true).unwrap_or(0xFFFF);
                entry.seen = now.max(asking);
                pages.extend((0..PAGES_PER_TILE).filter(|k| mask & (1 << k) != 0).map(|k| t + k * PAGE));
            }
            // Local (VICTORY_TILED views): a page shown at several addresses may be written
            // through another: the journal is asked about each of them too (with the tile's
            // `seen` from before this asking), and where there are many, it is hashed whole.
            if view {
                for (t, old) in alias_old {
                    let Some(entry) = self.mapped.get_mut(&t) else { continue };
                    let shown = crate::inplace::pages_of(t);
                    let mut mask = 0u16;
                    let mut latest = now;
                    if shown.len() > ALIASES_ASKED {
                        mask = 0xFFFF;
                        add(at::FORCED, PAGES_PER_TILE as u64);
                    } else {
                        for alias in shown.into_iter().filter(|&a| a != t) {
                            let Some((n, w)) = crate::watch::writes(alias, alias + TILE, true) else {
                                mask = 0xFFFF;
                                continue;
                            };
                            latest = latest.max(n);
                            for (p, asking) in w {
                                if p * TILE == alias && asking > old {
                                    mask |= crate::watch::stale_mask(p, old, true).unwrap_or(0xFFFF);
                                }
                            }
                        }
                    }
                    entry.seen = entry.seen.max(latest);
                    pages.extend((0..PAGES_PER_TILE).filter(|k| mask & (1 << k) != 0).map(|k| t + k * PAGE));
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        pages
    }

    /// The hashes of `pages` ([`hash_pages`]) compared with what was sent: the chunks that
    /// differ, as runs to send. A page merged into since `merges` was read is hashed again
    /// here (its hashes are of the bytes before the merge).
    fn apply_hashes(&mut self, pages: &[usize], hashes: &[Option<[u64; CHUNKS]>], merges: u64) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut page = [0u8; PAGE];
        for (&pg, h) in pages.iter().zip(hashes) {
            let t = pg & !(TILE - 1);
            let k = (pg - t) / PAGE;
            let Some(entry) = self.mapped.get_mut(&t) else { continue };
            add(at::HASHED, 1);
            if entry.merged > merges {
                if !crate::host::read_into(pg as *const u8, &mut page) {
                    entry.holes |= 1 << k;
                    continue;
                }
                entry.holes &= !(1 << k);
                scan_page(&page, pg, &mut entry.sent[k], &mut runs);
                continue;
            }
            let Some(h) = h else {
                entry.holes |= 1 << k;
                continue;
            };
            entry.holes &= !(1 << k);
            let slot = &mut entry.sent[k];
            if slot.is_none() {
                if h.iter().all(|&x| x == ZERO_HASH) {
                    continue;
                }
                *slot = Some(Box::new(ZERO_HASHES));
            }
            let Some(sent) = slot.as_deref_mut() else { continue };
            for c in 0..CHUNKS {
                if h[c] != sent[c] {
                    sent[c] = h[c];
                    push_run(&mut runs, pg + c * CHUNK, CHUNK);
                }
            }
        }
        runs
    }

    /// Clears the sent hashes of the pages of `[a, a + n)` (the card's bytes there are zeros).
    fn clear_sent(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        while pg < a + n {
            if let Some(e) = self.mapped.get_mut(&(pg & !(TILE - 1))) {
                e.sent[(pg % TILE) / PAGE] = None;
            }
            pg += PAGE;
        }
    }

    fn compare(&mut self) -> Option<&Compare> {
        if self.compare.is_none() {
            let made = (|| -> windows::core::Result<Compare> {
                let code = crate::host::compile(COMPARE, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
                // SAFETY: the root signature is inside the shader; the bytes live through the calls.
                unsafe {
                    let root: ID3D12RootSignature = self.dev.CreateRootSignature(0, &code)?;
                    let pso = self.dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                        ..Default::default()
                    })?;
                    Ok(Compare { root, pso })
                }
            })();
            match made {
                Ok(c) => self.compare = Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "VICTORY_TILED: no compare pass (shader writes are read back whole)");
                    return None;
                }
            }
        }
        self.compare.as_ref()
    }

    /// A pair of lists of ours for the host's queue, reset and open.
    fn take_host_lists(&mut self) -> Option<usize> {
        // SAFETY: a plain query.
        let done = unsafe { self.done_fence.GetCompletedValue() };
        let i = match self.host_lists.iter().position(|l| l.done <= done) {
            Some(i) => i,
            None => {
                // SAFETY: plain creation of our own objects.
                let made = unsafe {
                    (|| -> windows::core::Result<HostLists> {
                        let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                        let pre: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
                        pre.Close()?;
                        let check: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
                        check.Close()?;
                        Ok(HostLists { alloc, pre, check, done: 0 })
                    })()
                };
                self.host_lists.push(made.ok()?);
                self.host_lists.len() - 1
            }
        };
        Some(i)
    }

    /// Before a submission whose lists' shaders write `writes` (title ranges in the image): the
    /// two lists that go around its lists, and the check to read back after it.
    fn make_check(&mut self, writes: &[(usize, usize)]) -> Option<(ID3D12CommandList, ID3D12CommandList)> {
        // The ranges as whole 4 KiB pages, joined, cut at stretch pieces.
        let mut pages: Vec<(usize, usize)> = writes
            .iter()
            .filter(|&&(_, n)| n > 0)
            .map(|&(a, n)| (a & !(PAGE - 1), (a + n).next_multiple_of(PAGE)))
            .collect();
        pages.sort_unstable();
        let mut joined: Vec<(usize, usize)> = Vec::new();
        for (a, e) in pages {
            match joined.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(e),
                _ => joined.push((a, e)),
            }
        }
        let mut cut: Vec<(usize, usize, usize)> = Vec::new(); // (address, bytes, stretch base)
        for (a, e) in joined {
            let mut cur = a;
            while cur < e {
                let Some((b, len, _)) = self.stretch_of(cur) else { break };
                let to = e.min(b + len);
                cut.push((cur, to - cur, b));
                cur = to;
            }
        }
        if cut.is_empty() {
            return None;
        }
        let mut compared: Vec<(usize, usize, usize)> = Vec::new();
        let mut whole: Vec<(usize, usize)> = Vec::new();
        let mut total = 0;
        for (a, n, b) in cut {
            if total + n <= before_most() {
                compared.push((a, n, b));
                total += n;
            } else {
                whole.push((a, n));
            }
        }
        if self.compare().is_none() {
            whole.extend(compared.drain(..).map(|(a, n, _)| (a, n)));
            total = 0;
        }
        add(at::COMPARED, total as u64);
        add(at::WHOLE_BACK, whole.iter().map(|r| r.1 as u64).sum());
        // Local (VICTORY_TILED read-back): the compare hands back changed 64-byte chunks with
        // their bytes, room for every chunk compared up to `OUT_MOST`.
        let cap = (total / CHUNK).clamp(1, OUT_MOST).next_power_of_two();
        let mask_bytes = 4usize;
        // (title address, bytes, stretch base, first chunk, offset in `before`) of each compare.
        let mut items: Vec<(usize, usize, usize, u32, usize)> = Vec::with_capacity(compared.len());
        {
            let (mut off, mut first) = (0usize, 0u32);
            for &(a, n, b) in &compared {
                items.push((a, n, b, first, off));
                off += n;
                first += (n / CHUNK) as u32;
            }
        }
        // The scratch buffers: replaced by bigger ones when needed; the old ones kept until the
        // lists that may still use them are done.
        if total > 0 && self.before.as_ref().is_none_or(|b| b.1 < total) {
            let size = total.next_power_of_two().max(16 << 20);
            let Some(buf) = self.device_buffer(size) else {
                return None;
            };
            crate::host::name(&buf, || "tiled image compare: bytes before".into());
            if let Some((old, _)) = self.before.replace((buf, size)) {
                self.graveyard.push((self.done_value + 1, old));
            }
        }
        if self.mask.as_ref().is_none_or(|m| m.1 < mask_bytes) {
            let size = mask_bytes.next_power_of_two().max(64 << 10);
            let buf = self.device_buffer(size)?;
            crate::host::name(&buf, || "tiled image compare: mask".into());
            if let Some((old, _)) = self.mask.replace((buf, size)) {
                self.graveyard.push((self.done_value + 1, old));
            }
        }
        if !self.ensure_zero_buf() {
            return None;
        }
        // Local (VICTORY_TILED read-back): the output, from the pool or made.
        let out = match self.mask_pool.iter().position(|m| m.2 >= cap) {
            Some(i) => self.mask_pool.swap_remove(i),
            None => {
                let (r, p) = self.shared_buffer(OUT_HEAD + cap * (4 + CHUNK))?;
                crate::host::name(&r, || "tiled image compare: changed chunks".into());
                (r, p, cap)
            }
        };
        let cap = out.2;
        // SAFETY: the output's mapping; no list uses it (it was in the pool). The count's
        // pass writes it; zero until then.
        unsafe { (out.1 as *mut u32).write_volatile(0) };
        let li = self.take_host_lists()?;
        self.done_value += 1;
        let done = self.done_value;
        self.host_lists[li].done = done;
        let (alloc, pre, check) = (self.host_lists[li].alloc.clone(), self.host_lists[li].pre.clone(), self.host_lists[li].check.clone());
        let zero = self.zero_buf.clone()?;
        let (mask, _) = self.mask.clone()?;
        let before = self.before.as_ref().map(|b| b.0.clone()).filter(|_| !items.is_empty());
        let ranges: Vec<(usize, usize, u32)> = items.iter().map(|&(a, n, _, first, _)| (a, n / CHUNK, first)).collect();
        // SAFETY: lists of ours, done with (`take_host_lists`); live buffers; every copy and view
        // inside its buffer.
        unsafe {
            let _ = alloc.Reset();
            let _ = pre.Reset(&alloc, None);
            // The image's bytes before the lists, packed into `before`.
            if let Some(before) = before.as_ref() {
                for &(a, n, b, _, off) in &items {
                    if let Some(st) = self.stretches.get(&b) {
                        pre.CopyBufferRegion(before, off as u64, &st.res, (a - b) as u64, n as u64);
                    }
                }
            }
            pre.CopyBufferRegion(&mask, 0, &zero, 0, mask_bytes as u64);
            let mut back = vec![crate::host::transition(&mask, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)];
            if let Some(before) = before.as_ref() {
                back.push(crate::host::transition(before, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));
                let mut bases: Vec<usize> = items.iter().map(|c| c.2).collect();
                bases.dedup();
                for b in &bases {
                    if let Some(st) = self.stretches.get(b) {
                        back.push(crate::host::transition(&st.res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
                    }
                }
            }
            pre.ResourceBarrier(&back);
            pre.Close().ok()?;

            let _ = check.Reset(&alloc, None);
            check.ResourceBarrier(&[crate::indirect::uav_barrier()]);
            if let (Some(c), Some(before)) = (self.compare.as_ref(), before.as_ref()) {
                check.SetComputeRootSignature(&c.root);
                check.SetPipelineState(&c.pso);
                check.SetComputeRootUnorderedAccessView(3, mask.GetGPUVirtualAddress());
                check.SetComputeRootUnorderedAccessView(4, out.0.GetGPUVirtualAddress());
                for &(a, n, b, first, off) in &items {
                    let Some(st) = self.stretches.get(&b) else { continue };
                    let pages = n / PAGE;
                    let across = pages.clamp(1, 65535) as u32;
                    let rows = pages.div_ceil(across as usize) as u32;
                    let constants = [pages as u32, across, first, cap as u32, 0u32, 0, 0, 0];
                    check.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    check.SetComputeRootShaderResourceView(1, st.gpu + (a - b) as u64);
                    check.SetComputeRootShaderResourceView(2, before.GetGPUVirtualAddress() + off as u64);
                    check.Dispatch(across, rows, 1);
                }
                check.ResourceBarrier(&[crate::indirect::uav_barrier()]);
                // Local (VICTORY_TILED read-back): the count into the output's header.
                let constants = [0u32, 1, 0, cap as u32, 1u32, 0, 0, 0];
                check.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                check.Dispatch(1, 1, 1);
                check.ResourceBarrier(&[crate::indirect::uav_barrier()]);
            }
            check.Close().ok()?;
        }
        add(at::CHECKS, 1);
        self.checks.push_back(Check { done, out, ranges, whole });
        self.to_signal = Some(done);
        Some((pre.cast().ok()?, check.cast().ok()?))
    }

    /// Local (VICTORY_TILED fix): the read-back in two steps, so that the image's lock is not
    /// held while the card copies (it held up the submitting thread behind every read-back).
    /// This step: the pages of checks whose submission has run are copied into the read-back
    /// staging on our queue, as many as it holds. Returns the copies (title address, offset in
    /// the staging, bytes) and the fence value after which the staging holds them.
    fn land_take(&mut self) -> Option<(Vec<(usize, usize, usize)>, u64)> {
        // SAFETY: a plain query.
        let completed = unsafe { self.done_fence.GetCompletedValue() };
        while self.checks.front().is_some_and(|c| c.done <= completed) {
            let Some(check) = self.checks.pop_front() else { break };
            let mut pages: Vec<(usize, usize)> = check.whole.clone();
            // Local (VICTORY_TILED read-back): the changed chunks are in the output already
            // (the submission has run): merged from there, with no copy. Past its room, the
            // compared ranges are read back by page as before.
            let (_, p, cap) = check.out;
            // SAFETY: the output's mapping; the submission that writes it has run.
            let count = unsafe { (p as *const u32).read_volatile() } as usize;
            if count > cap {
                add(at::OVERFLOWS, 1);
                pages.extend(check.ranges.iter().map(|&(a, n, _)| (a, n * CHUNK)));
            } else if count > 0 {
                add(at::CARD_CHUNKS, count as u64);
                // SAFETY: inside the output (`OUT_HEAD + cap * (4 + CHUNK)` bytes).
                let numbers = unsafe { std::slice::from_raw_parts(p.add(OUT_HEAD) as *const u32, count) };
                let data = unsafe { p.add(OUT_HEAD + cap * 4) };
                let mut order: Vec<(usize, usize)> = numbers
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, &c)| {
                        let i = check.ranges.partition_point(|r| r.2 <= c).checked_sub(1)?;
                        let (a, n, first) = check.ranges[i];
                        let k = (c - first) as usize;
                        (k < n).then_some((a + k * CHUNK, slot))
                    })
                    .collect();
                order.sort_unstable();
                let mut i = 0;
                let mut chunks: Vec<(usize, *const u8)> = Vec::with_capacity(CHUNKS);
                while i < order.len() {
                    let pg = order[i].0 & !(PAGE - 1);
                    chunks.clear();
                    while i < order.len() && order[i].0 & !(PAGE - 1) == pg {
                        let (va, slot) = order[i];
                        // SAFETY: slot < count <= cap: inside the output's bytes.
                        chunks.push(((va - pg) / CHUNK, unsafe { data.add(slot * CHUNK) } as *const u8));
                        i += 1;
                    }
                    self.merge_page(pg, &chunks);
                }
                self.flush_writes();
            }
            self.land_left.extend(pages.into_iter().map(|(a, n)| (a, n, check.done)));
            self.mask_pool.push(check.out);
        }
        self.graveyard.retain(|(v, _)| *v > completed);
        if self.land_left.is_empty() {
            return None;
        }
        let (rb, _) = self.readback_buf()?;
        let mut pending: Vec<(usize, usize, usize)> = Vec::new();
        let mut used = 0usize;
        let mut after = 0u64;
        let mut open = false;
        while used < READBACK_STAGING {
            let Some((a, n, done)) = self.land_left.pop_front() else { break };
            let Some((b, len, _)) = self.stretch_of(a) else { continue };
            let tile = a & !(TILE - 1);
            let step = n.min(tile + TILE - a).min(b + len - a);
            if step < n {
                self.land_left.push_front((a + step, n - step, done));
            }
            if !self.mapped.contains_key(&tile) {
                continue;
            }
            let step = step.min(READBACK_STAGING - used);
            if step < n.min(tile + TILE - a).min(b + len - a) {
                // Cut by the staging's end: the rest at the next take.
                let rest = n.min(tile + TILE - a).min(b + len - a) - step;
                self.land_left.push_front((a + step, rest, done));
            }
            if !open {
                // SAFETY: the list's last work finished (each take is waited for).
                unsafe {
                    let _ = self.rb_alloc.Reset();
                    let _ = self.rb_list.Reset(&self.rb_alloc, None);
                }
                open = true;
            }
            after = after.max(done);
            let Some((_, _, res)) = self.stretch_of(a) else { break };
            // SAFETY: an open list of ours; live buffers; inside both.
            unsafe { self.rb_list.CopyBufferRegion(&rb, used as u64, res, (a - b) as u64, step as u64) };
            pending.push((a, used, step));
            used += step;
        }
        if !open {
            return None;
        }
        // SAFETY: our queue and fences; a list of ours, recorded. The host's queue signalled
        // `after` behind the lists (it has: the checks were taken only once it had).
        unsafe {
            let _ = self.queue.Wait(&self.done_fence, after);
            if self.rb_list.Close().is_ok()
                && let Ok(l) = self.rb_list.cast::<ID3D12CommandList>()
            {
                self.queue.ExecuteCommandLists(&[Some(l)]);
            }
        }
        self.signal();
        self.rb_last = self.value;
        Some((pending, self.value))
    }

    /// The second step of [`State::land_take`], once the card has copied: merged.
    fn land_merge(&mut self, pending: &[(usize, usize, usize)]) {
        self.reclaim_ring();
        let Some((_, p)) = self.readback.clone() else { return };
        for &(va, off, len) in pending {
            let mut at = va;
            while at < va + len {
                let pg = at & !(PAGE - 1);
                let step = (pg + PAGE).min(va + len) - at;
                // SAFETY: inside the read-back staging's mapping.
                let gpu = unsafe { std::slice::from_raw_parts(p.add(off + (at - va)), step) };
                self.merge(at, gpu);
                at += step;
            }
        }
        self.flush_writes();
    }

    /// The card's bytes `gpu` of title memory at `va` (inside one 4 KiB page, 64-byte aligned
    /// in practice) merged into title memory: a chunk the CPU changed since it was last sent
    /// keeps the CPU's bytes; another takes the card's, and its hash becomes the card's.
    fn merge(&mut self, va: usize, gpu: &[u8]) {
        let pg = va & !(PAGE - 1);
        let first = (va - pg) / CHUNK;
        let last = ((va - pg) + gpu.len()).div_ceil(CHUNK).min(CHUNKS);
        let mut chunks: Vec<(usize, *const u8)> = Vec::with_capacity(CHUNKS);
        for k in first..last {
            let (lo, hi) = (k * CHUNK, (k + 1) * CHUNK);
            // The card's bytes of this chunk, where `gpu` holds them.
            let (glo, ghi) = (lo.max(va - pg) - (va - pg), (hi - (va - pg)).min(gpu.len()));
            if ghi <= glo || ghi - glo != CHUNK {
                continue;
            }
            chunks.push((k, gpu[glo..].as_ptr()));
        }
        self.merge_page(pg, &chunks);
    }

    /// Local (VICTORY_TILED read-back): the card's bytes of chunks of the 4 KiB page `pg`
    /// ((chunk in the page, 64 bytes)) merged as [`State::merge`] says; title memory opened
    /// once for the page, and written with a plain copy where it can be.
    fn merge_page(&mut self, pg: usize, chunks: &[(usize, *const u8)]) {
        let t = pg & !(TILE - 1);
        let Some(entry) = self.mapped.get_mut(&t) else { return };
        let idx = (pg - t) / PAGE;
        if entry.holes & (1 << idx) != 0 || chunks.is_empty() {
            return;
        }
        let mut cpu = [0u8; PAGE];
        if !crate::host::read_into(pg as *const u8, &mut cpu) {
            return;
        }
        let sent = entry.sent[idx].get_or_insert_with(|| Box::new(ZERO_HASHES));
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for &(k, gp) in chunks {
            let (lo, hi) = (k * CHUNK, (k + 1) * CHUNK);
            // SAFETY: the caller's 64 bytes.
            let g = unsafe { std::slice::from_raw_parts(gp, CHUNK) };
            let c = &cpu[lo..hi];
            if chunk_hash(c) != sent[k] {
                // The CPU changed it since it was sent: the CPU's stands.
                if g != c {
                    add(at::CONFLICT_CHUNKS, 1);
                }
                continue;
            }
            if g != c {
                cpu[lo..hi].copy_from_slice(g);
                sent[k] = chunk_hash(g);
                push_run(&mut runs, lo, CHUNK);
            }
        }
        if !runs.is_empty() {
            self.merges += 1;
            entry.merged = self.merges;
            self.to_write.push((pg, Box::new(cpu), runs));
        }
        add(at::BACK_PAGES, 1);
    }

    /// Local (VICTORY_TILED read-back): the pages [`State::merge_page`] changed written into
    /// title memory, title memory told and opened once per 64 KiB block they share.
    ///
    /// Local (VICTORY_TILED fix): as the copies' merges do. Fills of constants taken ahead are
    /// told (`VICTORY_FILL_NO_WAIT`), and blocks fault tracking holds read-only are opened
    /// first: the kernel's write (`write_mapped`) does not change protections, and failed
    /// there without a word, so with the launcher's defaults the shaders' bytes never reached
    /// title memory (thousands of chunks a frame then counted as changed by the CPU). Written
    /// with the plain copy, the kernel's write where a page is not there.
    fn flush_writes(&mut self) {
        let mut pages = std::mem::take(&mut self.to_write);
        if pages.is_empty() {
            return;
        }
        let writing = std::time::Instant::now();
        pages.sort_unstable_by_key(|p| p.0);
        let mut i = 0;
        while i < pages.len() {
            let block = pages[i].0 & !(TILE - 1);
            let mut j = i;
            while j < pages.len() && pages[j].0 & !(TILE - 1) == block {
                j += 1;
            }
            let group = &pages[i..j];
            let lo = group[0].0 + group[0].2[0].0;
            let last = &group[group.len() - 1];
            let hi = last.0 + last.2.last().map_or(0, |r| r.0 + r.1);
            for (pg, cpu, runs) in group {
                let (a, b) = (runs[0].0, runs.last().map_or(0, |r| r.0 + r.1));
                crate::host::shaders_write(pg + a, &cpu[a..b]);
            }
            crate::watch::writing(lo, hi - lo);
            // Local (VICTORY_TILED views): the same bytes are at every other address that
            // shows the page: written there as far as anyone watching them knows.
            if is_aliased(block) {
                for alias in crate::inplace::pages_of(block) {
                    if alias != block {
                        crate::watch::writing(alias + (lo - block), hi - lo);
                    }
                }
            }
            for (pg, cpu, runs) in group {
                for &(off, n) in runs {
                    let (dst, src) = ((pg + off) as *mut u8, cpu[off..].as_ptr());
                    if !crate::host::write_alias(dst, src, n) {
                        crate::host::write_mapped(dst, src, n);
                    }
                    add(at::BACK_BYTES, n as u64);
                }
            }
            i = j;
        }
        add(at::WRITE_US, writing.elapsed().as_micros() as u64);
    }

    /// The ten-second line.
    fn say(&mut self) {
        let now = crate::host::now_ms();
        if now < self.line.1 + 10_000 {
            return;
        }
        let frame = crate::frame::current();
        let frames = frame.saturating_sub(self.line.0).max(1) as f64;
        let s: [u64; N_STATS] = std::array::from_fn(|i| STATS[i].load(Ordering::Relaxed));
        let d: [u64; N_STATS] = std::array::from_fn(|i| s[i].saturating_sub(self.line.2[i]));
        let reserved_mb = RESERVED.lock().unwrap_or_else(|e| e.into_inner()).values().map(|v| v.0).sum::<usize>() >> 20;
        let used_mb = self.stretches.values().map(|s| s.used.1.saturating_sub(s.used.0)).sum::<usize>() >> 20;
        tracing::info!(
            "tiled image: {} MB covered in {} stretches ({} MB reserved by the title), {} tiles mapped ({} MB of video memory in {} heaps, {} mapping calls in all); \
             per frame: {:.2} MB uploaded in {:.0} chunks, {:.1} ms scanning, {:.1} compares over {:.1} MB, {:.1} pages read back ({:.2} MB merged, {:.1} MB read back whole), \
             {:.0} chunks the CPU had changed under shader writes; uses on the image {} in all; fell back to copies: {} outside covered memory, {} across a stretch's end, \
             {} over VICTORY_TILED_MB, {} no device or reserved buffer; {} copies inside the image done on the CPU; {} spans not write-watched; {} tiles leaked; per frame: {:.1} ms asking the journal, {:.0} pages hashed, {:.1} ms mapping tiles, {:.1} ms sending up, {:.1} ms landing; {:.0} chunks the card changed handed back, {:.2} compares past their room; view memory: {} uses on the image, fell back: {} not all shown, {} shown at several addresses; {:.0} pages hashed whole a frame as shown at several; landing: {:.1} ms for the lock, {:.1} ms merging, of which {:.1} ms writing",
            used_mb,
            self.stretches.len(),
            reserved_mb,
            self.mapped.len(),
            (self.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize * TILE) >> 20,
            self.pools.iter().filter(|p| p.is_some()).count(),
            s[at::MAPS],
            d[at::UP_BYTES] as f64 / frames / (1 << 20) as f64,
            d[at::UP_RUNS] as f64 / frames,
            d[at::SCAN_US] as f64 / frames / 1000.0,
            d[at::CHECKS] as f64 / frames,
            d[at::COMPARED] as f64 / frames / (1 << 20) as f64,
            d[at::BACK_PAGES] as f64 / frames,
            d[at::BACK_BYTES] as f64 / frames / (1 << 20) as f64,
            d[at::WHOLE_BACK] as f64 / frames / (1 << 20) as f64,
            d[at::CONFLICT_CHUNKS] as f64 / frames,
            s[at::PLACED],
            s[at::NOT_COVERED],
            s[at::STRADDLES],
            s[at::OVER_BUDGET],
            s[at::NO_DEVICE],
            s[at::SAME_RESOURCE_COPY],
            s[at::UNWATCHED],
            s[at::LEAKED_TILES],
            d[at::ASK_US] as f64 / frames / 1000.0,
            d[at::HASHED] as f64 / frames,
            d[at::MAP_US] as f64 / frames / 1000.0,
            d[at::UP_US] as f64 / frames / 1000.0,
            d[at::LAND_US] as f64 / frames / 1000.0,
            d[at::CARD_CHUNKS] as f64 / frames,
            d[at::OVERFLOWS] as f64 / frames,
            s[at::VIEW_PLACED],
            s[at::VIEW_UNMAPPED],
            s[at::VIEW_ALIASED],
            d[at::FORCED] as f64 / frames,
            d[at::LAND_LOCK_US] as f64 / frames / 1000.0,
            d[at::LAND_TAKE_US] as f64 / frames / 1000.0,
            d[at::WRITE_US] as f64 / frames / 1000.0,
        );
        self.line = (frame, now, s);
    }
}

/// The compare pass: one group of 64 threads per 4 KiB page, a thread per 64 bytes. Local
/// (VICTORY_TILED read-back): a chunk whose bytes differ from the bytes before takes a slot
/// (the counter is in video memory) and writes its number and its bytes into the output, in
/// the PC's memory; plain stores only go there. With `mode` 1, one thread writes the count into
/// the output's header.
const COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Range : register(b0) {
    uint pages;
    uint across;
    uint first_chunk;
    uint cap;
    uint mode;
    uint unused0;
    uint unused1;
    uint unused2;
};
ByteAddressBuffer image : register(t0);
ByteAddressBuffer before : register(t1);
RWByteAddressBuffer counter : register(u0);
RWByteAddressBuffer output : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint t : SV_GroupIndex) {
    if (mode == 1u) {
        if (t == 0u) {
            output.Store(0, counter.Load(0));
        }
        return;
    }
    uint page = g.y * across + g.x;
    if (page >= pages) {
        return;
    }
    uint at = page * 4096u + t * 64u;
    uint4 a0 = image.Load4(at);
    uint4 a1 = image.Load4(at + 16u);
    uint4 a2 = image.Load4(at + 32u);
    uint4 a3 = image.Load4(at + 48u);
    uint4 b0 = before.Load4(at);
    uint4 b1 = before.Load4(at + 16u);
    uint4 b2 = before.Load4(at + 32u);
    uint4 b3 = before.Load4(at + 48u);
    if (any(a0 != b0) || any(a1 != b1) || any(a2 != b2) || any(a3 != b3)) {
        uint slot;
        counter.InterlockedAdd(0, 1u, slot);
        if (slot < cap) {
            output.Store(256u + slot * 4u, first_chunk + page * 64u + t);
            uint d = 256u + cap * 4u + slot * 64u;
            output.Store4(d, a0);
            output.Store4(d + 16u, a1);
            output.Store4(d + 32u, a2);
            output.Store4(d + 48u, a3);
        }
    }
}
"#;

// ---------------------------------------------------------------------------------------------
// Hooks for `crate::host`.

/// Before the host's queue runs a submission's lists: what the title changed in the image goes
/// up (on our queue; `queue` waits for it on the card), and for the ranges the lists' shaders
/// write in the image (`writes`), the two lists that go before and after them.
pub(crate) fn before_execute(queue: &ID3D12CommandQueue, uses: &[(usize, usize)], writes: &[(usize, usize)]) -> Option<(ID3D12CommandList, ID3D12CommandList)> {
    let _phase = Phase::enter(PH_BEFORE);
    // Local (VICTORY_TILED read-back): one submission at a time here; the image's lock is let
    // go while the pages are hashed.
    static SUBMITTING: Mutex<()> = Mutex::new(());
    let _submitting = SUBMITTING.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::time::Instant::now();
    let (mut zeros, pages, merges) = {
        let mut g = lock_state();
        let s = g.as_mut()?;
        // SAFETY: a plain query.
        let completed = unsafe { s.done_fence.GetCompletedValue() };
        s.graveyard.retain(|(v, _)| *v > completed);
        // Tiles taken again since the last submission (`place_for`): cleared, and their hashes
        // with them, before the scan. Tiles the scan itself maps are new entries (no hashes)
        // and are cleared with these, ahead of the scan's uploads.
        let zeros = std::mem::take(&mut s.to_zero);
        for &(a, n) in &zeros {
            s.clear_sent(a, n);
        }
        (zeros, s.scan(uses), s.merges)
    };
    let hashes = hash_pages(&pages);
    let mut g = lock_state();
    let s = g.as_mut()?;
    let runs = s.apply_hashes(&pages, &hashes, merges);
    zeros.append(&mut s.to_zero);
    add(at::SCAN_US, started.elapsed().as_micros() as u64);
    // Local (VICTORY_TILED fix): the tiles taken since the last submission mapped first.
    s.flush_maps();
    // Clears before the title's bytes: the copies of one queue run in order.
    let uploading = std::time::Instant::now();
    s.upload_zeros(&zeros);
    if !runs.is_empty() {
        s.upload_dirty(&runs);
    }
    add(at::UP_US, uploading.elapsed().as_micros() as u64);
    if s.value > s.waited {
        // SAFETY: the host's queue and our fence; the value is signalled on our queue already.
        unsafe {
            let _ = queue.Wait(&s.fence, s.value);
        }
        s.waited = s.value;
    }
    let lists = if writes.is_empty() { None } else { s.make_check(writes) };
    s.say();
    lists
}

/// Right after the host's queue was given a submission's lists, before its own signal: the done
/// value of the submission's check, if [`before_execute`] made one.
pub(crate) fn after_execute(queue: &ID3D12CommandQueue) {
    if !on() {
        return;
    }
    let _phase = Phase::enter(PH_AFTER);
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    if let Some(done) = s.to_signal.take() {
        // SAFETY: the host's queue and our fence.
        unsafe {
            let _ = queue.Signal(&s.done_fence, done);
        }
    }
}

/// Where host copies are copied back after submissions have run: the image's changed pages of
/// those submissions merged into title memory.
pub(crate) fn hand_back() {
    if !on() {
        return;
    }
    let _phase = Phase::enter(PH_LAND);
    // One landing at a time (the read-back staging is one).
    static LANDING: Mutex<()> = Mutex::new(());
    let _landing = LANDING.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::time::Instant::now();
    loop {
        let (job, fence) = {
            let waiting = std::time::Instant::now();
            let mut g = lock_state();
            add(at::LAND_LOCK_US, waiting.elapsed().as_micros() as u64);
            let Some(s) = g.as_mut() else { return };
            let taking = std::time::Instant::now();
            let job = s.land_take();
            add(at::LAND_TAKE_US, taking.elapsed().as_micros() as u64);
            (job, s.fence.clone())
        };
        let Some((pending, value)) = job else { break };
        // Waited for without the image's lock. SAFETY: our fence; a null event blocks until
        // the value is reached.
        unsafe {
            if fence.GetCompletedValue() < value {
                let _ = fence.SetEventOnCompletion(value, HANDLE::default());
            }
        }
        let mut g = lock_state();
        let Some(s) = g.as_mut() else { return };
        s.land_merge(&pending);
    }
    add(at::LAND_US, started.elapsed().as_micros() as u64);
}
