//! Clamps of many executions in one pass: the arguments of every indirect
//! execution between two points where they may change are bounded together,
//! at the first of them (`VICTORY_INDIRECT_BATCH`, `crate::draw::batches`).
//!
//! The executions that come after it are not known yet when the pass is
//! recorded, so it reads what to clamp from a table of jobs in upload
//! memory, which the CPU goes on filling while the list is recorded: the
//! pass is as wide as the table and its threads past the count do nothing.
//! An execution that asks for a record already in the table (the title's
//! tile classes) reads the same clamped one.
//!
//! What the records charge the list's budget is as for records clamped
//! alone, times the executions that read them.
//!
//! From cool_fox (github.com/codec-d/cool_fox, GPL-3.0-or-later):
//! `crates/victory-gpu/src/indirect/batch.rs`. (Neptune) The shader bounds a
//! record exactly as Neptune's clamp of one record does (`super::SOURCE`:
//! no `WIDEST`/`REST` switches, counts of a draw cut to 2^24, a mesh record
//! over 65,535 groups cut, not emptied), and its notes go to the words of
//! Neptune's budget buffer (`HostList::budget`): 4 and 8 the mesh notes, 12
//! and 16 the compute notes. The scratch buffers' places
//! ([`place`]) and their move back to the common state ([`close`]) live
//! here rather than in `crate::host`.

use windows::Win32::Graphics::Direct3D12::*;

use super::{Clamp, uav_barrier};
use crate::host::{HostList, host, transition};

/// The jobs of one pass.
pub(crate) const JOBS: u32 = 256;
/// The bytes of its table in upload memory: the count, then the jobs.
pub(crate) const TABLE: u64 = 16 + 16 * JOBS as u64;
/// The bytes of a clamped record's place in the scratch buffer.
pub(crate) const SLOT: u32 = 32;
/// The source buffers a pass can read.
pub(crate) const SOURCES: usize = 4;
/// The most executions charged by one job: the most a record costs (2^26)
/// as often still fits 32 bits.
pub(crate) const TIMES: u32 = 31;
/// The batches of clamped records in one scratch buffer ([`place`]).
const BATCHES: u32 = 8;
/// (Neptune) The longest record stride an execution that reads a batch may
/// name: the scratch buffer has that many bytes past its last place, so that
/// an execution of the last one reads inside it however the stride counts.
pub(crate) const STRIDE_MOST: u32 = 256;

// From cool_fox (github.com/codec-d/cool_fox, GPL-3.0-or-later), with the
// bounds and notes of Neptune's clamp of one record (`super::SOURCE`).
pub(crate) const SOURCE: &str = r#"
#define RS "SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0), UAV(u1)"
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
ByteAddressBuffer src2 : register(t2);
ByteAddressBuffer src3 : register(t3);
// The count, 12 bytes of nothing, then per job: source (bits 0-1), kind
// (4-7), words (8-11) and the executions that read it (16-31); the
// record's offset in the source; its place in the scratch buffer; nothing.
ByteAddressBuffer jobs : register(t4);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far, then its notes (Neptune's words: 4 the largest
// mesh record, 8 the mesh groups asked in all, 12 the largest x a compute
// record asked for, 16 the compute records cut in x).
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint mesh_largest;
groupshared uint mesh_all;
groupshared uint wide_largest;
groupshared uint wide_cut;
uint load(uint source, uint at) {
    if (source == 0) return src0.Load(at);
    if (source == 1) return src1.Load(at);
    if (source == 2) return src2.Load(at);
    return src3.Load(at);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint count = jobs.Load(0);
    bool active = id.x < count;
    uint4 job = uint4(0, 0, 0, 0);
    uint record[5] = {0, 0, 0, 0, 0};
    uint3 asked = uint3(0, 0, 0);
    uint cost = 0;
    uint kind = 0;
    uint words = 0;
    if (gi == 0) {
        mesh_largest = 0;
        mesh_all = 0;
        wide_largest = 0;
        wide_cut = 0;
    }
    // As Neptune's clamp of one execution (`super::SOURCE`) bounds them.
    if (active) {
        job = jobs.Load4(16 + id.x * 16);
        kind = (job.x >> 4) & 15;
        words = min((job.x >> 8) & 15, 5);
        for (uint w = 0; w < words; w++) {
            record[w] = load(job.x & 3, job.y + w * 4);
        }
        asked = uint3(record[0], record[1], record[2]);
        if (kind <= 1) {
            uint verts = min(record[0], 1u << 24);
            uint instances = min(record[1], (1u << 26) / max(verts, 1u));
            record[0] = verts;
            record[1] = instances;
            cost = verts * instances;
        } else if (kind == 2) {
            uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
            // x * y fits 32 bits; the whole count may not.
            if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64u;
            record[0] = g.x;
            record[1] = g.y;
            record[2] = g.z;
        } else {
            uint3 g = uint3(min(asked.x, 65535u), min(asked.y, 1u), min(asked.z, 1u));
            cost = g.x * g.y * g.z * 128u;
            record[0] = g.x;
            record[1] = g.y;
            record[2] = g.z;
        }
        cost *= job.x >> 16;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    // Notes, per record clamped (not per execution that reads it).
    if (active && kind == 2 && asked.x <= (1u << 24) && asked.y <= 64u && asked.z <= 64u) {
        uint was;
        InterlockedMax(wide_largest, asked.x, was);
        if (asked.x > 65535u) {
            InterlockedAdd(wide_cut, 1u, was);
        }
    }
    if (active && kind > 2) {
        uint was;
        InterlockedMax(mesh_largest, asked.x, was);
        InterlockedAdd(mesh_all, min(asked.x, 1u << 24), was);
    }
    uint first = id.x - gi;
    uint n = count > first ? min(64u, count - first) : 0u;
    if (gi == 0 && n > 0) {
        // What the group's records charge, from `before` on, in lane order:
        // a record that does not fit takes the rest and runs empty, as each
        // record clamped alone does in Neptune's clamp.
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next <= c) ? BUDGET : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && n > 0) {
        uint was;
        if (mesh_largest > 0 || mesh_all > 0) {
            budget.InterlockedMax(4, mesh_largest, was);
            budget.InterlockedAdd(8, mesh_all, was);
        }
        if (wide_largest > 0) {
            budget.InterlockedMax(12, wide_largest, was);
        }
        if (wide_cut > 0) {
            budget.InterlockedAdd(16, wide_cut, was);
        }
    }
    if (!active) {
        return;
    }
    if (fits[gi] == 0) {
        record[0] = 0;
        record[1] = 0;
        if (kind > 1) {
            record[2] = 0;
        }
    }
    for (uint s = 0; s < words; s++) {
        dst_args.Store(job.z + s * 4, record[s]);
    }
}
"#;

/// The words of a record of `kind` (PC argument numbering).
pub(crate) fn words(kind: u32) -> Option<u32> {
    super::min_stride(kind).map(|bytes| bytes / 4)
}

/// A job's first word.
pub(crate) fn job(source: u32, kind: u32, times: u32) -> u32 {
    source | (kind << 4) | (words(kind).unwrap_or(0) << 8) | (times << 16)
}

/// The jobs of a pass as the CPU fills them ([`SOURCE`]), and where it
/// keeps which record is where.
#[derive(Default)]
pub(crate) struct Table {
    /// The table in upload memory ([`TABLE`] bytes).
    at: usize,
    /// The place of the first clamped record in the scratch buffer.
    base: u32,
    /// The host copies it can read, by number ([`crate::shadow::Shadow::id`];
    /// 0: none).
    sources: [u64; SOURCES],
    /// How many executions each job is charged for so far.
    times: Vec<u32>,
    /// The job of a record (source, offset, kind).
    known: std::collections::HashMap<(u32, u32, u32), u32>,
}

impl Table {
    /// A table at `at`, with no job yet.
    ///
    /// # Safety
    /// `at` is [`TABLE`] bytes of mapped upload memory, 4-byte aligned, that
    /// live until the list that reads them is recorded and is not run
    /// before.
    pub(crate) unsafe fn new(at: *mut u8, base: u32, sources: [u64; SOURCES]) -> Table {
        // SAFETY: the caller's.
        unsafe { at.cast::<u32>().write_volatile(0) };
        Table { at: at as usize, base, sources, ..Table::default() }
    }

    /// The place in the scratch buffer of the record of `kind` at `offset`
    /// of the host copy `id`, once the pass has run: a job of it is added if
    /// there is none. `None` if the pass cannot read that copy or has no
    /// room for another job.
    pub(crate) fn place(&mut self, id: u64, offset: u64, kind: u32) -> Option<u64> {
        let source = self.sources.iter().position(|&s| s == id)? as u32;
        let offset = u32::try_from(offset).ok().filter(|o| o.is_multiple_of(4))?;
        offset.checked_add(words(kind)? * 4)?;
        let key = (source, offset, kind);
        let table = self.at as *mut u32;
        let place = |i: u32| u64::from(self.base + i * SLOT);
        if let Some(&i) = self.known.get(&key)
            && self.times[i as usize] < TIMES
        {
            self.times[i as usize] += 1;
            // SAFETY: the job's own word of the table ([`Table::new`]).
            unsafe { table.add(4 + 4 * i as usize).write_volatile(job(source, kind, self.times[i as usize])) };
            return Some(place(i));
        }
        let i = self.times.len() as u32;
        if i == JOBS {
            return None;
        }
        self.times.push(1);
        self.known.insert(key, i);
        // SAFETY: inside the table: fewer than [`JOBS`] jobs before it.
        unsafe {
            let to = table.add(4 + 4 * i as usize);
            to.write_volatile(job(source, kind, 1));
            to.add(1).write_volatile(offset);
            to.add(2).write_volatile(self.base + i * SLOT);
            to.add(3).write_volatile(0);
            table.write_volatile(i + 1);
        }
        Some(place(i))
    }
}

/// Records the pass over the table at `jobs` (a host address) into `dst`,
/// which is in state `before` and left in `INDIRECT_ARGUMENT`: after every
/// earlier shader write (the title's argument passes), before what reads
/// the clamped records.
pub(super) fn record(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    sources: [u64; SOURCES],
    jobs: u64,
) {
    let after = transition(
        dst,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    // SAFETY: an open list; our root signature and pipeline; live buffers:
    // the sources and the table are read in the common state (promoted).
    unsafe {
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for (i, source) in sources.into_iter().enumerate() {
            l.SetComputeRootShaderResourceView(i as u32, source);
        }
        l.SetComputeRootShaderResourceView(SOURCES as u32, jobs);
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 1, dst.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 2, budget);
        l.Dispatch(JOBS.div_ceil(64), 1, 1);
        // Nothing else the pass writes is read before the next full wait
        // (the budget, by the next pass): only the records have to be there.
        l.ResourceBarrier(&[after]);
    }
}

/// Where the next batch of clamped records goes: a device-local buffer
/// the list keeps from recording to recording, the batch's place in it
/// and the state the buffer is in. Every batch of a recording has a place
/// of its own: the executions before it may still read theirs.
///
/// From cool_fox (github.com/codec-d/cool_fox, GPL-3.0-or-later):
/// `HostList::batch_place`.
pub(crate) fn place(list: &HostList) -> Option<(ID3D12Resource, u32, D3D12_RESOURCE_STATES)> {
    let mut args = list.batch_args.lock().unwrap_or_else(|e| e.into_inner());
    let (buffer, i) = ((args.1 / BATCHES) as usize, args.1 % BATCHES);
    if buffer == args.0.len() {
        let size = u64::from(BATCHES * JOBS * SLOT + STRIDE_MOST);
        let made = host()?.uav_buffer(size)?;
        crate::host::name(&made, || format!("indirect batches {size:#x}"));
        args.0.push(made);
    }
    args.1 += 1;
    // (Its first batch finds it as the recording before left it: [`close`].)
    let state = if i == 0 { D3D12_RESOURCE_STATE_COMMON } else { D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT };
    Some((args.0[buffer].clone(), i * JOBS * SLOT, state))
}

/// Before the list closes: the scratch buffers of the batched clamps go
/// back to the common state, where the next recording expects them.
///
/// From cool_fox (github.com/codec-d/cool_fox, GPL-3.0-or-later):
/// `HostList::batches_common`.
pub(crate) fn close(list: &HostList) {
    let mut args = list.batch_args.lock().unwrap_or_else(|e| e.into_inner());
    let used = std::mem::take(&mut args.1).div_ceil(BATCHES) as usize;
    let barriers: Vec<D3D12_RESOURCE_BARRIER> = args.0[..used.min(args.0.len())]
        .iter()
        .map(|b| transition(b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON))
        .collect();
    if !barriers.is_empty() {
        // SAFETY: an open list and live buffers this recording left in
        // `INDIRECT_ARGUMENT`.
        unsafe { list.list.ResourceBarrier(&barriers) };
    }
}
