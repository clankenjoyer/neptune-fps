//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(test, windows))]
mod agility;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
mod coalesce;
mod epoch;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod ambient;
#[cfg(windows)]
mod cores;
#[cfg(windows)]
mod placement;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
pub mod frame;
pub mod frametime;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod overlay;
#[cfg(windows)]
mod pixels;
#[cfg(windows)]
mod prof;
#[cfg(windows)]
mod screenshot;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
// Local (VICTORY_INPLACE_ROLE)
#[cfg(windows)]
mod inplace_role;
// Ported from our build (VICTORY_LAZY_BACK).
#[cfg(windows)]
mod lazy;
// Ported from our build (VICTORY_BACK_ON_DEMAND): the per-copy stamp it needs.
#[cfg(windows)]
mod stamp;
// Ported from our build (VICTORY_QUEUE2).
#[cfg(windows)]
mod queue2;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod ramline;
#[cfg(windows)]
mod shadow;
#[cfg(windows)]
mod windowup;
#[cfg(windows)]
mod tiles;
// Local (VICTORY_TILED)
#[cfg(windows)]
pub mod tiled;
pub mod signature;
#[cfg(windows)]
mod texback;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
pub mod titlemem;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
// Ported from our build (VICTORY_LAZY_BACK): for the runtime's loads through views of its own.
#[cfg(windows)]
#[allow(unused_imports)]
pub use lazy::title_memory_overwriting;
// Ported from our build (VICTORY_FAULT_TRACK): for the runtime's allocation and free paths.
#[cfg(windows)]
#[allow(unused_imports)]
pub use watch::title_memory_allocated;
#[cfg(windows)]
pub use watch::{
    Remapping, notify_view_writes, title_memory_reading, title_memory_reading_range, title_memory_remapped,
    title_memory_remapping, title_memory_remapping_range, title_memory_written, writing as kernel_will_write,
    written_through_view,
};
