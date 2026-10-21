#!/usr/bin/env bash
# Qualify the exact staged bundle on Radeon VII, without gfx1151/XDNA fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."
native_dir="$(realpath "${1:?usage: test-gfx906-hardware.sh NATIVE_LIBRARY_DIRECTORY}")"
export HRX_AMDF_LIBRARY="$native_dir/libamdf.so"
export HRX_FABRIC_LIBRARY="$native_dir/libhrx_fabric.so"
export HRX_LOOM_LIBRARY="$native_dir/libloomc.so"
export HRX_TEST_LOOM="$HRX_LOOM_LIBRARY"
export HRX_OFFLINE=1
cargo run --locked --bin hrx -- info
# Bound each process too: a broken driver must not hang a qualification runner.
timeout 180 cargo test --locked --test gfx906 -- --ignored --test-threads=1
timeout 180 cargo test --locked --lib fabric::memory::file_fill_tests -- --ignored --test-threads=1
timeout 180 cargo test --locked --lib fabric::queue::serial::tests -- --ignored --test-threads=1
timeout 180 cargo test --locked --lib fabric::executable::tests -- --ignored --test-threads=1
for test in observing_an_event_retires_every_earlier_submission shared_queue_submission_waits_for_other_producers stream_queue_pool_is_bounded_shared_and_weak; do
    timeout 180 cargo test --locked --lib "$test" -- --ignored --test-threads=1
done
