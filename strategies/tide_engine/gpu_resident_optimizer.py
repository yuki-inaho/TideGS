import math
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - torch CUDA wheels normally provide Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_resident_adam_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        local_rows_ptr,
        state_rows_ptr,
        state_slots_ptr,
        bias_correction1_ptr,
        denom_scale_ptr,
        n_rows,
        learning_rate,
        beta1,
        beta2,
        eps,
        batch_size,
        WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = n_rows * WIDTH
        mask = offsets < total
        row = offsets // WIDTH
        col = offsets - row * WIDTH

        local_row = tl.load(local_rows_ptr + row, mask=mask, other=0)
        state_row = tl.load(state_rows_ptr + row, mask=mask, other=0)
        state_slot = tl.load(state_slots_ptr + row, mask=mask, other=0)

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = grad / batch_size

        state_offset = state_row * WIDTH + col
        param_offset = local_row * WIDTH + col
        exp_avg = tl.load(exp_avg_ptr + state_offset, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + state_offset, mask=mask, other=0.0).to(tl.float32)

        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

        bias_correction1 = tl.load(
            bias_correction1_ptr + state_slot, mask=mask, other=1.0
        ).to(tl.float32)
        denom_scale = tl.load(
            denom_scale_ptr + state_slot, mask=mask, other=1.0
        ).to(tl.float32)
        denom = tl.sqrt(exp_avg_sq) / denom_scale + eps

        param = tl.load(param_ptr + param_offset, mask=mask, other=0.0).to(tl.float32)
        param -= (learning_rate / bias_correction1) * exp_avg / denom

        tl.store(exp_avg_ptr + state_offset, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + state_offset, exp_avg_sq, mask=mask)
        tl.store(param_ptr + param_offset, param, mask=mask)


class GPUResidentAdam:
    """Adam with moments stored only for the current resident block set.

    Blocks keep stable state slots while resident. Eviction releases the slot and
    discards its moments; a later admission starts from zero. The update itself is
    vectorized over every touched row, avoiding the former block-by-component
    Python loop and its hundreds of thousands of tiny CUDA kernels.
    """

    COMPONENT_SPECS = (
        ("xyz", "_xyz", 3, 0),
        ("opacity", "_opacity", 1, 1),
        ("scaling", "_scaling", 3, 2),
        ("rotation", "_rotation", 4, 3),
        ("features_dc", "_features_dc", 3, 4),
        ("features_rest", "_features_rest", 45, 5),
    )
    TOTAL_WIDTH = sum(spec[2] for spec in COMPONENT_SPECS)

    def __init__(
        self,
        batch_size: int = 1,
        block_size: int = 4096,
        capacity_blocks: int = 0,
        device: str = "cuda",
    ):
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.capacity_blocks = max(0, int(capacity_blocks))
        self.device = torch.device(device)
        self.state_mode = "resident_blocks"

        self._resident_blocks = set()
        self._resident_streaks: Dict[int, int] = {}
        self._completed_streak_total = 0
        self._completed_streak_count = 0

        self._allocated_slots = 0
        self._block_to_slot: Dict[int, int] = {}
        self._slot_to_block: List[Optional[int]] = []
        self._free_slots: List[int] = []
        self._slot_steps: List[int] = []
        self._slot_initialized: List[bool] = []
        self._slot_row_counts: List[int] = []
        self._exp_avg: Dict[str, torch.Tensor] = {}
        self._exp_avg_sq: Dict[str, torch.Tensor] = {}
        self._bias_correction1: Optional[torch.Tensor] = None
        self._denom_scale: Optional[torch.Tensor] = None

        self._logical_state_blocks = 0
        self._logical_state_bytes = 0
        self._stats = {
            "state_mode": "resident_blocks",
            "resident_target_blocks": 0,
            "resident_state_blocks": 0,
            "resident_state_bytes": 0,
            "allocated_state_bytes": 0,
            "cold_restarts": 0,
            "state_evictions": 0,
            "optimizer_rows_touched_total": 0,
            "cold_restarted_rows_touched_total": 0,
            "mean_resident_streak": 0.0,
        }

    def _physical_state_bytes(self) -> int:
        return int(self._allocated_slots * self.block_size * self.TOTAL_WIDTH * 2 * 4)

    def _update_state_stats(self) -> None:
        self._stats["resident_target_blocks"] = len(self._resident_blocks)
        self._stats["resident_state_blocks"] = int(self._logical_state_blocks)
        self._stats["resident_state_bytes"] = int(self._logical_state_bytes)
        self._stats["allocated_state_bytes"] = self._physical_state_bytes()
        streak_total = self._completed_streak_total + sum(self._resident_streaks.values())
        streak_count = self._completed_streak_count + len(self._resident_streaks)
        self._stats["mean_resident_streak"] = float(streak_total) / max(1, streak_count)

    def _normalize_columns_lr(self, columns_lr) -> Dict[str, float]:
        if columns_lr is None:
            raise RuntimeError("[GPUResidentAdam] optimizer.columns_lr is required")
        if torch.is_tensor(columns_lr):
            cols = columns_lr.detach().to(device="cpu", dtype=torch.float32)
        else:
            cols = torch.tensor(columns_lr, dtype=torch.float32)
        cols = cols.flatten().contiguous()
        if cols.numel() == 6:
            grouped = cols
        elif cols.numel() == 59:
            grouped = cols[torch.tensor([0, 3, 4, 7, 11, 14])]
        else:
            raise RuntimeError(
                f"[GPUResidentAdam] Unexpected columns_lr width={cols.numel()}; "
                "expected 6 grouped or 59 expanded entries"
            )
        return {
            name: float(grouped[group_idx].item())
            for name, _, _, group_idx in self.COMPONENT_SPECS
        }

    def _ensure_storage(self, required_slots: int) -> None:
        if required_slots <= self._allocated_slots:
            return
        target = max(required_slots, self.capacity_blocks, max(1, self._allocated_slots * 2))
        old_slots = self._allocated_slots
        old_rows = old_slots * self.block_size
        new_rows = target * self.block_size

        new_exp_avg: Dict[str, torch.Tensor] = {}
        new_exp_avg_sq: Dict[str, torch.Tensor] = {}
        for name, _, width, _ in self.COMPONENT_SPECS:
            avg = torch.empty((new_rows, width), dtype=torch.float32, device=self.device)
            avg_sq = torch.empty((new_rows, width), dtype=torch.float32, device=self.device)
            if old_rows:
                avg[:old_rows].copy_(self._exp_avg[name])
                avg_sq[:old_rows].copy_(self._exp_avg_sq[name])
            new_exp_avg[name] = avg
            new_exp_avg_sq[name] = avg_sq

        bias = torch.empty((target,), dtype=torch.float32, device=self.device)
        denom = torch.empty((target,), dtype=torch.float32, device=self.device)
        if old_slots:
            bias[:old_slots].copy_(self._bias_correction1)
            denom[:old_slots].copy_(self._denom_scale)

        self._exp_avg = new_exp_avg
        self._exp_avg_sq = new_exp_avg_sq
        self._bias_correction1 = bias
        self._denom_scale = denom
        self._slot_to_block.extend([None] * (target - old_slots))
        self._slot_steps.extend([0] * (target - old_slots))
        self._slot_initialized.extend([False] * (target - old_slots))
        self._slot_row_counts.extend([0] * (target - old_slots))
        self._free_slots.extend(range(target - 1, old_slots - 1, -1))
        self._allocated_slots = target

    def _clear_slots(self, slots: List[int]) -> None:
        if not slots:
            return
        slot_ids = torch.tensor(slots, dtype=torch.long, device=self.device)
        for name, _, width, _ in self.COMPONENT_SPECS:
            self._exp_avg[name].view(
                self._allocated_slots, self.block_size, width
            ).index_fill_(0, slot_ids, 0.0)
            self._exp_avg_sq[name].view(
                self._allocated_slots, self.block_size, width
            ).index_fill_(0, slot_ids, 0.0)

    def set_resident_blocks(self, block_ids: List[int]) -> None:
        next_resident = {int(block_id) for block_id in block_ids}
        previous_resident = set(self._resident_blocks)
        evicted = previous_resident - next_resident
        kept = previous_resident & next_resident
        incoming = next_resident - previous_resident

        for block_id in evicted:
            streak = int(self._resident_streaks.pop(block_id, 0))
            if streak > 0:
                self._completed_streak_total += streak
                self._completed_streak_count += 1
            slot = self._block_to_slot.pop(block_id)
            if self._slot_initialized[slot]:
                self._stats["state_evictions"] += 1
                self._logical_state_blocks -= 1
                self._logical_state_bytes -= (
                    self._slot_row_counts[slot] * self.TOTAL_WIDTH * 2 * 4
                )
            self._slot_to_block[slot] = None
            self._slot_steps[slot] = 0
            self._slot_initialized[slot] = False
            self._slot_row_counts[slot] = 0
            self._free_slots.append(slot)

        for block_id in kept:
            self._resident_streaks[block_id] = self._resident_streaks.get(block_id, 0) + 1
        for block_id in incoming:
            self._resident_streaks[block_id] = 1

        self._ensure_storage(len(next_resident))
        incoming_slots: List[int] = []
        for block_id in sorted(incoming):
            if not self._free_slots:
                self._ensure_storage(self._allocated_slots + 1)
            slot = self._free_slots.pop()
            self._block_to_slot[block_id] = slot
            self._slot_to_block[slot] = block_id
            self._slot_steps[slot] = 0
            self._slot_initialized[slot] = False
            self._slot_row_counts[slot] = 0
            incoming_slots.append(slot)
        self._clear_slots(incoming_slots)

        self._resident_blocks = next_resident
        self._update_state_stats()

    def _working_set_state_rows(
        self, manager, local_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int], List[int]]:
        layout = []
        for block_id in manager.loaded_blocks:
            block_id = int(block_id)
            block_slice = manager.block_to_gpu_slice.get(block_id)
            if block_slice is None:
                continue
            slot = self._block_to_slot.get(block_id)
            if slot is None:
                raise RuntimeError(
                    f"[GPUResidentAdam] loaded block {block_id} has no resident state slot"
                )
            layout.append((int(block_slice.start), int(block_slice.stop), block_id, slot))
        layout.sort(key=lambda item: item[0])
        if not layout:
            raise RuntimeError("[GPUResidentAdam] working set contains no resident blocks")

        starts = torch.tensor([item[0] for item in layout], dtype=torch.long, device=self.device)
        ends = torch.tensor([item[1] for item in layout], dtype=torch.long, device=self.device)
        slots = torch.tensor([item[3] for item in layout], dtype=torch.long, device=self.device)
        block_positions = torch.searchsorted(ends, local_ids, right=True)
        row_in_block = local_ids - starts[block_positions]
        state_slots = slots[block_positions]
        state_rows = state_slots * self.block_size + row_in_block

        touched_positions, touched_counts = torch.unique_consecutive(
            block_positions, return_counts=True
        )
        positions_cpu = touched_positions.detach().cpu().tolist()
        counts_cpu = touched_counts.detach().cpu().tolist()
        touched_blocks = [layout[position][2] for position in positions_cpu]
        touched_slots = [layout[position][3] for position in positions_cpu]
        return state_rows, state_slots, touched_blocks, touched_slots, counts_cpu

    def _update_touched_slot_metadata(
        self,
        manager,
        touched_blocks: List[int],
        touched_slots: List[int],
        touched_counts: List[int],
        beta1: float,
        beta2: float,
    ) -> int:
        cold_rows = 0
        bias_values = []
        denom_values = []
        for block_id, slot, count in zip(touched_blocks, touched_slots, touched_counts):
            if not self._slot_initialized[slot]:
                self._slot_initialized[slot] = True
                block_slice = manager.block_to_gpu_slice[block_id]
                row_count = int(block_slice.stop - block_slice.start)
                self._slot_row_counts[slot] = row_count
                self._logical_state_blocks += 1
                self._logical_state_bytes += row_count * self.TOTAL_WIDTH * 2 * 4
                self._stats["cold_restarts"] += 1
                self._stats["cold_restarted_rows_touched_total"] += int(count)
                cold_rows += int(count)
            self._slot_steps[slot] += 1
            step = self._slot_steps[slot]
            bias_values.append(1.0 - beta1**step)
            denom_values.append(math.sqrt(1.0 - beta2**step))

        slot_ids = torch.tensor(touched_slots, dtype=torch.long, device=self.device)
        self._bias_correction1.index_copy_(
            0, slot_ids, torch.tensor(bias_values, dtype=torch.float32, device=self.device)
        )
        self._denom_scale.index_copy_(
            0, slot_ids, torch.tensor(denom_values, dtype=torch.float32, device=self.device)
        )
        return cold_rows

    def _vectorized_component_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        local_rows: torch.Tensor,
        state_rows: torch.Tensor,
        state_slots: torch.Tensor,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        avg = exp_avg.index_select(0, state_rows)
        avg_sq = exp_avg_sq.index_select(0, state_rows)
        scaled_grad = grad / float(self.batch_size)
        avg.mul_(beta1).add_(scaled_grad, alpha=1.0 - beta1)
        avg_sq.mul_(beta2).addcmul_(scaled_grad, scaled_grad, value=1.0 - beta2)
        exp_avg.index_copy_(0, state_rows, avg)
        exp_avg_sq.index_copy_(0, state_rows, avg_sq)
        bias = self._bias_correction1.index_select(0, state_slots).unsqueeze(1)
        denom_scale = self._denom_scale.index_select(0, state_slots).unsqueeze(1)
        update = (avg / (avg_sq.sqrt() / denom_scale + eps)) * (learning_rate / bias)
        values = param.index_select(0, local_rows) - update
        param.index_copy_(0, local_rows, values)

    def _component_step(
        self,
        *,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        local_rows: torch.Tensor,
        state_rows: torch.Tensor,
        state_slots: torch.Tensor,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
        width: int,
    ) -> None:
        if triton is None:
            self._vectorized_component_step(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                local_rows,
                state_rows,
                state_slots,
                learning_rate,
                beta1,
                beta2,
                eps,
            )
            return
        n_rows = int(local_rows.numel())
        grid = (triton.cdiv(n_rows * width, 256),)
        _fused_resident_adam_kernel[grid](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            local_rows,
            state_rows,
            state_slots,
            self._bias_correction1,
            self._denom_scale,
            n_rows,
            learning_rate,
            beta1,
            beta2,
            eps,
            float(self.batch_size),
            WIDTH=width,
            BLOCK_SIZE=256,
            num_warps=4,
        )

    def step(
        self,
        iteration: int,
        gaussians,
        sparse_grad_local_ids: torch.Tensor,
        sparse_grad_components: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        if sparse_grad_local_ids is None or sparse_grad_components is None:
            return {
                "updated_blocks": 0,
                "updated_block_ids": [],
                "touched_rows": 0,
                "cold_rows": 0,
            }
        if sparse_grad_local_ids.numel() == 0:
            return {
                "updated_blocks": 0,
                "updated_block_ids": [],
                "touched_rows": 0,
                "cold_rows": 0,
            }

        manager = getattr(gaussians, "gpu_working_set_manager", None)
        if manager is None or manager.local_to_global_idx is None:
            raise RuntimeError(
                "[GPUResidentAdam] gpu_working_set_manager with local_to_global_idx is required"
            )
        local_rows = sparse_grad_local_ids.to(
            device=self.device, dtype=torch.long
        ).contiguous()
        state_rows, state_slots, touched_blocks, touched_slots, touched_counts = (
            self._working_set_state_rows(manager, local_rows)
        )

        component_lrs = self._normalize_columns_lr(
            getattr(gaussians.optimizer, "columns_lr", None)
        )
        beta1, beta2 = gaussians.optimizer.param_groups[0]["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(gaussians.optimizer.param_groups[0]["eps"])
        cold_rows = self._update_touched_slot_metadata(
            manager,
            touched_blocks,
            touched_slots,
            touched_counts,
            beta1,
            beta2,
        )

        param_views = {
            name: getattr(gaussians, attr_name).data
            for name, attr_name, _, _ in self.COMPONENT_SPECS
        }
        for name, _, width, _ in self.COMPONENT_SPECS:
            if name not in sparse_grad_components:
                raise KeyError(
                    f"[GPUResidentAdam] Missing sparse_grad_components[{name!r}] "
                    f"at iter={iteration}"
                )
            param = param_views[name]
            grad = sparse_grad_components[name]
            if not param.is_contiguous() or not grad.is_contiguous():
                raise RuntimeError(
                    f"[GPUResidentAdam] {name} parameter and gradient must be contiguous"
                )
            self._component_step(
                param=param,
                grad=grad,
                exp_avg=self._exp_avg[name],
                exp_avg_sq=self._exp_avg_sq[name],
                local_rows=local_rows,
                state_rows=state_rows,
                state_slots=state_slots,
                learning_rate=component_lrs[name],
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                width=width,
            )

        touched_rows = int(local_rows.numel())
        self._stats["optimizer_rows_touched_total"] += touched_rows
        self._update_state_stats()
        return {
            "updated_blocks": len(touched_blocks),
            "updated_block_ids": list(touched_blocks),
            "touched_rows": touched_rows,
            "cold_rows": int(cold_rows),
        }

    def get_stats(self) -> Dict[str, Any]:
        self._update_state_stats()
        return dict(self._stats)
