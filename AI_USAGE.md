# AI Usage — generate_data.py

## How I used AI

I used Claude to write and iteratively improve `generate_data.py` across two sessions. The workflow was: prompt → review output → correct mistakes → prompt again.

---

## Prompts

### Prompt 1 — Initial generation

> Create a script `generate_data.py` that generates a CSV with 1–3 million rows using NumPy (no Python loops). Schema: event_id, user_id, ts, event_type, amount, country, device, session_id. Rules: purchase 2–5%, refund 0.1–0.5% (only if a purchase exists), lognormal amounts. Inject 0.5% dirty data for each of: invalid timestamps, null countries, invalid event_type ("???"), duplicate event_id. Explain how you vectorise event_type generation, enforce the refund→purchase dependency, inject dirty data without loops, and keep memory under control for 3M rows.

**What the AI produced:** A working scaffold with `np.searchsorted` for event types, `pd.Categorical` for low-cardinality columns, and a single `rng.random((4, n))` draw for dirty injection. It also wrote an invariant checker at the end that it ran itself. Reading the code and checking the output on the csv file, I noticed some bugs.

---

### Prompt 2 — Bug fixes (AI self-review)

After inspecting the AI code and the generated .csv file, I identified and fixed three problems:

**Bug 1 — Refund amounts were random, not mirrored from the linked purchase.**

```python
# BEFORE (wrong): independent lognormal for each refund
raw_amounts = rng.lognormal(...)
amounts = raw_amounts.astype(np.float32)
# refunds got a random negative, unrelated to their purchase

# AFTER (fixed): exact mirror via fancy indexing
amounts[refund_indices] = -amounts[linked_purchase_indices]
```

**Bug 2 — Dirty injection could corrupt purchase rows that refunds depended on.**

```python
# BEFORE: any row could get event_type = "???"
etype_mask = r[2] < DIRTY_RATE

# AFTER: linked purchases are shielded
etype_mask = (r[2] < DIRTY_RATE) & ~protected_purchase_mask
```

**Bug 3 — Duplicate injection was copying from already-mutated rows and sometimes copying a row onto itself.**

```python
# BEFORE
event_ids[dup_indices] = event_ids[src]   # reads mutated values, src may equal dup_indices

# AFTER: snapshot first, shift self-copies
ids_snapshot = event_ids.copy()
src[src == dup_indices] = (src[src == dup_indices] + 1) % n
event_ids[dup_indices] = ids_snapshot[src]
```

---

### Prompt 3 — Architecture review after I modified the file

> I noticed that it first links refunds to purchases using session_id, then only inside `generate()` it links user_id and fixes the amounts. Is it correct? Does it align with good code architecture? How would you change it?

**What the AI said:** The linking logic was split across two places with no clear boundary — `link_refunds_to_purchases()` handled `session_id`, while `user_id`, `timestamp`, and `amount` were scattered through `generate()`. It proposed replacing the raw 3-tuple return with a named `RefundLinks` dataclass and splitting into `_build_refund_links()` (plan) and `_apply_refund_links()` (apply all four constraints in one place).

---

### Prompt 4 — Apply the refactor

> Where and what should I change in the code?

The AI made 8 surgical changes: added the `RefundLinks` dataclass, replaced `link_refunds_to_purchases` with `_build_refund_links` + `_apply_refund_links`, simplified `generate()` to call both, and updated `report()` and `main()` to use `links` instead of two raw arrays. It ran 21 tests to verify correctness.

---

## Where I corrected the AI

### Correction 1 — `sys.argv` parsed at module level

The AI put `N_ROWS` and `OUTPUT_PATH` at the top of the file, outside any function. This means they execute on import, which breaks any test that does `from generate_data import generate`.

```python
# AI wrote (wrong — runs on import)
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000

# I moved it into main() where it belongs
def main():
    N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
```

### Correction 2 — Used global `N_ROWS` instead of parameter `n` inside `generate()`

```python
# AI wrote (wrong — ignores the function's own parameter)
protected_purchase_mask = np.zeros(N_ROWS, dtype=bool)

# I fixed it
protected_purchase_mask = np.zeros(n, dtype=bool)
```

This only worked in production because `main()` happened to pass `N_ROWS` as `n`. Any test calling `generate(100, rng)` would silently create a 3M-element mask.

### Correction 3 — Stale comment after changing a constant

```python
# AI wrote (rate is 0.005 = 0.5%, comment says 0.3%)
REFUND_RATE = 0.005   # 0.3 % → satisfies "0.1–0.5 %"

# I fixed the comment
REFUND_RATE = 0.005   # 0.5 % → satisfies "0.1–0.5 %"
```

### Correction 4 — Missing behavioural contracts for refunds

The AI's version never enforced that a refund belongs to the same user as its purchase, or that its timestamp comes after the purchase. I added both:

```python
user_ids[refund_indices] = user_ids[linked_purchase_indices]

offsets = rng.integers(3600, 30*86400, size=len(refund_indices), dtype=np.int64)
timestamps[refund_indices] = np.minimum(
    timestamps[linked_purchase_indices] + offsets, TS_END
)
```

---

## How I evaluated the output

**Running the invariant checker.** The AI wrote a `report()` function that checks `event_type == "purchase"` on all linked rows and `refund.amount == -purchase.amount` on all pairs. I ran this after every change and treated any `[FAIL]` as a blocker.

**Checking the dirty-data counts.** The output prints how many rows were corrupted per type. I verified each was close to 0.5% of 3M rows (~15,000), not zero or suspiciously low.

**Checking the return signature.** After seeing the AI return a 3-tuple `(session_ids, refund_idx, purchase_idx)` where the first element was immediately re-assigned by the caller, I noticed something was off. That observation is what led to the architecture prompt.

**Importing in tests.** The `sys.argv` bug revealed itself when I tried to import `generate` in a test — it crashed trying to parse pytest's arguments. That's how Correction 1 was found.