---
license: bsd-3-clause
tags:
  - isaac-lab
  - reinforcement-learning
  - robotics
  - peg-in-hole
  - contact-rich-manipulation
  - tacinsert
---

# TacInsert Checkpoints

This directory is reserved for optional pretrained checkpoints used for quick simulation smoke tests. Checkpoint files are large binary artifacts and are intentionally ignored by git. For anonymous review, checkpoint binaries should be distributed through the paper's anonymous supplementary material or an anonymous model-hosting repository.

If an anonymous checkpoint host is provided, download them into this directory before running the smoke-test commands below.

```powershell
hf download <ANONYMOUS_CHECKPOINT_REPO> TacInsert-LHole-III-Direct-v0.pth `
  --repo-type model `
  --local-dir source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints

hf download <ANONYMOUS_CHECKPOINT_REPO> TacInsert-Manipulation-Square-SingleHole-Direct-v0.pth `
  --repo-type model `
  --local-dir source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints
```

## Checkpoint Metadata

The following checkpoints were used to validate the current TacInsert simulation code:

| File | Task | Observation dim | Notes | SHA256 |
| --- | --- | ---: | --- | --- |
| `TacInsert-LHole-III-Direct-v0.pth` | `TacInsert-LHole-III-Direct-v0` | 23 | L-hole Tol. III with contact-force observation and `fixed_sigma: True` | `FC158B0D72AFDAB0D860CEB379F8E3EC2B0B693D49C9B24629608A5BEA4B58AC` |
| `TacInsert-Manipulation-Square-SingleHole-Direct-v0.pth` | `TacInsert-Manipulation-Square-SingleHole-Direct-v0` | 26 | ManipulationNet-style square/rectangle single-hole sampler policy with contact-force observation, tolerance one-hot, and `fixed_sigma: False` | `0301CD4C57848EDA73FAE2D9F019D879C6D8AA1F29315CCA8A12BDC5B36CA10E` |

## Smoke-Test Commands

Run from the repository root after installing the TacInsert extension.

### L-Hole Tol. III

```powershell
python scripts/rl_games/play.py `
  --task TacInsert-LHole-III-Direct-v0 `
  --num_envs 128 `
  --headless `
  --checkpoint source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints/TacInsert-LHole-III-Direct-v0.pth
```

Expected behavior: the terminal prints one summary per completed episode. In local validation with 128 environments, three episodes reached approximately 94-98 percent success.

### ManipulationNet-Style Square Single-Hole

```powershell
python scripts/rl_games/play.py `
  --task TacInsert-Manipulation-Square-SingleHole-Direct-v0 `
  --num_envs 128 `
  --headless `
  --checkpoint source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints/TacInsert-Manipulation-Square-SingleHole-Direct-v0.pth
```

Expected behavior with the default fixed Tol. I sampler: the terminal prints one summary per completed episode. In local validation with 128 environments, three episodes reached approximately 98-99 percent success.

To evaluate another fixed tolerance, edit `multi_hole_sample_weights` in `TacInsertManipulationSquareSingleHole` before running playback. For example, Tol. IV evaluation uses:

```python
multi_hole_sample_weights = [0.0, 0.0, 0.0, 1.0]
```

In local Tol. IV validation with 128 environments, three episodes reached approximately 61-63 percent success.

## Contact-Force Logging Check

For tasks with `contact_force["log_contact_force"] = True`, CSV logging is enabled only in evaluation mode with `--num_envs 1`:

```powershell
python scripts/rl_games/play.py `
  --task TacInsert-Manipulation-Square-SingleHole-Direct-v0 `
  --num_envs 1 `
  --headless `
  --checkpoint source/TacInsert/TacInsert/tasks/direct/tacinsert/checkpoints/TacInsert-Manipulation-Square-SingleHole-Direct-v0.pth
```

CSV files are written to `contact_force_logs/`, which is ignored by git.
