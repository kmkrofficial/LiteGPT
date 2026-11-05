import signal

import pytest
import torch

from litegpt.checkpoint import CheckpointError, CheckpointManager, GracefulInterrupt, config_hash
from litegpt.config import get_config
from litegpt.model import LiteGPT

CFG = get_config("litegpt-tiny")


def _batch(gen):
    x = torch.randint(0, CFG.vocab_size, (4, 32), generator=gen)
    return x, torch.roll(x, -1, 1)


def _train(model, opt, gen, steps):
    model.train()
    for _ in range(steps):
        x, y = _batch(gen)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)


def _fresh(seed):
    torch.manual_seed(seed)
    m = LiteGPT(CFG)
    return m, m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")


def test_resume_equivalence_bit_exact(tmp_path):
    N = 10
    m1, o1 = _fresh(0)
    g1 = torch.Generator().manual_seed(123)
    _train(m1, o1, g1, N)

    m2, o2 = _fresh(0)
    g2 = torch.Generator().manual_seed(123)
    _train(m2, o2, g2, N // 2)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model=m2, optimizer=o2, step=N // 2, tokens=N // 2 * 128, model_config=CFG.to_dict(),
             data_position={"gen_state": g2.get_state()})

    m3, o3 = _fresh(999)  # different init: everything must come from the checkpoint
    state = mgr.load(model=m3, optimizer=o3, expected_config=CFG.to_dict())
    assert state["step"] == N // 2
    g3 = torch.Generator()
    g3.set_state(state["data_position"]["gen_state"])
    _train(m3, o3, g3, N - N // 2)

    for (n, a), (_, b) in zip(m1.state_dict().items(), m3.state_dict().items()):
        assert torch.equal(a, b), f"{n} differs after resume"


def test_keep_last_and_milestones(tmp_path):
    m, o = _fresh(0)
    mgr = CheckpointManager(tmp_path, keep_last=2)
    for s in (1, 2, 3, 4):
        mgr.save(model=m, optimizer=o, step=s, tokens=s, model_config=CFG.to_dict())
    mgr.save(model=m, optimizer=o, step=4, tokens=4, model_config=CFG.to_dict(), milestone="stable_end")
    for s in (5, 6):
        mgr.save(model=m, optimizer=o, step=s, tokens=s, model_config=CFG.to_dict())
    assert [s for s, _ in mgr.regular_checkpoints()] == [5, 6]
    assert mgr.milestone_path("stable_end").exists()
    assert mgr.latest().name == "ckpt-00000006.pt"


def test_interrupted_save_leaves_no_valid_partial_checkpoint(tmp_path):
    m, o = _fresh(0)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model=m, optimizer=o, step=1, tokens=1, model_config=CFG.to_dict())
    (tmp_path / "ckpt-00000002.pt.tmp").write_bytes(b"garbage from a crash")
    mgr2 = CheckpointManager(tmp_path)  # a restart cleans leftovers and ignores them
    assert not list(tmp_path.glob("*.tmp")) and mgr2.latest().name == "ckpt-00000001.pt"


def test_config_and_spec_guards(tmp_path):
    m, o = _fresh(0)
    mgr = CheckpointManager(tmp_path, spec_sha256="a" * 64)
    mgr.save(model=m, optimizer=o, step=1, tokens=1, model_config=CFG.to_dict())
    with pytest.raises(CheckpointError, match="config differs"):
        mgr.load(model=m, expected_config=get_config("litegpt-tiny", n_layer=3).to_dict())
    other = CheckpointManager(tmp_path, spec_sha256="b" * 64)
    with pytest.raises(CheckpointError, match="spec sha256"):
        other.load(model=m)
    other.load(model=m, allow_spec_mismatch=True)  # explicit, recorded deviation
    with pytest.raises(CheckpointError, match="no checkpoint"):
        CheckpointManager(tmp_path / "empty").load()


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_rng_state_restored(tmp_path):
    import random
    import numpy as np
    m, o = _fresh(0)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model=m, optimizer=o, step=1, tokens=1, model_config=CFG.to_dict())
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    random.random(); np.random.rand(); torch.rand(3)  # advance everything
    mgr.load(model=m, optimizer=o)
    got = (random.random(), float(np.random.rand()), torch.rand(3))
    assert got[0] == expected[0] and got[1] == expected[1] and torch.equal(got[2], expected[2])


def test_graceful_interrupt_sets_flag_and_restores_handler():
    before = signal.getsignal(signal.SIGINT)
    with GracefulInterrupt() as gi:
        assert not gi.requested
        signal.raise_signal(signal.SIGINT)
        assert gi.requested
        with pytest.raises(KeyboardInterrupt):  # second Ctrl+C force-quits
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) == before
