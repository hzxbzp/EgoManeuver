"""Tests for the data scope (egokin/data/scope.py)."""

import numpy as np
import pandas as pd

from egokin.data import scope


def idx():
    return pd.DataFrame(dict(dataset=["pai", "pai", "hdd", "nuscenes", "eposh", "bdd"],
                             night=[False, True, False, False, False, False],
                             work_zone=[False, False, True, True, True, False]))


def test_filter_index_default(monkeypatch):
    monkeypatch.delenv("VEGO_SCOPE", raising=False)
    kept = scope.filter_index(idx())
    # night pai, short work-zone clip (nuscenes) and eposh dropped; long hdd session kept for frame masking
    assert kept.dataset.tolist() == ["pai", "hdd", "bdd"]


def test_filter_index_all(monkeypatch):
    monkeypatch.setenv("VEGO_SCOPE", "all")
    assert len(scope.filter_index(idx())) == 6


def npz(tmp_path, **arrays):
    p = tmp_path / "labels.npz"
    np.savez(p, **arrays)
    return np.load(p)


def test_frame_mask_work_zone_margin(tmp_path, monkeypatch):
    monkeypatch.delenv("VEGO_SCOPE", raising=False)
    wz = np.zeros(200, bool)
    wz[100] = True
    m = scope.frame_mask(npz(tmp_path, long=np.zeros(200, np.int8), ctx_work_zone=wz))
    assert m[100 - scope.MARGIN] and m[100 + scope.MARGIN] and not m[100 - scope.MARGIN - 1] and not m[100 + scope.MARGIN + 1]
    assert m.sum() == 2 * scope.MARGIN + 1


def test_frame_mask_night_and_all(tmp_path, monkeypatch):
    z = npz(tmp_path, long=np.zeros(50, np.int8), ctx_night=np.ones(50, bool))
    monkeypatch.delenv("VEGO_SCOPE", raising=False)
    assert scope.frame_mask(z).all()
    monkeypatch.setenv("VEGO_SCOPE", "all")
    assert not scope.frame_mask(z).any()


def test_official_keep(tmp_path, monkeypatch):
    monkeypatch.delenv("VEGO_SCOPE", raising=False)
    wz = np.zeros(300, bool)
    wz[150] = True
    z = npz(tmp_path, long=np.zeros(300, np.int8), ctx_work_zone=wz, official_t=np.array([1.0, 15.0, 25.0]))
    assert scope.official_keep(z).tolist() == [True, False, True]


def test_hdd_test_sessions_exempt(tmp_path, monkeypatch):
    """HDD official test sessions keep every frame (work zones included)."""
    from types import SimpleNamespace
    monkeypatch.delenv("VEGO_SCOPE", raising=False)
    wz = np.zeros(300, bool)
    wz[150] = True
    z = npz(tmp_path, long=np.zeros(300, np.int8), ctx_work_zone=wz, official_t=np.array([1.0, 15.0, 25.0]))
    assert not scope.frame_mask(z, SimpleNamespace(dataset="hdd", split="test")).any()
    assert scope.official_keep(z, SimpleNamespace(dataset="hdd", split="test")).all()
    assert scope.frame_mask(z, SimpleNamespace(dataset="hdd", split="train")).sum() == 2 * scope.MARGIN + 1
