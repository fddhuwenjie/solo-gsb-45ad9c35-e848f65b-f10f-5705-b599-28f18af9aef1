import os
import tempfile
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    os.environ["DB_PATH"] = path
    from app import db
    db.reset()
    from app.main import app
    with TestClient(app) as c:
        yield c
    db.reset()
    Path(path).unlink(missing_ok=True)


def add_cell(client, cell_id, chemistry="LFP", capacity=100.0, resistance=2.0, version="T1"):
    r = client.post("/cells", json={
        "cell_id": cell_id, "chemistry": chemistry, "capacity": capacity,
        "resistance": resistance, "test_version": version,
    })
    assert r.status_code == 201, r.text
    return r


def seed_six(client):
    # 6 节 LFP：容量两两差 <= 5，内阻差 <= 0.5
    data = [
        ("C001", 100.0, 2.0), ("C002", 99.0, 2.1), ("C003", 98.0, 2.2),
        ("C004", 97.0, 2.3), ("C005", 96.0, 2.4), ("C006", 95.5, 2.5),
    ]
    for cid, cap, res in data:
        add_cell(client, cid, capacity=cap, resistance=res)


def isolate(client, cell_id, reason="检出安全隐患"):
    return client.post(f"/cells/{cell_id}/isolate", json={"reason": reason})


PRESELECT4 = {
    "series_count": 4, "min_capacity": 90.0,
    "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
}


def test_register_limit_12(client):
    for i in range(12):
        add_cell(client, f"C{i:03d}")
    r = client.post("/cells", json={
        "cell_id": "C999", "chemistry": "LFP", "capacity": 1.0,
        "resistance": 1.0, "test_version": "T1",
    })
    assert r.status_code == 409
    assert "12" in r.json()["detail"]


def test_duplicate_cell_rejected(client):
    add_cell(client, "C001")
    r = client.post("/cells", json={
        "cell_id": "C001", "chemistry": "LFP", "capacity": 1.0,
        "resistance": 1.0, "test_version": "T1",
    })
    assert r.status_code == 409


def test_preselect_maximizes_min_capacity(client):
    seed_six(client)
    r = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    body = r.json()
    assert body["feasible"] is True
    # 最小容量最大的 4 节组合是 C001..C004（min=97）
    assert body["cell_ids"] == ["C001", "C002", "C003", "C004"]
    assert body["limited_capacity"] == 97.0
    assert all(c["test_version"] == "T1" for c in body["cells"])


def test_preselect_tie_break_lexicographic(client):
    # 4 节中任取 3 节最小容量都是 99：同值时按 ID 排序序列字典序取最小
    add_cell(client, "B001", capacity=100.0, resistance=2.0)
    add_cell(client, "B002", capacity=99.0, resistance=2.0)
    add_cell(client, "A001", capacity=100.0, resistance=2.0)
    add_cell(client, "A002", capacity=99.0, resistance=2.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 1.0, "max_resistance_diff": 0.5,
    })
    body = r.json()
    assert body["feasible"] is True
    assert body["limited_capacity"] == 99.0
    assert body["cell_ids"] == ["A001", "A002", "B001"]


def test_preselect_chemistry_isolated(client):
    seed_six(client)
    add_cell(client, "N001", chemistry="NCM", capacity=100.0, resistance=2.0)
    add_cell(client, "N002", chemistry="NCM", capacity=99.0, resistance=2.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5, "chemistry": "NCM",
    })
    body = r.json()
    assert body["feasible"] is False
    assert "不足 3 节" in body["reason"]


def test_preselect_infeasible_reasons(client):
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert r.json()["feasible"] is False
    assert "没有可用的未占用电芯" in r.json()["reason"]

    # 容量全部不达标
    add_cell(client, "C001", capacity=50.0)
    add_cell(client, "C002", capacity=50.0)
    add_cell(client, "C003", capacity=50.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert "最低单芯容量" in r.json()["reason"]

    # 数量够但差值约束不满足
    add_cell(client, "C004", capacity=200.0, resistance=9.0)
    add_cell(client, "C005", capacity=100.0, resistance=2.0)
    add_cell(client, "C006", capacity=100.0, resistance=2.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert "容量差上限" in r.json()["reason"] or "内阻差上限" in r.json()["reason"]


def test_confirm_success_and_snapshot(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 201, r.text
    group = r.json()
    assert group["group_id"] == 1
    assert group["limited_capacity"] == 97.0
    assert group["series_count"] == 4

    # 电芯被占用
    cells = client.get("/cells").json()["cells"]
    used = {c["cell_id"]: c for c in cells if c["occupied"]}
    assert set(used) == set(pre["cell_ids"])
    assert all(c["group_id"] == 1 for c in used.values())

    # 快照可按组编号查询
    snap = client.get("/groups/1").json()
    assert snap["cells"] == group["cells"]


def test_confirm_stale_version_rejected_atomically(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    # 预选后复测其中一节 -> 候选过期
    client.put("/cells/C002/test", json={
        "capacity": 99.0, "resistance": 2.1, "test_version": "T2"})
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 409
    assert "C002" in r.json()["detail"]
    # 整组拒绝：无任何部分占用，无组产生
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])
    assert client.get("/groups").json()["count"] == 0


def test_confirm_occupied_by_other_group_rejected(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    assert client.post("/groups/confirm", json=payload).status_code == 201
    # 别组已抢先占用 -> 再次确认同一候选被拒
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 409
    assert "占用" in r.json()["detail"]
    assert client.get("/groups").json()["count"] == 1


def test_revision_after_confirm_does_not_change_snapshot(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    group = client.post("/groups/confirm", json=payload).json()
    # 确认后复测已入组电芯：允许修订档案，但快照不变
    client.put("/cells/C001/test", json={
        "capacity": 88.0, "resistance": 3.0, "test_version": "T9"})
    snap = client.get(f"/groups/{group['group_id']}").json()
    c001 = next(c for c in snap["cells"] if c["cell_id"] == "C001")
    assert c001["capacity"] == 100.0
    assert c001["test_version"] == "T1"
    assert snap["limited_capacity"] == 97.0


def test_confirm_mixed_chemistry_rejected(client):
    add_cell(client, "C001", chemistry="LFP")
    add_cell(client, "C002", chemistry="LFP")
    add_cell(client, "C003", chemistry="NCM")
    r = client.post("/groups/confirm", json={"cells": [
        {"cell_id": "C001", "test_version": "T1"},
        {"cell_id": "C002", "test_version": "T1"},
        {"cell_id": "C003", "test_version": "T1"},
    ]})
    assert r.status_code == 409
    assert "化学体系" in r.json()["detail"]
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])


def test_series_count_validation(client):
    r = client.post("/preselect", json={
        "series_count": 2, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert r.status_code == 422
    r = client.post("/preselect", json={
        "series_count": 7, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert r.status_code == 422


def _begin_uncommitted_group(conn):
    """在当前线程的连接上模拟一个尚未提交的确认事务。"""
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO groups (chemistry, series_count, limited_capacity, created_at)"
        " VALUES ('LFP', 4, 97.0, '2026-10-05T00:00:00+00:00')")
    gid = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    for cid, cap, res in [("C001", 100.0, 2.0), ("C002", 99.0, 2.1),
                          ("C003", 98.0, 2.2), ("C004", 97.0, 2.3)]:
        conn.execute(
            "INSERT INTO group_cells (group_id, cell_id, capacity, resistance, test_version)"
            " VALUES (?, ?, ?, ?, 'T1')", (gid, cid, cap, res))
        conn.execute(
            "UPDATE cells SET occupied = 1, group_id = ? WHERE cell_id = ?", (gid, cid))
    return gid


def test_uncommitted_group_invisible_to_other_requests(client):
    """确认事务未提交时，其他请求看不到组记录、快照与临时占用；提交后整组一次性可见。"""
    seed_six(client)
    from app import db
    gid = _begin_uncommitted_group(db.get_conn())  # 本线程连接持有未提交事务

    # 未提交：组列表 / 按编号查询 / 电芯占用 / 预选都看不到事务的中间状态
    assert client.get("/groups").json()["count"] == 0
    assert client.get(f"/groups/{gid}").status_code == 404
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    assert pre["feasible"] is True
    assert pre["cell_ids"] == ["C001", "C002", "C003", "C004"]

    # 提交后：整组（组记录 + 快照 + 占用）一次性可见
    db.get_conn().commit()
    assert client.get("/groups").json()["count"] == 1
    snap = client.get(f"/groups/{gid}").json()
    assert [c["cell_id"] for c in snap["cells"]] == ["C001", "C002", "C003", "C004"]
    used = {c["cell_id"] for c in client.get("/cells").json()["cells"] if c["occupied"]}
    assert used == {"C001", "C002", "C003", "C004"}


def test_rolled_back_group_leaves_no_trace(client):
    """确认回滚后，组记录、快照与占用对外毫无痕迹。"""
    seed_six(client)
    from app import db
    conn = db.get_conn()
    gid = _begin_uncommitted_group(conn)
    conn.rollback()
    assert client.get("/groups").json()["count"] == 0
    assert client.get(f"/groups/{gid}").status_code == 404
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])


def test_concurrent_confirm_and_reads_do_not_interfere(client):
    """并发确认同一候选：只有一个成功；并发查询只看到完整快照，互不干扰。"""
    seed_six(client)
    from app.main import app
    payload = {"cells": [{"cell_id": f"C00{i}", "test_version": "T1"} for i in range(1, 5)]}
    statuses = []
    broken = []

    def worker():
        with TestClient(app) as c:
            r = c.post("/groups/confirm", json=payload)
            statuses.append(r.status_code)
            for _ in range(10):
                # 组列表里出现的任何组都必须带完整快照
                for g in c.get("/groups").json()["groups"]:
                    if len(g["cells"]) != g["series_count"]:
                        broken.append(g)
                # 已提交的占用必须整组可见，不允许只看到部分占用
                members = [cell["cell_id"] for cell in c.get("/cells").json()["cells"]
                           if cell["occupied"]]
                if members and len(members) != 4:
                    broken.append(members)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses.count(201) == 1
    assert statuses.count(409) == 3
    assert not broken
    assert client.get("/groups").json()["count"] == 1


# ---------- 安全隔离 ----------

def test_isolate_cell_records_isolation(client):
    add_cell(client, "C001")
    r = isolate(client, "C001", "外壳鼓包")
    assert r.status_code == 201
    body = r.json()
    assert body["cell_id"] == "C001"
    assert body["reason"] == "外壳鼓包"
    assert body["isolated_at"]
    # 电芯档案带隔离标记（新增字段，原字段保持不变）
    cell = client.get("/cells/C001").json()
    assert cell["isolated"] is True
    assert cell["occupied"] is False
    # 隔离列表可追溯
    lst = client.get("/isolations").json()
    assert lst["count"] == 1
    assert lst["isolations"][0]["cell_id"] == "C001"
    assert lst["isolations"][0]["reason"] == "外壳鼓包"


def test_isolate_missing_cell_404(client):
    r = isolate(client, "NOPE")
    assert r.status_code == 404
    assert "NOPE" in r.json()["detail"]


def test_repeat_isolation_returns_original_record(client):
    add_cell(client, "C001")
    first = isolate(client, "C001", "首次原因")
    assert first.status_code == 201
    # 重复隔离（即使原因不同）返回首次隔离的原记录，不新增、不改写
    again = isolate(client, "C001", "不同的新原因")
    assert again.status_code == 200
    assert again.json() == first.json()
    assert client.get("/isolations").json()["count"] == 1


def test_isolation_irreversible_no_delete_endpoint(client):
    add_cell(client, "C001")
    isolate(client, "C001")
    assert client.delete("/cells/C001/isolate").status_code == 405
    assert client.get("/isolations").json()["count"] == 1


def test_retest_cannot_clear_isolation(client):
    add_cell(client, "C001")
    isolate(client, "C001", "热失控风险")
    # 复测只更新检测档案，隔离记录保持不变
    r = client.put("/cells/C001/test", json={
        "capacity": 98.0, "resistance": 2.0, "test_version": "T2"})
    assert r.status_code == 200
    assert r.json()["isolated"] is True
    lst = client.get("/isolations").json()
    assert lst["count"] == 1
    assert lst["isolations"][0]["reason"] == "热失控风险"


def test_preselect_excludes_isolated_cells(client):
    seed_six(client)
    isolate(client, "C001", "待复检")
    body = client.post("/preselect", json=PRESELECT4).json()
    assert body["feasible"] is True
    # 最优组合跳过 C001，顺延为 C002~C005
    assert body["cell_ids"] == ["C002", "C003", "C004", "C005"]
    assert body["limited_capacity"] == 96.0
    # 隔离到剩余不足成组时，预选判不可行
    for cid in ("C002", "C003", "C004"):
        isolate(client, cid, "待复检")
    body = client.post("/preselect", json=PRESELECT4).json()
    assert body["feasible"] is False
    assert "不足 4 节" in body["reason"]


def test_confirm_rechecks_isolation_and_rejects_atomically(client):
    seed_six(client)
    pre = client.post("/preselect", json=PRESELECT4).json()
    # 预选后、确认前隔离候选中的一节 -> 确认事务重查时命中，整组拒绝
    isolate(client, "C002", "复检不合格")
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 409
    assert "C002" in r.json()["detail"]
    # 整组拒绝：不产生组，也不留任何部分占用
    assert client.get("/groups").json()["count"] == 0
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])


def _confirm(client, ids):
    return client.post("/groups/confirm", json={
        "cells": [{"cell_id": i, "test_version": "T1"} for i in ids]})


def test_isolating_grouped_cell_blocks_group_snapshot_retained(client):
    seed_six(client)
    assert _confirm(client, ["C001", "C002", "C003"]).status_code == 201
    assert _confirm(client, ["C004", "C005", "C006"]).status_code == 201

    # 隔离已入组（组 1）的电芯
    assert isolate(client, "C002", "抽检发现漏液").status_code == 201

    groups = {g["group_id"]: g for g in client.get("/groups").json()["groups"]}
    g1, g2 = groups[1], groups[2]
    # 组 1 已阻断：显示涉及的电芯 ID 与原因
    assert g1["blocked"] is True
    assert [b["cell_id"] for b in g1["blocked_cells"]] == ["C002"]
    assert g1["blocked_cells"][0]["reason"] == "抽检发现漏液"
    assert g1["blocked_cells"][0]["isolated_at"]
    # 原检测快照保留不变
    assert [c["cell_id"] for c in g1["cells"]] == ["C001", "C002", "C003"]
    c002 = next(c for c in g1["cells"] if c["cell_id"] == "C002")
    assert c002["capacity"] == 99.0 and c002["test_version"] == "T1"
    assert g1["limited_capacity"] == 98.0
    # 组占用保留
    cells = {c["cell_id"]: c for c in client.get("/cells").json()["cells"]}
    assert cells["C002"]["occupied"] is True
    assert cells["C002"]["group_id"] == 1
    # 未受影响的组 2 保持可放行
    assert g2["blocked"] is False
    assert g2["blocked_cells"] == []

    # 按组编号查询同样显示阻断信息与完整快照
    one = client.get("/groups/1").json()
    assert one["blocked"] is True
    assert [b["cell_id"] for b in one["blocked_cells"]] == ["C002"]
    assert one["cells"] == g1["cells"]
    assert client.get("/groups/2").json()["blocked"] is False


def test_uncommitted_isolation_invisible_to_other_requests(client):
    """隔离事务未提交时，其他请求看不到；提交后预选立即排除该电芯。"""
    seed_six(client)
    from app import db
    conn = db.get_conn()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO isolations (cell_id, reason, created_at)"
        " VALUES ('C001', '排查中', '2026-10-06T00:00:00+00:00')")
    # 未提交：预选、隔离列表、电芯档案都看不到这条隔离
    pre = client.post("/preselect", json=PRESELECT4).json()
    assert pre["cell_ids"] == ["C001", "C002", "C003", "C004"]
    assert client.get("/isolations").json()["count"] == 0
    assert client.get("/cells/C001").json()["isolated"] is False
    # 提交后：一次性可见，预选排除 C001
    conn.commit()
    pre = client.post("/preselect", json=PRESELECT4).json()
    assert pre["cell_ids"] == ["C002", "C003", "C004", "C005"]
    assert client.get("/isolations").json()["count"] == 1
    assert client.get("/cells/C001").json()["isolated"] is True


def test_isolation_and_confirm_race_commit_order_decides(client):
    """隔离与确认并发：以提交先后决定是拒绝候选还是阻断新组，查询不见半成品。"""
    seed_six(client)
    from app.main import app
    payload = {"cells": [{"cell_id": f"C00{i}", "test_version": "T1"} for i in range(1, 5)]}
    results = {}
    broken = []

    def do_confirm():
        with TestClient(app) as c:
            results["confirm"] = c.post("/groups/confirm", json=payload).status_code

    def do_isolate():
        with TestClient(app) as c:
            results["isolate"] = c.post(
                "/cells/C002/isolate", json={"reason": "并发隔离"}).status_code
            for _ in range(10):
                # 任何可见的组都必须带完整快照，阻断标记与隔离记录一致
                for g in c.get("/groups").json()["groups"]:
                    if len(g["cells"]) != g["series_count"]:
                        broken.append(g)
                    if g["blocked"] and not g["blocked_cells"]:
                        broken.append(g)

    threads = [threading.Thread(target=do_confirm), threading.Thread(target=do_isolate)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not broken
    assert results["isolate"] == 201  # 隔离本身总是成功
    groups = client.get("/groups").json()["groups"]
    if results["confirm"] == 201:
        # 确认先提交：新组产生，随后的隔离使其阻断；快照完整、占用保留
        assert len(groups) == 1
        g = groups[0]
        assert g["blocked"] is True
        assert [b["cell_id"] for b in g["blocked_cells"]] == ["C002"]
        assert len(g["cells"]) == 4
        used = {c["cell_id"] for c in client.get("/cells").json()["cells"] if c["occupied"]}
        assert used == {"C001", "C002", "C003", "C004"}
    else:
        # 隔离先提交：确认整组拒绝，不产生组或部分占用
        assert results["confirm"] == 409
        assert groups == []
        assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])


# ---------- 单次查询的快照一致性（可控读写交错） ----------

def _install_row_hook(monkeypatch, target_cell):
    """在 _row_to_cell 读取 target_cell 的隔离标志前插入一个可控阻塞点。

    阻塞点位于 cells 行已取出、isolations 查询尚未发出之间：
    修复前两条 SELECT 各取最新快照，窗口内的提交会被后半段读到而拼出混合状态；
    修复后整个响应包在同一只读事务中，快照在 cells 查询时即固定，窗口内提交不可见。
    返回 (started, proceed)：阻塞开始时 set(started)，等待 set(proceed) 后继续。
    """
    from app import main
    started = threading.Event()
    proceed = threading.Event()
    orig = main._row_to_cell

    def patched(conn, row):
        if row["cell_id"] == target_cell and not patched.fired:
            patched.fired = True
            started.set()
            assert proceed.wait(10), "交错窗口等待超时"
        return orig(conn, row)
    patched.fired = False
    monkeypatch.setattr(main, "_row_to_cell", patched)
    return started, proceed


def _assert_cell_coherent(cell):
    """电芯档案三态不变量：占用/组编号/隔离不得跨快照拼接。"""
    if cell["occupied"]:
        assert cell["group_id"] is not None, f"已占用却无组编号：{cell}"
    else:
        assert cell["group_id"] is None, f"未占用却带组编号：{cell}"
    # 从未提交过的状态：未占用（确认前）不可能同时已隔离（隔离在确认之后）
    assert not (not cell["occupied"] and cell["isolated"]), (
        f"拼出不存在的状态 occupied=false, isolated=true：{cell}")


def test_list_cells_reads_one_snapshot_across_confirm_then_isolate(client, monkeypatch):
    """GET /cells 交错窗口：先读确认前的 cells，窗口内确认+隔离先后提交。

    响应必须整体停留在确认前快照；修复前会返回 C001 occupied=false, isolated=true。
    """
    seed_six(client)
    started, proceed = _install_row_hook(monkeypatch, "C001")
    from app.main import app

    result = {}

    def do_list():
        with TestClient(app) as c:
            r = c.get("/cells")
            result["status"] = r.status_code
            result["cells"] = {c["cell_id"]: c for c in r.json()["cells"]}

    t = threading.Thread(target=do_list)
    t.start()
    assert started.wait(10)

    # cells 已读、C001 隔离标志未读的窗口内：确认含 C001 的组并提交，再隔离 C001 并提交
    assert _confirm(client, ["C001", "C002", "C003", "C004"]).status_code == 201
    assert isolate(client, "C001", "交错窗口隔离").status_code == 201
    proceed.set()
    t.join(10)
    assert not t.is_alive()

    assert result["status"] == 200
    snapshot = result["cells"]
    # 整体为确认前快照：4 节候选全部未占用、无组、未隔离
    for cid in ("C001", "C002", "C003", "C004"):
        c = snapshot[cid]
        _assert_cell_coherent(c)
        assert (c["occupied"], c["group_id"], c["isolated"]) == (False, None, False)

    # 窗口关闭后再查：完整进入隔离后状态（已占用 + 有组 + 已隔离，整组一致）
    after = {c["cell_id"]: c for c in client.get("/cells").json()["cells"]}
    _assert_cell_coherent(after["C001"])
    assert (after["C001"]["occupied"], after["C001"]["group_id"],
            after["C001"]["isolated"]) == (True, 1, True)
    for cid in ("C002", "C003", "C004"):
        assert (after[cid]["occupied"], after[cid]["group_id"],
                after[cid]["isolated"]) == (True, 1, False)


def test_get_cell_reads_one_snapshot_across_confirm_then_isolate(client, monkeypatch):
    """GET /cells/C001 同一交错窗口：占用、组编号与隔离标志须同属确认前快照。"""
    seed_six(client)
    started, proceed = _install_row_hook(monkeypatch, "C001")
    from app.main import app

    result = {}

    def do_get():
        with TestClient(app) as c:
            r = c.get("/cells/C001")
            result["status"] = r.status_code
            result["cell"] = r.json()

    t = threading.Thread(target=do_get)
    t.start()
    assert started.wait(10)

    assert _confirm(client, ["C001", "C002", "C003", "C004"]).status_code == 201
    assert isolate(client, "C001", "交错窗口隔离").status_code == 201
    proceed.set()
    t.join(10)
    assert not t.is_alive()

    assert result["status"] == 200
    _assert_cell_coherent(result["cell"])
    assert (result["cell"]["occupied"], result["cell"]["group_id"],
            result["cell"]["isolated"]) == (False, None, False)

    after = client.get("/cells/C001").json()
    _assert_cell_coherent(after)
    assert (after["occupied"], after["group_id"], after["isolated"]) == (True, 1, True)


def test_retest_response_reads_one_snapshot_in_race_window(client, monkeypatch):
    """复测入口的电芯响应复用同一口径：回读窗口内确认+隔离提交，不拼混合状态。"""
    seed_six(client)
    started, proceed = _install_row_hook(monkeypatch, "C001")
    from app.main import app

    result = {}

    def do_retest():
        with TestClient(app) as c:
            r = c.put("/cells/C001/test", json={
                "capacity": 100.0, "resistance": 2.0, "test_version": "T2"})
            result["status"] = r.status_code
            result["cell"] = r.json()

    t = threading.Thread(target=do_retest)
    t.start()
    assert started.wait(10)

    # 复测已提交（T2）、响应隔离标志未读：用新版本确认含 C001 的组，再隔离 C001
    r = client.post("/groups/confirm", json={"cells": [
        {"cell_id": "C001", "test_version": "T2"},
        {"cell_id": "C002", "test_version": "T1"},
        {"cell_id": "C003", "test_version": "T1"},
        {"cell_id": "C004", "test_version": "T1"}]})
    assert r.status_code == 201, r.text
    assert isolate(client, "C001", "复测回读窗口隔离").status_code == 201
    proceed.set()
    t.join(10)
    assert not t.is_alive()

    assert result["status"] == 200
    cell = result["cell"]
    assert cell["test_version"] == "T2"
    _assert_cell_coherent(cell)
    # 响应固定在复测提交后、确认前的快照：未占用、无组、未隔离
    assert (cell["occupied"], cell["group_id"], cell["isolated"]) == (False, None, False)


def test_create_response_reads_one_snapshot_in_race_window(client, monkeypatch):
    """录入入口的电芯响应复用同一口径：回读窗口内确认+隔离提交，不拼混合状态。"""
    seed_six(client)  # 6 节在库，仍可继续录入
    started, proceed = _install_row_hook(monkeypatch, "C007")
    from app.main import app

    result = {}

    def do_create():
        with TestClient(app) as c:
            r = c.post("/cells", json={
                "cell_id": "C007", "chemistry": "LFP", "capacity": 96.0,
                "resistance": 2.45, "test_version": "T1"})
            result["status"] = r.status_code
            result["cell"] = r.json()

    t = threading.Thread(target=do_create)
    t.start()
    assert started.wait(10)

    # C007 插入已提交、响应隔离标志未读：确认含 C007 的 3 节组（约束满足），再隔离
    assert _confirm(client, ["C007", "C005", "C006"]).status_code == 201
    assert isolate(client, "C007", "录入回读窗口隔离").status_code == 201
    proceed.set()
    t.join(10)
    assert not t.is_alive()

    assert result["status"] == 201
    cell = result["cell"]
    _assert_cell_coherent(cell)
    assert (cell["occupied"], cell["group_id"], cell["isolated"]) == (False, None, False)

    after = client.get("/cells/C007").json()
    _assert_cell_coherent(after)
    assert (after["occupied"], after["group_id"], after["isolated"]) == (True, 1, True)


def test_concurrent_cell_reads_never_show_mixed_state(client):
    """随机并发：确认与隔离接连提交期间轮询电芯档案，任何响应都不得出现混合状态。"""
    seed_six(client)
    from app.main import app
    broken = []

    def reader():
        with TestClient(app) as c:
            for _ in range(50):
                body = c.get("/cells").json()
                for cell in body["cells"]:
                    try:
                        _assert_cell_coherent(cell)
                    except AssertionError:
                        broken.append(cell)
                one = c.get("/cells/C001")
                if one.status_code == 200:
                    try:
                        _assert_cell_coherent(one.json())
                    except AssertionError:
                        broken.append(one.json())

    rthread = threading.Thread(target=reader)
    rthread.start()
    assert _confirm(client, ["C001", "C002", "C003", "C004"]).status_code == 201
    assert isolate(client, "C001", "并发轮询隔离").status_code == 201
    rthread.join(10)
    assert not rthread.is_alive()
    assert not broken
