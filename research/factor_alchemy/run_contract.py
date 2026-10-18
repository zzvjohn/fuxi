# -*- coding: utf-8 -*-
"""Atomic single-writer and lineage contract for Fuxi research runs."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


ALCHEMY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALCHEMY_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR / "fuxi_runs"
LOCK_PATH = DATA_DIR / "fuxi_runtime.lock"

_CORE_CODE = (
    ALCHEMY_DIR / "run_v4_pipeline.py",
    ALCHEMY_DIR / "ralph_loop.py",
    ALCHEMY_DIR / "multi_stage_validator.py",
    ALCHEMY_DIR / "factor_ic_computer.py",
    ALCHEMY_DIR / "s5_joint_filter.py",
    ALCHEMY_DIR / "llm_generator.py",
    ALCHEMY_DIR / "forge" / "__init__.py",
    ALCHEMY_DIR / "forge" / "expression.py",
    ALCHEMY_DIR / "forge" / "primitives.py",
)

_DATA_SENTINELS = (
    DATA_DIR / "raw" / "daily_prices.csv",
    DATA_DIR / "raw" / "moneyflow_daily.csv",
    # [!] 2026-09-25 (P4): 下面四个此前不在标准集里, 而它们都是**判据输入**:
    #   suspend_d.csv 决定「整日停牌」判定; delist_dates.csv 是退市权威通道;
    #   stock_list.csv 是那张已知过期的证券主表 (正因为它过期, 更要留指纹才能证明是哪一版);
    #   trade_calendar.csv 决定会话集合。旧 manifest 声称「覆盖数据」却对这四者一无所知。
    DATA_DIR / "raw" / "suspend_d.csv",
    DATA_DIR / "raw" / "delist_dates.csv",
    DATA_DIR / "raw" / "stock_list.csv",
    DATA_DIR / "raw" / "trade_calendar.csv",
    DATA_DIR / "stage1_factor_proposals.json",
    DATA_DIR / "mab_scheduler_state.json",
    DATA_DIR / "experience_memory.json",
)

_TERMINAL_STATES = {"completed", "failed", "aborted"}
WRITER_LEASE_ENV = "FUXI_WRITER_LEASE"
LOCAL_LEASE_ATTRIBUTE = "_fuxi_governed_writer_leases"
RAW_OWNER_ATTRIBUTE = "_fuxi_owned_writer_locks"


class ActiveFuxiRunError(RuntimeError):
    """Raised when another live writer owns the Fuxi runtime lock."""


class RunReuseError(RuntimeError):
    """Raised when a run id would overwrite an existing run manifest."""


class InvalidFuxiLeaseError(ActiveFuxiRunError):
    """A claimed inherited owner could not be verified from the live lock."""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    with open(path, "r", encoding="utf-8") as handle:
        json.load(handle)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _manifest_key(path: Path) -> str:
    """manifest 里用的键: 项目内用相对路径, 项目外用绝对路径。

    [!] 2026-09-25 (P4): 原实现直接 `path.relative_to(PROJECT_ROOT)`, 对项目外路径会抛
    ValueError ⇒ 一个批次想登记自己放在别处的脚本/数据时, 只能**放弃登记**(静默不覆盖)。
    改成绝对路径兜底, 让「登记」这件事不因路径位置而静默失败。
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_declared(paths: Iterable[Path], *, role: str) -> tuple:
    """把批次显式声明的附加路径解析成**必须存在**的绝对路径元组。

    🔴 fail-closed: 声明了却不存在 ⇒ 直接抛错。若允许「不存在也记一行 exists=False」,
    声明就退化成了一句无需兑现的话 —— 与「静默跳过」同类。
    """
    out = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise ValueError('%s declares %r but it is not a file' % (role, str(raw)))
        out.append(path)
    return tuple(out)


def _file_fingerprint(path: Path, *, content_hash: bool) -> Dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    stat = path.stat()
    out: Dict[str, Any] = {
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        out["sha256"] = _sha256_bytes(path.read_bytes())
    return out


def _process_create_time(pid: int) -> Optional[float]:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _lock_owner_is_live(meta: Dict[str, Any]) -> bool:
    try:
        pid = int(meta["pid"])
        recorded = float(meta["process_create_time"])
    except (KeyError, TypeError, ValueError):
        return False
    actual = _process_create_time(pid)
    return actual is not None and abs(actual - recorded) < 1.0


def _verified_owner_is_live(meta):
    import psutil
    try:
        if not isinstance(meta.get("token"), str) or not meta["token"] or meta.get("hostname") != socket.gethostname():
            raise ValueError("invalid owner metadata")
        pid, created = int(meta["pid"]), float(meta["process_create_time"])
        if pid <= 0 or not created > 0:
            raise ValueError("invalid owner process identity")
        # 🔴 2026-10-11 (P-20261011-014): 容差 .001(1ms)→1.0(1s)。
        #   根因: enforce 全量持锁 1h43m 期间内存膨胀至 ~10GB, Windows 下 psutil
        #   读取高负载进程的 create_time 出现亚秒级抖动 ⇒ 1ms 容差把**活跃** owner
        #   误判为死 ⇒ 触发 stale reclamation 把活锁 rename 走 ⇒ enforce 后续
        #   写队列读锁 Errno2 FATAL (37/54)。对齐死代码 _lock_owner_is_live 的 1.0s
        #   口径; 进程创建时间跨调用恒等, 1s 容差足以区分「真活」与「PID 复用」。
        return abs(psutil.Process(pid).create_time() - created) < 1.0
    except psutil.NoSuchProcess:
        return False
    except Exception as exc:
        raise ActiveFuxiRunError("Cannot verify existing Fuxi lock owner") from exc


def validate_writer_lease(path: Path, lease: Dict[str, Any], *, allow_self: bool = False) -> Dict[str, Any]:
    """Verify bytes, live process identity and actual ancestry; an env claim is insufficient."""
    try:
        import psutil
        resolved = Path(path).resolve()
        if Path(lease["lock_path"]).resolve() != resolved:
            raise ValueError("lease names another lock")
        owner = json.loads(resolved.read_text(encoding="utf-8"))
        for key in ("token", "run_id", "pid", "process_create_time"):
            if owner[key] != lease[key]:
                raise ValueError("lease/owner identity differs: " + key)
        if not owner["token"] or not owner["run_id"] or owner.get("hostname") != socket.gethostname():
            raise ValueError("invalid owner identity")
        process = psutil.Process(int(owner["pid"]))
        # 🔴 2026-10-11 (P-20261011-014): 容差 .001(1ms)→1.0(1s), 与 _verified_owner_is_live 对齐。
        #   同根因: Windows 高负载下 create_time 读取抖动 ⇒ 1ms 会把**活跃** owner 误判为
        #   「进程被替换」⇒ 继承校验失败。create_time 跨调用恒等, 1s 足够区分 PID 复用。
        if abs(process.create_time() - float(owner["process_create_time"])) >= 1.0:
            raise ValueError("owner process was replaced")
        current = psutil.Process(os.getpid())
        if process.pid == current.pid:
            if not allow_self:
                raise ValueError("same-process env inheritance requires an active local context")
        elif not any(p.pid == process.pid and abs(p.create_time() - process.create_time()) < 1.0
                     for p in current.parents()):
            raise ValueError("owner is not an ancestor of this process")
        # A second read closes the ordinary owner-change window before a borrower proceeds.
        if json.loads(resolved.read_text(encoding="utf-8")) != owner:
            raise ValueError("lock owner changed during validation")
        return owner
    except Exception as exc:
        raise InvalidFuxiLeaseError("Inherited Fuxi writer refused: " + str(exc)) from exc


def environment_writer_lease():
    raw = os.environ.get(WRITER_LEASE_ENV)
    if raw is None:
        return None
    try:
        lease = json.loads(raw)
        if not isinstance(lease, dict):
            raise ValueError("object required")
        return lease
    except Exception as exc:
        raise InvalidFuxiLeaseError("Malformed Fuxi writer lease") from exc


def local_writer_lease(path):
    """Only a governance scope in this exact thread may borrow its own process lock."""
    leases = getattr(threading.current_thread(), LOCAL_LEASE_ATTRIBUTE, {})
    return leases.get(str(Path(path).resolve()))


def current_thread_writer(path):
    """Return a strictly verified raw lock owned by this thread, or None without an acquisition."""
    owners = getattr(threading.current_thread(), RAW_OWNER_ATTRIBUTE, {})
    lease = owners.get(str(Path(path).resolve()))
    if lease is None:
        return None
    return validate_writer_lease(path, lease, allow_self=True)


def formula_sha256(formula: str) -> str:
    """Hash the exact full formula; no truncation or whitespace rewriting."""
    return _sha256_bytes(str(formula or "").encode("utf-8"))


def candidate_identity(candidate: Dict[str, Any]) -> Dict[str, str]:
    formula = candidate.get("formula") or candidate.get("expression") or ""
    fhash = formula_sha256(str(formula))
    identity = {
        "formula_sha256": fhash,
        "source": str(candidate.get("_source") or candidate.get("source") or "unknown"),
        "natural_freq": str(candidate.get("natural_freq") or "daily"),
        "paradigm": str(candidate.get("paradigm") or candidate.get("_mab_paradigm") or ""),
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity["candidate_id"] = "cand_" + _sha256_bytes(canonical.encode("utf-8"))[:20]
    return identity


def research_action_id(
    *, generator: str, paradigm: str, fitness_mode: str, natural_freq: str = ""
) -> str:
    payload = {
        "generator": str(generator or ""),
        "paradigm": str(paradigm or ""),
        "fitness_mode": str(fitness_mode or ""),
        "natural_freq": str(natural_freq or ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "action_" + _sha256_bytes(raw.encode("utf-8"))[:20]


def make_experiment_id(parameters: Dict[str, Any]) -> str:
    raw = json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str,
                     separators=(",", ":"))
    return "exp_" + _sha256_bytes(raw.encode("utf-8"))[:20]


class FuxiWriterLock(AbstractContextManager):
    """Auditable O_EXCL lock that distinguishes stale files from live owners."""

    def __init__(self, path: Path = LOCK_PATH, run_id: str = "", *, inheritance=None,
                 allow_self: bool = False, lease_scope=None):
        self.path = Path(path)
        self.run_id = run_id or "fuxi_lock_" + uuid.uuid4().hex
        self.token = uuid.uuid4().hex
        self.meta: Dict[str, Any] = {}
        self.acquired = False
        self.inherited = False
        self.inheritance = inheritance
        self.allow_self = allow_self
        self.lease_scope = lease_scope
        self.owner_thread = None

    def acquire(self) -> Dict[str, Any]:
        if self.acquired or self.inherited:
            raise ActiveFuxiRunError("A Fuxi lock object cannot be acquired twice")
        local = local_writer_lease(self.path)
        lease = self.inheritance if self.inheritance is not None else local or environment_writer_lease()
        if lease is not None:
            self.meta = validate_writer_lease(self.path, lease, allow_self=self.allow_self or local is not None)
            self.meta = dict(self.meta, borrowed_run_id=self.run_id, borrowed_by_pid=os.getpid())
            self.inherited = True
            return dict(self.meta)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        created = _process_create_time(pid)
        if created is None:
            raise ActiveFuxiRunError("Cannot verify this writer's real process creation time")
        self.meta = {
            "schema_version": 1,
            "token": self.token,
            "run_id": self.run_id,
            "pid": pid,
            "process_create_time": created,
            "hostname": socket.gethostname(),
            "started_at": _now(),
            "argv": list(sys.argv),
        }
        if self.lease_scope is not None:
            self.meta["governance_scope"] = dict(self.lease_scope)
        raw = (json.dumps(self.meta, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, raw)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                self.acquired = True
                self.owner_thread = threading.get_ident()
                thread = threading.current_thread()
                owners = getattr(thread, RAW_OWNER_ATTRIBUTE, {})
                setattr(thread, RAW_OWNER_ATTRIBUTE, {**owners,
                        str(self.path.resolve()): {"lock_path": str(self.path.resolve()), **self.meta}})
                return dict(self.meta)
            except FileExistsError:
                try:
                    owner = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    raise ActiveFuxiRunError("Existing Fuxi lock is incomplete or unreadable; refuse reclamation")
                if _verified_owner_is_live(owner):
                    raise ActiveFuxiRunError(
                        "Fuxi writer is active: "
                        f"pid={owner.get('pid')} run_id={owner.get('run_id')} "
                        f"started_at={owner.get('started_at')}"
                    )
                reclaim = self.path.with_name(self.path.name + ".reclaim")
                try:
                    reclaim_fd = os.open(str(reclaim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError as exc:
                    raise ActiveFuxiRunError("Fuxi stale-owner reclamation is already active") from exc
                try:
                    try:
                        current = json.loads(self.path.read_text(encoding="utf-8"))
                    except FileNotFoundError:
                        continue
                    if current != owner or _verified_owner_is_live(current):
                        raise ActiveFuxiRunError("Fuxi owner changed during stale-lock reclamation")
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    stale = self.path.with_name(self.path.name + f".stale.{stamp}")
                    os.replace(self.path, stale)
                finally:
                    os.close(reclaim_fd)
                    reclaim.unlink()

    def release(self) -> None:
        if self.inherited:
            # Borrowers never remove or mutate an ancestor's lock, including on failure.
            self.inherited = False
            return
        if not self.acquired:
            return
        if threading.get_ident() != self.owner_thread:
            raise ActiveFuxiRunError("Only the acquiring thread may release its Fuxi lock")
        # [!] 2026-10-15 (P-20261015-001): release 幂等化。长跑任务退出时锁文件
        #   可能已不存在（外部删除/宿主清理，4 次事故实证均发生在数据全部落账后）。
        #   释放的目标状态 = 锁文件不存在 + 本线程注册清除；文件已不在时目标已达成，
        #   清除注册并正常返回，不再抛 FileNotFoundError 把已完成的运行误报为失败。
        #   文件存在但 token 不符的严格校验保留在下方, 防止误删他人锁。
        if not self.path.exists():
            print("[fuxi-lock] release idempotent: lock file already absent for "
                  f"token={self.token[:8]} pid={os.getpid()}", file=sys.stderr, flush=True)
            self.acquired = False
            thread = threading.current_thread()
            owners = dict(getattr(thread, RAW_OWNER_ATTRIBUTE, {}))
            owners.pop(str(self.path.resolve()), None)
            setattr(thread, RAW_OWNER_ATTRIBUTE, owners)
            self.owner_thread = None
            return
        owned = current_thread_writer(self.path)
        if owned is None or owned.get("token") != self.token:
            raise ActiveFuxiRunError("This thread no longer owns the Fuxi acquisition token")
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
        if current.get("token") != self.token:
            raise ActiveFuxiRunError(
                "Refusing to release a Fuxi lock whose ownership token changed"
            )
        self.path.unlink()
        self.acquired = False
        thread = threading.current_thread()
        owners = dict(getattr(thread, RAW_OWNER_ATTRIBUTE, {}))
        owners.pop(str(self.path.resolve()), None)
        setattr(thread, RAW_OWNER_ATTRIBUTE, owners)
        self.owner_thread = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class RunContract(AbstractContextManager):
    """Single-writer context plus atomically persisted run/phase lineage."""

    def __init__(
        self,
        *,
        parameters: Dict[str, Any],
        run_id: str = "",
        experiment_id: str = "",
        resume: bool = False,
        runs_dir: Path = RUNS_DIR,
        lock_path: Path = LOCK_PATH,
        extra_code: Iterable[Path] = (),
        extra_data: Iterable[Path] = (),
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"fuxi_{stamp}_{uuid.uuid4().hex[:8]}"
        self.experiment_id = experiment_id or make_experiment_id(parameters)
        self.parameters = dict(parameters)
        self.resume = bool(resume)
        self.run_dir = Path(runs_dir) / self.run_id
        self.manifest_path = self.run_dir / "manifest.json"
        self.lock = FuxiWriterLock(Path(lock_path), self.run_id)
        # [!] 2026-09-25 (P4): 批次可显式声明「本批自己的脚本/数据」并登记进 manifest。
        #     原实现只能登记两张固定清单, 所以「本批脚本不在 code_fingerprints 里」
        #     不是偶发遗漏, 而是**结构上无法登记**。声明即必须存在 (fail-closed)。
        self.extra_code = _resolve_declared(extra_code, role="extra_code")
        self.extra_data = _resolve_declared(extra_data, role="extra_data")
        self.manifest: Dict[str, Any] = {}

    def _new_manifest(self) -> Dict[str, Any]:
        switches = {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("FUXI_") and key != WRITER_LEASE_ENV
        }
        code_paths = tuple(_CORE_CODE) + self.extra_code
        data_paths = tuple(_DATA_SENTINELS) + self.extra_data
        # [!] 2026-09-25 (P4/修正): extra_* 必须**去重后分类**。批次声明的路径有可能
        #     本来就在标准集里 (例如本批为了证明「退市通道属于判据输入」而显式声明
        #     delist_dates.csv)。若把这种「重复声明」也算成「新增」, 就会两处出错:
        #       · standard_data 与 len(keys) 对不上;
        #       · 读者会把「本来就有的键」误读成「本批扩进来的键」。
        #     故拆成 already_standard / newly_added 两个清单, 两者并集 == 声明集。
        extra_code_keys = [_manifest_key(p) for p in self.extra_code]
        extra_data_keys = [_manifest_key(p) for p in self.extra_data]
        std_code_keys = {_manifest_key(p) for p in _CORE_CODE}
        std_data_keys = {_manifest_key(p) for p in _DATA_SENTINELS}
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": "created",
            "created_at": _now(),
            "updated_at": _now(),
            "parameters": self.parameters,
            "public_switches": switches,
            "code_fingerprints": {
                _manifest_key(path): _file_fingerprint(path, content_hash=True)
                for path in code_paths
            },
            "data_fingerprints": {
                _manifest_key(path): _file_fingerprint(path, content_hash=False)
                for path in data_paths
            },
            # [!] 2026-09-25 (P4): 把「标准集」与「本批追加」分开报 —— 否则读者无法回答
            #     「这个 manifest 到底覆盖了什么」, 只能靠数键猜。
            "fingerprint_coverage": {
                "core_code": len(_CORE_CODE),
                "extra_code": extra_code_keys,
                "extra_code_already_standard": [k for k in extra_code_keys
                                                if k in std_code_keys],
                "extra_code_newly_added": [k for k in extra_code_keys
                                           if k not in std_code_keys],
                "standard_data": len(_DATA_SENTINELS),
                "extra_data": extra_data_keys,
                "extra_data_already_standard": [k for k in extra_data_keys
                                                if k in std_data_keys],
                "extra_data_newly_added": [k for k in extra_data_keys
                                           if k not in std_data_keys],
                "code_hashes_included": True,
                "data_hashes_included": False,
                "note": ("data 只记 bytes/mtime_ns 摘要, 不记 sha256; "
                         "extra_* 是**声明**而非自动发现, 未声明的输入依旧不在集内; "
                         "extra_*_already_standard 是「重复声明」, 它不是新增"),
            },
            "phases": {},
            "candidates": [],
            "events": [],
        }

    def _write(self) -> None:
        self.manifest["updated_at"] = _now()
        _atomic_json(self.manifest_path, self.manifest)

    def event(self, event: str, **details: Any) -> None:
        self.manifest.setdefault("events", []).append(
            {"at": _now(), "event": event, "details": details}
        )
        self._write()

    def phase(self, name: str, status: str, **details: Any) -> None:
        if status not in {"not_started", "running", "completed", "failed", "skipped"}:
            raise ValueError(f"Invalid phase status: {status}")
        record = dict(self.manifest.setdefault("phases", {}).get(name, {}))
        record.update(details)
        record["status"] = status
        record["updated_at"] = _now()
        self.manifest["phases"][name] = record
        self._write()

    def record_candidates(self, candidates: Iterable[Dict[str, Any]]) -> None:
        rows = []
        for candidate in candidates:
            identity = candidate_identity(candidate)
            candidate.update(identity)
            rows.append({
                **identity,
                "factor_name": str(candidate.get("factor_name") or candidate.get("name") or ""),
                "research_action_id": str(candidate.get("research_action_id") or ""),
            })
        self.manifest["candidates"] = rows
        self._write()

    def record_evaluation(
        self, summary: Dict[str, Any], results: Iterable[Any]
    ) -> None:
        rows = []
        for result in results:
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            rows.append(payload)
        self.manifest["evaluation"] = {
            "summary": dict(summary or {}),
            "results": rows,
        }
        self._write()

    def __enter__(self):
        self.lock.acquire()
        try:
            if self.manifest_path.exists():
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if not self.resume or existing.get("status") in _TERMINAL_STATES:
                    raise RunReuseError(
                        f"Run id {self.run_id!r} already exists with status "
                        f"{existing.get('status')!r}; choose a new id"
                    )
                self.manifest = existing
                self.event("resumed")
            else:
                self.run_dir.mkdir(parents=True, exist_ok=False)
                self.manifest = self._new_manifest()
            self.manifest["status"] = "running"
            self.manifest["started_at"] = self.manifest.get("started_at") or _now()
            self.event("started", lock=self.lock.meta)
            return self
        except BaseException:
            self.lock.release()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is None:
                self.manifest["status"] = "completed"
                self.manifest["completed_at"] = _now()
                self.event("completed")
            else:
                self.manifest["status"] = "failed"
                self.manifest["failed_at"] = _now()
                self.event("failed", error_type=getattr(exc_type, "__name__", ""),
                           error=str(exc)[:1000])
        finally:
            self.lock.release()
        return False
