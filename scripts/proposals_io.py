# -*- coding: utf-8 -*-
"""stage1_factor_proposals.json 安全读写模块 — 2026-08-21 防护加固
背景: 14:48 外部进程瞬时覆盖事故 — 多脚本共享该 JSON, 各自「读全量→改→写全量」无锁,
并发时后写覆盖先写 (我的手工编辑被流水线脚本覆盖过一次)。
防护: ① 跨进程文件锁 (O_EXCL 锁文件 + 过期清理) ② 原子写 (tmp + os.replace, 断电/崩溃不写坏)
     ③ 写后读回 JSON 校验 ④ load_and_modify 锁内闭环 (读最新→改→写, 杜绝 TOCTOU 丢改)
用法:
    from proposals_io import load_and_modify
    data = load_and_modify(lambda d: d)   # 返回锁内修改后的数据
"""
import json, os, time

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'stage1_factor_proposals.json'))
LOCK = PATH + '.lock'
LOCK_STALE_SEC = 600          # 锁文件超过 10 分钟视为进程崩溃残留, 自动清理
ACQUIRE_TIMEOUT = 60          # 最多等 60s


def _acquire():
    start = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, ('%d %s' % (os.getpid(), time.strftime('%Y-%m-%d %H:%M:%S'))).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > LOCK_STALE_SEC:
                    os.remove(LOCK)          # 崩溃残留锁 → 清理重试
                    continue
            except OSError:
                continue
            if time.time() - start > ACQUIRE_TIMEOUT:
                raise TimeoutError(
                    'stage1_factor_proposals.json 被其他进程锁定超过 %ds (锁: %s)' % (ACQUIRE_TIMEOUT, LOCK))
            time.sleep(1)


def _release():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def load():
    _acquire()
    try:
        with open(PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    finally:
        _release()


def save(data):
    _acquire()
    try:
        _atomic_write(data)
    finally:
        _release()


def _atomic_write(data):
    tmp = PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PATH)
    with open(PATH, 'r', encoding='utf-8') as f:   # 写后读回校验
        json.load(f)


def load_and_modify(modifier):
    """锁内闭环: 读最新 → modifier 修改 → 原子写 → 校验。返回修改后的数据。"""
    _acquire()
    try:
        with open(PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = modifier(data)
        _atomic_write(data)
        return data
    finally:
        _release()


if __name__ == '__main__':
    d = load_and_modify(lambda x: x)
    print('OK, total proposals:', len(d.get('proposals', [])))
