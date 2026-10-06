"""Circuit Bender's JPEG bitstream: a marker parser, a structural validator and
the five databends, working on the bytes of a real baseline JPEG.

An independent implementation from the JPEG standard (ITU-T T.81 / ISO
10918-1, and the JFIF container): SOI, DQT (B.2.4.1), SOF0 (B.2.2), DHT
(B.2.4.2), SOS (B.2.3), the entropy-coded segment with its 0xFF00 byte
stuffing and RSTn markers (B.1.1.5, F.1.2.3), EOI, and the zigzag sequence of
Figure A.6. The effect NAMES come from the CyberShot Cam guide by
@lixofuturista / @cebolander (github.com/cebola4444/cybershot-cam), which
inspired this mode. That repository carries no licence, so neither its code
nor its numbers are used: each bend is designed from the standard, with this
mode's own scheme (a fold, a rank, a mirror, golden-ratio cuts).

This is the desktop's source of truth for the bends, and a port of the same
design the browser page runs (cyborg-garden-site src/scripts/bender/jpeg.js).
The two are held byte-identical by `dtouch/shaders/bender/bender_goldens.json`
(dtouch.bender_looks writes it from this module; the site's tests replay it
through jpeg.js). So the arithmetic copies the browser's exactly: JS
`Math.round` is `_jround` (half up, not Python's half even), and the seeded
generator is the same mulberry32.

Every mutator takes bytes and returns NEW bytes (the input is never touched),
and amount 0 returns an identical copy. Pure numpy: no GL, no cv2.
"""
from __future__ import annotations

import math

import numpy as np

# T.81 Figure A.6: ZIGZAG[k] is the natural (row-major) index of the k-th
# coefficient in zigzag order. DQT tables are stored in this order.
ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
)

# Marker codes (T.81 Table B.1).
SOI, EOI, SOS, DQT, DHT, DRI = 0xD8, 0xD9, 0xDA, 0xDB, 0xC4, 0xDD
SOF0, SOF1, SOF2 = 0xC0, 0xC1, 0xC2

_M32 = 0xFFFFFFFF


def _is_rst(m):
    return 0xD0 <= m <= 0xD7


def _standalone(m):
    # markers with no length field (T.81 B.1.1.3)
    return m in (SOI, EOI) or _is_rst(m) or m == 0x01


class JpegError(Exception):
    """The bytes are not a JPEG this module can walk."""


def _u16(b, i):
    return (b[i] << 8) | b[i + 1]


def _jround(x):
    """JavaScript's Math.round: half rounds up (toward +inf)."""
    return int(math.floor(x + 0.5))


def _clamp(v, lo, hi):
    return min(hi, max(lo, v))


def _amt(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(a):
        return 0.0 if math.isnan(a) else (1.0 if a > 0 else 0.0)
    return _clamp(a, 0.0, 1.0)


# ----- the parser -----

def parse_jpeg(b):
    """Parse the marker structure into offsets (never copies). Keys:
    segments [{marker, at, body, end}], dqt [{id, precision, at}],
    dht [{cls, id, counts, symbols, n}], sof {width, height, comps},
    sos {comps}, scan {start, end}, eoi (-1 when cut short), dri.
    Raises JpegError on anything it cannot walk."""
    n = len(b)
    if n < 4 or b[0] != 0xFF or b[1] != SOI:
        raise JpegError("no SOI")
    out = dict(segments=[], dqt=[], dht=[], sof=None, sos=None, scan=None,
               eoi=-1, dri=0)
    i = 2
    while i < n:
        if b[i] != 0xFF:
            raise JpegError(f"expected a marker at {i}")
        while i + 1 < n and b[i + 1] == 0xFF:      # fill bytes (B.1.1.2)
            i += 1
        if i + 1 >= n:
            raise JpegError("stream ends inside a marker")
        at = i
        m = b[i + 1]
        i += 2
        if m == EOI:
            out["eoi"] = at
            out["segments"].append(dict(marker=m, at=at, body=i, end=i))
            return out
        if _standalone(m):
            out["segments"].append(dict(marker=m, at=at, body=i, end=i))
            continue
        if i + 2 > n:
            raise JpegError("segment length missing")
        ln = _u16(b, i)
        if ln < 2:
            raise JpegError(f"segment length {ln}")
        body, end = i + 2, i + ln
        if end > n:
            raise JpegError(f"segment 0x{m:x} runs past the end")
        seg = dict(marker=m, at=at, body=body, end=end)
        out["segments"].append(seg)
        if m == DQT:
            _parse_dqt(b, seg, out)
        elif m == DHT:
            _parse_dht(b, seg, out)
        elif m in (SOF0, SOF1):
            out["sof"] = _parse_sof(b, seg)
        elif m == SOF2 or (0xC3 <= m <= 0xCF and m not in (DHT, 0xC8, 0xCC)):
            raise JpegError("not a baseline JPEG")
        elif m == DRI:
            out["dri"] = _u16(b, body)
        i = end
        if m == SOS:
            out["sos"] = _parse_sos(b, seg)
            # the entropy-coded segment runs until a marker that is neither a
            # stuffed 0xFF00 nor an RSTn (B.1.1.5). Every 0xFF in it is looked
            # at (a skipped byte is a 0x00 or an RSTn code, never an 0xFF),
            # so the end is the first 0xFF followed by anything else.
            start = i
            seg = np.frombuffer(bytes(b[start:n]), dtype=np.uint8)
            nx = seg[1:]
            stop = np.flatnonzero((seg[:-1] == 0xFF) & (nx != 0x00) & (nx != 0xFF)
                                  & ((nx < 0xD0) | (nx > 0xD7)))
            j = start + int(stop[0]) if stop.size else n - 1
            if j >= n - 1:
                out["scan"] = dict(start=start, end=n)
                return out                        # no EOI: cut short
            out["scan"] = dict(start=start, end=j)
            i = j
    return out


def _parse_dqt(b, seg, out):
    p = seg["body"]
    while p < seg["end"]:
        pq, tid = b[p] >> 4, b[p] & 15
        size = 128 if pq else 64
        if pq > 1 or tid > 3 or p + 1 + size > seg["end"]:
            raise JpegError("bad DQT")
        out["dqt"].append(dict(id=tid, precision=16 if pq else 8, at=p + 1))
        p += 1 + size


def _parse_dht(b, seg, out):
    p = seg["body"]
    while p < seg["end"]:
        cls, tid = b[p] >> 4, b[p] & 15
        if cls > 1 or tid > 3 or p + 17 > seg["end"]:
            raise JpegError("bad DHT")
        n = sum(b[p + 1:p + 17])
        if n > 256 or p + 17 + n > seg["end"]:
            raise JpegError("bad DHT counts")
        out["dht"].append(dict(cls=cls, id=tid, counts=p + 1, symbols=p + 17, n=n))
        p += 17 + n


def _parse_sof(b, seg):
    p = seg["body"]
    if b[p] != 8:
        raise JpegError("not 8-bit")
    height, width, nf = _u16(b, p + 1), _u16(b, p + 3), b[p + 5]
    if seg["end"] - p < 6 + 3 * nf:
        raise JpegError("bad SOF")
    comps = [dict(id=b[p + 6 + 3 * k], h=b[p + 7 + 3 * k] >> 4,
                  v=b[p + 7 + 3 * k] & 15, tq=b[p + 8 + 3 * k]) for k in range(nf)]
    return dict(width=width, height=height, comps=comps)


def _parse_sos(b, seg):
    p = seg["body"]
    ns = b[p]
    if seg["end"] - p < 4 + 2 * ns:
        raise JpegError("bad SOS")
    return dict(comps=[dict(id=b[p + 1 + 2 * k], td=b[p + 2 + 2 * k] >> 4,
                            ta=b[p + 2 + 2 * k] & 15) for k in range(ns)])


# ----- the validator -----

def validate_jpeg(b):
    """Structural checks a decoder needs to get through the stream at all:
    the frame, every table the scan names, quantiser values of at least 1,
    Huffman counts that form a prefix code (Kraft, C.2), and an entropy-coded
    segment in which every 0xFF is stuffed or an RSTn. Returns a dict with
    `ok` (and `why` when not). It says nothing about whether it is pretty."""
    try:
        j = parse_jpeg(b)
    except JpegError as e:
        return dict(ok=False, why=str(e))
    if not j["sof"]:
        return dict(ok=False, why="no SOF0")
    if not j["sos"] or not j["scan"]:
        return dict(ok=False, why="no scan")
    if j["eoi"] < 0:
        return dict(ok=False, why="no EOI")
    sof = j["sof"]
    if not sof["width"] or not sof["height"]:
        return dict(ok=False, why="empty frame")
    for t in j["dqt"]:
        for k in range(64):
            if _read_q(b, t, k) < 1:
                return dict(ok=False, why=f"DQT {t['id']} has a zero at {k}")
    for c in sof["comps"]:
        if not any(t["id"] == c["tq"] for t in j["dqt"]):
            return dict(ok=False, why=f"no DQT {c['tq']}")
    for h in j["dht"]:
        room = 1
        for ln in range(1, 17):
            room = room * 2 - b[h["counts"] + ln - 1]
            if room < 0:
                return dict(ok=False, why=f"DHT {h['cls']}/{h['id']} over-full at length {ln}")
    for c in j["sos"]["comps"]:
        if not any(h["cls"] == 0 and h["id"] == c["td"] for h in j["dht"]):
            return dict(ok=False, why=f"no DC table {c['td']}")
        if not any(h["cls"] == 1 and h["id"] == c["ta"] for h in j["dht"]):
            return dict(ok=False, why=f"no AC table {c['ta']}")
    start, end = j["scan"]["start"], j["scan"]["end"]
    i = start
    while i < end:
        if b[i] == 0xFF:
            nx = b[i + 1] if i + 1 < end else -1
            if nx == 0x00:
                i += 1
            elif _is_rst(nx):
                if not j["dri"]:
                    return dict(ok=False, why=f"RST at {i} without DRI")
                i += 1
            else:
                return dict(ok=False, why=f"unstuffed 0xFF at {i}")
        i += 1
    return dict(ok=True, width=sof["width"], height=sof["height"],
                scan_bytes=end - start)


# ----- helpers for the mutators -----

def rng32(seed):
    """A seeded [0, 1) generator (mulberry32, the browser's own), so a bend
    can be held still or replayed: the same seed and amount give the same
    bytes on both hosts."""
    a = int(seed) & _M32

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & _M32
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & _M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & _M32)) & _M32)) & _M32
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0
    return nxt


def _read_q(b, t, k):
    return _u16(b, t["at"] + 2 * k) if t["precision"] == 16 else b[t["at"] + k]


def _write_q(b, t, k, v):
    v = _clamp(_jround(v), 1, 32767 if t["precision"] == 16 else 255)
    if t["precision"] == 16:
        b[t["at"] + 2 * k] = v >> 8
        b[t["at"] + 2 * k + 1] = v & 255
    else:
        b[t["at"] + k] = v


def _chroma_tables(j):
    """The table ids the chroma components quantise with (every component
    after the first). An encoder that quantises chroma with the luma table
    leaves no chroma table of its own; then the shared table is the only
    lever, and luma is raised with it."""
    ids = set()
    comps = j["sof"]["comps"] if j["sof"] else []
    for c in comps[1:]:
        ids.add(c["tq"])
    ids.discard(comps[0]["tq"] if comps else 0)
    if not ids:
        ids = {t["id"] for t in j["dqt"]}
    return ids


# ----- the five databends -----
#
# Four of them edit what the decoder is TOLD, not the picture: the encoder
# quantised with one table and Huffman-coded with another, and the decoder
# reads the edited ones, so the stream stays structurally valid by
# construction. SCAN SWAP moves whole MCUs of the entropy-coded data, cut
# where each begins, so the decoder never loses its place either; only its
# byte-level fallback desynchronises it, and repairs the stuffing after.

def fold_span(a):
    """ZIGZAG PERM's span: a short fold at a hint, the whole AC sequence
    reversed at 1; 0 is the identity."""
    a = _amt(a)
    return 0 if a == 0 else _jround(6 + a * 57)


def zigzag_perm(b, amount):
    """Fold the low end of each table's AC steps back on itself along the
    zigzag sequence: the first `span` AC places are read in reverse, so fine
    frequencies decode with coarse steps and the other way round. Broad
    shapes swell into hard, embossed tiles while fine detail fades."""
    a = _amt(amount)
    out = bytearray(b)
    if a == 0:
        return bytes(out)
    j = parse_jpeg(out)
    span = fold_span(a)
    for t in j["dqt"]:
        run = [_read_q(out, t, k) for k in range(1, span + 1)]
        for k in range(1, span + 1):
            _write_q(out, t, k, run[span - k])
    return bytes(out)


def dqt_erosion(b, amount):
    """Rank each table's 63 AC steps by size and erode both ends: the finest
    go to 1 (colour fields go soft), the coarsest are multiplied up toward
    the ceiling (surviving high frequency blows into hard grain). At amount 1
    the 16 finest go to 1 and the 35 coarsest are raised up to x41. DC is
    kept, so the mean of every block survives."""
    a = _amt(amount)
    out = bytearray(b)
    if a == 0:
        return bytes(out)
    j = parse_jpeg(out)
    fine = _jround(a * 16)
    coarse = max(1, _jround(a * 35))
    gain = 1 + 40 * a
    for t in j["dqt"]:
        rank = sorted(((_read_q(out, t, k), k) for k in range(1, 64)))
        for r in range(fine):
            _write_q(out, t, rank[r][1], 1)
        for r in range(63 - coarse, 63):
            _write_q(out, t, rank[r][1], rank[r][0] * gain)
    return bytes(out)


def dht_remap(b, amount):
    """Inside each AC Huffman table, the symbols that share a coefficient
    SIZE are listed from the shortest code to the longest. A remap mirrors
    that list from its two ends inward, so the commonest code of each touched
    size trades with its rarest. A code still carries the same number of
    extra bits, so the decoder never loses its place; only the zero-run in
    front of each coefficient changes. DC tables and EOB / ZRL are left
    alone; chroma tables trade half as many pairs as luma."""
    a = _amt(amount)
    out = bytearray(b)
    if a == 0:
        return bytes(out)
    j = parse_jpeg(out)
    sos = j["sos"]
    luma_ac = sos["comps"][0]["ta"] if sos and sos["comps"] else 0
    # small amounts touch only the big, rare coefficients (the edges)
    frm = max(4, _jround(10 - 8 * a))
    for h in j["dht"]:
        if h["cls"] != 1 or h["n"] < 2:
            continue
        reach = a if h["id"] == luma_ac else a / 2
        for size in range(frm, 11):
            at = [h["symbols"] + k for k in range(h["n"])
                  if (out[h["symbols"] + k] & 15) == size]
            n = len(at)
            if n < 2:
                continue
            pairs = _clamp(_jround(reach * (n >> 1)), 1, n >> 1)
            for k in range(pairs):
                x, y = at[k], at[n - 1 - k]
                out[x], out[y] = out[y], out[x]
    return bytes(out)


# ----- the entropy-coded segment, MCU by MCU -----

def unstuff(b, start, end):
    """Remove the stuffed 0x00 after every 0xFF in [start, end). A skipped
    byte is always 0x00, never an 0xFF, so one vectorised mask is exact."""
    seg = np.frombuffer(bytes(b[start:end]), dtype=np.uint8)
    if seg.size == 0:
        return seg.copy()
    drop = np.zeros(seg.size, dtype=bool)
    drop[1:] = (seg[:-1] == 0xFF) & (seg[1:] == 0x00)
    return seg[~drop].copy()


def stuff(u):
    """Stuff a plain entropy byte string: 0xFF becomes 0xFF 0x00."""
    u = np.asarray(u, dtype=np.uint8)
    ff = np.flatnonzero(u == 0xFF)
    return np.insert(u, ff + 1, 0).astype(np.uint8)


_HUFF_CACHE = {}
_HUFF_CACHE_MAX = 16


def _huff_lut(b, h):
    """The peek table for one DHT entry, cached on the table's own bytes (the
    16 counts and the symbols): an encoder writes the same tables on every
    frame, and building one is a 65536-entry pass."""
    key = bytes(b[h["counts"]:h["counts"] + 16]) + bytes(b[h["symbols"]:h["symbols"] + h["n"]])
    lut = _HUFF_CACHE.get(key)
    if lut is None:
        lut = _build_huff_lut(key)
        if len(_HUFF_CACHE) >= _HUFF_CACHE_MAX:
            _HUFF_CACHE.pop(next(iter(_HUFF_CACHE)))
        _HUFF_CACHE[key] = lut
    return lut


def _build_huff_lut(key):
    """A 16-bit peek table from a DHT entry's bytes (16 counts, then the
    symbols; canonical codes, C.2 and F.2.2.3): sym[v] / ln[v] for the code
    at the top of the 16 bits v, ln 0 where no code of 16 bits or fewer
    matches (the browser's decode -1)."""
    counts = list(key[:16])
    vals = np.frombuffer(key[16:], dtype=np.uint8)
    peek = np.arange(65536, dtype=np.int64)
    sym = np.zeros(65536, dtype=np.int32)
    ln = np.zeros(65536, dtype=np.int32)
    code = k = 0
    for length in range(1, 17):
        n = counts[length - 1]
        if n:
            lo, hi = code, code + n - 1
            top = peek >> (16 - length)
            hit = (ln == 0) & (top >= lo) & (top <= hi)
            idx = k + (top[hit] - lo)
            ok = idx < vals.size
            sel = np.flatnonzero(hit)[ok]
            sym[sel] = vals[idx[ok]]
            ln[sel] = length
        code += n
        k += n
        code <<= 1
    return sym.tolist(), ln.tolist()


def mcu_offsets(b, j=None):
    """Walk the scan the way a decoder does (T.81 F.2.2) and return the bit
    offset (into the unstuffed data) of every MCU's first bit plus one past
    the last: dict(offsets, u, mcus, complete). `complete` is False when it
    met a code the tables do not hold before the last MCU. Past the end the
    walk reads 1 bits (the fill), as a real decoder does."""
    if j is None:
        j = parse_jpeg(b)
    if not j["sof"] or not j["sos"] or not j["scan"]:
        raise JpegError("no scan")
    if j["dri"]:
        raise JpegError("restart intervals are not walked")
    u = unstuff(b, j["scan"]["start"], j["scan"]["end"])
    tables = {(h["cls"], h["id"]): h for h in j["dht"]}
    luts = {}
    sof, sos = j["sof"], j["sos"]
    single = len(sos["comps"]) == 1
    hmax = max(c["h"] for c in sof["comps"])
    vmax = max(c["v"] for c in sof["comps"])
    plan = []
    for sc in sos["comps"]:
        fc = next((c for c in sof["comps"] if c["id"] == sc["id"]), None)
        if fc is None:
            raise JpegError("scan names an unknown component")
        dc, ac = tables.get((0, sc["td"])), tables.get((1, sc["ta"]))
        if dc is None or ac is None:
            raise JpegError("missing Huffman table")
        for key, h in (((0, sc["td"]), dc), ((1, sc["ta"]), ac)):
            if key not in luts:
                luts[key] = _huff_lut(b, h)
        blocks = 1 if single else fc["h"] * fc["v"]
        plan.extend([(luts[(0, sc["td"])], luts[(1, sc["ta"])])] * blocks)
    if single:
        n = math.ceil(sof["width"] / 8) * math.ceil(sof["height"] / 8)
    else:
        n = math.ceil(sof["width"] / (8 * hmax)) * math.ceil(sof["height"] / (8 * vmax))
    total = int(u.size) * 8
    # 32-bit windows at every byte, padded with 1 bits past the end
    pad = np.concatenate([u, np.full(8, 0xFF, dtype=np.uint8)]).astype(np.uint32)
    win = ((pad[:-3] << 24) | (pad[1:-2] << 16) | (pad[2:-1] << 8) | pad[3:]).tolist()
    nwin = len(win)
    offsets = []

    def partial(pos):
        return dict(offsets=offsets + [min(pos, total)], u=u, mcus=n, complete=False)

    pos = 0
    for _ in range(n):
        if pos >= total:
            return partial(pos)
        offsets.append(pos)
        for (dsym, dln), (asym, aln) in plan:
            wi = pos >> 3
            v = (win[wi] >> (16 - (pos & 7))) & 0xFFFF if wi < nwin else 0xFFFF
            ln = dln[v]
            if not ln:
                # the browser's decode reads all 16 lengths before it fails
                offsets.pop()
                return partial(pos + 16)
            s = dsym[v]
            pos += ln
            if s > 11:
                offsets.pop()
                return partial(pos)
            pos += s
            k = 1
            while k < 64:
                wi = pos >> 3
                v = (win[wi] >> (16 - (pos & 7))) & 0xFFFF if wi < nwin else 0xFFFF
                ln = aln[v]
                if not ln:
                    offsets.pop()
                    return partial(pos + 16)
                rs = asym[v]
                pos += ln
                sz = rs & 15
                if sz == 0:
                    if (rs >> 4) != 15:
                        break               # EOB
                    k += 16                 # ZRL
                    continue
                k += (rs >> 4) + 1
                pos += sz
    offsets.append(min(pos, total))
    return dict(offsets=offsets, u=u, mcus=n, complete=True)


def reorder_mcus(b, j, walk, order):
    """Rebuild the file with its MCUs in a new order: order[i] names the
    source MCU written at position i. Header kept, the last byte padded with
    1 bits (F.1.2.3), restuffed, then EOI."""
    offsets, u = walk["offsets"], walk["u"]
    bits = np.unpackbits(u)
    total = bits.size
    parts = []
    for m in order:
        a, z = offsets[m], offsets[m + 1]
        part = bits[a:min(z, total)]
        if z > total:   # a walk that ran into the fill reads 1 bits there
            part = np.concatenate([part, np.ones(z - max(a, total), dtype=np.uint8)])
        parts.append(part)
    cat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    padn = (-cat.size) % 8
    if padn:
        cat = np.concatenate([cat, np.ones(padn, dtype=np.uint8)])
    scan = stuff(np.packbits(cat)).tobytes()
    start = j["scan"]["start"]
    return bytes(b[:start]) + scan + bytes((0xFF, EOI))


GOLDEN = 0.6180339887498949


def _cut_at(c, cuts, phase, rng):
    """Where cut c of `cuts` lands, as a fraction of the scan: golden-ratio
    steps, jittered a little by the rng, slid together by `phase`."""
    x = phase + c * GOLDEN + (rng() - 0.5) * (0.3 / cuts)
    # JS ((x % 1) + 1) % 1, rounding included: % is a truncating fmod there
    return math.fmod(math.fmod(x, 1.0) + 1.0, 1.0)


def scan_swap(b, amount, rng=None, phase=0.0):
    """Runs of MCUs are replaced by runs copied from elsewhere in the scan,
    cut exactly on MCU boundaries, so every frame decodes. Each block's DC is
    a difference from the one before (F.1.2.1), so after a cut the following
    blocks inherit a colour offset and bands run to the edge of the frame.
    Falls back to scan_smear on a scan it cannot walk."""
    a = _amt(amount)
    if a == 0:
        return bytes(b)
    rng = rng or _sys_rng
    j = parse_jpeg(b)
    try:
        walk = mcu_offsets(b, j)
    except JpegError:
        walk = None
    n = len(walk["offsets"]) - 1 if walk else 0
    if not walk or not walk["complete"] or n < 8:
        return scan_smear(b, a, rng, phase)
    order = list(range(n))
    cuts = 1 + int(math.floor(a * 4 * (0.5 + rng()) + 0.5))       # 1 to 7
    run = max(1, _jround(n * (0.012 + 0.06 * a) * (0.6 + 0.8 * rng())))
    span = n - run
    for c in range(cuts):
        r = int(math.floor(_cut_at(c, cuts, phase, rng) * span))
        d = r + _jround(n * (0.33 + 0.33 * rng()))
        if d > span:
            d -= span
        for k in range(run):
            order[r + k] = d + k
    return reorder_mcus(b, j, walk, order)


def scan_smear(b, amount, rng=None, phase=0.0):
    """The byte-level stutter: a stretch of the entropy-coded bytes written
    again over the bytes that follow it, a few stretches on, like a stuck
    read pointer. The decoder loses its place. Afterwards any 0xFF no longer
    followed by 0x00 becomes 0xFE, so the stream stays walkable (it never
    grows or shrinks, and no stray RST or EOI can appear)."""
    a = _amt(amount)
    out = bytearray(b)
    if a == 0:
        return bytes(out)
    rng = rng or _sys_rng
    j = parse_jpeg(out)
    if not j["scan"]:
        return bytes(out)
    start, end = j["scan"]["start"], j["scan"]["end"]
    ln = end - start
    if ln < 256:
        return bytes(out)
    cuts = 1 + int(math.floor(a * 2.5 * rng() + 0.5))
    chunk = _clamp(_jround(ln * (0.003 + 0.025 * a) * (0.6 + 0.8 * rng())), 16, 2048)
    for c in range(cuts):
        r = start + int(math.floor(_cut_at(c, cuts, phase, rng) * (ln - chunk)))
        lag = chunk * (1 + int(math.floor(rng() * 5)))
        d = r - lag if r - lag >= start else min(r + lag, end - chunk)
        if out[r - 1] == 0xFF:      # never start between a stuffed pair
            r += 1
        if out[d - 1] == 0xFF:
            d += 1
        l = min(chunk, end - r, end - d)
        if l > 0 and d != r:
            out[r:r + l] = out[d:d + l]
    repair_stuffing(out, start, end)
    return bytes(out)


def repair_stuffing(b, start, end):
    """In [start, end) every 0xFF must be followed by 0x00; one that is not
    becomes 0xFE, which is ordinary data, never a marker. In place."""
    # vectorised: the verdict on each 0xFF reads only the byte after it, and
    # rewriting one 0xFF to 0xFE never turns its neighbour into a 0x00
    seg = np.frombuffer(bytes(b[start:end]), dtype=np.uint8).copy()
    if seg.size == 0:
        return b
    ff = seg == 0xFF
    stuffed = np.zeros(seg.size, dtype=bool)
    stuffed[:-1] = seg[1:] == 0x00
    seg[ff & ~stuffed] = 0xFE
    b[start:end] = seg.tobytes()
    return b


def chroma_amp(b, amount):
    """Raise the chroma table's DC and its AC steps along a curve that starts
    at full gain on the first AC step and halves every 3 to 9 steps along the
    zigzag. Chroma was quantised finely and decodes coarsely, so colour is
    multiplied out toward full saturation while luma (sharpness) is
    untouched. DC is held to x5 so a face never floods to one flat colour."""
    a = _amt(amount)
    out = bytearray(b)
    if a == 0:
        return bytes(out)
    j = parse_jpeg(out)
    ids = _chroma_tables(j)
    dc_gain = 1 + 4 * a
    peak = 12 * a
    half = 3 + 6 * a
    for t in j["dqt"]:
        if t["id"] not in ids:
            continue
        _write_q(out, t, 0, _read_q(out, t, 0) * dc_gain)
        for k in range(1, 64):
            _write_q(out, t, k, _read_q(out, t, k) * (1 + peak * 2 ** (-(k - 1) / half)))
    return bytes(out)


# ----- the table -----

# Effects in the order the CyberShot guide lists them. STACK composes in this
# mode's own order: the scan first, while it can still be walked MCU by MCU
# (so the cut is always the clean SCAN SWAP), then the tables.
EFFECTS = ("zigzag", "erosion", "remap", "swap", "chroma", "stack")
EFFECT_TITLES = {
    "zigzag": "zigzag perm", "erosion": "dqt erosion", "remap": "dht remap",
    "swap": "scan swap", "chroma": "chroma amp", "stack": "stack",
}

# STACK's hands: lighter each, or five full bends pile into mud
STACK_WEIGHTS = (("swap", 0.6), ("remap", 0.45), ("chroma", 0.3),
                 ("erosion", 0.3), ("zigzag", 0.4))


def _sys_rng():
    return float(np.random.random())


def bend(b, effect, amount, rng=None, phase=0.0):
    """One bend: `effect` from EFFECTS, amount 0..1, a seeded rng (rng32) and
    a phase for SCAN SWAP. Raises JpegError when the input does not parse."""
    rng = rng or _sys_rng
    if effect == "zigzag":
        return zigzag_perm(b, amount)
    if effect == "erosion":
        return dqt_erosion(b, amount)
    if effect == "remap":
        return dht_remap(b, amount)
    if effect == "swap":
        return scan_swap(b, amount, rng, phase)
    if effect == "chroma":
        return chroma_amp(b, amount)
    if effect == "stack":
        s = _amt(amount)
        out = b
        for name, w in STACK_WEIGHTS:
            out = (scan_swap(out, s * w, rng, phase) if name == "swap"
                   else bend(out, name, s * w))
        return out
    return bytes(b)
