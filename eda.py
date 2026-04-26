#!/usr/bin/env python3
"""
EDA — Evrensel Dataset Arşivi  (Earth Data Archive)
"""

import argparse
import lzma
import struct
import sys
import io
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

# ── Sabitler ──────────────────────────────────────────────────────────────────

FLAG_BYTE  = 0x01
EDA_MAGIC  = b"EDA\x01"
ENTRY_FILE = b"F"
ENTRY_DIR  = b"D"
CODECS_DIR = Path(__file__).parent / "codecs"
META_NAME  = "eda_data.txt"          # arşiv kök dizinindeki meta dosya adı


# ── Codec ──────────────────────────────────────────────────────────────────────

def find_codec(name: str) -> Path:
    """
    İsim çözümleme sırası:
      1. Verilen yol doğrudan varsa kullan
      2. ./codecs/<name>  (uzantısız verilmişse .edc eklenir)
    """
    p = Path(name)
    if p.exists():
        return p.resolve()
    stem = p.stem if p.suffix == ".edc" else p.name
    candidate = CODECS_DIR / (stem + ".edc")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Codec bulunamadı: '{name}'\n"
        f"  Denenen: {candidate}\n"
        f"  Mevcut codec'ler için:  eda list-codecs"
    )


def load_codec(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"EDAC", "Geçersiz codec dosyası (.edc)"
        f.read(1)                                      # version
        f.read(1)                                      # flag_byte
        n = struct.unpack("<H", f.read(2))[0]
        for _ in range(n):
            tid  = struct.unpack("<H", f.read(2))[0]
            tlen = struct.unpack("B",  f.read(1))[0]
            tok  = f.read(tlen).decode("utf-8")
            vocab[tok] = tid
    return vocab


def codec_name_from_path(path: Path) -> str:
    """TR_tr.edc  →  'TR_tr'"""
    return path.stem


def codec_id_bytes(name: str) -> bytes:
    """6 byte, null-padded ASCII codec kimliği."""
    return name[:5].ljust(5, "\x00").encode("ascii", errors="replace") + b"\x00"


# ── Automaton & encode/decode ──────────────────────────────────────────────────

def build_automaton(vocab: dict[str, int]):
    try:
        import ahocorasick
        A = ahocorasick.Automaton()
        for tok, tid in vocab.items():
            A.add_word(tok, (len(tok), len(tok.encode("utf-8")), tid))
        A.make_automaton()
        return ("aho", A)
    except ImportError:
        by_len = sorted(vocab.items(), key=lambda x: len(x[0]), reverse=True)
        return ("simple", by_len)


def encode_bytes(text: str, automaton) -> bytes:
    out        = bytearray()
    text_bytes = text.encode("utf-8")
    n          = len(text_bytes)

    if automaton[0] == "aho":
        A = automaton[1]
        char_to_byte: list[int] = []
        b_off = 0
        for ch in text:
            char_to_byte.append(b_off)
            b_off += len(ch.encode("utf-8"))
        char_to_byte.append(b_off)

        match_at: dict[int, tuple[int, int]] = {}
        for end_char, (char_len, byte_len, slot_id) in A.iter(text):
            bstart = char_to_byte[end_char - char_len + 1]
            if bstart not in match_at or match_at[bstart][0] < byte_len:
                match_at[bstart] = (byte_len, slot_id)

        i = 0
        while i < n:
            if i in match_at:
                blen, sid = match_at[i]
                out.append(FLAG_BYTE)
                out.append(sid >> 4)
                out.append(sid & 0xF)
                i += blen
            else:
                b = text_bytes[i]
                if b == FLAG_BYTE:
                    out += b"\x01\x00\x01"
                else:
                    out.append(b)
                i += 1
    else:
        by_len = automaton[1]
        i = 0
        while i < n:
            matched = False
            for tok, tid in by_len:
                tb = tok.encode("utf-8")
                tl = len(tb)
                if text_bytes[i:i + tl] == tb:
                    out.append(FLAG_BYTE)
                    out.append(tid >> 4)
                    out.append(tid & 0xF)
                    i += tl
                    matched = True
                    break
            if not matched:
                b = text_bytes[i]
                if b == FLAG_BYTE:
                    out += b"\x01\x00\x01"
                else:
                    out.append(b)
                i += 1

    return bytes(out)


def decode_bytes(data: bytes, slot_to_tok: dict[int, str]) -> str:
    out = bytearray()
    i   = 0
    n   = len(data)
    while i < n:
        if data[i] == FLAG_BYTE and i + 2 < n:
            b2, b3 = data[i + 1], data[i + 2]
            if b2 == 0x00 and b3 == FLAG_BYTE:
                out.append(FLAG_BYTE)
                i += 3
            else:
                slot_id = (b2 << 4) | b3
                tok = slot_to_tok.get(slot_id)
                if tok is not None:
                    out += tok.encode("utf-8")
                    i += 3
                else:
                    out.append(data[i])
                    i += 1
        else:
            out.append(data[i])
            i += 1
    return out.decode("utf-8", errors="replace")


# ── LZMA ───────────────────────────────────────────────────────────────────────

CHUNK = 256 * 1024   # 256 KB — her adımda işlenen blok boyutu


def lzma_compress(data: bytes, preset: int, label: str = "LZMA") -> bytes:
    filters = [{"id": lzma.FILTER_LZMA2, "preset": preset,
                 "dict_size": 16 * 1024 * 1024}]
    comp   = lzma.LZMACompressor(format=lzma.FORMAT_XZ, filters=filters)
    total  = len(data)
    out    = bytearray()
    bar    = tqdm(total=total, desc=f"  {label} ↓", unit="B",
                  unit_scale=True, unit_divisor=1024, leave=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    pos = 0
    while pos < total:
        chunk  = data[pos:pos + CHUNK]
        out   += comp.compress(chunk)
        pos   += len(chunk)
        bar.update(len(chunk))
    out += comp.flush()
    bar.close()
    return bytes(out)


def lzma_decompress(data: bytes, label: str = "LZMA") -> bytes:
    decomp = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    total  = len(data)
    out    = bytearray()
    bar    = tqdm(total=total, desc=f"  {label} ↑", unit="B",
                  unit_scale=True, unit_divisor=1024, leave=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    pos = 0
    while pos < total:
        chunk  = data[pos:pos + CHUNK]
        out   += decomp.decompress(chunk)
        pos   += len(chunk)
        bar.update(len(chunk))
    bar.close()
    return bytes(out)


# ── Meta (eda_data.txt) ────────────────────────────────────────────────────────

def build_meta(codec_name: str, src: Path, is_dir: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"codec={codec_name}",
        f"source_name={src.name}",
        f"source_type={'directory' if is_dir else 'file'}",
    ]
    if not is_dir:
        lines.append(f"source_extension={src.suffix}")
    lines.append(f"archived_at={now}")
    return "\n".join(lines) + "\n"


def parse_meta(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


# ── Tekli dosya ────────────────────────────────────────────────────────────────

def compress_file(src: Path, dst: Path, automaton,
                  codec_name: str, preset: int):
    text    = src.read_text(encoding="utf-8", errors="ignore")
    encoded = encode_bytes(text, automaton)
    meta    = build_meta(codec_name, src, is_dir=False)

    # İçerik: [codec_name\0][meta_bytes][payload]
    # Paket: codec_id(6) + F + meta_len(2) + meta + enc_size(8) + lzma(encoded)
    meta_b   = meta.encode("utf-8")
    payload  = lzma_compress(encoded, preset, label="Sıkıştır")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(EDA_MAGIC)
        codec_name_b = codec_name.encode("utf-8")
        f.write(struct.pack("<H", len(codec_name_b)))
        f.write(codec_name_b)
        f.write(ENTRY_FILE)
        f.write(struct.pack("<H", len(meta_b)))
        f.write(meta_b)
        f.write(struct.pack("<Q", len(encoded)))
        f.write(payload)


def read_file_header(src: Path) -> dict:
    """Tekli .eda dosyasının header + meta bilgisini döner, veri okumaz."""
    with open(src, "rb") as f:
        assert f.read(4) == EDA_MAGIC, "Geçersiz .eda dosyası"
        cname_len  = struct.unpack("<H", f.read(2))[0]
        codec_name = f.read(cname_len).decode("utf-8")
        entry      = f.read(1)
        assert entry == ENTRY_FILE, "Klasör arşivi (D), tekli dosya (F) beklendi"
        meta_len   = struct.unpack("<H", f.read(2))[0]
        meta_text  = f.read(meta_len).decode("utf-8")
        enc_size   = struct.unpack("<Q", f.read(8))[0]
        lzma_size  = src.stat().st_size - f.tell()
    meta = parse_meta(meta_text)
    return {
        "codec"     : codec_name,
        "meta"      : meta,
        "meta_text" : meta_text,
        "enc_size"  : enc_size,
        "lzma_size" : lzma_size,
    }


def decompress_file(src: Path, dst: Path, slot_to_tok: dict[int, str],
                    verbose: bool):
    with open(src, "rb") as f:
        assert f.read(4) == EDA_MAGIC, "Geçersiz .eda dosyası"
        cname_len  = struct.unpack("<H", f.read(2))[0]
        codec_name = f.read(cname_len).decode("utf-8")
        entry      = f.read(1)
        assert entry == ENTRY_FILE, "Bu .eda bir klasör arşivi (D), dosya beklendi (F)"
        meta_len   = struct.unpack("<H", f.read(2))[0]
        meta_text  = f.read(meta_len).decode("utf-8")
        _enc_size  = struct.unpack("<Q", f.read(8))[0]
        payload    = f.read()

    if verbose:
        print(f"  Codec    : {codec_name}")
        for line in meta_text.strip().splitlines():
            print(f"  {line}")

    encoded = lzma_decompress(payload, label="Aç    ")
    text    = decode_bytes(encoded, slot_to_tok)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


# ── Klasör ────────────────────────────────────────────────────────────────────

def _worker_encode(args: tuple) -> tuple[str, bytes]:
    fpath_str, vocab = args
    fpath     = Path(fpath_str)
    text      = fpath.read_text(encoding="utf-8", errors="ignore")
    automaton = build_automaton(vocab)
    encoded   = encode_bytes(text, automaton)
    return (fpath_str, encoded)


def compress_dir(src: Path, dst: Path, vocab: dict[str, int],
                 codec_name: str, preset: int, jobs: int, verbose: bool):
    files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        print("Hata: Klasörde dosya bulunamadı.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(files)} dosya bulundu, encode ediliyor ({jobs} worker)...")
    args = [(str(f), vocab) for f in files]
    results: list[tuple[str, bytes]] = []

    with Pool(processes=jobs) as pool:
        for item in tqdm(pool.imap(_worker_encode, args),
                         total=len(files), desc="Encode", unit="dosya"):
            results.append(item)

    # Bellek içi arşiv:
    #   [codec_name\0][4B count]
    #   her dosya: [2B yol_len][yol][8B enc_size][encoded]
    #   son giriş: eda_data.txt meta dosyası
    print("  Arşiv birleştiriliyor...")
    meta     = build_meta(codec_name, src, is_dir=True)
    meta_enc = meta.encode("utf-8")

    buf = io.BytesIO()
    codec_name_b = codec_name.encode("utf-8")
    buf.write(struct.pack("<H", len(codec_name_b)))
    buf.write(codec_name_b)
    # dosya sayısı + 1 (meta dahil)
    buf.write(struct.pack("<I", len(results) + 1))

    # meta dosyası ilk sıraya
    meta_path_b = META_NAME.encode("utf-8")
    buf.write(struct.pack("<H", len(meta_path_b)))
    buf.write(meta_path_b)
    buf.write(struct.pack("<Q", len(meta_enc)))
    buf.write(meta_enc)

    for fpath_str, encoded in results:
        rel       = Path(fpath_str).relative_to(src)
        rel_bytes = str(rel).encode("utf-8")
        buf.write(struct.pack("<H", len(rel_bytes)))
        buf.write(rel_bytes)
        buf.write(struct.pack("<Q", len(encoded)))
        buf.write(encoded)
        if verbose:
            print(f"    + {rel}  ({len(encoded):,} B)")

    archive = buf.getvalue()
    print(f"  LZMA sıkıştırılıyor (preset={preset})...")
    payload = lzma_compress(archive, preset, label="Sıkıştır")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(EDA_MAGIC)
        f.write(ENTRY_DIR)
        f.write(payload)

    raw_size = sum(Path(p).stat().st_size for p, _ in results)
    print(f"\n  Ham      : {raw_size / 1e6:.1f} MB")
    print(f"  Arşiv    : {len(archive) / 1e6:.1f} MB")
    print(f"  .eda     : {dst.stat().st_size / 1e6:.1f} MB  "
          f"({100 * dst.stat().st_size / raw_size:.1f}%)")


def decompress_dir(src: Path, dst: Path, codec_path_override: Path | None,
                   verbose: bool):
    with open(src, "rb") as f:
        assert f.read(4) == EDA_MAGIC, "Geçersiz .eda dosyası"
        entry   = f.read(1)
        assert entry == ENTRY_DIR, "Bu .eda tekli dosya (F), klasör beklendi (D)"
        payload = f.read()

    archive = lzma_decompress(payload, label="Aç    ")
    view    = memoryview(archive)
    pos     = 0

    cname_len  = struct.unpack_from("<H", view, pos)[0]; pos += 2
    codec_name = bytes(view[pos:pos + cname_len]).decode("utf-8"); pos += cname_len
    count      = struct.unpack_from("<I", view, pos)[0]; pos += 4

    print(f"  Codec    : {codec_name}")
    print(f"  {count - 1} dosya + meta çıkarılıyor...")

    # codec'i bul
    if codec_path_override:
        codec_path = codec_path_override
    else:
        try:
            codec_path = find_codec(codec_name)
        except FileNotFoundError as e:
            print(f"Hata: {e}", file=sys.stderr)
            sys.exit(1)

    vocab       = load_codec(codec_path)
    slot_to_tok = {v: k for k, v in vocab.items()}

    dst.mkdir(parents=True, exist_ok=True)

    for _ in tqdm(range(count), desc="Çıkar", unit="dosya"):
        rel_len  = struct.unpack_from("<H", view, pos)[0]; pos += 2
        rel_path = bytes(view[pos:pos + rel_len]).decode("utf-8"); pos += rel_len
        enc_size = struct.unpack_from("<Q", view, pos)[0]; pos += 8
        data     = bytes(view[pos:pos + enc_size]); pos += enc_size

        out_path = dst / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if rel_path == META_NAME:
            # meta dosyası: codec encode edilmemiş, ham metin
            out_path.write_bytes(data)
            if verbose:
                print(f"\n--- {META_NAME} ---")
                print(data.decode("utf-8"))
        else:
            text = decode_bytes(data, slot_to_tok)
            out_path.write_text(text, encoding="utf-8")
            if verbose:
                print(f"  → {rel_path}")


# ── summarize ─────────────────────────────────────────────────────────────────

def cmd_summarize(args):
    src = Path(args.source).resolve()
    if not src.exists():
        print(f"Hata: Dosya bulunamadı: {src}", file=sys.stderr)
        sys.exit(1)

    _print_banner("EDA Summarize")
    print(f"  Dosya   : {src}")
    print(f"  Boyut   : {src.stat().st_size:,} B\n")

    with open(src, "rb") as f:
        magic = f.read(4)
        if magic != EDA_MAGIC:
            print("Hata: Geçersiz .eda dosyası", file=sys.stderr)
            sys.exit(1)
        first = f.read(1)
        is_dir = (first == ENTRY_DIR)

    if not is_dir:
        # ── Tekli dosya ──
        hdr  = read_file_header(src)
        meta = hdr["meta"]
        print("  Tür     : Tekli dosya")
        print(f"  Codec   : {hdr['codec']}")
        for k, v in meta.items():
            print(f"  {k:<20}: {v}")
        print(f"\n  Encoded boyut : {hdr['enc_size']:,} B")
        print(f"  LZMA boyutu   : {hdr['lzma_size']:,} B")
        ratio = hdr['lzma_size'] / hdr['enc_size'] * 100 if hdr['enc_size'] else 0
        print(f"  Sıkıştırma    : {ratio:.1f}%")
    else:
        # ── Klasör arşivi — LZMA açılmadan dizin tablosu okunur ──
        with open(src, "rb") as f:
            f.read(4)   # magic
            f.read(1)   # D
            payload = f.read()

        archive = lzma_decompress(payload, label="Aç    ")
        view    = memoryview(archive)
        pos     = 0

        cname_len  = struct.unpack_from("<H", view, pos)[0]; pos += 2
        codec_name = bytes(view[pos:pos + cname_len]).decode("utf-8"); pos += cname_len
        count      = struct.unpack_from("<I", view, pos)[0]; pos += 4

        print("  Tür     : Klasör arşivi")
        print(f"  Codec   : {codec_name}")
        print(f"  Girdi   : {count} kayıt (meta dahil)\n")

        files = []
        meta_text = ""
        for _ in range(count):
            rel_len  = struct.unpack_from("<H", view, pos)[0]; pos += 2
            rel_path = bytes(view[pos:pos + rel_len]).decode("utf-8"); pos += rel_len
            enc_size = struct.unpack_from("<Q", view, pos)[0]; pos += 8
            if rel_path == META_NAME:
                meta_text = bytes(view[pos:pos + enc_size]).decode("utf-8")
            else:
                files.append((rel_path, enc_size))
            pos += enc_size

        if meta_text:
            print("  ── eda_data.txt ──")
            for line in meta_text.strip().splitlines():
                k, _, v = line.partition("=")
                print(f"  {k:<20}: {v}")
            print()

        total_enc = sum(s for _, s in files)
        eda_size  = src.stat().st_size
        print(f"  Dosya sayısı  : {len(files)}")
        print(f"  Encoded toplam: {total_enc / 1e6:.2f} MB")
        print(f"  .eda boyutu   : {eda_size / 1e6:.2f} MB")
        print(f"  Sıkıştırma    : {100 * eda_size / total_enc:.1f}%" if total_enc else "")

        if args.list:
            print(f"\n  {'Dosya':<45} {'Encoded':>12}")
            print(f"  {'-'*45} {'-'*12}")
            for rel, enc in files:
                print(f"  {rel:<45} {enc:>12,} B")


# ── list-codecs ────────────────────────────────────────────────────────────────

def cmd_list_codecs(_args):
    print("=== Mevcut Codec'ler ===")
    print(f"  Dizin: {CODECS_DIR}\n")
    edcs = sorted(CODECS_DIR.glob("*.edc"))
    if not edcs:
        print("  (Hiç codec bulunamadı)")
        return
    for edc in edcs:
        vocab = load_codec(edc)
        size  = edc.stat().st_size
        print(f"  {edc.stem:<20}  {len(vocab):>5} token   {size/1024:.1f} KB   {edc.name}")
    print(f"\n  Toplam: {len(edcs)} codec")


# ── compress ───────────────────────────────────────────────────────────────────

def cmd_compress(args):
    src = Path(args.source).resolve()
    if not src.exists():
        print(f"Hata: Kaynak bulunamadı: {src}", file=sys.stderr)
        sys.exit(1)

    try:
        codec_path = find_codec(args.codec)
    except FileNotFoundError as e:
        print(f"Hata: {e}", file=sys.stderr)
        sys.exit(1)

    codec_name = codec_name_from_path(codec_path)

    if args.output:
        dst = Path(args.output).resolve()
    elif src.is_file():
        dst = src.with_suffix(src.suffix + ".eda")
    else:
        dst = src.parent / (src.name + ".eda")

    jobs = args.jobs or cpu_count()

    _print_banner("EDA Compress")
    print(f"  Kaynak  : {src}")
    print(f"  Çıktı   : {dst}")
    print(f"  Codec   : {codec_name}  ({codec_path.name})")
    print(f"  Preset  : {args.preset}   Jobs: {jobs}\n")

    vocab = load_codec(codec_path)
    print(f"  Codec yüklendi: {len(vocab)} token\n")

    if src.is_file():
        automaton = build_automaton(vocab)
        compress_file(src, dst, automaton, codec_name, args.preset)
        raw = src.stat().st_size
        out = dst.stat().st_size
        print(f"  Ham  : {raw:,} B")
        print(f"  .eda : {out:,} B  ({100*out/raw:.1f}%)")
    else:
        compress_dir(src, dst, vocab, codec_name, args.preset, jobs, args.verbose)

    print(f"\nTamamlandı → {dst}")


# ── decompress ─────────────────────────────────────────────────────────────────

def cmd_decompress(args):
    src = Path(args.source).resolve()
    if not src.exists():
        print(f"Hata: Dosya bulunamadı: {src}", file=sys.stderr)
        sys.exit(1)

    # entry tipini peek et
    with open(src, "rb") as f:
        assert f.read(4) == EDA_MAGIC, "Geçersiz .eda dosyası"
        first = f.read(1)
        is_dir_archive = (first == ENTRY_DIR)

    # -c yalnızca kullanıcı açıkça verdiyse override say
    user_gave_codec = "--codec" in sys.argv or "-c" in sys.argv
    codec_override = None
    if user_gave_codec:
        try:
            codec_override = find_codec(args.codec)
        except FileNotFoundError as e:
            print(f"Hata: {e}", file=sys.stderr)
            sys.exit(1)

    if args.output:
        dst = Path(args.output).resolve()
    elif is_dir_archive:
        # klasör: kaynak_adı/ (eda_data.txt'deki source_name kullanılmaz, klasör adı yeterli)
        dst = src.parent / src.stem
    else:
        # tekli dosya: orijinal isim + uzantıyı header'dan oku
        hdr  = read_file_header(src)
        meta = hdr["meta"]
        name = meta.get("source_name", src.stem)
        ext  = meta.get("source_extension", "")
        dst  = src.parent / (name if name.endswith(ext) else name + ext)

    _print_banner("EDA Decompress")
    print(f"  Kaynak  : {src}")
    print(f"  Çıktı   : {dst}\n")

    if is_dir_archive:
        decompress_dir(src, dst, codec_override, args.verbose)
    else:
        hdr        = read_file_header(src)
        codec_name = hdr["codec"]
        if codec_override:
            codec_path = codec_override
        else:
            try:
                codec_path = find_codec(codec_name)
            except FileNotFoundError as e:
                print(f"Hata: {e}", file=sys.stderr)
                sys.exit(1)

        vocab       = load_codec(codec_path)
        slot_to_tok = {v: k for k, v in vocab.items()}
        print(f"  Codec   : {codec_name}\n")
        decompress_file(src, dst, slot_to_tok, args.verbose)

    print(f"Tamamlandı → {dst}")


# ── Yardımcılar ────────────────────────────────────────────────────────────────

def _print_banner(title: str):
    en = f"EDA — Earth Data Archive  |  {title}"
    tr = f"EDA — Evrensel Dataset Arşivi  |  {title}"
    width = max(len(en), len(tr)) + 4
    bar = "=" * width
    print(bar)
    print(f"  {en}")
    print(f"  {tr}")
    print(bar)


# ── CLI ────────────────────────────────────────────────────────────────────────

HELP_EPILOG = """
────────────────────────────────────────────────────────────
  EDA — Earth Data Archive / Evrensel Dataset Arşivi
────────────────────────────────────────────────────────────

CODEC:
  Codec dosyaları ./codecs/ dizininde bulunur (.edc uzantılı).
  -c seçeneğine sadece adı verin (örn: TR_tr) ya da tam yol.
  Mevcut codec'leri görmek için:  eda list-codecs

DOSYA FORMATI (.eda):
  Tekli dosya  :  [magic][codec_adı][F][meta][lzma(encoded)]
  Klasör arşivi:  [magic][D][lzma([codec_adı][N][meta+dosyalar])]

  Klasör arşivinde kök dizine eda_data.txt eklenir:
    codec=TR_tr
    source_name=wiki_output
    source_type=directory
    archived_at=2026-04-26T10:00:00Z

ÖRNEKLER:
  eda compress dosya.txt                    # varsayılan codec (TR_tr)
  eda compress dosya.txt -c EN_en           # farklı codec
  eda compress klasör/ -o arsiv.eda -j 8   # klasör → tek .eda
  eda compress klasör/ -p 9                 # maksimum sıkıştırma
  eda decompress arsiv.eda                  # otomatik çıkartma
  eda decompress arsiv.eda -o /hedef/       # belirli konum
  eda summarize arsiv.eda                   # içeriği göster (açmadan)
  eda summarize arsiv.eda -l                # + dosya listesi
  eda list-codecs                           # mevcut codec listesi

KISALTMALAR:
  eda c  →  compress
  eda d  →  decompress
  eda s  →  summarize
  eda lc →  list-codecs
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eda",
        description=(
            "EDA — Earth Data Archive / Evrensel Dataset Arşivi\n"
            "Codec tabanlı n-gram ön-kodlama + LZMA sıkıştırma aracı."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )

    sub = parser.add_subparsers(dest="command", required=True,
                                title="komutlar")

    # Ortak seçenekler (parents için)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "-c", "--codec",
        default="TR_tr",
        metavar="CODEC",
        help="Codec adı veya .edc dosya yolu  (varsayılan: TR_tr)",
    )
    shared.add_argument("-o", "--output", default=None, metavar="YOL",
                        help="Çıktı yolu")
    shared.add_argument("-v", "--verbose", action="store_true",
                        help="Ayrıntılı çıktı")

    # ── compress ──
    cp = sub.add_parser(
        "compress", aliases=["c"],
        parents=[shared],
        help="Dosya veya klasörü sıkıştır → .eda",
        description=(
            "Bir dosya veya klasörü EDA formatına sıkıştırır.\n"
            "Klasör modunda tüm alt dosyalar önce codec ile encode edilir,\n"
            "ardından tek bir LZMA akışına paketlenir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  eda compress dosya.txt\n"
            "  eda compress klasör/ -o arsiv.eda -j 8 -p 9\n"
            "  eda c dosya.txt -c TR_tr\n"
        ),
    )
    cp.add_argument("source", metavar="KAYNAK",
                    help="Sıkıştırılacak dosya veya klasör")
    cp.add_argument(
        "-j", "--jobs", type=int, default=None, metavar="N",
        help=f"Paralel worker sayısı  (varsayılan: CPU sayısı = {cpu_count()})",
    )
    cp.add_argument(
        "-p", "--preset", type=int, default=9,
        choices=range(10), metavar="0-9",
        help="LZMA sıkıştırma seviyesi 0(hızlı)–9(maksimum)  (varsayılan: 6)",
    )
    cp.set_defaults(func=cmd_compress)

    # ── decompress ──
    dp = sub.add_parser(
        "decompress", aliases=["d"],
        parents=[shared],
        help=".eda dosyasını aç",
        description=(
            "Bir .eda dosyasını açar. Codec adı arşiv içinden okunur;\n"
            "-c ile override edilebilir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  eda decompress arsiv.eda\n"
            "  eda decompress arsiv.eda -o /hedef/klasör\n"
            "  eda d arsiv.eda -v\n"
        ),
    )
    dp.add_argument("source", metavar="KAYNAK", help=".eda dosyası")
    dp.set_defaults(func=cmd_decompress)

    # ── summarize ──
    sp = sub.add_parser(
        "summarize", aliases=["s"],
        help=".eda dosyasının içeriğini göster (açmadan)",
        description=(
            "Bir .eda dosyasını açmadan meta bilgilerini ve içindeki\n"
            "dosya listesini gösterir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  eda summarize arsiv.eda\n"
            "  eda summarize arsiv.eda -l   # dosya listesiyle\n"
            "  eda s arsiv.eda\n"
        ),
    )
    sp.add_argument("source", metavar="KAYNAK", help=".eda dosyası")
    sp.add_argument("-l", "--list", action="store_true",
                    help="Klasör arşivlerinde dosya listesini göster")
    sp.set_defaults(func=cmd_summarize)

    # ── list-codecs ──
    lc = sub.add_parser(
        "list-codecs", aliases=["lc"],
        help="Mevcut codec'leri listele",
        description="./codecs/ dizinindeki tüm .edc dosyalarını listeler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    lc.set_defaults(func=cmd_list_codecs)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
