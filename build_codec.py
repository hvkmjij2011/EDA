#!/usr/bin/env python3
"""
EDA — Evrensel Dataset Arşivi
build_codec.py — Codec builder CLI

Kullanım:
  build_codec.py <kaynak> <codec_adı> [seçenekler]

  <kaynak>     : Düz metin dosyası veya dizin
  <codec_adı>  : Çıktı codec adı (örn: TR_tr, EN_en)
"""

import argparse
import gc
import json
import struct
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

FLAG_BYTE = 0x01
MAX_SLOTS = 4096
MIN_LEN   = 3
MAX_LEN   = 16

CODECS_DIR = Path(__file__).parent / "codecs"

# Düz metin tespiti için eşikler
TEXT_SAMPLE_BYTES  = 8192    # ilk N byte'a bakılır
TEXT_MIN_PRINTABLE = 0.85    # en az %85 yazdırılabilir karakter
TEXT_MAX_NULL      = 0.01    # en fazla %1 null byte


# ── Düz metin tespiti ──────────────────────────────────────────────────────────

def is_text_file(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:TEXT_SAMPLE_BYTES]
    except OSError:
        return False
    if not sample:
        return False
    null_ratio = sample.count(0) / len(sample)
    if null_ratio > TEXT_MAX_NULL:
        return False
    printable = sum(
        1 for b in sample
        if (0x09 <= b <= 0x0D) or (0x20 <= b <= 0x7E) or b >= 0x80
    )
    return (printable / len(sample)) >= TEXT_MIN_PRINTABLE


# ── Kaynak dosyaları topla ────────────────────────────────────────────────────

def collect_files(source: Path) -> list[Path]:
    if source.is_file():
        if not is_text_file(source):
            print(f"Hata: '{source}' düz metin dosyası değil.", file=sys.stderr)
            sys.exit(1)
        return [source]

    all_files  = sorted(p for p in source.rglob("*") if p.is_file())
    text_files = [p for p in all_files if is_text_file(p)]
    skipped    = len(all_files) - len(text_files)

    if skipped:
        print(f"  {skipped} ikili/boş dosya atlandı.")
    if not text_files:
        print(f"Hata: '{source}' içinde düz metin dosyası bulunamadı.", file=sys.stderr)
        sys.exit(1)

    return text_files


# ── Slot hesabı ───────────────────────────────────────────────────────────────

def is_valid_slot(token_id: int) -> bool:
    if (token_id >> 4)  == FLAG_BYTE: return False
    if (token_id & 0xFF) == FLAG_BYTE: return False
    if (token_id >> 8)  == 0x1:       return False
    if (token_id & 0xF) == 0x0:       return False
    return True


def get_valid_slots() -> list[int]:
    valid = [i for i in range(MAX_SLOTS) if is_valid_slot(i)]
    print(f"  Geçerli slot: {len(valid)} / {MAX_SLOTS}")
    return valid


# ── Slot kotası ───────────────────────────────────────────────────────────────

def compute_quotas(n_parts: int, valid_slots: list[int]) -> list[tuple[int, int]]:
    base   = len(valid_slots) // n_parts
    quotas = []
    start  = 0
    for i in range(n_parts):
        count = (len(valid_slots) - start) if i == n_parts - 1 else base
        quotas.append((start, count))
        start += count
    return quotas


# ── RAM tahmini ───────────────────────────────────────────────────────────────

def available_ram_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e6
    except ImportError:
        # /proc/meminfo fallback
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 2000.0   # bilinmiyorsa 2 GB varsay


def dynamic_prune_threshold(counter_len: int) -> int:
    """
    Mevcut RAM'e göre counter'ı prune etmek için min_freq eşiği döner.
    RAM azaldıkça daha agresif prune yapılır.
    """
    ram = available_ram_mb()
    if ram > 4000:
        return 2    # bolca RAM: sadece tekil olanları at
    elif ram > 2000:
        return 3
    elif ram > 1000:
        return 5
    elif ram > 500:
        return 8
    else:
        return 15   # kritik: agresif


# ── Worker ────────────────────────────────────────────────────────────────────

def worker_count(args: tuple) -> Counter:
    """
    Bir dosya grubunu tarar, n-gram Counter döner.
    RAM durumuna göre dinamik prune eşiği kullanır.
    """
    file_paths, min_freq = args
    counter: Counter = Counter()

    for fpath_str in tqdm(file_paths, desc=Path(file_paths[0]).parent.name,
                          position=0, leave=False, unit="dosya"):
        fpath = Path(fpath_str)
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        length = len(text)
        for n in range(MIN_LEN, min(MAX_LEN, length) + 1):
            for i in range(length - n + 1):
                counter[text[i: i + n]] += 1

        # Dinamik prune: counter büyüdükçe RAM'e bak
        if len(counter) > 2_000_000:
            threshold = dynamic_prune_threshold(len(counter))
            counter   = Counter({k: v for k, v in counter.items()
                                 if v >= threshold})
            gc.collect()

    if min_freq > 1:
        counter = Counter({k: v for k, v in counter.items() if v >= min_freq})

    print(f"  [{Path(file_paths[0]).parent.name}] {len(counter):,} n-gram", flush=True)
    return counter


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_counters(counters: list[Counter]) -> Counter:
    merged: Counter = Counter()
    for c in counters:
        merged.update(c)
    return merged


# ── Vocab ─────────────────────────────────────────────────────────────────────

def build_vocab_partitioned(
    counters: list[Counter],
    valid_slots: list[int],
    quotas: list[tuple[int, int]],
) -> dict:
    vocab: dict = {}
    carry = 0

    for part_idx, (counter, (slot_start, slot_count)) in enumerate(
        zip(counters, quotas)
    ):
        alloc   = slot_count + carry
        carry   = 0
        slots   = valid_slots[slot_start: slot_start + alloc]
        slot_it = iter(slots)
        filled  = 0

        for gram, _ in counter.most_common():
            if gram in vocab:
                continue
            try:
                slot_id = next(slot_it)
            except StopIteration:
                break
            vocab[gram] = slot_id
            filled += 1

        shortfall = alloc - filled
        if shortfall > 0:
            carry += shortfall
            print(f"  Parça {part_idx}: {filled}/{alloc} slot doldu, "
                  f"{shortfall} carry-over")
        else:
            print(f"  Parça {part_idx}: {filled}/{alloc} slot doldu")

    if carry > 0:
        print(f"  Carry-over temizleme: {carry} slot kaldı, global merge'den alınıyor")
        global_counter  = merge_counters(counters)
        remaining_slots = valid_slots[len(vocab):]
        slot_it         = iter(remaining_slots)
        for gram, _ in global_counter.most_common():
            if gram in vocab:
                continue
            try:
                vocab[gram] = next(slot_it)
            except StopIteration:
                break

    print(f"  Toplam vocab: {len(vocab)} token")
    return vocab


# ── Codec kaydet ──────────────────────────────────────────────────────────────

def save_codec(vocab: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"EDAC")
        f.write(struct.pack("B", 1))
        f.write(struct.pack("B", FLAG_BYTE))
        f.write(struct.pack("<H", len(vocab)))
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            tb = token.encode("utf-8")
            f.write(struct.pack("<H", token_id))
            f.write(struct.pack("B", len(tb)))
            f.write(tb)
    print(f"  .edc  → {output_path}  ({output_path.stat().st_size:,} byte)")

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"flag_byte": hex(FLAG_BYTE),
             "vocab": {str(v): k for k, v in vocab.items()}},
            f, ensure_ascii=False, indent=2,
        )
    print(f"  .json → {json_path}")


# ── Dosyaları worker'lara böl ─────────────────────────────────────────────────

def partition_files(files: list[Path], n: int) -> list[list[str]]:
    """
    Dosyaları n gruba böler.
    Boyuta göre sıralar, büyük dosyalar farklı gruplara dağılsın.
    """
    sorted_files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    groups: list[list[str]] = [[] for _ in range(n)]
    for i, f in enumerate(sorted_files):
        groups[i % n].append(str(f))
    return [g for g in groups if g]   # boş grupları at


# ── CLI ───────────────────────────────────────────────────────────────────────

HELP_EPILOG = """
────────────────────────────────────────────────────────────
  EDA — Earth Data Archive / Evrensel Dataset Arşivi
  Codec Builder
────────────────────────────────────────────────────────────

KAYNAK:
  Tek dosya veya dizin verilebilir.
  Dizin verilirse tüm alt dosyalar taranır.
  İkili/boş dosyalar otomatik atlanır.
  Hiç düz metin bulunamazsa hata verilir.

ÇIKTI:
  Codec ./codecs/<ad>.edc ve ./codecs/<ad>.json olarak kaydedilir.
  -o ile farklı dizin belirtilebilir.

RAM YÖNETİMİ:
  Worker sayısı her zaman CPU sayısına eşittir.
  Counter prune eşiği mevcut RAM'e göre dinamik ayarlanır:
    > 4 GB  → min_freq = 2
    > 2 GB  → min_freq = 3
    > 1 GB  → min_freq = 5
    > 500MB → min_freq = 8
    kritik  → min_freq = 15

ÖRNEKLER:
  build_codec.py corpus/ TR_tr
  build_codec.py metin.txt EN_en -f 3
  build_codec.py /data/wiki DE_de -o /data/codecs
  build_codec.py corpus/ TR_tr --min-len 2 --max-len 12
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_codec",
        description=(
            "EDA — Codec Builder\n"
            "Metin corpus'undan n-gram tabanlı .edc codec dosyası üretir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    parser.add_argument("source", metavar="KAYNAK",
                        help="Metin dosyası veya dizin")
    parser.add_argument("name", metavar="CODEC_ADI",
                        help="Codec adı (örn: TR_tr, EN_en)")
    parser.add_argument("-o", "--output", default=None, metavar="DİZİN",
                        help=f"Çıktı dizini  (varsayılan: ./codecs/)")
    parser.add_argument("-f", "--min-freq", type=int, default=5, metavar="N",
                        help="Minimum n-gram frekansı  (varsayılan: 5)")
    parser.add_argument("--min-len", type=int, default=MIN_LEN, metavar="N",
                        help=f"Minimum token uzunluğu  (varsayılan: {MIN_LEN})")
    parser.add_argument("--max-len", type=int, default=MAX_LEN, metavar="N",
                        help=f"Maksimum token uzunluğu  (varsayılan: {MAX_LEN})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Ayrıntılı slot kota çıktısı")
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Global parametreleri güncelle
    global MIN_LEN, MAX_LEN
    MIN_LEN = args.min_len
    MAX_LEN = args.max_len

    n_workers = cpu_count()
    source    = Path(args.source).resolve()
    out_dir   = Path(args.output).resolve() if args.output else CODECS_DIR
    out_path  = out_dir / (args.name + ".edc")

    en = "EDA — Earth Data Archive  |  Codec Builder"
    tr = "EDA — Evrensel Dataset Arşivi  |  Codec Builder"
    bar = "=" * (max(len(en), len(tr)) + 4)
    print(bar)
    print(f"  {en}")
    print(f"  {tr}")
    print(bar)
    print(f"\n  Kaynak   : {source}")
    print(f"  Codec    : {args.name}  →  {out_path}")
    print(f"  Workers  : {n_workers} (CPU sayısı)")
    print(f"  min_freq : {args.min_freq}")
    print(f"  n-gram   : {MIN_LEN}–{MAX_LEN}\n")

    # 1) Dosyaları topla
    print("[1/4] Dosyalar taranıyor...")
    files = collect_files(source)
    total_size = sum(f.stat().st_size for f in files)
    print(f"  {len(files)} düz metin dosyası  ({total_size / 1e6:.1f} MB)")

    # 2) Slot hesabı
    print("\n[2/4] Geçerli 12-bit slot'lar hesaplanıyor...")
    valid_slots = get_valid_slots()

    # 3) Dosyaları worker'lara böl
    groups  = partition_files(files, n_workers)
    n_parts = len(groups)
    print(f"\n[3/4] Slot kotaları hesaplanıyor ({n_parts} grup, {n_workers} worker)...")
    quotas = compute_quotas(n_parts, valid_slots)
    if args.verbose:
        for i, (s, c) in enumerate(quotas):
            print(f"  Grup {i}: slot[{s}..{s+c-1}] ({c} slot)  "
                  f"— {len(groups[i])} dosya")

    # 4) Corpus tara
    print(f"\n[4/4] Corpus taranıyor ({n_workers} paralel worker)...")
    ram_mb = available_ram_mb()
    print(f"  Mevcut RAM: {ram_mb:.0f} MB")

    worker_args = [(group, args.min_freq) for group in groups]

    with Pool(processes=n_workers) as pool:
        counters = list(tqdm(
            pool.imap(worker_count, worker_args),
            total=len(worker_args),
            desc="Gruplar",
            unit="grup",
        ))

    # 5) Vocab & kaydet
    print("\n[5/4] Vocab oluşturuluyor ve kaydediliyor...")
    vocab = build_vocab_partitioned(counters, valid_slots, quotas)
    save_codec(vocab, out_path)

    print(f"\nTamamlandı → {out_path}")


if __name__ == "__main__":
    main()
