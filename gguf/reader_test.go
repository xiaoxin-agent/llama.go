package gguf

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// buildSample writes a small, byte-exact GGUF v3 file and returns its path.
// Layout:
//
//	metadata: general.alignment=32, general.name="test", tokenizer.ggml.tokens=["a","bb"]
//	tensors:  "w" f32 [4] = {1,2,3,4}, "q" q8_0 [32]
func buildSample(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.WriteMagic()
	w.WriteVersion(3)
	w.WriteUint64(2) // tensor count
	w.WriteUint64(3) // kv count

	// general.alignment
	w.WriteString("general.alignment")
	w.WriteUint32(uint32(ValUint32))
	w.WriteUint32(32)

	// general.name
	w.WriteString("general.name")
	w.WriteUint32(uint32(ValString))
	w.WriteString("test")

	// tokenizer.ggml.tokens = ["a", "bb"]
	w.WriteString("tokenizer.ggml.tokens")
	w.WriteUint32(uint32(ValArray))
	w.WriteUint32(uint32(ValString))
	w.WriteUint64(2)
	w.WriteString("a")
	w.WriteString("bb")

	// tensor info: w, f32, dims [4]
	w.WriteString("w")
	w.WriteUint32(1)
	w.WriteUint64(4)
	w.WriteUint32(uint32(TypeF32))
	w.WriteUint64(0)

	// tensor info: q, q8_0, dims [32], offset 32 (after w: 4*4=16, padded to 32)
	w.WriteString("q")
	w.WriteUint32(1)
	w.WriteUint64(32)
	w.WriteUint32(uint32(TypeQ8_0))
	w.WriteUint64(32)

	w.WritePadding(32)

	// tensor data
	for _, f := range []float32{1, 2, 3, 4} {
		w.WriteFloat32(f)
	}
	w.WritePadding(32) // pad from 16 to offset 32

	// q8_0 block: fp16 scale 0.5, then 32 int8 values 0..31
	w.WriteRaw([]byte{0x00, 0x38}) // 0.5 in fp16 (little-endian)
	for i := 0; i < 32; i++ {
		w.WriteRaw([]byte{byte(int8(i))})
	}
	if err := w.Err(); err != nil {
		t.Fatalf("build sample: %v", err)
	}

	p := filepath.Join(t.TempDir(), "sample.gguf")
	if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write sample: %v", err)
	}
	return p
}

func TestOpenSample(t *testing.T) {
	p := buildSample(t)
	f, err := Open(p)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	if got := f.Version(); got != 3 {
		t.Errorf("Version = %d, want 3", got)
	}
	if got := f.Alignment(); got != 32 {
		t.Errorf("Alignment = %d, want 32", got)
	}
	if got := f.TensorCount(); got != 2 {
		t.Errorf("TensorCount = %d, want 2", got)
	}
	if name, ok := f.String("general.name"); !ok || name != "test" {
		t.Errorf("general.name = %q,%v want test,true", name, ok)
	}
	toks, ok := f.Strings("tokenizer.ggml.tokens")
	if !ok || len(toks) != 2 || toks[0] != "a" || toks[1] != "bb" {
		t.Errorf("tokens = %v,%v", toks, ok)
	}

	w, ok := f.Tensor("w")
	if !ok {
		t.Fatal("tensor w missing")
	}
	if w.Type != TypeF32 || len(w.Dims) != 1 || w.Dims[0] != 4 {
		t.Errorf("w = %v", w)
	}

	raw, err := f.ReadRaw(w)
	if err != nil {
		t.Fatalf("ReadRaw(w): %v", err)
	}
	if len(raw) != 16 {
		t.Fatalf("len(w) = %d, want 16", len(raw))
	}
}

func TestReadRawQ8(t *testing.T) {
	p := buildSample(t)
	f, err := Open(p)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	q, ok := f.Tensor("q")
	if !ok {
		t.Fatal("tensor q missing")
	}
	raw, err := f.ReadRaw(q)
	if err != nil {
		t.Fatalf("ReadRaw(q): %v", err)
	}
	if len(raw) != 34 {
		t.Fatalf("len(q) = %d, want 34", len(raw))
	}
	if raw[0] != 0x00 || raw[1] != 0x38 {
		t.Errorf("scale bytes = %#x %#x, want 0x00 0x38", raw[0], raw[1])
	}
	if raw[2] != 0 || raw[33] != 31 {
		t.Errorf("quants = %d..%d, want 0..31", int8(raw[2]), int8(raw[33]))
	}
}

func TestDefaultAlignment(t *testing.T) {
	// File without general.alignment must fall back to 32.
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.WriteMagic()
	w.WriteVersion(3)
	w.WriteUint64(0) // no tensors
	w.WriteUint64(0) // no kv
	w.WritePadding(32)

	p := filepath.Join(t.TempDir(), "noalign.gguf")
	if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	f, err := Open(p)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()
	if got := f.Alignment(); got != DefaultAlignment {
		t.Errorf("Alignment = %d, want %d", got, DefaultAlignment)
	}
}

func TestErrors(t *testing.T) {
	dir := t.TempDir()

	t.Run("bad magic", func(t *testing.T) {
		p := filepath.Join(dir, "badmagic.gguf")
		os.WriteFile(p, []byte("NOPE"+"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"), 0o600)
		_, err := Open(p)
		if !errors.Is(err, ErrBadMagic) {
			t.Errorf("err = %v, want ErrBadMagic", err)
		}
	})

	t.Run("unsupported version", func(t *testing.T) {
		var buf bytes.Buffer
		w := NewWriter(&buf)
		w.WriteMagic()
		w.WriteVersion(2)
		w.WriteUint64(0)
		w.WriteUint64(0)
		p := filepath.Join(dir, "v2.gguf")
		os.WriteFile(p, buf.Bytes(), 0o600)
		_, err := Open(p)
		if !errors.Is(err, ErrUnsupportedVersion) {
			t.Errorf("err = %v, want ErrUnsupportedVersion", err)
		}
	})

	t.Run("truncated kv", func(t *testing.T) {
		var buf bytes.Buffer
		w := NewWriter(&buf)
		w.WriteMagic()
		w.WriteVersion(3)
		w.WriteUint64(0)
		w.WriteUint64(1) // claims 1 kv pair but writes none
		p := filepath.Join(dir, "trunc.gguf")
		os.WriteFile(p, buf.Bytes(), 0o600)
		_, err := Open(p)
		if !errors.Is(err, ErrTruncated) {
			t.Errorf("err = %v, want ErrTruncated", err)
		}
	})

	t.Run("missing file", func(t *testing.T) {
		if _, err := Open(filepath.Join(dir, "nope.gguf")); err == nil {
			t.Error("expected error for missing file")
		}
	})
}

func TestTypeSize(t *testing.T) {
	cases := []struct {
		t    Type
		n    uint64
		want uint64
		err  bool
	}{
		{TypeF32, 10, 40, false},
		{TypeF16, 10, 20, false},
		{TypeQ8_0, 32, 34, false},
		{TypeQ8_0, 64, 68, false},
		{TypeQ4_0, 32, 18, false},
		{TypeQ4_0, 64, 36, false},
		{TypeQ4_0, 33, 0, true},
		{TypeQ8_0, 31, 0, true},
		{TypeQ4_K, 256, 0, true},
	}
	for _, c := range cases {
		got, err := TypeSize(c.t, c.n)
		if c.err {
			if err == nil {
				t.Errorf("TypeSize(%s,%d) = %d, want error", c.t, c.n, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("TypeSize(%s,%d): %v", c.t, c.n, err)
		} else if got != c.want {
			t.Errorf("TypeSize(%s,%d) = %d, want %d", c.t, c.n, got, c.want)
		}
	}
}

// TestAlignmentValidation covers hostile and malformed general.alignment
// values. A bad alignment must be rejected, never used in arithmetic: a value
// like 2^63 overflows int64 padding and used to panic on slicing.
func TestAlignmentValidation(t *testing.T) {
	cases := []struct {
		align  uint64
		reject bool
	}{
		{1 << 63, true},
		{^uint64(0), true},
		{1 << 40, true},
		{48, true}, // not a power of two
		{0, true},
		{32, false},
		{16, false},
		{64, false},
	}
	for _, c := range cases {
		var buf bytes.Buffer
		w := NewWriter(&buf)
		w.WriteMagic()
		w.WriteVersion(3)
		w.WriteUint64(0)
		w.WriteUint64(1)
		w.WriteString("general.alignment")
		w.WriteUint32(uint32(ValUint64))
		w.WriteUint64(c.align)
		w.WriteRaw(make([]byte, 64)) // room for alignment padding
		p := filepath.Join(t.TempDir(), "align.gguf")
		if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("alignment %d panicked: %v", c.align, r)
				}
			}()
			f, err := Open(p)
			if err == nil {
				f.Close()
			}
			if c.reject && err == nil {
				t.Errorf("alignment %d: accepted, want rejection", c.align)
			}
			if !c.reject && err != nil {
				t.Errorf("alignment %d: rejected (%v), want accepted", c.align, err)
			}
			if c.reject && err != nil && !strings.Contains(err.Error(), "alignment") {
				t.Errorf("alignment %d: error does not mention alignment: %v", c.align, err)
			}
		}()
	}
}

// TestAlignmentOverflowBoundsCheck drives a file whose tensor-data start would
// exceed the file size, so the uint64 bounds check at the top of the parse tail
// is exercised. The alignment value itself is legal, so this path is reachable
// without any validation help: it is the second layer of defence against a
// negative dataStart and would otherwise be untested.
func TestAlignmentOverflowBoundsCheck(t *testing.T) {
	// One tensor, one dimension of 8 elements of f32 = 32 bytes, but no payload
	// written. The data section start is aligned past the end of the file.
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.WriteMagic()
	w.WriteVersion(3)
	w.WriteUint64(1) // 1 tensor
	w.WriteUint64(1) // 1 kv
	w.WriteString("general.alignment")
	w.WriteUint32(uint32(ValUint32))
	w.WriteUint32(1048576) // legal maximum, forces a large pad
	w.WriteString("w")
	w.WriteUint32(1)                  // n_dims
	w.WriteUint64(8)                  // dim
	w.WriteUint32(uint32(ValFloat32)) // type
	w.WriteUint64(0)                  // offset
	if err := w.Err(); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "bounds.gguf")
	if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked instead of returning an error: %v", r)
			}
		}()
		f, err := Open(p)
		if err == nil {
			f.Close()
			t.Fatal("expected error for data section starting past EOF, got nil")
		}
		if !strings.Contains(err.Error(), "data section starts at") {
			t.Fatalf("wrong error, want data-section bounds failure, got: %v", err)
		}
		t.Logf("rejected as expected: %v", err)
	}()
}

// TestAlignmentZeroPadBoundary ensures a legal alignment that lands exactly on a
// boundary still parses, i.e. the bounds check does not false-reject.
func TestAlignmentZeroPadBoundary(t *testing.T) {
	full := buildAlignedGGUF(t, 32, 0)
	f, err := Open(full)
	if err != nil {
		t.Fatalf("legal file rejected: %v", err)
	}
	defer f.Close()
	if f.Alignment() != 32 {
		t.Fatalf("alignment = %d, want 32", f.Alignment())
	}
}

// buildAlignedGGUF writes a minimal valid file: one f32 tensor of 8 elements,
// with the given alignment, and pads the payload area so the tensor fits.
func buildAlignedGGUF(t *testing.T, align uint64, payloadPad int) string {
	t.Helper()
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.WriteMagic()
	w.WriteVersion(3)
	w.WriteUint64(1) // 1 tensor
	w.WriteUint64(1) // 1 kv
	w.WriteString("general.alignment")
	w.WriteUint32(uint32(ValUint32))
	w.WriteUint32(uint32(align))
	w.WriteString("w")
	w.WriteUint32(1) // n_dims
	w.WriteUint64(8) // dim
	w.WriteUint32(uint32(TypeF32))
	w.WriteUint64(0) // offset
	w.WritePadding(align)
	w.WriteRaw(make([]byte, 32+payloadPad)) // 8 f32 values
	if err := w.Err(); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "aligned.gguf")
	if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestAlignmentUint32NotTruncated ensures a general.alignment stored in a narrow
// integer type is not silently truncated before validation, which would surface
// as a misleading "alignment is zero" error instead of the real value.
func TestAlignmentUint32NotTruncated(t *testing.T) {
	cases := []struct {
		name string
		tag  ValueType
		val  uint64
	}{
		{"uint32 max", ValUint32, 1<<32 - 1},
		{"uint16 max", ValUint16, 1<<16 - 1},
		{"uint64 2^40", ValUint64, 1 << 40},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var buf bytes.Buffer
			w := NewWriter(&buf)
			w.WriteMagic()
			w.WriteVersion(3)
			w.WriteUint64(0) // 0 tensors
			w.WriteUint64(1) // 1 kv
			w.WriteString("general.alignment")
			w.WriteUint32(uint32(c.tag))
			switch c.tag {
			case ValUint8:
				w.write(uint8(c.val))
			case ValUint16:
				w.write(uint16(c.val))
			case ValUint32:
				w.WriteUint32(uint32(c.val))
			case ValUint64:
				w.WriteUint64(c.val)
			}
			w.WriteRaw(make([]byte, 64))
			if err := w.Err(); err != nil {
				t.Fatal(err)
			}
			p := filepath.Join(t.TempDir(), "narrow.gguf")
			if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
				t.Fatal(err)
			}
			f, err := Open(p)
			if err == nil {
				al := f.Alignment()
				f.Close()
				if al != c.val {
					t.Errorf("%s: alignment read back as %d, want %d (truncated)", c.name, al, c.val)
				}
				return
			}
			want := strconv.FormatUint(c.val, 10)
			if !strings.Contains(err.Error(), want) {
				t.Errorf("%s: error does not mention the real value %s: %v", c.name, want, err)
			}
			t.Logf("%s: rejected with real value: %v", c.name, err)
		})
	}
}
