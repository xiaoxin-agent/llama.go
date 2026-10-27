package gguf

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
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
