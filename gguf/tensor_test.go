package gguf

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// buildTypedSample writes a one-tensor GGUF v3 file whose single tensor uses
// typ. Payload is one align-sized block of zeros, which is enough for the
// parser's structural checks: what matters here is the type tag, not the data.
func buildTypedSample(t *testing.T, typ Type) string {
	t.Helper()
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.WriteMagic()
	w.WriteVersion(3)
	w.WriteUint64(1) // tensor count
	w.WriteUint64(1) // kv count

	w.WriteString("general.alignment")
	w.WriteUint32(uint32(ValUint32))
	w.WriteUint32(32)

	w.WriteString("t")
	w.WriteUint32(1)  // 1 dimension
	w.WriteUint64(32) // 32 elements
	w.WriteUint32(uint32(typ))
	w.WriteUint64(0) // offset

	w.WritePadding(32)
	w.WriteRaw(make([]byte, 32))
	if err := w.Err(); err != nil {
		t.Fatalf("build typed sample: %v", err)
	}
	p := filepath.Join(t.TempDir(), "typed.gguf")
	if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write sample: %v", err)
	}
	return p
}

// TestTypeSizeErrorClassification pins the difference between a ggml type this
// package has not implemented and a tag ggml never assigned.
//
// The two used to share ErrUnknownType, so a correct file containing q4_k and
// a corrupt file containing type 47 produced the same message. That wording
// ("unknown element type: q6_k") sent the reader looking for a parsing bug when
// the only thing missing was a size formula.
func TestTypeSizeErrorClassification(t *testing.T) {
	// Defined by ggml, named in typeNames, but not sized by TypeSize.
	unimplemented := []Type{TypeQ4_1, TypeQ5_0, TypeQ5_1, TypeQ8_1, TypeQ2_K, TypeQ3_K, TypeQ4_K, TypeQ5_K}
	for _, typ := range unimplemented {
		_, err := TypeSize(typ, 256)
		if !errors.Is(err, ErrUnimplementedType) {
			t.Errorf("TypeSize(%s) error = %v, want ErrUnimplementedType", typ, err)
		}
		if errors.Is(err, ErrInvalidType) {
			t.Errorf("TypeSize(%s) classified as ErrInvalidType; it is a real ggml type", typ)
		}
	}

	// Never assigned in ggml.
	invalid := []Type{Type(15), Type(47), Type(200)}
	for _, typ := range invalid {
		_, err := TypeSize(typ, 256)
		if !errors.Is(err, ErrInvalidType) {
			t.Errorf("TypeSize(%v) error = %v, want ErrInvalidType", typ, err)
		}
		if errors.Is(err, ErrUnimplementedType) {
			t.Errorf("TypeSize(%v) classified as ErrUnimplementedType; ggml never assigned it", typ)
		}
	}

	// The message must carry the type name, so a log line alone is actionable.
	if _, err := TypeSize(TypeQ4_K, 256); err == nil {
		t.Fatal("expected an error for q4_k")
	} else if want := "q4_k"; !contains(err.Error(), want) {
		t.Errorf("error %q does not mention %q", err, want)
	}
}

// TestInvalidTypeRejectsOpen checks that a file with an unassigned type tag is
// refused, while an unimplemented-but-real type still opens. Tolerating the
// former would hide corruption; refusing the latter would make real quantised
// models unopenable.
func TestInvalidTypeRejectsOpen(t *testing.T) {
	path := buildTypedSample(t, Type(47))
	if _, err := Open(path); !errors.Is(err, ErrInvalidType) {
		t.Fatalf("Open with type 47: err = %v, want ErrInvalidType", err)
	}

	path = buildTypedSample(t, TypeQ4_K)
	if _, err := Open(path); err != nil {
		t.Fatalf("Open with q4_k: %v, want success (type is real, just unsized)", err)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
