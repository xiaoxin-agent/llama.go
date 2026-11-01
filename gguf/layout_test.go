package gguf

import (
	"os"
	"testing"
)

// TestRealFileLayoutSum checks a property that no per-tensor assertion can:
// the declared payload sizes must add up to exactly the bytes that follow the
// tensor-data offset.
//
// If one TypeSize formula is wrong the sum is short or long by that amount. If
// the offset table skips or repeats a region, the contiguous walk catches it.
// A per-tensor ReadRaw test cannot detect either: reading one tensor only
// proves that tensor's extent is readable, and a size that is too small still
// yields a short, plausible-looking payload.
//
// The model path comes from LLAMA_GO_TEST_GGUF. Set it to any .gguf you want
// the invariant applied to; CI points it at the small Q8_0 model. The check is
// skipped when unset so the package still tests standalone.
func TestRealFileLayoutSum(t *testing.T) {
	path := os.Getenv("LLAMA_GO_TEST_GGUF")
	if path == "" {
		t.Skip("set LLAMA_GO_TEST_GGUF to run layout checks against a real model")
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	fileSize := info.Size()

	f, err := Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	dataStart := f.DataStart()
	if dataStart < 0 || dataStart > fileSize {
		t.Fatalf("data offset %d outside file of %d bytes", dataStart, fileSize)
	}
	tail := fileSize - dataStart

	// 1. Every tensor must resolve to a size, and those sizes must sum to the
	//    tail exactly.
	names := f.TensorNames()
	if len(names) != f.TensorCount() {
		t.Fatalf("TensorNames returned %d entries, TensorCount is %d",
			len(names), f.TensorCount())
	}

	byOffset := make([]TensorInfo, 0, len(names))
	var sum uint64
	for _, name := range names {
		ti, ok := f.Tensor(name)
		if !ok {
			t.Fatalf("tensor %q listed but not found", name)
		}
		nb, err := ti.NBytes()
		if err != nil {
			t.Fatalf("tensor %q (%s dims=%v): NBytes: %v",
				ti.Name, ti.Type, ti.Dims, err)
		}
		if ti.Offset+nb > uint64(tail) {
			t.Fatalf("tensor %q: offset %d + size %d exceeds data section of %d bytes",
				ti.Name, ti.Offset, nb, tail)
		}
		byOffset = append(byOffset, ti)
		sum += nb
	}
	if sum != uint64(tail) {
		t.Fatalf("tensor payloads sum to %d bytes, data section is %d (difference %d); "+
			"a TypeSize formula or the offset table disagrees with the file",
			sum, tail, int64(tail)-int64(sum))
	}

	// 2. Walking the tensors in offset order must be gapless and in-bounds.
	//    This is what rules out a size that is too small: the sum could still
	//    match if another tensor absorbed the slack, but a gap cannot hide.
	sortByOffset(byOffset)
	var cursor uint64
	for _, ti := range byOffset {
		if ti.Offset != cursor {
			t.Fatalf("tensor %q starts at offset %d, expected %d (gap or overlap)",
				ti.Name, ti.Offset, cursor)
		}
		nb, err := ti.NBytes()
		if err != nil {
			t.Fatalf("tensor %q: NBytes: %v", ti.Name, err)
		}
		cursor += nb
	}
	if cursor != uint64(tail) {
		t.Fatalf("contiguous walk ended at %d, data section is %d", cursor, tail)
	}

	// 3. Offsets respect the declared alignment.
	align := f.Alignment()
	for _, ti := range byOffset {
		if align > 0 && ti.Offset%align != 0 {
			t.Fatalf("tensor %q offset %d is not %d-byte aligned",
				ti.Name, ti.Offset, align)
		}
	}
}

// sortByOffset orders tensors by their data-section offset. Insertion sort is
// deliberate: the slice is already in file order in practice, and a
// dependency-free helper keeps this test readable.
func sortByOffset(s []TensorInfo) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j].Offset < s[j-1].Offset; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
