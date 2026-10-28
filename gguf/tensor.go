package gguf

import (
	"errors"
	"fmt"
	"strconv"
)

// ErrNoTensor reports a missing tensor lookup.
type ErrNoTensor struct{ Name string }

func (e ErrNoTensor) Error() string { return "gguf: no tensor named " + strconv.Quote(e.Name) }

// ErrUnimplementedType reports a ggml element type this package can name but
// cannot size yet.
//
// It is deliberately distinct from ErrInvalidType. "q4_k is not implemented"
// and "type 47 does not exist" are different facts, and a caller that cannot
// tell them apart will either treat a real ggml type as corrupt data or waste
// time looking for a bug in a file that is fine. The previous single
// ErrUnknownType made both cases print "unknown element type: q6_k", which
// reads as "the file is broken" when the file was correct all along.
var ErrUnimplementedType = errors.New("gguf: element type not implemented")

// ErrInvalidType reports a type tag that is not assigned in ggml.
var ErrInvalidType = errors.New("gguf: invalid element type")

// TypeSize returns the byte size of n elements of type t.
//
// Types that ggml defines but this package does not size return
// ErrUnimplementedType; tags that ggml does not define return ErrInvalidType.
// Both wrap the type so callers can report it. Partial blocks are rejected:
// GGUF tensor payloads are always a whole number of blocks.
func TypeSize(t Type, n uint64) (uint64, error) {
	switch t {
	case TypeF32:
		return 4 * n, nil
	case TypeF16:
		return 2 * n, nil
	case TypeQ8_0:
		if n%32 != 0 {
			return 0, fmt.Errorf("q8_0: element count %d is not a multiple of 32", n)
		}
		return (n / 32) * 34, nil
	case TypeQ4_0:
		if n%32 != 0 {
			return 0, fmt.Errorf("q4_0: element count %d is not a multiple of 32", n)
		}
		return (n / 32) * 18, nil
	case TypeQ6_K:
		// block_q6_K: ql[128] qh[64] scales[16] d(2) = 210 bytes per 256 weights.
		if n%256 != 0 {
			return 0, fmt.Errorf("q6_k: element count %d is not a multiple of 256", n)
		}
		return (n / 256) * 210, nil
	}
	return 0, unsupportedType(t)
}

// unsupportedType classifies a type tag that fell through TypeSize's switch.
// The test is membership in typeNames, the same table String uses, so the two
// cannot disagree about which types ggml defines.
func unsupportedType(t Type) error {
	if _, known := typeNames[t]; known {
		return fmt.Errorf("%w: %s", ErrUnimplementedType, t)
	}
	return fmt.Errorf("%w: %s", ErrInvalidType, t)
}

// ElementCount returns the total number of elements in t.
func (t TensorInfo) ElementCount() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}
