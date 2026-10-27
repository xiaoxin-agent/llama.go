package gguf

import (
	"errors"
	"fmt"
	"strconv"
)

// ErrNoTensor reports a missing tensor lookup.
type ErrNoTensor struct{ Name string }

func (e ErrNoTensor) Error() string { return "gguf: no tensor named " + strconv.Quote(e.Name) }

// ErrUnknownType reports an element type this package cannot size yet.
var ErrUnknownType = errors.New("gguf: unknown element type")

// TypeSize returns the byte size of n elements of type t.
//
// Unknown types return ErrUnknownType together with the type so callers can
// report it. Partial blocks are rejected: GGUF tensor payloads are always a
// whole number of blocks.
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
	}
	return 0, fmt.Errorf("%w: %s", ErrUnknownType, t)
}

// ElementCount returns the total number of elements in t.
func (t TensorInfo) ElementCount() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}
