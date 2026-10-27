package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// Magic is the four-byte GGUF file signature: 'G' 'G' 'U' 'F'.
const Magic uint32 = 0x46554747

// CurrentVersion is the GGUF spec version implemented by this package.
const CurrentVersion = 3

var (
	// ErrBadMagic means the file does not start with "GGUF".
	ErrBadMagic = errors.New("gguf: bad magic, not a GGUF file")
	// ErrUnsupportedVersion means the file version is newer or older than supported.
	ErrUnsupportedVersion = errors.New("gguf: unsupported version")
	// ErrTruncated means the file ended before the declared structure did.
	ErrTruncated = errors.New("gguf: truncated file")
)

// Type is the ggml tensor element type.
//
// Values match ggml_type in ggml.h and MUST NOT be renumbered: they appear
// verbatim in GGUF files.
type Type uint32

const (
	TypeF32  Type = 0
	TypeF16  Type = 1
	TypeQ4_0 Type = 2
	TypeQ4_1 Type = 3
	TypeQ5_0 Type = 6
	TypeQ5_1 Type = 7
	TypeQ8_0 Type = 8
	TypeQ8_1 Type = 9
	TypeQ2_K Type = 10
	TypeQ3_K Type = 11
	TypeQ4_K Type = 12
	TypeQ5_K Type = 13
	TypeQ6_K Type = 14
)

// String returns the ggml type name, or "type(<n>)" for unknown values.
func (t Type) String() string {
	if s, ok := typeNames[t]; ok {
		return s
	}
	return fmt.Sprintf("type(%d)", uint32(t))
}

var typeNames = map[Type]string{
	TypeF32:  "f32",
	TypeF16:  "f16",
	TypeQ4_0: "q4_0",
	TypeQ4_1: "q4_1",
	TypeQ5_0: "q5_0",
	TypeQ5_1: "q5_1",
	TypeQ8_0: "q8_0",
	TypeQ8_1: "q8_1",
	TypeQ2_K: "q2_k",
	TypeQ3_K: "q3_k",
	TypeQ4_K: "q4_k",
	TypeQ5_K: "q5_k",
	TypeQ6_K: "q6_k",
}

// ValueType is the GGUF metadata value tag.
type ValueType uint32

const (
	ValUint8   ValueType = 0
	ValInt8    ValueType = 1
	ValUint16  ValueType = 2
	ValInt16   ValueType = 3
	ValUint32  ValueType = 4
	ValInt32   ValueType = 5
	ValFloat32 ValueType = 6
	ValBool    ValueType = 7
	ValString  ValueType = 8
	ValArray   ValueType = 9
	ValUint64  ValueType = 10
	ValInt64   ValueType = 11
	ValFloat64 ValueType = 12
)

var valueTypeNames = map[ValueType]string{
	ValUint8:   "uint8",
	ValInt8:    "int8",
	ValUint16:  "uint16",
	ValInt16:   "int16",
	ValUint32:  "uint32",
	ValInt32:   "int32",
	ValFloat32: "float32",
	ValBool:    "bool",
	ValString:  "string",
	ValArray:   "array",
	ValUint64:  "uint64",
	ValInt64:   "int64",
	ValFloat64: "float64",
}

func (v ValueType) String() string {
	if s, ok := valueTypeNames[v]; ok {
		return s
	}
	return fmt.Sprintf("valuetype(%d)", uint32(v))
}

// Writer serialises GGUF v3 structures in little-endian order.
//
// It exists so tests and the reference-fixture generator can build small,
// byte-exact GGUF files without depending on the reader.
type Writer struct {
	w   io.Writer
	err error
	// n counts bytes successfully written; callers use it to compute
	// alignment padding.
	n int64
}

// NewWriter returns a Writer that emits to w.
func NewWriter(w io.Writer) *Writer { return &Writer{w: w} }

// Err returns the first error encountered, if any.
func (w *Writer) Err() error { return w.err }

// Offset returns the number of bytes written so far.
func (w *Writer) Offset() int64 { return w.n }

func (w *Writer) write(v any) {
	if w.err != nil {
		return
	}
	before := w.n
	switch t := v.(type) {
	case uint8:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n++
	case int8:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n++
	case uint16:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n += 2
	case int16:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n += 2
	case uint32:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n += 4
	case int32:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n += 4
	case uint64:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n += 8
	case int64:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n += 8
	case float32:
		w.err = binary.Write(w.w, binary.LittleEndian, t)
		w.n += 4
	case []byte:
		var n int
		n, w.err = w.w.Write(t)
		w.n += int64(n)
	default:
		w.err = fmt.Errorf("gguf: writer cannot encode %T", v)
	}
	if w.err != nil && w.n != before {
		// Partial write: keep the count but stop accepting more data.
		return
	}
}

// WriteMagic writes the file magic.
func (w *Writer) WriteMagic() { w.write(Magic) }

// WriteVersion writes the format version.
func (w *Writer) WriteVersion(v uint32) { w.write(v) }

// WriteUint32 writes a little-endian uint32.
func (w *Writer) WriteUint32(v uint32) { w.write(v) }

// WriteUint64 writes a little-endian uint64.
func (w *Writer) WriteUint64(v uint64) { w.write(v) }

// WriteRaw appends raw bytes without any framing.
func (w *Writer) WriteRaw(b []byte) { w.write(b) }

// WriteFloat32 writes a little-endian float32.
func (w *Writer) WriteFloat32(v float32) { w.write(v) }

// WriteString writes a GGUF string: uint64 length followed by UTF-8 bytes.
func (w *Writer) WriteString(s string) {
	w.write(uint64(len(s)))
	w.write([]byte(s))
}

// WritePadding writes zero bytes until the offset is a multiple of align.
func (w *Writer) WritePadding(align uint64) {
	if w.err != nil || align == 0 {
		return
	}
	rem := uint64(w.n) % align
	if rem == 0 {
		return
	}
	pad := make([]byte, align-rem)
	w.write(pad)
}
