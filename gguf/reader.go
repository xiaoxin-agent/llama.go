package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// TensorInfo describes one tensor entry in the GGUF tensor-info table.
//
// Dims is in GGML order: Dims[0] is the innermost (fastest-varying) dimension.
// So the byte count of element (i0, i1, ...) is the row-major index with the
// LAST dimension varying slowest. For a weight of shape [rows][cols] as written
// by llama.cpp, Dims[0] == cols and Dims[1] == rows.
type TensorInfo struct {
	Name   string
	Dims   []uint64
	Type   Type
	Offset uint64 // relative to the start of the tensor-data section
}

// NBytes returns the size in bytes of the tensor payload given its type.
func (t TensorInfo) NBytes() (uint64, error) {
	n, err := t.ElementCountChecked()
	if err != nil {
		return 0, err
	}
	return TypeSize(t.Type, n)
}

// ElementCountChecked multiplies the dimensions and reports overflow instead of
// wrapping. A declared shape like 2^32 x 2^32 x 2^32 wraps to zero elements,
// which would make a tensor look like a zero-byte payload and pass every bounds
// check while describing something impossible.
func (t TensorInfo) ElementCountChecked() (uint64, error) {
	n := uint64(1)
	for i, d := range t.Dims {
		if d != 0 && n > math.MaxUint64/d {
			return 0, fmt.Errorf("dimension product overflows: dims %v (at dimension %d)", t.Dims, i)
		}
		n *= d
	}
	return n, nil
}

// String renders the tensor for diagnostics.
func (t TensorInfo) String() string {
	return fmt.Sprintf("%s dims=%v type=%s offset=%d", t.Name, t.Dims, t.Type, t.Offset)
}

// File is an open, parsed GGUF file. It reads tensor payloads on demand and
// never loads weights itself.
type File struct {
	path      string
	version   uint32
	alignment uint64
	tensors   map[string]TensorInfo
	order     []string // tensor names in file order
	meta      map[string]Value
	metaOrder []string // metadata keys in file order
	dataStart int64    // absolute file offset of the tensor-data section
	f         *os.File
	data      []byte // tensor-data section (mmap-backed, sub-slice of whole)
	whole     []byte // full-file mapping; the only slice legal for Munmap
	mapLen    int    // length of the whole-file mapping
}

// Value is a decoded metadata value. Exactly one field is meaningful,
// selected by Kind.
type Value struct {
	Kind   ValueType
	Uint   uint64
	Int    int64
	Float  float64
	Bool   bool
	Str    string
	Strs   []string
	Floats []float64
	Uints  []uint64
	Ints   []int64
	Bools  []bool
}

// String renders a scalar value for diagnostics.
func (v Value) String() string {
	switch v.Kind {
	case ValString:
		return v.Str
	case ValArray:
		return fmt.Sprintf("array(len=%d)", len(v.Strs)+len(v.Floats)+len(v.Uints)+len(v.Ints)+len(v.Bools))
	case ValFloat32, ValFloat64:
		// Always keep a decimal point: a reference dump prints 10000.0, and
		// printing 10000 would show up as a spurious difference in a line-by-line
		// comparison even though the parsed value is identical.
		s := strconv.FormatFloat(v.Float, 'g', -1, 64)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return s
	default:
		return fmt.Sprintf("%v", v.any())
	}
}

func (v Value) any() any {
	switch v.Kind {
	case ValUint8, ValUint16, ValUint32, ValUint64:
		return v.Uint
	case ValInt8, ValInt16, ValInt32, ValInt64:
		return v.Int
	case ValFloat32, ValFloat64:
		return v.Float
	case ValBool:
		return v.Bool
	case ValString:
		return v.Str
	}
	return nil
}

// DefaultAlignment is the alignment llama.cpp writes when the metadata key
// "general.alignment" is absent.
const DefaultAlignment = 32

// Open parses path's GGUF header, metadata and tensor table, and maps the
// tensor-data section read-only. Close must be called to release it.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	gf := &File{path: path, f: f}
	if err := gf.parse(); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return gf, nil
}

func (f *File) parse() error {
	st, err := f.f.Stat()
	if err != nil {
		return err
	}
	size := st.Size()

	br := bufio.NewReaderSize(f.f, 1<<16)
	var consumed int64
	read := func(v any) error {
		err := binary.Read(br, binary.LittleEndian, v)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return ErrTruncated
		}
		if err == nil {
			consumed += int64(binary.Size(v))
		}
		return err
	}

	var u32 uint32
	if err := read(&u32); err != nil {
		return err
	}
	if u32 != Magic {
		return ErrBadMagic
	}
	if err := read(&u32); err != nil {
		return err
	}
	f.version = u32
	if f.version != CurrentVersion {
		return fmt.Errorf("%w: got %d, want %d", ErrUnsupportedVersion, f.version, CurrentVersion)
	}

	var tensorCount, kvCount uint64
	if err := read(&tensorCount); err != nil {
		return err
	}
	if err := read(&kvCount); err != nil {
		return err
	}
	// Sanity: each KV needs at least 8 (key len) + 4 (type) bytes, and each
	// tensor needs at least 8 + 4 + 4 + 4 + 8 bytes. Reject obvious garbage
	// before allocating.
	if kvCount > uint64(size) || tensorCount > uint64(size) {
		return fmt.Errorf("%w: implausible counts (tensors=%d, kv=%d, size=%d)", ErrTruncated, tensorCount, kvCount, size)
	}

	f.meta = make(map[string]Value, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(br, &consumed)
		if err != nil {
			return err
		}
		if _, dup := f.meta[key]; dup {
			return fmt.Errorf("duplicate metadata key %q", key)
		}
		v, err := readValue(br, &consumed)
		if err != nil {
			return fmt.Errorf("key %q: %w", key, err)
		}
		if _, dup := f.meta[key]; dup {
			return fmt.Errorf("duplicate metadata key %q", key)
		}
		f.meta[key] = v
		f.metaOrder = append(f.metaOrder, key)
	}

	f.alignment = DefaultAlignment
	if v, ok := f.Uint("general.alignment"); ok {
		// A corrupt or hostile alignment must be rejected, never trusted.
		// It is used as a power-of-two mask for padding and as a divisor, so
		// non-power-of-two or absurd values would overflow the int64 padding
		// arithmetic further down (2^63 wraps negative and defeats the
		// '> size' bounds check, then panics on slicing).
		if v == 0 {
			return fmt.Errorf("general.alignment is zero")
		}
		if v > 1<<20 {
			return fmt.Errorf("general.alignment %d is implausibly large (max %d)", v, 1<<20)
		}
		if v&(v-1) != 0 {
			return fmt.Errorf("general.alignment %d is not a power of two", v)
		}
		f.alignment = v
	}

	f.tensors = make(map[string]TensorInfo, tensorCount)
	f.order = make([]string, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readString(br, &consumed)
		if err != nil {
			return err
		}
		if len(name) > 64 {
			return fmt.Errorf("tensor %d: name longer than 64 bytes", i)
		}
		var nDims uint32
		if err := read(&nDims); err != nil {
			return err
		}
		if nDims == 0 {
			return fmt.Errorf("tensor %q: %d dimensions, want at least 1", name, nDims)
		}
		if nDims > 4 {
			return fmt.Errorf("tensor %q: %d dimensions, max 4", name, nDims)
		}
		dims := make([]uint64, nDims)
		for d := range dims {
			if err := read(&dims[d]); err != nil {
				return err
			}
			if dims[d] == 0 {
				return fmt.Errorf("tensor %q: zero dimension", name)
			}
		}
		var typ uint32
		if err := read(&typ); err != nil {
			return err
		}
		var off uint64
		if err := read(&off); err != nil {
			return err
		}
		if _, dup := f.tensors[name]; dup {
			return fmt.Errorf("duplicate tensor %q", name)
		}
		ti := TensorInfo{Name: name, Dims: dims, Type: Type(typ), Offset: off}
		f.tensors[name] = ti
		f.order = append(f.order, name)
	}

	// The tensor-data section starts at the next alignment boundary after
	// the tensor-info table. Computed in uint64 and bounded before narrowing to
	// int64 so no overflow can reach the slice expression below.
	pos := uint64(consumed)
	pad := (f.alignment - pos%f.alignment) % f.alignment
	start := pos + pad
	if start < pos || start > uint64(size) {
		return fmt.Errorf("%w: data section starts at %d, file is %d bytes", ErrTruncated, start, size)
	}
	f.dataStart = int64(start)

	// Validate every tensor fits inside the file.
	dataLen := size - f.dataStart
	if dataLen == 0 {
		// No tensor payload section at all (metadata-only file). Nothing to
		// map; ReadRaw will fail on any tensor lookup, which is correct since
		// every tensor was already validated against zero available bytes.
		if len(f.order) > 0 {
			return fmt.Errorf("%w: %d tensors declared but data section is empty", ErrTruncated, len(f.order))
		}
		return nil
	}
	for _, name := range f.order {
		ti := f.tensors[name]
		// Alignment and bounds are checked for every tensor, including ones
		// whose element type is not implemented yet: those facts do not depend
		// on knowing the type. Only the payload size does.
		if ti.Offset%f.alignment != 0 {
			return fmt.Errorf("tensor %q: offset %d is not a multiple of alignment %d", name, ti.Offset, f.alignment)
		}
		n, err := ti.NBytes()
		if err != nil {
			if errors.Is(err, ErrUnknownType) {
				// Keep the tensor in the table and let the caller decide.
				// Refusing to open the file would make correctly parsed
				// metadata unreachable just because one element type is
				// unimplemented - which is the situation of every real
				// quantised model, where embeddings stay at Q6_K.
				//
				// The payload cannot be bounds-checked, but tensor payloads are
				// read on demand (ReadRaw) and that path errors on unknown
				// types, so nothing can read past the data section.
				continue
			}
			return fmt.Errorf("tensor %q: %w", name, err)
		}
		if ti.Offset > uint64(dataLen) || n > uint64(dataLen)-ti.Offset {
			return fmt.Errorf("tensor %q: payload [%d,+%d) exceeds data section (%d bytes)", name, ti.Offset, n, dataLen)
		}
	}

	// Reject tensors whose payloads overlap. Every tensor was already checked
	// to fit inside the data section, so two payloads that start at the same
	// offset, or whose ranges are ordered contradictorily, mean the file is
	// corrupt. Reading through such a file returns legal bytes that belong to
	// a different tensor, which silently produces wrong results.
	if err := f.checkNoOverlap(); err != nil {
		return err
	}

	// Note: syscall.Mmap on Linux requires a zero offset. Map the whole file
	// and reach the tensor-data section by slicing, so the offsets we report
	// stay identical to the on-disk layout.
	whole, err := syscall.Mmap(int(f.f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("mmap %s: %w", f.path, err)
	}
	f.mapLen = int(size)
	f.whole = whole
	f.data = whole[f.dataStart:]
	return nil
}

func readString(br *bufio.Reader, consumed *int64) (string, error) {
	var n uint64
	if err := binary.Read(br, binary.LittleEndian, &n); err != nil {
		return "", ErrTruncated
	}
	*consumed += 8
	if n > 1<<28 {
		return "", fmt.Errorf("%w: string length %d too large", ErrTruncated, n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(br, buf); err != nil {
		return "", ErrTruncated
	}
	*consumed += int64(n)
	return string(buf), nil
}

func readValue(br *bufio.Reader, consumed *int64) (Value, error) {
	var tag uint32
	if err := binary.Read(br, binary.LittleEndian, &tag); err != nil {
		return Value{}, ErrTruncated
	}
	*consumed += 4
	v, err := readValueBody(br, consumed, ValueType(tag))
	if err != nil {
		return Value{}, err
	}
	v.Kind = ValueType(tag)
	return v, nil
}

func readValueBody(br *bufio.Reader, consumed *int64, t ValueType) (Value, error) {
	var v Value
	switch t {
	case ValUint8, ValUint16, ValUint32, ValUint64:
		var tmp uint64
		var n int
		switch t {
		case ValUint8:
			var b uint8
			n = binary.Size(b)
			err := binary.Read(br, binary.LittleEndian, &b)
			if err != nil {
				return v, ErrTruncated
			}
			tmp = uint64(b)
		case ValUint16:
			var b uint16
			n = binary.Size(b)
			if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
				return v, ErrTruncated
			}
			tmp = uint64(b)
		case ValUint32:
			var b uint32
			n = binary.Size(b)
			if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
				return v, ErrTruncated
			}
			tmp = uint64(b)
		case ValUint64:
			var b uint64
			n = binary.Size(b)
			if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
				return v, ErrTruncated
			}
			tmp = b
		}
		*consumed += int64(n)
		v.Uint = tmp
	case ValInt8, ValInt16, ValInt32, ValInt64:
		var tmp int64
		var n int
		switch t {
		case ValInt8:
			var b int8
			n = binary.Size(b)
			if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
				return v, ErrTruncated
			}
			tmp = int64(b)
		case ValInt16:
			var b int16
			n = binary.Size(b)
			if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
				return v, ErrTruncated
			}
			tmp = int64(b)
		case ValInt32:
			var b int32
			n = binary.Size(b)
			if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
				return v, ErrTruncated
			}
			tmp = int64(b)
		case ValInt64:
			var b int64
			n = binary.Size(b)
			if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
				return v, ErrTruncated
			}
			tmp = b
		}
		*consumed += int64(n)
		v.Int = tmp
	case ValFloat32:
		var b float32
		if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
			return v, ErrTruncated
		}
		*consumed += 4
		v.Float = float64(b)
	case ValFloat64:
		var b float64
		if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
			return v, ErrTruncated
		}
		*consumed += 8
		v.Float = b
	case ValBool:
		var b uint8
		if err := binary.Read(br, binary.LittleEndian, &b); err != nil {
			return v, ErrTruncated
		}
		*consumed++
		if b > 1 {
			return v, fmt.Errorf("invalid bool byte %d", b)
		}
		v.Bool = b == 1
	case ValString:
		s, err := readString(br, consumed)
		if err != nil {
			return v, err
		}
		v.Str = s
	case ValArray:
		var elemTag uint32
		if err := binary.Read(br, binary.LittleEndian, &elemTag); err != nil {
			return v, ErrTruncated
		}
		*consumed += 4
		var n uint64
		if err := binary.Read(br, binary.LittleEndian, &n); err != nil {
			return v, ErrTruncated
		}
		*consumed += 8
		if n > 1<<28 {
			return v, fmt.Errorf("%w: array length %d too large", ErrTruncated, n)
		}
		et := ValueType(elemTag)
		for i := uint64(0); i < n; i++ {
			e, err := readValueBody(br, consumed, et)
			if err != nil {
				return v, fmt.Errorf("array[%d] (elem %s): %w", i, et, err)
			}
			switch et {
			case ValString:
				v.Strs = append(v.Strs, e.Str)
			case ValFloat32, ValFloat64:
				v.Floats = append(v.Floats, e.Float)
			case ValUint8, ValUint16, ValUint32, ValUint64:
				v.Uints = append(v.Uints, e.Uint)
			case ValInt8, ValInt16, ValInt32, ValInt64:
				v.Ints = append(v.Ints, e.Int)
			case ValBool:
				v.Bools = append(v.Bools, e.Bool)
			default:
				return v, fmt.Errorf("unsupported array element type %s", et)
			}
		}
	default:
		return v, fmt.Errorf("unknown value type %s", t)
	}
	return v, nil
}

// Close unmaps the data section and closes the file.
func (f *File) Close() error {
	var errs []error
	if f.mapLen > 0 && f.f != nil {
		// Munmap needs the original whole-file mapping, not the sub-slice.
		whole := f.whole
		if err := syscall.Munmap(whole); err != nil {
			errs = append(errs, err)
		}
		f.whole = nil
		f.data = nil
		f.mapLen = 0
	}
	if f.f != nil {
		if err := f.f.Close(); err != nil {
			errs = append(errs, err)
		}
		f.f = nil
	}
	return errors.Join(errs...)
}

// Version returns the GGUF format version.
func (f *File) Version() uint32 { return f.version }

// Path returns the file path given to Open.
func (f *File) Path() string { return f.path }

// Alignment returns the tensor-data alignment in bytes.
func (f *File) Alignment() uint64 { return f.alignment }

// DataStart returns the absolute file offset where the tensor-data section
// begins, i.e. the first byte after the aligned end of the metadata and
// tensor-info tables. Exposed so callers can verify layout accounting.
func (f *File) DataStart() int64 { return f.dataStart }

// Has reports whether the metadata key is present.
func (f *File) Has(key string) bool {
	_, ok := f.meta[key]
	return ok
}

// TensorCount returns the number of tensors.
func (f *File) TensorCount() int { return len(f.order) }

// TensorNames returns tensor names in file order.
func (f *File) TensorNames() []string { return append([]string(nil), f.order...) }

// Tensor looks up a tensor by name.
func (f *File) Tensor(name string) (TensorInfo, bool) {
	t, ok := f.tensors[name]
	return t, ok
}

// MetadataKeys returns metadata keys in file order.
//
// File order matters: it is part of the on-disk format, reference dumps such as
// gguf_dump emit keys in the same order, and a sorted list cannot be compared
// line by line against them. Use SortedMetadataKeys when a deterministic
// ordering independent of the file is wanted.
func (f *File) MetadataKeys() []string {
	return append([]string(nil), f.metaOrder...)
}

// SortedMetadataKeys returns metadata keys sorted lexicographically.
func (f *File) SortedMetadataKeys() []string {
	keys := append([]string(nil), f.metaOrder...)
	sortStrings(keys)
	return keys
}

// Lookup returns the raw metadata value for key.
func (f *File) Lookup(key string) (Value, bool) {
	v, ok := f.meta[key]
	return v, ok
}

// checkNoOverlap rejects files where two tensor payloads claim overlapping
// byte ranges in the data section. Tensors are visited in declaration order;
// each range is compared against every earlier one. The tensor count is small
// (hundreds at most), so the quadratic scan is not worth optimizing.
func (f *File) checkNoOverlap() error {
	type span struct {
		name string
		start,
		end uint64
	}
	spans := make([]span, 0, len(f.order))
	for _, name := range f.order {
		ti := f.tensors[name]
		n, err := ti.NBytes()
		if err != nil {
			// Unknown element type: the payload size is unknown, so this
			// tensor cannot participate in an overlap check. Reading it is
			// already refused by ReadRaw.
			continue
		}
		start, end := ti.Offset, ti.Offset+n
		for _, s := range spans {
			if start < s.end && s.start < end {
				return fmt.Errorf("tensor %q payload [%d,%d) overlaps tensor %q payload [%d,%d)",
					name, start, end, s.name, s.start, s.end)
			}
		}
		spans = append(spans, span{name: name, start: start, end: end})
	}
	return nil
}
