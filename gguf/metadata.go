package gguf

import "sort"

func sortStrings(s []string) { sort.Strings(s) }

// String returns the string value for key.
func (f *File) String(key string) (string, bool) {
	v, ok := f.meta[key]
	if !ok || v.Kind != ValString {
		return "", false
	}
	return v.Str, true
}

// Uint returns an unsigned integer metadata value.
func (f *File) Uint(key string) (uint64, bool) {
	v, ok := f.meta[key]
	if !ok {
		return 0, false
	}
	switch v.Kind {
	case ValUint8, ValUint16, ValUint32, ValUint64:
		return v.Uint, true
	case ValInt8, ValInt16, ValInt32, ValInt64:
		return uint64(v.Int), true
	}
	return 0, false
}

// Int returns a signed integer metadata value.
func (f *File) Int(key string) (int64, bool) {
	v, ok := f.meta[key]
	if !ok {
		return 0, false
	}
	switch v.Kind {
	case ValInt8, ValInt16, ValInt32, ValInt64:
		return v.Int, true
	case ValUint8, ValUint16, ValUint32, ValUint64:
		return int64(v.Uint), true
	}
	return 0, false
}

// Float returns a floating-point metadata value.
func (f *File) Float(key string) (float64, bool) {
	v, ok := f.meta[key]
	if !ok || (v.Kind != ValFloat32 && v.Kind != ValFloat64) {
		return 0, false
	}
	return v.Float, true
}

// Bool returns a boolean metadata value.
func (f *File) Bool(key string) (bool, bool) {
	v, ok := f.meta[key]
	if !ok || v.Kind != ValBool {
		return false, false
	}
	return v.Bool, true
}

// Strings returns a string-array metadata value.
func (f *File) Strings(key string) ([]string, bool) {
	v, ok := f.meta[key]
	if !ok || v.Kind != ValArray {
		return nil, false
	}
	return v.Strs, true
}

// Uints returns an unsigned-integer-array metadata value.
func (f *File) Uints(key string) ([]uint64, bool) {
	v, ok := f.meta[key]
	if !ok || v.Kind != ValArray {
		return nil, false
	}
	return v.Uints, true
}

// Floats returns a float-array metadata value.
func (f *File) Floats(key string) ([]float64, bool) {
	v, ok := f.meta[key]
	if !ok || v.Kind != ValArray {
		return nil, false
	}
	return v.Floats, true
}

// ReadRaw returns the tensor payload bytes without dequantisation.
// The returned slice aliases the mmap; callers must not modify it.
func (f *File) ReadRaw(t TensorInfo) ([]byte, error) {
	n, err := t.NBytes()
	if err != nil {
		return nil, err
	}
	if t.Offset > uint64(len(f.data)) || n > uint64(len(f.data))-t.Offset {
		return nil, ErrTruncated
	}
	return f.data[t.Offset : t.Offset+n], nil
}

// ReadRawByName is ReadRaw with a name lookup.
func (f *File) ReadRawByName(name string) ([]byte, error) {
	t, ok := f.Tensor(name)
	if !ok {
		return nil, ErrNoTensor{Name: name}
	}
	return f.ReadRaw(t)
}
