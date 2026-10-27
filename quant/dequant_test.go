package quant

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// --- independent reference implementations -------------------------------
//
// These deliberately re-derive the arithmetic from the ggml source text
// instead of calling the production helpers, so a bug in the production
// helper cannot hide itself.

// refDequantQ4_0 is written straight from ggml-quants.c's
// dequantize_row_q4_0, with no shared code.
func refDequantQ4_0(raw []byte) []float32 {
	nb := len(raw) / 18
	out := make([]float32, nb*32)
	for i := 0; i < nb; i++ {
		// fp16 -> float32 via IEEE bit surgery (independent of Float16ToFloat32).
		h := binary.LittleEndian.Uint16(raw[i*18 : i*18+2])
		exp := int(h>>10) & 0x1f
		mant := int(h) & 0x3ff
		var d float64
		switch exp {
		case 0:
			d = math.Ldexp(float64(mant), -24)
		case 31:
			d = math.Inf(1)
		default:
			d = math.Ldexp(float64(mant|0x400), exp-15-10)
		}
		if h&0x8000 != 0 {
			d = -d
		}
		for j := 0; j < 16; j++ {
			q := raw[i*18+2+j]
			x0 := int(q&0x0F) - 8
			x1 := int(q>>4) - 8
			out[i*32+j] = float32(float64(x0) * d)
			out[i*32+j+16] = float32(float64(x1) * d)
		}
	}
	return out
}

func refDequantQ8_0(raw []byte) []float32 {
	nb := len(raw) / 34
	out := make([]float32, nb*32)
	for i := 0; i < nb; i++ {
		h := binary.LittleEndian.Uint16(raw[i*34 : i*34+2])
		exp := int(h>>10) & 0x1f
		mant := int(h) & 0x3ff
		var d float64
		switch exp {
		case 0:
			d = math.Ldexp(float64(mant), -24)
		case 31:
			d = math.Inf(1)
		default:
			d = math.Ldexp(float64(mant|0x400), exp-15-10)
		}
		if h&0x8000 != 0 {
			d = -d
		}
		for j := 0; j < 32; j++ {
			out[i*32+j] = float32(float64(int8(raw[i*34+2+j])) * d)
		}
	}
	return out
}

func maxAbsDiff(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// --- tests ---------------------------------------------------------------

func TestFloat16Conversion(t *testing.T) {
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x3C00, 1},
		{0xBC00, -1},
		{0x4000, 2},
		{0x3800, 0.5},   // used by the gguf reader test fixture
		{0x7BFF, 65504}, // max finite
		{0x0001, float32(math.Ldexp(1, -24))},
		{0x03FF, float32(math.Ldexp(1023, -24))},
	}
	for _, c := range cases {
		if got := Float16ToFloat32(c.bits); got != c.want {
			t.Errorf("Float16ToFloat32(%#04x) = %v, want %v", c.bits, got, c.want)
		}
	}

	// Specials.
	if v := Float16ToFloat32(0x7C00); !math.IsInf(float64(v), 1) {
		t.Errorf("+inf: got %v", v)
	}
	if v := Float16ToFloat32(0xFC00); !math.IsInf(float64(v), -1) {
		t.Errorf("-inf: got %v", v)
	}
	if v := Float16ToFloat32(0x7E00); !math.IsNaN(float64(v)) {
		t.Errorf("NaN: got %v", v)
	}
}

func TestFloat16AllBitPatterns(t *testing.T) {
	// Exhaustive: every one of the 65536 half values must convert to the
	// correctly-rounded float32.
	for bits := 0; bits <= 0xFFFF; bits++ {
		got := Float16ToFloat32(uint16(bits))
		want := refHalfToFloat(uint16(bits))
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("bits %#04x: want NaN, got %v", bits, got)
			}
			continue
		}
		if got != want {
			t.Fatalf("bits %#04x: got %v, want %v", bits, got, want)
		}
	}
}

// refHalfToFloat is an independent half->float conversion using math/big-free
// bit arithmetic, written as a second opinion for the exhaustive test.
func refHalfToFloat(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		// Normalise the subnormal.
		e := -14
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | uint32(e+127)<<23 | mant<<13)
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp-15+127)<<23 | mant<<13)
	}
}

func TestDequantQ4_0AgainstReference(t *testing.T) {
	scales := []uint16{0x3C00, 0x3800, 0xBC00, 0x0000, 0x2C00}
	for _, scale := range scales {
		for nb := 1; nb <= 4; nb++ {
			raw := make([]byte, nb*18)
			for i := 0; i < nb; i++ {
				binary.LittleEndian.PutUint16(raw[i*18:], scale)
				for j := 0; j < 16; j++ {
					// Cover both nibbles across the full 0..15 range.
					lo := byte((i*16 + j) % 16)
					hi := byte((j*3 + i) % 16)
					raw[i*18+2+j] = hi<<4 | lo
				}
			}
			got, err := Dequantize(Q4_0, raw, nb*32)
			if err != nil {
				t.Fatalf("Dequantize: %v", err)
			}
			want := refDequantQ4_0(raw)
			if d := maxAbsDiff(got, want); d > 0 {
				t.Errorf("scale %#04x nb=%d: max abs diff %v", scale, nb, d)
			}
		}
	}
}

func TestDequantQ4_0NibbleOrder(t *testing.T) {
	// A block whose first byte is 0x10 (low=0 -> -8, high=1 -> -7) with scale
	// 1.0. Element 0 must be -8 and element 16 must be -7. Swapping these is
	// the classic q4_0 bug, so assert it explicitly.
	raw := make([]byte, 18)
	binary.LittleEndian.PutUint16(raw[0:], 0x3C00)
	raw[2] = 0x10
	dst := make([]float32, 32)
	if err := DequantizeInto(Q4_0, raw, dst); err != nil {
		t.Fatal(err)
	}
	if dst[0] != -8 {
		t.Errorf("dst[0] = %v, want -8", dst[0])
	}
	if dst[16] != -7 {
		t.Errorf("dst[16] = %v, want -7", dst[16])
	}
	if dst[1] != -8 || dst[17] != -8 {
		t.Errorf("unwritten nibbles: dst[1]=%v dst[17]=%v, want -8", dst[1], dst[17])
	}
}

func TestDequantQ8_0AgainstReference(t *testing.T) {
	for nb := 1; nb <= 4; nb++ {
		raw := make([]byte, nb*34)
		for i := 0; i < nb; i++ {
			binary.LittleEndian.PutUint16(raw[i*34:], 0x3800) // 0.5
			for j := 0; j < 32; j++ {
				raw[i*34+2+j] = byte(int8(j*8 - 127))
			}
		}
		got, err := Dequantize(Q8_0, raw, nb*32)
		if err != nil {
			t.Fatalf("Dequantize: %v", err)
		}
		want := refDequantQ8_0(raw)
		if d := maxAbsDiff(got, want); d > 0 {
			t.Errorf("nb=%d: max abs diff %v", nb, d)
		}
	}
}

func TestDequantF32F16(t *testing.T) {
	f32raw := make([]byte, 12)
	for i, v := range []float32{1.5, -2.25, 0} {
		binary.LittleEndian.PutUint32(f32raw[i*4:], math.Float32bits(v))
	}
	got, err := Dequantize(F32, f32raw, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{1.5, -2.25, 0} {
		if got[i] != v {
			t.Errorf("f32[%d] = %v, want %v", i, got[i], v)
		}
	}

	f16raw := make([]byte, 6)
	for i, v := range []uint16{0x3C00, 0xBC00, 0x3800} {
		binary.LittleEndian.PutUint16(f16raw[i*2:], v)
	}
	got, err = Dequantize(F16, f16raw, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{1, -1, 0.5} {
		if got[i] != v {
			t.Errorf("f16[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestDequantErrors(t *testing.T) {
	cases := []struct {
		name string
		kind Kind
		raw  []byte
		n    int
	}{
		{"q4_0 partial block", Q4_0, make([]byte, 18), 33},
		{"q4_0 short payload", Q4_0, make([]byte, 17), 32},
		{"q4_0 args mismatch", Q4_0, make([]byte, 36), 32},
		{"q8_0 partial block", Q8_0, make([]byte, 34), 31},
		{"q8_0 short payload", Q8_0, make([]byte, 33), 32},
		{"f32 short", F32, make([]byte, 3), 1},
		{"f16 short", F16, make([]byte, 3), 2},
		{"unknown kind", Kind(99), make([]byte, 4), 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Dequantize(c.kind, c.raw, c.n); err == nil {
				t.Error("expected error, got nil")
			} else if !errors.Is(err, err) {
				t.Errorf("unexpected error type: %v", err)
			}
		})
	}
}

func TestRowStride(t *testing.T) {
	cases := []struct {
		kind Kind
		cols int
		want int
		ok   bool
	}{
		{F32, 7, 28, true},
		{F16, 7, 14, true},
		{Q8_0, 32, 34, true},
		{Q8_0, 2048, 2176, true},
		{Q4_0, 2048, 1152, true},
		{Q4_0, 33, 0, false},
		{Kind(99), 32, 0, false},
	}
	for _, c := range cases {
		got, err := RowStride(c.kind, c.cols)
		if !c.ok {
			if err == nil {
				t.Errorf("RowStride(%s,%d) = %d, want error", c.kind, c.cols, got)
			}
			continue
		}
		if err != nil || got != c.want {
			t.Errorf("RowStride(%s,%d) = %d,%v want %d", c.kind, c.cols, got, err, c.want)
		}
	}
}

func BenchmarkDequantQ4_0(b *testing.B) {
	raw := make([]byte, 2048/32*18) // one 2048-wide row
	for i := 0; i < len(raw); i += 18 {
		binary.LittleEndian.PutUint16(raw[i:], 0x3C00)
	}
	dst := make([]float32, 2048)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := DequantizeInto(Q4_0, raw, dst); err != nil {
			b.Fatal(err)
		}
	}
}
