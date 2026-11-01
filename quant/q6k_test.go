package quant

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// maxQ6KValue bounds any dequantised Q6_K weight by construction:
// the 6-bit field spans [-32, 31], so |q| <= 32, and each of the 16 groups has
// an int8 scale, so |scale| <= 127. Everything is then multiplied by the
// super-block scale d. A value above this cannot come from a correct decode.
const maxQ6KQuarter = 32 * 127

// TestDequantQ6_KWithinBounds exists because of two real bugs, both of which
// produced arrays of the right length full of finite numbers:
//
//   - scales read through []byte without an int8 conversion, turning -15 into
//     241 and inflating weights by up to 256x;
//   - index arithmetic that wrote l, l+32, l+64, l+96 as adjacent slots,
//     leaving three quarters of the block at whatever the buffer already held.
//
// Neither is detectable by checking the output length or looking for NaN.
// Both are trivially detectable by bounding the magnitude.
func TestDequantQ6_KWithinBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 8; trial++ {
		nb := 1 + rng.Intn(4)
		n := nb * QK_K
		raw := make([]byte, nb*SizeOfQ6_K)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		// Give each block a sane fp16 scale (1/2^12 .. 1) so the bound is
		// meaningful; random fp16 bits can encode inf/NaN.
		d := float32(math.Pow(2, -float64(1+rng.Intn(13)))) // 2^-1 .. 2^-13
		for b := 0; b < nb; b++ {
			h := f32ToF16(d)
			binary.LittleEndian.PutUint16(raw[b*SizeOfQ6_K+208:], h)
		}

		dst, err := Dequantize(Q6_K, raw, n)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if len(dst) != n {
			t.Fatalf("trial %d: got %d values, want %d", trial, len(dst), n)
		}
		limit := float32(maxQ6KQuarter) * d
		for i, v := range dst {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("trial %d index %d: non-finite value %v", trial, i, v)
			}
			if abs := float32(math.Abs(float64(v))); abs > limit {
				t.Fatalf("trial %d index %d: |%v| exceeds bound %v (d=%v); "+
					"a decode bug, not rounding", trial, i, v, limit, d)
			}
		}
	}
}

// TestDequantQ6_KScaleSign covers the []byte-versus-int8 mistake directly by
// using two scales of equal magnitude and opposite sign.
func TestDequantQ6_KScaleSign(t *testing.T) {
	raw := make([]byte, SizeOfQ6_K)
	binary.LittleEndian.PutUint16(raw[208:], f32ToF16(1.0))
	// First half's eight scales: +5, -5, +3, -3, +7, -7, +1, -1.
	// Alternating signs catch a uint8-vs-int8 read; distinct magnitudes catch a
	// slot mix-up, since a wrong column would then multiply by the wrong one.
	scales := []int8{5, -5, 3, -3, 7, -7, 1, -1}
	for i, v := range scales {
		raw[192+i] = byte(v)
	}
	// ql/qh all zero: every 6-bit field decodes to 0, minus 32 gives -32.
	dst, err := Dequantize(Q6_K, raw, QK_K)
	if err != nil {
		t.Fatal(err)
	}

	// Within a half, is = l/16 picks one of two columns and the four output
	// slots add 0, 2, 4 or 6 to it, so the eight scales of a half map to
	// outputs as:
	//
	//	dst[l+ 0] -> sc[is+0]     dst[l+64] -> sc[is+4]
	//	dst[l+32] -> sc[is+2]     dst[l+96] -> sc[is+6]
	//
	// Every check below therefore needs the scale that slot actually reads.
	checks := []struct {
		idx  int
		sc   int // scale index used
		want float32
	}{
		// q is 0 - 32 = -32 for every weight here, so each expected value is
		// -32 times the scale that slot actually reads.
		{0, 0, -32 * 5},   // sc[0] = +5
		{16, 1, -32 * -5}, // sc[1] = -5
		{32, 2, -32 * 3},  // sc[2] = +3
		{48, 3, -32 * -3}, // sc[3] = -3
	}
	for _, c := range checks {
		if got := dst[c.idx]; got != c.want {
			t.Errorf("dst[%d] = %v, want %v (reads sc[%d])",
				c.idx, got, c.want, c.sc)
		}
	}
}

// TestDequantQ6_KNoUnwrittenSlots catches the "only a quarter of the block is
// written" bug by checking that every element of a block is touched. It uses a
// nonzero scale and nonzero quants so every output is distinguishable from an
// untouched slot.
func TestDequantQ6_KNoUnwrittenSlots(t *testing.T) {
	raw := make([]byte, SizeOfQ6_K)
	binary.LittleEndian.PutUint16(raw[208:], f32ToF16(1.0))
	for i := 0; i < 128; i++ {
		raw[i] = 0xFF // every nibble = 15
	}
	for i := 0; i < 64; i++ {
		raw[128+i] = 0xFF // every high pair = 3
	}
	for i := 0; i < 16; i++ {
		raw[192+i] = 1
	}

	dst, err := Dequantize(Q6_K, raw, QK_K)
	if err != nil {
		t.Fatal(err)
	}
	// All nibbles 15 and all high bits 3 gives (15 | 3<<4) - 32 = 63 - 32 = 31.
	want := float32(31 * 1)
	for i, v := range dst {
		if v != want {
			t.Fatalf("index %d = %v, want %v: that slot was not written by the "+
				"decode loop", i, v, want)
		}
	}
}

// f32ToF16 converts a float32 to IEEE-754 binary16 bits. Truncating rather
// than rounding is fine here because the tests only use exact powers of two.
func f32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xFF) - 127
	mant := bits & 0x7FFFFF
	// Subnormals are out of scope for the test helper: the scales used here are
	// normal powers of two between 2^-14 and 1.
	if exp < -14 || exp > 15 {
		panic("f32ToF16: exponent out of range for test use")
	}
	return sign | uint16(exp+15)<<10 | uint16(mant>>13)
}

// TestDequantQ6_KLengthGuards covers the two rejection paths. They matter more
// than usual here: a *shorter* payload silently dequantises as a truncated
// tensor if the length is never checked, and every downstream matmul then
// reads zeros it believes came from the model.
func TestDequantQ6_KLengthGuards(t *testing.T) {
	raw := make([]byte, 2*SizeOfQ6_K)
	binary.LittleEndian.PutUint16(raw[208:], f32ToF16(1.0))
	binary.LittleEndian.PutUint16(raw[SizeOfQ6_K+208:], f32ToF16(1.0))

	cases := []struct {
		name string
		raw  []byte
		n    int
	}{
		{name: "element count not a multiple of QK_K", raw: raw, n: QK_K + 1},
		{name: "payload one byte short", raw: raw[:len(raw)-1], n: 2 * QK_K},
		{name: "payload one byte long", raw: append(raw, 0), n: 2 * QK_K},
		{name: "payload empty", raw: nil, n: QK_K},
		{name: "block count disagrees with payload", raw: raw, n: QK_K},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Dequantize(Q6_K, c.raw, c.n); err == nil {
				t.Fatalf("Dequantize(q6_k, %d bytes, n=%d) = nil error, want rejection",
					len(c.raw), c.n)
			}
		})
	}

	// DequantizeInto shares the same guards; a caller that preallocated the
	// destination must not get a partially filled buffer back as success.
	dst := make([]float32, QK_K+1)
	if err := DequantizeInto(Q6_K, raw, dst); err == nil {
		t.Fatal("DequantizeInto(q6_k, n not a multiple of QK_K) = nil error, want rejection")
	}
}
