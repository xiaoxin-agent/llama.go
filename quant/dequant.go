package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// GGML block geometry. These match ggml/src/ggml-common.h exactly.
const (
	QK4_0 = 32
	QK8_0 = 32

	// SizeOfQ4_0 is sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0/2.
	SizeOfQ4_0 = 2 + QK4_0/2 // 18
	// SizeOfQ8_0 is sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0.
	SizeOfQ8_0 = 2 + QK8_0 // 34

	// QK_K is the super-block size of the K-quant family.
	QK_K = 256

	// SizeOfQ6_K is sizeof(block_q6_K) == ql(128) + qh(64) + scales(16) + d(2).
	SizeOfQ6_K = 210
)

// Kind mirrors the subset of ggml_type this package can dequantise.
type Kind uint32

const (
	F32  Kind = 0
	F16  Kind = 1
	Q4_0 Kind = 2
	Q8_0 Kind = 8
	Q6_K Kind = 14
)

// String returns the ggml type name.
func (k Kind) String() string {
	switch k {
	case F32:
		return "f32"
	case F16:
		return "f16"
	case Q4_0:
		return "q4_0"
	case Q8_0:
		return "q8_0"
	case Q6_K:
		return "q6_k"
	}
	return fmt.Sprintf("kind(%d)", uint32(k))
}

// Float16ToFloat32 converts an IEEE-754 binary16 bit pattern to float32.
// Subnormals, infinities and NaN are handled; no flush-to-zero.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: value = mant * 2^-24, mant in [1, 1023].
		// Normalise: find the leading bit position k = floor(log2(mant)).
		// value = (mant >> k) . (mant & (2^k-1)) * 2^(k-24), exponent unbias k-24.
		m := uint32(mant)
		k := int32(0)
		for m >= 2 {
			m >>= 1
			k++
		}
		// Restore the leading 1 and keep 10 mantissa bits below it.
		mant32 := (uint32(mant) << (23 - uint32(k))) & 0x7FFFFF
		e := uint32(k - 24 + 127)
		return math.Float32frombits(sign<<31 | e<<23 | mant32)
	case 0x1f:
		if mant == 0 {
			return math.Float32frombits(sign<<31 | 0x7f800000)
		}
		return math.Float32frombits(sign<<31 | 0x7f800000 | mant<<13)
	default:
		// Normal: re-bias exponent from 15 to 127.
		return math.Float32frombits(sign<<31 | (exp-15+127)<<23 | mant<<13)
	}
}

// Dequantize converts raw bytes of type kind into n float32 values.
//
// n must be a multiple of the block size for quantised kinds and must match
// len(raw)/stride exactly; violations return an error rather than a short read.
func Dequantize(kind Kind, raw []byte, n int) ([]float32, error) {
	dst := make([]float32, n)
	if err := DequantizeInto(kind, raw, dst); err != nil {
		return nil, err
	}
	return dst, nil
}

// DequantizeInto dequantises into a caller-provided slice of length n.
func DequantizeInto(kind Kind, raw []byte, dst []float32) error {
	n := len(dst)
	switch kind {
	case F32:
		if len(raw) != 4*n {
			return fmt.Errorf("quant: f32 payload %d bytes, want %d", len(raw), 4*n)
		}
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return nil
	case F16:
		if len(raw) != 2*n {
			return fmt.Errorf("quant: f16 payload %d bytes, want %d", len(raw), 2*n)
		}
		for i := range dst {
			dst[i] = Float16ToFloat32(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		return nil
	case Q4_0:
		return dequantQ4_0(raw, dst)
	case Q8_0:
		return dequantQ8_0(raw, dst)
	case Q6_K:
		return dequantQ6_K(raw, dst)
	}
	return fmt.Errorf("quant: unsupported kind %s", kind)
}

// dequantQ4_0 mirrors dequantize_row_q4_0 in ggml/src/ggml-quants.c:
//
//	d = fp16(x[i].d)
//	for j in [0, qk/2): x0 = (qs[j] & 0x0F) - 8; x1 = (qs[j] >> 4) - 8
//	                    y[j] = x0*d;  y[j + qk/2] = x1*d
//
// The low nibble feeds the FIRST half of the block, the high nibble the second
// half. Byte j therefore carries elements j and j+16, not 2j and 2j+1.
func dequantQ4_0(raw []byte, dst []float32) error {
	n := len(dst)
	if n%QK4_0 != 0 {
		return fmt.Errorf("quant: q4_0 element count %d is not a multiple of %d", n, QK4_0)
	}
	nb := n / QK4_0
	if len(raw) != nb*SizeOfQ4_0 {
		return fmt.Errorf("quant: q4_0 payload %d bytes, want %d", len(raw), nb*SizeOfQ4_0)
	}
	for b := 0; b < nb; b++ {
		blk := raw[b*SizeOfQ4_0:]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2 : 2+QK4_0/2]
		base := b * QK4_0
		for j := 0; j < QK4_0/2; j++ {
			q := qs[j]
			dst[base+j] = float32(int(q&0x0F)-8) * d
			dst[base+j+QK4_0/2] = float32(int(q>>4)-8) * d
		}
	}
	return nil
}

// dequantQ8_0 mirrors dequantize_row_q8_0: y[i] = int8(qs[i]) * d.
func dequantQ8_0(raw []byte, dst []float32) error {
	n := len(dst)
	if n%QK8_0 != 0 {
		return fmt.Errorf("quant: q8_0 element count %d is not a multiple of %d", n, QK8_0)
	}
	nb := n / QK8_0
	if len(raw) != nb*SizeOfQ8_0 {
		return fmt.Errorf("quant: q8_0 payload %d bytes, want %d", len(raw), nb*SizeOfQ8_0)
	}
	for b := 0; b < nb; b++ {
		blk := raw[b*SizeOfQ8_0:]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(blk[0:2]))
		base := b * QK8_0
		for j := 0; j < QK8_0; j++ {
			dst[base+j] = float32(int8(blk[2+j])) * d
		}
	}
	return nil
}

// dequantQ6_K mirrors dequantize_row_q6_K in ggml/src/ggml-quants.c.
//
// Block layout (210 bytes, 256 weights):
//
//	ql[128]     lower 4 bits of every weight
//	qh[64]      upper 2 bits
//	scales[16]  int8, one per group of 16 weights
//	d           fp16 super-block scale
//
// The 256 weights are decoded in two halves of 128. Within a half, index
// l in [0,32) writes to l, l+32, l+64 and l+96, and is = l/16 selects one of
// two columns of the scale window, so the eight scales of a half map to
// output slots as l+0 -> sc[is+0], l+32 -> sc[is+2], l+64 -> sc[is+4],
// l+96 -> sc[is+6]. Mapping the two halves or the four columns incorrectly
// still yields plausible-looking weights, only permuted.
func dequantQ6_K(raw []byte, dst []float32) error {
	n := len(dst)
	if n%QK_K != 0 {
		return fmt.Errorf("quant: q6_k element count %d is not a multiple of %d", n, QK_K)
	}
	nb := n / QK_K
	if len(raw) != nb*SizeOfQ6_K {
		return fmt.Errorf("quant: q6_k payload %d bytes, want %d", len(raw), nb*SizeOfQ6_K)
	}

	for b := 0; b < nb; b++ {
		blk := raw[b*SizeOfQ6_K:]
		ql := blk[0:128]
		qh := blk[128:192]
		scales := blk[192:208]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(blk[208:210]))

		base := b * QK_K
		for half := 0; half < 2; half++ {
			qlh := ql[half*64 : half*64+64]
			qhh := qh[half*32 : half*32+32]
			off := half * 128

			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int(qlh[l]&0x0F | ((qhh[l]>>0)&3)<<4)
				q2 := int(qlh[l+32]&0x0F | ((qhh[l]>>2)&3)<<4)
				q3 := int(qlh[l]>>4 | ((qhh[l]>>4)&3)<<4)
				q4 := int(qlh[l+32]>>4 | ((qhh[l]>>6)&3)<<4)

				// Each group of 16 weights has its own int8 scale, so the four
				// output slots of this l read four different scales.
				//
				// scales is []byte, so it must go through int8: a negative
				// scale read as uint8 becomes 256+|s|, which inflates the
				// weight by up to 256x and still produces finite values that
				// look like plausible weights until compared against a
				// reference. int8(...) is the conversion, not int(...).
				sc := scales[half*8:]
				s0 := float32(int8(sc[is+0]))
				s2 := float32(int8(sc[is+2]))
				s4 := float32(int8(sc[is+4]))
				s6 := float32(int8(sc[is+6]))
				dst[base+off+l+0] = d * s0 * float32(q1-32)
				dst[base+off+l+32] = d * s2 * float32(q2-32)
				dst[base+off+l+64] = d * s4 * float32(q3-32)
				dst[base+off+l+96] = d * s6 * float32(q4-32)
			}
		}
	}
	return nil
}

// RowStride returns the byte size of one row of cols elements for kind.
func RowStride(kind Kind, cols int) (int, error) {
	switch kind {
	case F32:
		return 4 * cols, nil
	case F16:
		return 2 * cols, nil
	case Q4_0:
		if cols%QK4_0 != 0 {
			return 0, fmt.Errorf("quant: q4_0 row of %d elements is not block-aligned", cols)
		}
		return (cols / QK4_0) * SizeOfQ4_0, nil
	case Q8_0:
		if cols%QK8_0 != 0 {
			return 0, fmt.Errorf("quant: q8_0 row of %d elements is not block-aligned", cols)
		}
		return (cols / QK8_0) * SizeOfQ8_0, nil
	}
	return 0, fmt.Errorf("quant: unsupported kind %s", kind)
}
