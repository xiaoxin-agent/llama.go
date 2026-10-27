// gguf_info prints the metadata and tensor table of a GGUF file.
//
// Usage:
//
//	go run ./tools/gguf_info.go /path/to/model.gguf
//
// It reports format version, alignment, metadata key/type/value summary and
// the first names tensors with their dimensions, type and payload size. The
// output is intended for cross-checking against `python -m gguf.scripts.gguf_dump`.
//
//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/xiaoxin-agent/llama.go/gguf"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "usage: go run ./tools/gguf_info.go <model.gguf>\n")
		os.Exit(2)
	}
	f, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	fmt.Printf("version   = %d\n", f.Version())
	fmt.Printf("alignment = %d\n", f.Alignment())
	fmt.Printf("tensors   = %d\n", f.TensorCount())
	fmt.Printf("metadata  = %d keys\n\n", len(f.MetadataKeys()))

	fmt.Println("--- metadata ---")
	for _, k := range f.MetadataKeys() {
		v, _ := f.Lookup(k)
		s := v.String()
		if len(s) > 64 {
			s = s[:64] + "..."
		}
		fmt.Printf("%-45s %-8s %s\n", k, v.Kind, s)
	}

	fmt.Println("\n--- tensors (first 20) ---")
	names := f.TensorNames()
	for i, n := range names {
		if i >= 20 {
			fmt.Printf("... and %d more\n", len(names)-20)
			break
		}
		t, _ := f.Tensor(n)
		nb, err := t.NBytes()
		if err != nil {
			fmt.Printf("%-28s dims=%v ERR=%v\n", t.Name, t.Dims, err)
			continue
		}
		fmt.Printf("%-28s dims=%-18v %-6s bytes=%d\n", t.Name, t.Dims, t.Type, nb)
	}
}
