// Command gguf-inspect prints the header, metadata and tensor table of a GGUF
// file in a stable, diffable form.
//
// The output is designed to be compared, line by line, against a reference
// dump produced by llama.cpp's gguf_dump or by an independent parser. Every
// line is "<key> = <value>" or a fixed-column table row, so a plain diff shows
// exactly which field disagrees.
//
// With --meta-only the tool stops after the metadata section and never looks at
// tensor element types. That matters because a real quantised model mixes types
// (Q4_0 weights alongside Q6_K embeddings), and a parser that rejects one
// unknown type should not be prevented from reporting metadata that it parsed
// correctly. M1's gate compares metadata, so it must be reachable even when the
// tensor types are not yet all supported.
//
// Usage:
//
//	gguf-inspect [-meta-only] [-json] <model.gguf>
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/xiaoxin-agent/llama.go/gguf"
)

func main() {
	metaOnly := flag.Bool("meta-only", false, "print metadata only; do not resolve tensor element types")
	asJSON := flag.Bool("json", false, "emit JSON instead of text")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s [-meta-only] [-json] <model.gguf>\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(flag.Arg(0), *metaOnly, *asJSON); err != nil {
		fmt.Fprintf(os.Stderr, "gguf-inspect: %v\n", err)
		os.Exit(1)
	}
}

type metaEntry struct {
	Key   string `json:"key"`
	Type  string `json:"type"`
	Value string `json:"value"`
	Len   int    `json:"array_len,omitempty"`
}

type tensorEntry struct {
	Name   string   `json:"name"`
	Dims   []uint64 `json:"dims"`
	Type   string   `json:"type,omitempty"`
	Bytes  uint64   `json:"bytes,omitempty"`
	Offset uint64   `json:"offset"`
	Note   string   `json:"note,omitempty"`
}

type report struct {
	Version   uint32        `json:"version"`
	Alignment uint64        `json:"alignment"`
	DataStart int64         `json:"data_start"`
	Tensors   int           `json:"tensor_count"`
	Metadata  []metaEntry   `json:"metadata"`
	TensorTab []tensorEntry `json:"tensors"`
}

func run(path string, metaOnly, asJSON bool) error {
	f, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	r := report{
		Version:   f.Version(),
		Alignment: f.Alignment(),
		DataStart: f.DataStart(),
		Tensors:   f.TensorCount(),
	}

	for _, k := range f.MetadataKeys() {
		v, _ := f.Lookup(k)
		e := metaEntry{Key: k, Type: v.Kind.String(), Value: v.String()}
		switch v.Kind {
		case gguf.ValArray:
			e.Len = arrayLen(v)
			e.Value = fmt.Sprintf("[len=%d]", e.Len)
		case gguf.ValString:
			// Strings are printed in full: tokenizer vocabularies are the
			// main thing this tool exists to verify, and truncating them
			// would hide the exact bytes under comparison.
		}
		r.Metadata = append(r.Metadata, e)
	}

	for _, name := range f.TensorNames() {
		t, _ := f.Tensor(name)
		e := tensorEntry{Name: name, Dims: t.Dims, Offset: t.Offset, Type: t.Type.String()}
		if metaOnly {
			e.Type, e.Bytes = "", 0
			e.Note = "" // element types deliberately not resolved
		} else {
			nb, err := t.NBytes()
			if err != nil {
				e.Bytes = 0
				e.Note = "unsupported: " + err.Error()
			} else {
				e.Bytes = nb
			}
		}
		r.TensorTab = append(r.TensorTab, e)
	}

	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(r)
	}
	printText(r, metaOnly)
	return nil
}

func printText(r report, metaOnly bool) {
	fmt.Printf("version    = %d\n", r.Version)
	fmt.Printf("alignment  = %d\n", r.Alignment)
	fmt.Printf("data_start = %d\n", r.DataStart)
	fmt.Printf("tensors    = %d\n", r.Tensors)
	fmt.Printf("metadata   = %d keys\n", len(r.Metadata))

	fmt.Println("\n--- metadata ---")
	for _, e := range r.Metadata {
		fmt.Printf("%s = %s\n", e.Key, e.Value)
	}

	fmt.Println("\n--- tensors ---")
	if metaOnly {
		fmt.Println("(element types not resolved: --meta-only)")
	}
	for _, e := range r.TensorTab {
		fmt.Printf("%-40s dims=%-20v offset=%-12d", e.Name, e.Dims, e.Offset)
		if metaOnly {
			fmt.Println()
			continue
		}
		if e.Note != "" {
			fmt.Printf("type=%-7s %s\n", e.Type, e.Note)
			continue
		}
		fmt.Printf("type=%-7s bytes=%d\n", e.Type, e.Bytes)
	}
}

// arrayLen returns the element count of an array value.
func arrayLen(v gguf.Value) int {
	return len(v.Strs) + len(v.Floats) + len(v.Uints) + len(v.Ints) + len(v.Bools)
}
