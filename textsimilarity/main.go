package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"sort"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/nlpodyssey/spago/mat"
)

var targets = []string{
	"Instructions for internet use, please?",
	"Point me to the refreshments area?",
	"Location of the lavatories?",
	"I'd like to join additional learning sessions.",
	"When do the main talks begin?",
}

type Pair struct {
	Text  string
	Score float32
}

func main() {
	m, err := tasks.Load[textencoding.Interface](&tasks.Config{
		ModelsDir: os.Args[1],
		ModelName: "sentence-transformers/all-MiniLM-L6-v2",
	})

	if err != nil {
		log.Fatalf("error loading model: %s", err)
	}

	targetVectors := make([]mat.Matrix, len(targets))

	// Precompute target vectors
	for i, text := range targets {
		vector, err := m.Encode(context.Background(), text, int(bert.MeanPooling))
		if err != nil {
			log.Fatal(err)
		}
		targetVectors[i] = vector.Vector.Normalize2()
	}

	fn := func(text string) error {
		result, err := m.Encode(context.Background(), text, int(bert.MeanPooling))
		if err != nil {
			return err
		}
		vector := result.Vector.Normalize2()

		fmt.Println("Encoding: first 10 of", len(vector.Data().F32()), "dims:", vector.Data().F32()[:10])

		hits := make([]Pair, len(targets))
		for i, target := range targets {
			hits[i] = Pair{Text: target, Score: vector.DotUnitary(targetVectors[i]).Item().F32()}
		}
		sort.Slice(hits, func(i, j int) bool {
			return hits[j].Score < hits[i].Score
		})
		fmt.Println(MarshalJSON(hits))
		return nil
	}

	if err = ForEachInput(os.Stdin, fn); err != nil {
		log.Fatal(err)
	}
}

func MarshalJSON(data any) string {
	m, _ := json.MarshalIndent(data, "", "  ")
	return string(m)
}

func ForEachInput(r io.Reader, callback func(text string) error) error {
	scanner := bufio.NewScanner(r)
	for {
		fmt.Print("> ")
		scanner.Scan()
		text := scanner.Text()
		if text == "" {
			break
		}
		if err := callback(text); err != nil {
			return err
		}
	}
	return nil
}
