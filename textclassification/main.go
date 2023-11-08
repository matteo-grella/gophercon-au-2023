package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
)

func main() {
	m, err := tasks.Load[textclassification.Interface](&tasks.Config{
		ModelsDir: os.Args[1],
		ModelName: "nlpodyssey/bert-multilingual-uncased-geo-countries-headlines",
	})
	if err != nil {
		log.Fatalf("error loading model: %v", err)
	}

	fn := func(text string) error {
		result, err := m.Classify(context.Background(), text)
		if err != nil {
			return err
		}
		fmt.Println(MarshalJSON(LimitSize(result, 5)))
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

func LimitSize(r textclassification.Response, maxSize int) textclassification.Response {
	newLabels := make([]string, 0, maxSize)
	newScores := make([]float64, 0, maxSize)

	copySize := min(maxSize, len(r.Labels))
	if copySize > 0 {
		newLabels = append(newLabels, r.Labels[:copySize]...)
		newScores = append(newScores, r.Scores[:copySize]...)
	}

	return textclassification.Response{Labels: newLabels, Scores: newScores}
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
