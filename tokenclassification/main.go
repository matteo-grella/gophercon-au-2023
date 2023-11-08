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
	"github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification"
)

var models = []string{
	"dslim/bert-base-NER",                           // Loc, Org, Misc, Per
	"ml6team/bert-base-uncased-city-country-ner",    // City, Country
	"QCRI/bert-base-multilingual-cased-pos-english", // Part-of-speech tagging
}

func main() {
	m, err := tasks.Load[tokenclassification.Interface](&tasks.Config{
		ModelsDir: os.Args[1],
		ModelName: models[0],
	})
	if err != nil {
		log.Fatalf("error loading model: %s", err)
	}

	params := tokenclassification.Parameters{
		AggregationStrategy: tokenclassification.AggregationStrategySimple,
	}

	fn := func(text string) error {
		result, err := m.Classify(context.Background(), text, params)
		if err != nil {
			return err
		}
		fmt.Println(MarshalJSON(result))
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
