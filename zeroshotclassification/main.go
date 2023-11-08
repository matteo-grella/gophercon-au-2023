package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatalf("usage: %s <dir> <labels>", os.Args[0])
	}

	m, err := tasks.Load[zeroshotclassifier.Interface](&tasks.Config{
		ModelsDir: os.Args[1],
		ModelName: "valhalla/distilbart-mnli-12-3",
	})
	if err != nil {
		log.Fatalf("error loading model: %v", err)
	}

	params := zeroshotclassifier.Parameters{
		CandidateLabels:    strings.Split(os.Args[2], ","),
		HypothesisTemplate: zeroshotclassifier.DefaultHypothesisTemplate,
		MultiLabel:         true,
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
