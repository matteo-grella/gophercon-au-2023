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
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
)

func main() {
	m, err := tasks.Load[languagemodeling.Interface](&tasks.Config{
		ModelsDir: os.Args[1],
		ModelName: "bert-large-cased",
	})
	if err != nil {
		log.Fatalf("error loading model: %s", err)
	}

	fn := func(text string) error {
		result, err := m.Predict(context.Background(), text, languagemodeling.Parameters{K: 10})
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
