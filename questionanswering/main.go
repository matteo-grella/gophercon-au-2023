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
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
)

const document = `Cloud computing is a technology that allows individuals and businesses to access computing resources over the Internet. It enables users to utilize hardware and software that are managed by third parties at remote locations. Services provided by cloud computing include storage solutions, databases, and computing power, which can be used on a pay-per-use basis. This model offers flexibility and scalability, reducing the need for large upfront investments in infrastructure. Major providers of cloud computing services include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP).`

func main() {
	m, err := tasks.Load[questionanswering.Interface](&tasks.Config{
		ModelsDir: os.Args[1],
		ModelName: "deepset/bert-base-cased-squad2",
	})

	if err != nil {
		log.Fatalf("error loading model: %s", err)
	}

	opts := &questionanswering.Options{}

	fn := func(question string) error {
		result, err := m.ExtractAnswer(context.Background(), question, document, opts)
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
