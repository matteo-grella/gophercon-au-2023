package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textgeneration"
)

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("usage: %s <dir> <source> <target>", os.Args[0])
	}

	m, err := tasks.Load[textgeneration.Interface](&tasks.Config{
		ModelsDir: os.Args[1],
		ModelName: textgeneration.DefaultModelForMachineTranslation(os.Args[2], os.Args[3]),
	})

	if err != nil {
		log.Fatalf("error loading model: %v", err)
	}

	opts := textgeneration.DefaultOptions()

	fn := func(text string) error {
		result, err := m.Generate(context.Background(), text, opts)
		if err != nil {
			return err
		}
		fmt.Println(result.Texts[0])
		return nil
	}

	if err = ForEachInput(os.Stdin, fn); err != nil {
		log.Fatal(err)
	}
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
