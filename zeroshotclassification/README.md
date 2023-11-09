# Zero-shot Text Classification

This example shows how to use a pre-trained BART model for zero-shot text classification.

```console
go run . models sport,music,politics,business,technology,entertainment,science,health,war
```

You can then enter a text and a list of labels and the model will predict the label that best fits the text, e.g.:

- I like to play football and tennis.
- The battery of my phone is dead.
- New research shows that the playing music can help you sleep better.
- Military forces are preparing to attack the enemy.

You can also use it for sentiment analysis (positive/negative/neutral):

- I absolutely loved the friendly service at the cafe!
- The movie was a complete waste of time and money.





