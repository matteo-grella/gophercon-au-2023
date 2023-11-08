# Text Similarity

This example demonstrates how to utilize a pre-trained BART model for measuring text similarity.

To execute the program, use the following command in the terminal:

```console
go run . models
```

Upon running the model, you can input a sentence, and the model will encode it into a vector. It will then compare this vector with the vectors of a set of predefined sentences to determine similarity. For instance:

- Instructions for internet use, please?
- Point me to the refreshments area?
- Location of the lavatories?
- I'd like to join additional learning sessions.
- When do the main talks begin?

Here are some examples of input sentences you might test against the model:

- How does one go online here?
- Where can I find snacks?
- Need to find the washroom.
- How to sign up for more classes?
- What's the schedule for keynote speeches?
