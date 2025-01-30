# Peek MT [![Paper](https://img.shields.io/badge/📜%20paper-481.svg)](https://vilda.net/papers/mt_peek.pdf)

Code for the small exploratory project project [Machine Translation that Peeks at the Reference](https://vilda.net/papers/mt_peek.pdf).

> **Abstract:** Machine translation with lexical constraints is a popular research topic, especially for terminology translation.
> Existing approaches for lexical control in MT are usually complex and not easily applicable to all existing MT toolkits.
> We propose an off-the-shelf baseline approach, *Peek MT*, for lexical constraints.
> During training, the model is provided with access to some of the words in the reference, allowing it to produce better translations.
> During inference, the user can specify which words they would like the translation to contain.
> Depending on the amount of additional tokens, the MT performance is improved by 1.3-4.4 BLEU points per revealed token.
> Despite these being very soft constraints, they are fulfilled ≈66% of the time.
> Notably, the same approach can also be used to control the output translation length without tinkering with the decoder.
> Finally, from analysis point of view, this method allows us to establish that the knowledge of particular word in the reference, such as verbs and organization names boosts the MT performance the most.

<img alt="Example of an incorrect and correct translation of a German sentence given additional info." src="https://github.com/zouharvi/mt-peek/assets/7661193/3b946ab8-bad2-42af-b3d8-e364ca1b5f23" width=400em>
