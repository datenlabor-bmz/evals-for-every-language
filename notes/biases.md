We want to make a benchmark that objectively

- shows the level of how well-supported or neglected different languages are;
- shows for each language the performance of different AI models;
- shows how performant different AI models generally are regarding language support.

Turns out this is really difficult to do without having some kind of bias.

Here's a list of approaches:

- Translate **from English** to `evaluated_language`: This will give an advantage to languages that are similar to English, and makes it impossible to evaluate English itself.
- Translate from **every other language** to `evaluated_language`: This will give an advantage to language families that have a lot of members.
- Translate from every other language **(weighted by population)** to `evaluated language`: The metrics are usually not comparable across evaluated languages, specifically ...
  - **BLEU**: May support some languages better than others.
  - **BertScore**: Will give an advantage to English, which it is primarily trained on.
  - **Multilingual** BertScore: Will give an advantage to the languages primarily trained on; may recognize semantic similarity even for untranslated words.
  - **ChrF++**: Better than BLEU.
  - **Sp**BLEU, SentencePiece-based tokenizers trained separately on `evaluated_language`, as provided for the FLORES+ dataset: Seems okay.
- Translate **from** `evaluated_language` to every other language (weighted by population), evaluate using any metric: For 2 very similar sister languages that are equally well translated, this gives an advantage to the smaller language (because it has the big sister language as an easy target, whereas the other sister language only has the small sister language as an easy target).
- Translate from `evaluated_language` to every language (`evaluated_language` **itself included**, weighted by population), evaluate using any metric: Gives an advantage to big languages that trivially get a high score for translating to themselves; but this is fair in terms of objectively showing "to how many people can I communicate (and to what extent) given the AI model".
- Rather than translation, use **masked language modeling** just on `evaluated_language` itself: This still depends on an evaluation metric, which is usually not comparable across languages, see the problems above.
- Use **categorization** of sentences in `evaluated_language` (where the same categories are used for all languages): Categories may be tied to certain cultures and thus languages, giving an advantage to the language/culture in which the categories are created.
- Use **culture-independent categories** for categorization of sentences in `evaluated_language` using zero-shot prompting with a given set of category labels: The labels should be language-independent but consistent at the same time, which may be difficult; in practice, English labels may be just fine.
- Use culture-independent categorization of sentences in `evaluated_language` using **few-shot prompting**: This seems okay.