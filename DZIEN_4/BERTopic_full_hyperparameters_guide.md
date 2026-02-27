# BERTopic -- pełne wyjaśnienie hiperparametrów (praktycznie i technicznie)

Ten dokument opisuje wszystkie kluczowe parametry używane w BERTopic ---
szczególnie w kontekście klastrowania logów systemowych.

------------------------------------------------------------------------

# 1️⃣ Główne parametry konstruktora BERTopic

## embedding_model

Określa model generujący embeddingi zdań.

Najczęściej: - all-MiniLM-L6-v2 (szybki, 384D, dobry do krótkich
tekstów) - all-mpnet-base-v2 (dokładniejszy, wolniejszy)

Wpływ: - Lepszy model → czystsze klastry semantyczne - Gorszy model →
mieszanie znaczeń

------------------------------------------------------------------------

## language

Określa język do: - stopwords - tokenizacji - c-TF-IDF

Dla logów ma umiarkowane znaczenie.

------------------------------------------------------------------------

## verbose

Pokazuje przebieg: - embedding - redukcję wymiaru (UMAP) - klasteryzację
(HDBSCAN)

Nie wpływa na wynik.

------------------------------------------------------------------------

## nr_topics

Steruje liczbą tematów.

Opcje: - "auto" → automatyczne scalanie podobnych tematów - liczba
całkowita → wymuszenie liczby tematów

Wpływ: - "auto" → naturalna struktura - stała liczba → kontrola
prezentacyjna

------------------------------------------------------------------------

## min_topic_size

Minimalna liczba dokumentów w temacie.

Małe wartości → więcej tematów, więcej szumu\
Duże wartości → mniej tematów, większa czystość

Dla 6000 logów: 20--50 rozsądne

------------------------------------------------------------------------

## calculate_probabilities

Jeśli True: - liczy prawdopodobieństwa przynależności dokumentu do
tematów

Koszt: - większe zużycie RAM - wolniejsze działanie

Dla logów zwykle False wystarcza.

------------------------------------------------------------------------

## top_n\_words

Ile słów wyświetlać jako reprezentację tematu.

Standard: 10\
Dla logów często 8--15

------------------------------------------------------------------------

# 2️⃣ Parametry UMAP (redukcja wymiaru)

## n_neighbors

Liczba sąsiadów w grafie.

Małe wartości (5--10): - więcej lokalnych klastrów

Duże (15--50): - bardziej globalna struktura

Dla logów: 10--20

------------------------------------------------------------------------

## n_components

Liczba wymiarów po redukcji.

Domyślnie: 5

Więcej wymiarów → dokładniejsza struktura\
Mniej → szybsze działanie

------------------------------------------------------------------------

## min_dist

Jak ciasno punkty są w embeddingu UMAP.

Małe (0.0--0.1): - gęste klastry

Większe (0.3--0.5): - luźniejsze

------------------------------------------------------------------------

# 3️⃣ Parametry HDBSCAN (klasteryzacja)

## min_cluster_size

Minimalna wielkość klastra.

Małe → więcej małych tematów\
Duże → mniej, stabilniejsze

------------------------------------------------------------------------

## min_samples

Im większe: - bardziej konserwatywne klastry - więcej dokumentów jako -1
(noise)

------------------------------------------------------------------------

## metric

Metryka odległości. Najczęściej: euclidean

Dla embeddingów działa dobrze.

------------------------------------------------------------------------

# 4️⃣ Vectorizer Model

Można przekazać własny CountVectorizer.

Parametry warte kontroli: - ngram_range - stop_words - min_df - max_df

Dla logów warto dodać własne stopwords: - num - ts - hex - trace -
latency_ms

------------------------------------------------------------------------

# 5️⃣ Pipeline wewnętrzny BERTopic

1.  Embedding (SentenceTransformer)
2.  Redukcja wymiaru (UMAP)
3.  Klasteryzacja (HDBSCAN)
4.  c-TF-IDF (opis tematów)

------------------------------------------------------------------------

# 6️⃣ Rekomendowana konfiguracja dla logów produkcyjnych

``` python
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="english",
    nr_topics="auto",
    min_topic_size=40,
    calculate_probabilities=False,
    verbose=True
)
```

Dodatkowo: - n_neighbors = 15 - min_cluster_size = 40 - agresywna
normalizacja tokenów

------------------------------------------------------------------------

# 7️⃣ Jak ocenić jakość tematów?

Dobre tematy: - stabilne słowa kluczowe - powtarzalne wzorce (service,
code, route) - mało szumu

Złe tematy: - dominują placeholdery (num, ts, hex) - brak spójności
semantycznej

------------------------------------------------------------------------

# 8️⃣ Wskazówki praktyczne

Jeśli klastry są zbyt rozbite: - zwiększ min_topic_size - zwiększ
min_cluster_size

Jeśli są zbyt ogólne: - zmniejsz min_topic_size - zmniejsz n_neighbors

------------------------------------------------------------------------

Dokument przygotowany pod kątem analizy logów systemowych i monitoringu
produkcyjnego.
