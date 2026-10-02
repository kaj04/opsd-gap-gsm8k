# Piano di lavoro — OPSD gap su GSM8K

## Obiettivo
Misurare il *predittore* della legge OPSD di Tufa Labs (gap self-teacher vs student,
misurato PRIMA del training) su un dominio che il paper non copre: la matematica (GSM8K),
invece del coding (LiveCodeBench v6).

Riferimento: He, Sieber, Saponati — "A Predictive Law for On-Policy Self-Distillation
From World Feedback", RLxF @ ICML 2026. arXiv:2605.30070
Repo: https://github.com/Tufalabs/opsd-predictive-law

## Cosa misuriamo (e cosa no)
- SI: accuratezza del modello con e senza contesto privilegiato -> gap per categoria.
- NO: il training OPSD, quindi NON misuriamo il guadagno finale.
- Claim difendibile: "SE la legge regge fuori dal coding, ALLORA in matematica la
  configurazione migliore sarebbe X, perche ha il gap maggiore".

## Setup
- Colab, GPU T4 (fallback: Qwen3-0.6B se la memoria stringe)
- Modello: Qwen3-1.7B-Instruct, vLLM
- Dati: GSM8K test, primi 200 problemi
- Campionamento: 4 campioni/problema, temperature 0.7, max_new_tokens 512
- Metrica: acc per condizione; gap = acc(cond) - acc(student); CI via bootstrap

## Condizioni (categorie del paper, realizzate per la matematica)
| nome             | cosa vede il teacher                          | origine          |
|------------------|-----------------------------------------------|------------------|
| student          | solo la domanda                               | baseline         |
| hint             | una frase sul METODO, senza numeri            | generato offline |
| partial          | primi 1-2 passaggi della soluzione            | dal dataset      |
| feedback_binary  | proprio tentativo errato + "e sbagliato"      | generato         |
| feedback_diag    | proprio tentativo errato + DOVE si rompe      | generato         |
| solution         | svolgimento completo                          | dal dataset      |

Controllo anti answer-leakage: hint e partial non devono contenere il numero finale.

## Serate
1. Prompt del repo Tufa + scaffolding + condizioni gratuite (student/partial/solution)
   + smoke test su 10 problemi + ispezione manuale dei prompt.
2. Generazione offline di hint e feedback + run completo 6 condizioni x 200 x 4
   + tabella, grafico, CI. (Opzionale: ripetere con 0.6B per stabilita dell'ordine.)
3. README onesto (limiti inclusi) + repo pubblica + mail a Benjamin.

## Rischi / piano B
- Gap tutti ~0 (GSM8K troppo facile): filtrare sui problemi che lo student sbaglia.
- Formato risposta non rispettato: parser tollerante, deciso dopo lo smoke test.
- T4 lenta: scendere a 100 problemi o a 0.6B.

## Deliverable
Repo pubblica + figures/gaps.png + results/gaps.csv + README + paragrafo per la mail.
