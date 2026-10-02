# Design: dal setup Tufa (coding) al nostro (matematica)

Fonte: https://github.com/Tufalabs/opsd-predictive-law
File letti: README.md, eval_self_teacher_gap_1step.sbatch,
experiments/rich_feedback/run_eval_self_teacher_gap_1step.sh,
verl/trainer/ppo/ray_trainer.py (_resolve_self_teacher_inputs),
verl/trainer/config/actor/actor.yaml (template).

## Come misurano il gap (loro)
- 131 problemi LCB v6, 8 rollout/problema, punteggio sparso 0/1.
- UNO step di training con lr=0 (pesi congelati): serve solo a far passare il
  trainer dal percorso "student rollout -> costruzione reprompt -> teacher rollout".
- Metriche lette: critic/score/mean (student) vs self-teacher-priv<METHOD>/score/mean.
- gap = teacher - student. Nel paper l'improvement e' Delta mean@4.
- enable_thinking: false; le tracce <think> vengono rimosse dalle dimostrazioni.

## Template del teacher (verbatim dai loro config)
reprompt_template:      "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
solution_template:      "\nCorrect solution:\n\n{successful_previous_attempt}"
feedback_template:      "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
hint_extraction_template: estrae "5-10 short, specific key hints" da una soluzione corretta,
                        senza rivelare la risposta completa; lista numerata.
static_teacher_template: preambolo da esperto (competitive programming) + metodologia.

## Regole di gating (importanti)
- dont_reprompt_on_self_success=True -> il contesto privilegiato va SOLO ai campioni
  il cui rollout e' fallito. Il gap si misura dove lo student sbaglia.
- peer solution = un ALTRO rollout dello stesso modello, risultato corretto nello stesso
  gruppo (NON la ground truth del dataset).
- environment_feedback_only_without_solution=True -> in "all", il feedback compare solo
  quando non c'e' soluzione da mostrare.

## Mappa delle condizioni: loro -> nostre (GSM8K)
| Loro (CONSTRUCTION_METHOD)          | Coding                                   | Nostro nome        | Matematica                                                        |
|-------------------------------------|------------------------------------------|--------------------|-------------------------------------------------------------------|
| none                                | prompt nudo                              | none               | prompt nudo (= baseline student)                                   |
| static_teacher1                     | preambolo esperto CP                     | static_math        | preambolo esperto di matematica (riscritto, dichiarato)            |
| feedback_only                       | errore/test falliti dall'esecuzione      | feedback_binary    | "la tua risposta X e' sbagliata" (verifica del numero finale)      |
| (nuova, non loro)                   | -                                        | feedback_diag      | come sopra + DOVE si rompe il ragionamento (generato dal modello)  |
| correct_solution_processed          | 5-10 hint estratti da soluzione peer     | hints              | stesso template, su soluzione peer corretta                        |
| correct_solution_only               | soluzione peer completa                  | peer_solution      | rollout peer corretto dello stesso modello                         |
| all                                 | soluzione peer + feedback                | all                | combinazione delle due                                             |
| -                                   | -                                        | gt_solution        | soluzione del dataset: TETTO, non e' una loro condizione           |

feedback_binary vs feedback_diag e' il confronto che ci interessa: misura quanto conta
che il feedback sia DIAGNOSTICO. Nel coding lo e' per costruzione (compilatore),
in matematica di default e' solo binario.

## Il nostro setup
- Colab T4, Qwen3-1.7B-Instruct (fallback 0.6B), vLLM, enable_thinking=False.
- GSM8K test, 200 problemi, 8 rollout/problema a temperature 0.7 (come loro), max 512 token.
- Verifica: match del numero finale dopo ####.
- acc(cond) calcolata sui soli campioni che ricevono contesto (= rollout falliti),
  coerente con dont_reprompt_on_self_success.
- gap = acc(teacher | cond) - acc(student), CI via bootstrap.

## Cosa NON facciamo (da dichiarare nel README)
- Nessun training OPSD: misuriamo il predittore, non il guadagno.
- Un solo modello, un solo dataset, nessun seed multiplo (al massimo 2 se avanza tempo).
- Il preambolo esperto e' riscritto per la matematica, quindi non e' identico al loro.
