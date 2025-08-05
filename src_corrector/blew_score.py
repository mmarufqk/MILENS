from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# Daftar Ref dan Fixed dari data kamu
refs = [
    "Esplen is part of Pittsburgh and is in the Pittsburgh City School district.",
    "Safronov is the nearest rural locality.",
    "Weather forecasting is another critical aspect of sailing yacht management.",
    "The group then sing the bridge, and end the song repeating the chorus twice.",
    "Typically it encloses a metal grommet for reinforcement and to reduce wear.",
    "The department is, historically, a prominent producer of gold, silver, and copper.",
    "According to an old account, there was an important exception to the rule.",
    "One skeleton dances part of the Charleston.",
    "The ghettoization was completed within a week.",
    "He then started to study Business administration but broke off after a few semesters.",
    "Then you're not the man.",
    "The local baseball field is named for him.",
    "These circuits are very frequently fed from transformers, and have significant resistance.",
    "He won his first career race at New Hampshire and finished eighth in points.",
    "She is married to Mathieu Sweeney.",
    "“These others,” he said in a voice of extreme irritation.",
    "He is quickly killed by Superboy-Prime amidst the chaos.",
    "He attended Iowa State University, where he played defense on the school's football team.",
    "Many highly successful television series have been known as period pieces.",
    "Lenny Hart was also the Grateful Dead's original money manager.",
]

fixeds = [
    "Aspirin is part of Pittsburgh and is in the Pittsburgh City School District.",
    "Saffron of is the nearest rural locality.",
    "weather forecasting is another critical aspect of sailing yacht management",
    "The group then saw the bridge and ended the song repeating the chorus twice.",
    "Typically, it encloses a metal grommets for reinforcement and to reduce wear.",
    "The department is historically a prominent producer of gold, silver, and copper.",
    "According to an old account there was an important exception to the rule.",
    "Once collected thus is part of the Charleston.",
    "The garage station was completed within a week.",
    "He then started to study business administration, but broke off after a few semesters.",
    "Then you're not the man.",
    "the local baseball field is named for him",
    "They are so cute are very frequently fed from transformers and acid navy contrast these days.",
    "He won his first career was at New Hampshire and finished eighth in points.",
    "She is married to Matthew Sweep.",
    "these others he said in a voice of extreme irritation",
    "He is quickly killed by superboy prime amidst the chaos.",
    "He attended Iowa State University where he played defense on the school's football team.",
    "Many highly successful TV series have been known as period pieces.",
    "lenny hart was also the grateful dead's original money manager",
]

# Menghitung BLEU score untuk setiap pasangan
smoothie = SmoothingFunction().method4
scores = []

for ref, pred in zip(refs, fixeds):
    ref_tokens = ref.lower().split()
    pred_tokens = pred.lower().split()
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    scores.append(score)

# Menampilkan hasil
average_bleu = np.mean(scores)
print(f"Rata-rata BLEU Score: {average_bleu:.4f}")
