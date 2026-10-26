---

🧭 Mediator: Persona- and Act-Guided Direct Preference Optimization (PA-DPO)

This repository implements \*\*PA-DPO (Persona-Act Direct Preference Optimization)\*\* — a method for faithful and personalized dialogue generation in agent–traveler negotiation settings.    
The approach extends standard DPO to incorporate \*\*persona grounding\*\* and \*\*dialogue act consistency\*\*, ensuring coherent, personalized, and contextually faithful negotiation dialogues.

\---

🔍 Overview

Given a conversational history:

\\\[  
H \= (A\_1, U\_1, \\dots, A\_{t-1}, U\_{t-1})  
\\\]

where \\(A\\) and \\(U\\) denote the agent and user turns respectively, and a user’s current turn \\(U\_t\\), the model generates:

1\. The traveler’s **persona triplets** \\((p\_1, \\dots, p\_k)\\)  
2\. The agent’s **dialogue act** \\(a\_t\\)  
3\. The agent’s **response** \\(A\_t\\)

Each training sample is represented as:

\- **Input**:    
  \\(X \= \\text{INST} \\oplus H \\oplus U\_t\\)

\- **Output**:    
  \\\[  
  Y \= \[BOS\]\[BPER\]p\_1…p\_k\[EPER\]\[BACT\]a\_t\[EACT\]\[BRES\]A\_t\[ERES\]\[EOS\]  
  \\\]

Special tokens (\`\[BPER\]\`, \`\[BACT\]\`, \`\[BRES\]\`, etc.) are used to mark persona, dialogue act, and response spans.

\---

🧠 Methodology Summary

1\. Supervised Fine-Tuning (SFT)  
We fine-tune **Qwen2.5-3B-Instruct** on the supervised dataset \\(D=\\{(X\_n,Y\_n)\\}\\) by minimizing:

\\\[  
\\mathcal{L}\_{SFT} \= \-E\_{(X,Y)\\sim D}\\left\[\\sum\_i \\log \\pi\_\\theta(y\_i|y\_{\<i},X)\\right\]  
\\\]

to obtain a base model \\( \\pi\_{\\text{SFT}} \\).

2\. Preference Data Generation  
We construct preference pairs \\((y^+, y^-)\\) through:  
\- **Selective Masking** — removing persona/act cues from preferred responses.  
\- **Persona-Act Retrieval** — retrieving semantically similar samples for persona and act grounding.  
\- **Hallucination Elicitation** — regenerating responses that violate persona or act consistency.

3\. PA-DPO Training  
Using the preference dataset    
\\(D\_{\\text{pref}} \= \\{(x, y^+, y^-, pa^+, pa^-)\\}\\),    
we optimize the combined objective:

\\\[  
\\mathcal{L}\_{PA\\text{-DPO}} \= \\mathcal{L}\_{\\text{DPO}} \+ \\mathcal{L}\_{PA}  
\\\]

where \\(\\mathcal{L}\_{PA}\\) incorporates persona and act likelihoods to reinforce fidelity and constraint adherence.

\---

📂 Repository Structure

mediator/  
 ├── baselines/  
 │ ├── final\_dpo.py  
 │ ├── final\_kto.py  
 │ ├── final\_ppo.py  
 │ └── final\_sft\_llama.py  
 │  
 ├── eval/  
 │ ├── eval\_padpo.py  
 │ ├── eval\_ppo.py  
 │ ├── final\_eval\_1.py  
 │ ├── final\_eval\_2.py  
 │ ├── pacons\_classifier.py  
 │ └── pacons\_pre.py  
 │  
 ├── inference/  
 │ ├── final\_infer\_pa\_dpo.py  
 │ ├── final\_infer\_sft\_qwen.py  
 │ ├── final\_infer\_ppo.py  
 │ ├── final\_infer\_gpt5.py  
 │ ├── final\_infer\_cpu.py  
 │ └── final\_infer.py  
 │  
 ├── PA-DPO/  
 │ └── final\_padpo.py  
 │  
 ├── preference\_data/  
 │ ├── final\_gen\_triplets.py  
 │ └── final\_pref\_dataset\_gen.py  
 │  
 ├── sft/  
 │ ├── final\_preprocess\_sft.py  
 │ └── final\_sft\_qwen25.py  
 │  
 └── README.md

\---

⚙️ Setup

1\. Environment

\`\`\`bash  
conda create \-n mediator python=3.12  
conda activate mediator  
pip install torch torchvision torchaudio  
pip install transformers datasets accelerate peft  
pip install pandas numpy tqdm scikit-learn sentence-transformers

(Optional) For multi-GPU training:

pip install deepspeed

### **2\. Dataset Format**

The dataset is stored as a **CSV file** with alternating traveler and agent utterances in multi-turn dialogues.  
 The **first row** of each dialogue contains traveler persona information:

| dialogue\_id | turn | speaker | utterance | persona\_evidences | persona\_triplets |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 1 | 1 | traveler | I’m planning a trip to Paris. | adventurous, budget-conscious | (traveler, likes, adventure), (traveler, budget, low) |
| 1 | 2 | agent | Great\! Let’s find something exciting yet affordable. | — | — |

---

## **🚀 Running the Pipeline**

### **Step 1: Preprocessing**

Prepare the dataset for SFT:

python sft/final\_preprocess\_sft.py \--input\_path data/dialogues.csv \--output\_path data/sft\_ready.json

---

### **Step 2: Supervised Fine-Tuning**

Train the SFT model (Qwen2.5-3B-Instruct):

python sft/final\_sft\_qwen25.py \\  
    \--train\_file data/sft\_ready.json \\  
    \--model\_name Qwen2.5-3B-Instruct \\  
    \--output\_dir outputs/sft\_qwen25

---

### **Step 3: Preference Data Generation**

Generate persona-act based preference pairs:

python preference\_data/final\_pref\_dataset\_gen.py \\  
    \--input data/sft\_ready.json \\  
    \--output data/pref\_data.json

(Optional) Generate persona triplets:

python preference\_data/final\_gen\_triplets.py

---

### **Step 4: Persona-Act DPO Training**

Run PA-DPO fine-tuning:

python PA-DPO/final\_padpo.py \\  
    \--model\_path outputs/sft\_qwen25 \\  
    \--pref\_data data/pref\_data.json \\  
    \--output\_dir outputs/pa\_dpo

---

### **Step 5: Inference**

Generate model outputs on the test set:

python inference/final\_infer\_pa\_dpo.py \\  
    \--model\_dir outputs/pa\_dpo \\  
    \--test\_file data/test.csv \\  
    \--save\_path results/padpo\_generations.json

If you only need raw text output (no formatting cleanup):

python inference/final\_infer\_pa\_dpo.py \--raw\_output

---

### **Step 6: Evaluation**

Evaluate generated responses for persona and act consistency:

python eval/eval\_padpo.py \\  
    \--pred\_file results/padpo\_generations.json \\  
    \--ref\_file data/test.csv

Other scripts (`final_eval_1.py`, `final_eval_2.py`, `pacons_classifier.py`) provide alternate or ablation-specific evaluation metrics.

---

## **📈 Baselines**

| Model | Script | Description |
| ----- | ----- | ----- |
| **SFT (LLaMA / Qwen)** | `baselines/final_sft_llama.py` | Standard supervised fine-tuning baseline |
| **DPO** | `baselines/final_dpo.py` | Vanilla DPO without persona/act grounding |
| **KTO** | `baselines/final_kto.py` | KTO baseline for preference learning |
| **PPO** | `baselines/final_ppo.py` | Reinforcement learning baseline using PPO |
| **PA-DPO** | `PA-DPO/final_padpo.py` | Persona- and act-guided preference optimization (ours) |

---

## **💬 Citation**

If you use this work, please cite:

---

## **🧩 Key Features**

* ✅ Persona and dialogue act grounding for negotiation dialogues

* ✅ Retrieval-guided hallucination elicitation for controlled preference data

* ✅ Joint DPO \+ Persona-Act preference optimization

* ✅ Modular code: SFT, PA-DPO, preference data, inference, evaluation

---

## **🧑‍💻 Author**

---

## **🏁 License**

This project is for academic research use only.

\---

# MEDIATOR
